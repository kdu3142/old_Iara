import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Optional

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

from tts_mlx_isolated import TTSMLXIsolated

load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


DEFAULT_SYSTEM_PROMPT = (
    "You are Pipecat, a friendly, helpful chatbot.\n\n"
    "Your input is text transcribed in realtime from the user's voice. There may be "
    "transcription errors. Adjust your responses automatically to account for these "
    "errors.\n\n"
    "Your output will be converted to audio so don't include special characters in "
    "your answers and do not use any markdown or special formatting.\n\n"
    "Respond to what the user said in a creative and helpful way. Keep your responses "
    "brief unless you are explicitly asked for long or detailed responses. Normally "
    "you should use one or two sentences at most. Keep each sentence short. Prefer "
    "simple sentences. Try not to use long sentences with multiple comma clauses.\n\n"
    'Start the conversation by saying, "Hello, I\'m Pipecat!" Then stop and wait '
    "for the user."
)


CONFIG_PATH = os.environ.get(
    "VOICE_UI_CONFIG_PATH",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "client", ".config", "voice-ui-config.json")
    ),
)

SUPPORTED_WHISPER_MODELS = {
    MLXModel.TINY.value,
    MLXModel.MEDIUM.value,
    MLXModel.LARGE_V3.value,
    MLXModel.LARGE_V3_TURBO.value,
    MLXModel.DISTIL_LARGE_V3.value,
    MLXModel.LARGE_V3_TURBO_Q4.value,
}

SUPPORTED_WHISPER_LANGUAGES = {
    "ar",
    "bn",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sv",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "zh",
}

LANGUAGE_MAP = {
    "ar": Language.AR,
    "bn": Language.BN,
    "cs": Language.CS,
    "da": Language.DA,
    "de": Language.DE,
    "el": Language.EL,
    "en": Language.EN,
    "es": Language.ES,
    "fa": Language.FA,
    "fi": Language.FI,
    "fr": Language.FR,
    "hi": Language.HI,
    "hu": Language.HU,
    "id": Language.ID,
    "it": Language.IT,
    "ja": Language.JA,
    "ko": Language.KO,
    "nl": Language.NL,
    "pl": Language.PL,
    "pt": Language.PT,
    "ro": Language.RO,
    "ru": Language.RU,
    "sk": Language.SK,
    "sv": Language.SV,
    "th": Language.TH,
    "tr": Language.TR,
    "uk": Language.UK,
    "ur": Language.UR,
    "vi": Language.VI,
    "zh": Language.ZH,
}

SUPPORTED_TTS_MODELS = {
    "mlx-community/Kokoro-82M-bf16",
    "Marvis-AI/marvis-tts-250m-v0.1",
}

KOKORO_VOICE_LIST = [
    ("af_heart", "en-US"),
    ("af_alloy", "en-US"),
    ("af_aoede", "en-US"),
    ("af_bella", "en-US"),
    ("af_jessica", "en-US"),
    ("af_kore", "en-US"),
    ("af_nicole", "en-US"),
    ("af_nova", "en-US"),
    ("af_river", "en-US"),
    ("af_sarah", "en-US"),
    ("af_sky", "en-US"),
    ("am_adam", "en-US"),
    ("am_echo", "en-US"),
    ("am_eric", "en-US"),
    ("am_fenrir", "en-US"),
    ("am_liam", "en-US"),
    ("am_michael", "en-US"),
    ("am_onyx", "en-US"),
    ("am_puck", "en-US"),
    ("am_santa", "en-US"),
    ("bf_alice", "en-GB"),
    ("bf_emma", "en-GB"),
    ("bf_isabella", "en-GB"),
    ("bf_lily", "en-GB"),
    ("bm_daniel", "en-GB"),
    ("bm_fable", "en-GB"),
    ("bm_george", "en-GB"),
    ("bm_lewis", "en-GB"),
    ("jf_alpha", "ja"),
    ("jf_gongitsune", "ja"),
    ("jf_nezumi", "ja"),
    ("jf_tebukuro", "ja"),
    ("jm_kumo", "ja"),
    ("zf_xiaobei", "zh"),
    ("zf_xiaoni", "zh"),
    ("zf_xiaoxiao", "zh"),
    ("zf_xiaoyi", "zh"),
    ("zm_yunjian", "zh"),
    ("zm_yunxi", "zh"),
    ("zm_yunxia", "zh"),
    ("zm_yunyang", "zh"),
    ("ef_dora", "es"),
    ("em_alex", "es"),
    ("em_santa", "es"),
    ("ff_siwis", "fr"),
    ("hf_alpha", "hi"),
    ("hf_beta", "hi"),
    ("hm_omega", "hi"),
    ("hm_psi", "hi"),
    ("if_sara", "it"),
    ("im_nicola", "it"),
    ("pf_dora", "pt-BR"),
    ("pm_alex", "pt-BR"),
    ("pm_santa", "pt-BR"),
]

KOKORO_VOICES = {voice_id for voice_id, _ in KOKORO_VOICE_LIST}
KOKORO_VOICE_LANGUAGE = {
    voice_id: language for voice_id, language in KOKORO_VOICE_LIST
}
KOKORO_VOICES_BY_LANGUAGE = {}
for voice_id, language in KOKORO_VOICE_LIST:
    KOKORO_VOICES_BY_LANGUAGE.setdefault(language, []).append(voice_id)

MARVIS_VOICES = {"conversational_a"}

SUPPORTED_TTS_LANGUAGES = set(KOKORO_VOICES_BY_LANGUAGE.keys())
TTS_LANGUAGE_ALIASES = {
    "pt": "pt-BR",
    "pt-br": "pt-BR",
    "pt br": "pt-BR",
    "pt_br": "pt-BR",
    "ptbr": "pt-BR",
}

DEFAULT_CONFIG = {
    "whisperModel": MLXModel.LARGE_V3_TURBO_Q4.value,
    "whisperLanguage": "en",
    "ttsModel": "mlx-community/Kokoro-82M-bf16",
    "ttsLanguage": "en-US",
    "ttsVoice": "af_heart",
    "ttsSentenceStreaming": False,
    "llmProvider": "openai-compatible",
    "llmBaseUrl": "http://127.0.0.1:1234/v1",
    "llmModel": "gemma-3n-e4b-it-text",
    "systemPrompt": DEFAULT_SYSTEM_PROMPT,
}


def _normalize_tts_language(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    return TTS_LANGUAGE_ALIASES.get(lowered, normalized)


def _default_voice_for_language(language: str) -> Optional[str]:
    voices = KOKORO_VOICES_BY_LANGUAGE.get(language)
    if voices:
        return voices[0]
    return None


def _load_active_config() -> dict:
    try:
        with open(CONFIG_PATH, "r") as file:
            parsed = json.load(file)
        presets = parsed.get("presets", [])
        active_id = parsed.get("activePresetId")
        active = None
        for preset in presets:
            if preset.get("id") == active_id:
                active = preset
                break
        if not active and presets:
            active = presets[0]
        values = active.get("values", {}) if active else {}
    except Exception:
        values = {}

    config = DEFAULT_CONFIG.copy()
    for key in config.keys():
        if key in values:
            config[key] = values[key]

    if config["whisperModel"] not in SUPPORTED_WHISPER_MODELS:
        config["whisperModel"] = DEFAULT_CONFIG["whisperModel"]

    if config["whisperLanguage"] not in SUPPORTED_WHISPER_LANGUAGES:
        config["whisperLanguage"] = DEFAULT_CONFIG["whisperLanguage"]

    if config["ttsModel"] not in SUPPORTED_TTS_MODELS:
        config["ttsModel"] = DEFAULT_CONFIG["ttsModel"]

    normalized_tts_language = _normalize_tts_language(config.get("ttsLanguage"))
    if normalized_tts_language not in SUPPORTED_TTS_LANGUAGES:
        inferred_language = KOKORO_VOICE_LANGUAGE.get(config.get("ttsVoice", ""))
        config["ttsLanguage"] = inferred_language or DEFAULT_CONFIG["ttsLanguage"]
    else:
        config["ttsLanguage"] = normalized_tts_language

    if config["ttsModel"].startswith("Marvis-AI"):
        if config["ttsVoice"] not in MARVIS_VOICES:
            config["ttsVoice"] = None
    else:
        if config["ttsVoice"] not in KOKORO_VOICES:
            config["ttsVoice"] = (
                _default_voice_for_language(config["ttsLanguage"])
                or DEFAULT_CONFIG["ttsVoice"]
            )
        else:
            expected_language = KOKORO_VOICE_LANGUAGE.get(config["ttsVoice"])
            if expected_language and expected_language != config["ttsLanguage"]:
                config["ttsVoice"] = (
                    _default_voice_for_language(config["ttsLanguage"])
                    or DEFAULT_CONFIG["ttsVoice"]
                )

    if config["llmProvider"] not in {"openai-compatible", "ollama"}:
        config["llmProvider"] = DEFAULT_CONFIG["llmProvider"]

    if not isinstance(config.get("llmBaseUrl"), str) or not config["llmBaseUrl"]:
        config["llmBaseUrl"] = DEFAULT_CONFIG["llmBaseUrl"]

    if not isinstance(config.get("llmModel"), str) or not config["llmModel"]:
        config["llmModel"] = DEFAULT_CONFIG["llmModel"]

    if not isinstance(config.get("systemPrompt"), str) or not config["systemPrompt"].strip():
        config["systemPrompt"] = DEFAULT_CONFIG["systemPrompt"]

    if not isinstance(config.get("ttsSentenceStreaming"), bool):
        config["ttsSentenceStreaming"] = DEFAULT_CONFIG["ttsSentenceStreaming"]

    return config


async def run_bot(webrtc_connection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path="",  # Download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )

    config = _load_active_config()
    stt = WhisperSTTServiceMLX(
        model=config["whisperModel"],
        language=LANGUAGE_MAP.get(config["whisperLanguage"], Language.EN),
    )

    tts = TTSMLXIsolated(
        model=config["ttsModel"],
        voice=config["ttsVoice"],
        sample_rate=24000,
        sentence_streaming_enabled=config["ttsSentenceStreaming"],
    )

    llm = OpenAILLMService(
        api_key="dummyKey",
        model=config["llmModel"],
        # model="google/gemma-3-12b",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url=config["llmBaseUrl"],
        max_tokens=4096,
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": config["systemPrompt"],
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # a de minimus value since we don't expect any transcript aggregation to be
        # necessary.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
