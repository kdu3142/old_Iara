import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Optional

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))
import sitecustomize  # noqa: F401

# Transformers compatibility shim for smart-turn v2 on older versions.
try:
    from transformers.modeling_utils import PreTrainedModel

    def _get_all_tied(self):
        keys = getattr(self, "_all_tied_weights_keys", None)
        if keys is not None:
            return keys
        keys = getattr(self, "_tied_weights_keys", None)
        if keys is None:
            return {}
        if isinstance(keys, dict):
            return keys
        if isinstance(keys, (list, tuple, set)):
            return {k: None for k in keys}
        return {}

    def _set_all_tied(self, value):
        object.__setattr__(self, "_all_tied_weights_keys", value)

    PreTrainedModel.all_tied_weights_keys = property(_get_all_tied, _set_all_tied)
except Exception:
    pass

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
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.frames.frames import LLMTextFrame

from tts_mlx_isolated import TTSMLXIsolated
from assistant_sentence_aggregator import AssistantSentenceAggregator

load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

WARMUP_STATE = {"status": "idle", "error": None}
WARMUP_SIGNATURE = None
WARMUP_TASK: Optional[asyncio.Task] = None


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
    "mlx-community/Qwen3-TTS",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
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
    "turnTaking": {
        "vad": {
            "confidence": 0.7,
            "startSecs": 0.2,
            "stopSecs": 0.2,
            "minVolume": 0.6,
        },
        "smartTurn": {
            "enabled": True,
            "stopSecs": 3,
            "preSpeechMs": 0,
            "maxDurationSecs": 8,
        },
    },
    "qwenTts": {
        "mode": "base",
        "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "language": "english",
        "speaker": "Ryan",
        "instruct": "",
        "refAudioPath": "",
        "refText": "",
        "seed": 1234,
        "temperature": 0.0,
        "topK": 50,
        "topP": 1.0,
        "repetitionPenalty": 1.05,
        "maxTokens": 0,
        "doSample": False,
        "speed": 1.0,
        "sttModel": "",
        "xVectorOnlyMode": False,
    },
    "llmProvider": "openai-compatible",
    "llmBaseUrl": "http://127.0.0.1:1234/v1",
    "llmModel": "gemma-3n-e4b-it-text",
    "llmOllamaThink": True,
    "systemPrompt": DEFAULT_SYSTEM_PROMPT,
}

QWEN_TTS_MODELS = {
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}

QWEN_TTS_SENTINEL = "mlx-community/Qwen3-TTS"

QWEN_TTS_LANGUAGES = {
    "auto",
    "english",
    "chinese",
    "japanese",
    "korean",
    "portuguese",
    "french",
    "german",
    "italian",
    "spanish",
    "russian",
}
QWEN_TTS_SPEAKERS = {"Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"}
QWEN_TTS_MODES = {"base", "customVoice", "voiceDesign", "voiceCloning"}

QWEN_LANGUAGE_ALIASES = {
    "en": "english",
    "english": "english",
    "pt": "portuguese",
    "pt-br": "portuguese",
    "pt_br": "portuguese",
    "ptbr": "portuguese",
    "portuguese (brazil)": "portuguese",
    "portuguese": "portuguese",
    "chinese": "chinese",
    "zh": "chinese",
    "japanese": "japanese",
    "ja": "japanese",
    "korean": "korean",
    "ko": "korean",
    "french": "french",
    "fr": "french",
    "german": "german",
    "de": "german",
    "italian": "italian",
    "it": "italian",
    "spanish": "spanish",
    "es": "spanish",
    "russian": "russian",
    "ru": "russian",
    "auto": "auto",
}


class LoggingOpenAILLMService(OpenAILLMService):
    def __init__(self, *args, log_raw_chunks: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_raw_chunks = log_raw_chunks

    async def get_chat_completions(self, *args, **kwargs):
        chunks = await super().get_chat_completions(*args, **kwargs)
        if not self._log_raw_chunks:
            return chunks

        async def generator():
            async for chunk in chunks:
                payload = None
                try:
                    payload = chunk.model_dump()
                except Exception:
                    try:
                        payload = chunk.to_dict()
                    except Exception:
                        payload = str(chunk)
                logger.debug(f"{self}: raw chunk: {payload}")
                yield chunk

        return generator()


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


def _normalize_qwen_language(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().lower()
    if not normalized:
        return ""
    return QWEN_LANGUAGE_ALIASES.get(normalized, normalized)


def _qwen_model_size(model: str) -> str:
    if "1.7B" in model:
        return "1.7B"
    if "0.6B" in model:
        return "0.6B"
    return ""


def _qwen_model_mode(model: str) -> str:
    if "VoiceDesign" in model:
        return "voiceDesign"
    if "CustomVoice" in model:
        return "customVoice"
    return "base"


def _qwen_pick_model_for_mode(mode: str, current_model: str) -> str:
    size = _qwen_model_size(current_model) or "0.6B"
    if mode == "voiceDesign":
        return "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    if mode == "customVoice":
        return (
            "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
            if size == "1.7B"
            else "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
        )
    # base + voiceCloning use base models
    return (
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        if size == "1.7B"
        else "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    )


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

    is_qwen_model = config["ttsModel"].startswith("mlx-community/Qwen3-TTS")
    if is_qwen_model:
        qwen_config = config.get("qwenTts") if isinstance(config.get("qwenTts"), dict) else {}
        config["qwenTts"] = DEFAULT_CONFIG["qwenTts"].copy()
        for key in config["qwenTts"].keys():
            if key in qwen_config:
                config["qwenTts"][key] = qwen_config[key]
        if config["ttsModel"].startswith("mlx-community/Qwen3-TTS-"):
            config["qwenTts"]["model"] = config["ttsModel"]
        config["ttsModel"] = QWEN_TTS_SENTINEL
        if config["qwenTts"]["mode"] not in QWEN_TTS_MODES:
            config["qwenTts"]["mode"] = DEFAULT_CONFIG["qwenTts"]["mode"]
        if config["qwenTts"]["model"] not in QWEN_TTS_MODELS:
            config["qwenTts"]["model"] = DEFAULT_CONFIG["qwenTts"]["model"]
        expected_mode = _qwen_model_mode(config["qwenTts"]["model"])
        if config["qwenTts"]["mode"] != expected_mode:
            config["qwenTts"]["model"] = _qwen_pick_model_for_mode(
                config["qwenTts"]["mode"], config["qwenTts"]["model"]
            )
        normalized_qwen_language = _normalize_qwen_language(config["qwenTts"].get("language"))
        if normalized_qwen_language not in QWEN_TTS_LANGUAGES:
            normalized_qwen_language = DEFAULT_CONFIG["qwenTts"]["language"]
        config["qwenTts"]["language"] = normalized_qwen_language
        if config["qwenTts"]["speaker"] not in QWEN_TTS_SPEAKERS:
            config["qwenTts"]["speaker"] = DEFAULT_CONFIG["qwenTts"]["speaker"]
        if not isinstance(config["qwenTts"].get("instruct"), str):
            config["qwenTts"]["instruct"] = ""
        if not isinstance(config["qwenTts"].get("refAudioPath"), str):
            config["qwenTts"]["refAudioPath"] = ""
        if not isinstance(config["qwenTts"].get("refText"), str):
            config["qwenTts"]["refText"] = ""
        seed = config["qwenTts"].get("seed")
        if isinstance(seed, bool):
            seed = int(seed)
        if not isinstance(seed, int):
            seed = DEFAULT_CONFIG["qwenTts"]["seed"]
        config["qwenTts"]["seed"] = seed
        temperature = config["qwenTts"].get("temperature")
        if isinstance(temperature, bool):
            temperature = float(temperature)
        if not isinstance(temperature, (int, float)):
            temperature = DEFAULT_CONFIG["qwenTts"]["temperature"]
        config["qwenTts"]["temperature"] = float(temperature)
        top_k = config["qwenTts"].get("topK")
        if isinstance(top_k, bool):
            top_k = int(top_k)
        if not isinstance(top_k, int):
            top_k = DEFAULT_CONFIG["qwenTts"]["topK"]
        config["qwenTts"]["topK"] = max(0, top_k)
        top_p = config["qwenTts"].get("topP")
        if isinstance(top_p, bool):
            top_p = float(top_p)
        if not isinstance(top_p, (int, float)):
            top_p = DEFAULT_CONFIG["qwenTts"]["topP"]
        config["qwenTts"]["topP"] = float(top_p)
        repetition_penalty = config["qwenTts"].get("repetitionPenalty")
        if isinstance(repetition_penalty, bool):
            repetition_penalty = float(repetition_penalty)
        if not isinstance(repetition_penalty, (int, float)):
            repetition_penalty = DEFAULT_CONFIG["qwenTts"]["repetitionPenalty"]
        config["qwenTts"]["repetitionPenalty"] = float(repetition_penalty)
        max_tokens = config["qwenTts"].get("maxTokens")
        if isinstance(max_tokens, bool):
            max_tokens = int(max_tokens)
        if not isinstance(max_tokens, int):
            max_tokens = DEFAULT_CONFIG["qwenTts"]["maxTokens"]
        config["qwenTts"]["maxTokens"] = max(0, max_tokens)
        do_sample = config["qwenTts"].get("doSample")
        if isinstance(do_sample, str):
            normalized = do_sample.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                do_sample = True
            elif normalized in {"false", "0", "no", "off"}:
                do_sample = False
        if not isinstance(do_sample, bool):
            do_sample = DEFAULT_CONFIG["qwenTts"]["doSample"]
        config["qwenTts"]["doSample"] = do_sample
        speed = config["qwenTts"].get("speed")
        if isinstance(speed, bool):
            speed = float(speed)
        if not isinstance(speed, (int, float)):
            speed = DEFAULT_CONFIG["qwenTts"]["speed"]
        config["qwenTts"]["speed"] = float(speed)
        stt_model = config["qwenTts"].get("sttModel")
        if not isinstance(stt_model, str):
            stt_model = DEFAULT_CONFIG["qwenTts"]["sttModel"]
        config["qwenTts"]["sttModel"] = stt_model
        x_vector_only_mode = config["qwenTts"].get("xVectorOnlyMode")
        if isinstance(x_vector_only_mode, str):
            normalized = x_vector_only_mode.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                x_vector_only_mode = True
            elif normalized in {"false", "0", "no", "off"}:
                x_vector_only_mode = False
        if not isinstance(x_vector_only_mode, bool):
            x_vector_only_mode = DEFAULT_CONFIG["qwenTts"]["xVectorOnlyMode"]
        config["qwenTts"]["xVectorOnlyMode"] = x_vector_only_mode
    else:
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

    def _parse_bool(value: object, fallback: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return fallback

    def _parse_number(value: object, fallback: float) -> float:
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return fallback
        return fallback

    def _clamp_min(value: float, minimum: float) -> float:
        return value if value >= minimum else minimum

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    turn_defaults = DEFAULT_CONFIG["turnTaking"]
    turn_config = config.get("turnTaking") if isinstance(config.get("turnTaking"), dict) else {}
    vad_config = turn_config.get("vad") if isinstance(turn_config.get("vad"), dict) else {}
    smart_config = (
        turn_config.get("smartTurn") if isinstance(turn_config.get("smartTurn"), dict) else {}
    )

    vad_confidence = _clamp01(
        _parse_number(vad_config.get("confidence"), turn_defaults["vad"]["confidence"])
    )
    vad_start_secs = _clamp_min(
        _parse_number(vad_config.get("startSecs"), turn_defaults["vad"]["startSecs"]), 0.0
    )
    vad_stop_secs = _clamp_min(
        _parse_number(vad_config.get("stopSecs"), turn_defaults["vad"]["stopSecs"]), 0.0
    )
    vad_min_volume = _clamp01(
        _parse_number(vad_config.get("minVolume"), turn_defaults["vad"]["minVolume"])
    )

    smart_enabled = _parse_bool(
        smart_config.get("enabled"), turn_defaults["smartTurn"]["enabled"]
    )
    smart_stop_secs = _clamp_min(
        _parse_number(smart_config.get("stopSecs"), turn_defaults["smartTurn"]["stopSecs"]), 0.0
    )
    smart_pre_speech_ms = _clamp_min(
        _parse_number(
            smart_config.get("preSpeechMs"), turn_defaults["smartTurn"]["preSpeechMs"]
        ),
        0.0,
    )
    smart_max_duration_secs = _clamp_min(
        _parse_number(
            smart_config.get("maxDurationSecs"),
            turn_defaults["smartTurn"]["maxDurationSecs"],
        ),
        0.0,
    )

    config["turnTaking"] = {
        "vad": {
            "confidence": vad_confidence,
            "startSecs": vad_start_secs,
            "stopSecs": vad_stop_secs,
            "minVolume": vad_min_volume,
        },
        "smartTurn": {
            "enabled": smart_enabled,
            "stopSecs": smart_stop_secs,
            "preSpeechMs": smart_pre_speech_ms,
            "maxDurationSecs": smart_max_duration_secs,
        },
    }

    if config["llmProvider"] not in {"openai-compatible", "ollama"}:
        config["llmProvider"] = DEFAULT_CONFIG["llmProvider"]

    if not isinstance(config.get("llmBaseUrl"), str) or not config["llmBaseUrl"]:
        config["llmBaseUrl"] = DEFAULT_CONFIG["llmBaseUrl"]

    if not isinstance(config.get("llmModel"), str) or not config["llmModel"]:
        config["llmModel"] = DEFAULT_CONFIG["llmModel"]

    ollama_think = config.get("llmOllamaThink")
    if isinstance(ollama_think, str):
        normalized = ollama_think.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            ollama_think = True
        elif normalized in {"false", "0", "no", "off"}:
            ollama_think = False
    if not isinstance(ollama_think, bool):
        ollama_think = DEFAULT_CONFIG["llmOllamaThink"]
    config["llmOllamaThink"] = ollama_think

    if not isinstance(config.get("systemPrompt"), str) or not config["systemPrompt"].strip():
        config["systemPrompt"] = DEFAULT_CONFIG["systemPrompt"]

    if not isinstance(config.get("ttsSentenceStreaming"), bool):
        config["ttsSentenceStreaming"] = DEFAULT_CONFIG["ttsSentenceStreaming"]

    return config


def _warmup_signature(config: dict) -> tuple:
    return (
        config.get("whisperModel"),
        config.get("whisperLanguage"),
        config.get("ttsModel"),
        config.get("ttsLanguage"),
        config.get("ttsVoice"),
        json.dumps(config.get("qwenTts", {}), sort_keys=True),
    )


async def _warmup_stt_service(stt: WhisperSTTServiceMLX) -> None:
    # 1 second of 16-bit PCM silence at 16kHz.
    silence = b"\x00" * (16000 * 2)
    async for _ in stt.run_stt(silence):
        break


async def _warmup_models(config: dict) -> None:
    stt = WhisperSTTServiceMLX(
        model=config["whisperModel"],
        language=LANGUAGE_MAP.get(config["whisperLanguage"], Language.EN),
    )

    is_qwen_model = config["ttsModel"].startswith("mlx-community/Qwen3-TTS")
    tts_language = (
        config.get("ttsLanguage") if config["ttsModel"].startswith("Marvis-AI") else None
    )
    qwen_settings = config.get("qwenTts") if is_qwen_model else None
    tts_model_name = (
        config["qwenTts"]["model"]
        if is_qwen_model and isinstance(config.get("qwenTts"), dict)
        else config["ttsModel"]
    )
    tts = TTSMLXIsolated(
        model=tts_model_name,
        voice=None if is_qwen_model else config["ttsVoice"],
        language=tts_language,
        qwen_settings=qwen_settings,
        sample_rate=24000,
        sentence_streaming_enabled=config["ttsSentenceStreaming"],
        aggregate_sentences=False,
    )

    await asyncio.gather(_warmup_stt_service(stt), tts.warmup())
    qwen_mode = None
    if is_qwen_model and isinstance(qwen_settings, dict):
        qwen_mode = qwen_settings.get("mode")
    if qwen_mode != "voiceCloning":
        await tts.warmup_generate("Hello")


async def run_bot(webrtc_connection):
    config = _load_active_config()
    turn_config = config.get("turnTaking", {})
    vad_config = turn_config.get("vad", {}) if isinstance(turn_config, dict) else {}
    smart_config = (
        turn_config.get("smartTurn", {}) if isinstance(turn_config, dict) else {}
    )
    vad_params = VADParams(
        confidence=vad_config.get("confidence", DEFAULT_CONFIG["turnTaking"]["vad"]["confidence"]),
        start_secs=vad_config.get("startSecs", DEFAULT_CONFIG["turnTaking"]["vad"]["startSecs"]),
        stop_secs=vad_config.get("stopSecs", DEFAULT_CONFIG["turnTaking"]["vad"]["stopSecs"]),
        min_volume=vad_config.get("minVolume", DEFAULT_CONFIG["turnTaking"]["vad"]["minVolume"]),
    )
    smart_turn_enabled = smart_config.get(
        "enabled", DEFAULT_CONFIG["turnTaking"]["smartTurn"]["enabled"]
    )
    smart_turn_params = (
        SmartTurnParams(
            stop_secs=smart_config.get(
                "stopSecs", DEFAULT_CONFIG["turnTaking"]["smartTurn"]["stopSecs"]
            ),
            pre_speech_ms=smart_config.get(
                "preSpeechMs", DEFAULT_CONFIG["turnTaking"]["smartTurn"]["preSpeechMs"]
            ),
            max_duration_secs=smart_config.get(
                "maxDurationSecs", DEFAULT_CONFIG["turnTaking"]["smartTurn"]["maxDurationSecs"]
            ),
        )
        if smart_turn_enabled
        else None
    )
    turn_analyzer = (
        LocalSmartTurnAnalyzerV2(
            smart_turn_model_path="",  # Download from HuggingFace
            params=smart_turn_params,
        )
        if smart_turn_enabled
        else None
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
            turn_analyzer=turn_analyzer,
        ),
    )
    stt = WhisperSTTServiceMLX(
        model=config["whisperModel"],
        language=LANGUAGE_MAP.get(config["whisperLanguage"], Language.EN),
    )

    is_qwen_model = config["ttsModel"].startswith("mlx-community/Qwen3-TTS")
    tts_language = config.get("ttsLanguage") if config["ttsModel"].startswith("Marvis-AI") else None
    qwen_settings = config.get("qwenTts") if is_qwen_model else None
    tts_model_name = (
        config["qwenTts"]["model"]
        if is_qwen_model and isinstance(config.get("qwenTts"), dict)
        else config["ttsModel"]
    )
    tts = TTSMLXIsolated(
        model=tts_model_name,
        voice=None if is_qwen_model else config["ttsVoice"],
        language=tts_language,
        qwen_settings=qwen_settings,
        sample_rate=24000,
        sentence_streaming_enabled=config["ttsSentenceStreaming"],
        aggregate_sentences=False,
    )

    llm_params = None
    if config["llmProvider"] == "ollama":
        ollama_think = config.get("llmOllamaThink", True)
        logger.info(f"Ollama think enabled: {ollama_think}")
        llm_params = BaseOpenAILLMService.InputParams(
            extra={"extra_body": {"think": ollama_think}}
        )

    llm_service = LoggingOpenAILLMService if config["llmProvider"] == "ollama" else OpenAILLMService
    llm = llm_service(
        api_key="dummyKey",
        model=config["llmModel"],
        # model="google/gemma-3-12b",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url=config["llmBaseUrl"],
        max_tokens=4096,
        params=llm_params,
        log_raw_chunks=False,
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
    assistant_sentence_aggregator = AssistantSentenceAggregator(
        enabled=config["ttsSentenceStreaming"],
        min_words=3,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            assistant_sentence_aggregator,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    class FilteredRTVIObserver(RTVIObserver):
        def __init__(self, *args, llm_sources=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._llm_sources = set(llm_sources or [])

        async def on_push_frame(self, data):
            if isinstance(data.frame, LLMTextFrame):
                if data.source not in self._llm_sources:
                    return
            await super().on_push_frame(data)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[FilteredRTVIObserver(rtvi, llm_sources={assistant_sentence_aggregator})],
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


async def _run_warmup(config: dict, signature: tuple) -> None:
    global WARMUP_SIGNATURE, WARMUP_TASK
    try:
        await _warmup_models(config)
        WARMUP_STATE["status"] = "ready"
        WARMUP_STATE["error"] = None
        WARMUP_SIGNATURE = signature
    except Exception as exc:
        logger.exception("Warmup failed")
        WARMUP_STATE["status"] = "error"
        WARMUP_STATE["error"] = str(exc)
    finally:
        WARMUP_TASK = None


@app.post("/api/warmup")
async def warmup_models():
    global WARMUP_SIGNATURE, WARMUP_TASK

    config = _load_active_config()
    signature = _warmup_signature(config)

    if WARMUP_STATE["status"] == "ready" and signature == WARMUP_SIGNATURE:
        return WARMUP_STATE

    if WARMUP_TASK:
        if signature != WARMUP_SIGNATURE:
            WARMUP_TASK.cancel()
            WARMUP_TASK = None
        else:
            return WARMUP_STATE

    WARMUP_STATE["status"] = "loading"
    WARMUP_STATE["error"] = None
    WARMUP_SIGNATURE = signature
    WARMUP_TASK = asyncio.create_task(_run_warmup(config, signature))

    return WARMUP_STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    def _env_port(default: int) -> int:
        raw = os.environ.get("IARA_SERVER_PORT") or os.environ.get("PORT")
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid port in environment: %s", raw)
            return default

    default_host = os.environ.get("IARA_SERVER_HOST", "localhost")
    default_port = _env_port(7860)

    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host",
        default=default_host,
        help=f"Host for HTTP server (default: {default_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port for HTTP server (default: {default_port})",
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
