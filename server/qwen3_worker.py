#!/usr/bin/env python3
"""
Standalone Qwen3 TTS worker process.

This worker runs in complete isolation to avoid Metal threading conflicts.
It communicates via JSON over stdin/stdout.

Commands:
    {"cmd": "init", "model": "...", "mode": "base", "language": "English", "speaker": "Ryan"}
    {"cmd": "configure", "mode": "...", "language": "...", "speaker": "...", "instruct": "...", "refAudioPath": "...", "refText": "..."}
    {"cmd": "generate", "text": "...", "mode": "...", "language": "...", "speaker": "...", "instruct": "...", "refAudioPath": "...", "refText": "..."}
"""

import sys
import json
import base64
import numpy as np
import random
import inspect
import os

import logging

logging.basicConfig(level=logging.INFO, format="WORKER: %(message)s")

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False


class Worker:
    def __init__(self):
        self.model = None
        self.mode = "base"
        self.language = "english"
        self.speaker = "Ryan"
        self.instruct = ""
        self.ref_audio_path = ""
        self.ref_text = ""
        self.seed = None
        self.max_tokens = None
        self.do_sample = None
        self.speed = None
        self.stt_model = ""
        self.x_vector_only_mode = None
        self.supported_speakers = None
        self.supported_languages = None

    def initialize(
        self,
        model_name,
        mode=None,
        language=None,
        speaker=None,
        seed=None,
        max_tokens=None,
        do_sample=None,
        speed=None,
        stt_model=None,
        x_vector_only_mode=None,
    ):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            if mode:
                self.mode = mode
            if language:
                self.language = language
            if speaker:
                self.speaker = speaker
            if seed is not None:
                self.seed = seed
            if max_tokens is not None:
                self.max_tokens = max_tokens
            if do_sample is not None:
                self.do_sample = do_sample
            if speed is not None:
                self.speed = speed
            if stt_model is not None:
                self.stt_model = stt_model
            if x_vector_only_mode is not None:
                self.x_vector_only_mode = x_vector_only_mode
            self.supported_speakers = (
                self.model.get_supported_speakers()
                if hasattr(self.model, "get_supported_speakers")
                else None
            )
            self.supported_languages = (
                self.model.get_supported_languages()
                if hasattr(self.model, "get_supported_languages")
                else None
            )
            return {"success": True}
        except Exception as e:
            message = str(e)
            if "qwen3_tts" in message and "not supported" in message:
                message = (
                    f"{message}. Update mlx-audio to a version that supports Qwen3-TTS "
                    "(for example: pip install -U mlx-audio)."
                )
            return {"error": message}

    def configure(self, settings: dict):
        for key, value in settings.items():
            if value is None:
                continue
            if key == "mode":
                self.mode = value
            elif key == "language":
                self.language = value
            elif key == "speaker":
                self.speaker = value
            elif key == "instruct":
                self.instruct = value
            elif key == "refAudioPath":
                self.ref_audio_path = value
            elif key == "refText":
                self.ref_text = value
            elif key == "seed":
                self.seed = value
            elif key == "maxTokens":
                self.max_tokens = value
            elif key == "doSample":
                self.do_sample = value
            elif key == "speed":
                self.speed = value
            elif key == "sttModel":
                self.stt_model = value
            elif key == "xVectorOnlyMode":
                self.x_vector_only_mode = value
        return {"success": True}

    def _resolve_settings(self, req: dict):
        mode = req.get("mode") or self.mode
        language = req.get("language") or self.language
        speaker = req.get("speaker") or self.speaker
        instruct = req.get("instruct") or self.instruct
        ref_audio_path = req.get("refAudioPath") or self.ref_audio_path
        ref_text = req.get("refText") or self.ref_text
        seed = req.get("seed")
        if seed is None:
            seed = self.seed
        max_tokens = req.get("maxTokens")
        if max_tokens is None:
            max_tokens = self.max_tokens
        do_sample = req.get("doSample")
        if do_sample is None:
            do_sample = self.do_sample
        speed = req.get("speed")
        if speed is None:
            speed = self.speed
        stt_model = req.get("sttModel")
        if stt_model is None:
            stt_model = self.stt_model
        x_vector_only_mode = req.get("xVectorOnlyMode")
        if x_vector_only_mode is None:
            x_vector_only_mode = self.x_vector_only_mode
        temperature = req.get("temperature")
        top_k = req.get("topK")
        top_p = req.get("topP")
        repetition_penalty = req.get("repetitionPenalty")
        return (
            mode,
            language,
            speaker,
            instruct,
            ref_audio_path,
            ref_text,
            seed,
            max_tokens,
            do_sample,
            speed,
            stt_model,
            x_vector_only_mode,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )

    def _apply_seed(self, seed):
        if seed is None:
            return
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            return
        random.seed(seed_value)
        np.random.seed(seed_value)
        if mx is not None:
            try:
                mx.random.seed(seed_value)
            except Exception:
                pass

    def _normalize_language(self, value):
        if not value:
            return "auto"
        raw = str(value).strip().lower()
        aliases = {
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
        normalized = aliases.get(raw, raw)
        if self.supported_languages:
            supported = {lang.lower() for lang in self.supported_languages}
            if normalized not in supported and normalized != "auto":
                logging.info(
                    f"Unsupported language '{value}'. Using auto. Supported: {self.supported_languages}"
                )
                return "auto"
        return normalized

    def _normalize_speaker(self, speaker, mode):
        if not speaker:
            return None
        if self.supported_speakers is None:
            return speaker
        if not self.supported_speakers:
            logging.info(
                "Model does not expose predefined speakers. Ignoring speaker selection."
            )
            return None
        normalized = str(speaker)
        supported_lower = {s.lower(): s for s in self.supported_speakers}
        if normalized.lower() not in supported_lower:
            if mode == "customVoice":
                return "INVALID"
            logging.info(
                f"Speaker '{speaker}' not supported. Available: {self.supported_speakers}"
            )
            return None
        return supported_lower[normalized.lower()]

    def _resolve_sampling(self, temperature, top_k, top_p, repetition_penalty):
        temp = 0.9 if temperature is None else float(temperature)
        top_k_val = 50 if top_k is None else int(top_k)
        top_p_val = 1.0 if top_p is None else float(top_p)
        rep_penalty = 1.05 if repetition_penalty is None else float(repetition_penalty)
        return temp, top_k_val, top_p_val, rep_penalty

    def _normalize_optional_int(self, value):
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _normalize_optional_float(self, value, min_value=None):
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if min_value is not None and parsed < min_value:
            return None
        return parsed

    def _normalize_optional_bool(self, value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return None

    def _signature_info(self, func):
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return set(), True
        params = sig.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        return set(params.keys()), accepts_kwargs

    def _filter_kwargs(self, params, accepts_kwargs, kwargs):
        cleaned = {key: value for key, value in kwargs.items() if value is not None}
        if accepts_kwargs:
            return cleaned
        return {key: value for key, value in cleaned.items() if key in params}

    def _add_language_kw(self, params, accepts_kwargs, kwargs, language, default_key):
        if language is None:
            return
        if "lang_code" in params:
            kwargs["lang_code"] = language
        elif "language" in params:
            kwargs["language"] = language
        elif accepts_kwargs:
            kwargs[default_key] = language

    def _add_speaker_kw(self, params, accepts_kwargs, kwargs, speaker, default_key):
        if not speaker:
            return
        if "speaker" in params:
            kwargs["speaker"] = speaker
        elif "voice" in params:
            kwargs["voice"] = speaker
        elif accepts_kwargs:
            kwargs[default_key] = speaker

    def _add_ref_audio_kw(self, params, accepts_kwargs, kwargs, ref_audio):
        if ref_audio is None:
            return
        if isinstance(ref_audio, (list, tuple, np.ndarray)) and len(ref_audio) == 0:
            return
        if "ref_audio" in params:
            kwargs["ref_audio"] = ref_audio
        elif "ref_audio_path" in params:
            kwargs["ref_audio_path"] = ref_audio
        elif accepts_kwargs:
            kwargs["ref_audio"] = ref_audio

    def _add_ref_text_kw(self, params, accepts_kwargs, kwargs, ref_text):
        if not ref_text:
            return
        if "ref_text" in params:
            kwargs["ref_text"] = ref_text
        elif "prompt_text" in params:
            kwargs["prompt_text"] = ref_text
        elif accepts_kwargs:
            kwargs["ref_text"] = ref_text

    def _add_max_tokens_kw(self, params, accepts_kwargs, kwargs, max_tokens):
        if max_tokens is None:
            return
        if "max_tokens" in params:
            kwargs["max_tokens"] = max_tokens
        elif "max_new_tokens" in params:
            kwargs["max_new_tokens"] = max_tokens
        elif accepts_kwargs:
            kwargs["max_new_tokens"] = max_tokens

    def _add_kw(self, params, accepts_kwargs, kwargs, key, value):
        if value is None:
            return
        if key in params or accepts_kwargs:
            kwargs[key] = value

    def generate(self, req: dict):
        try:
            if not self.model:
                return {"error": "Not initialized"}
            text = req.get("text", "")
            if not text:
                return {"error": "Missing text"}
            (
                mode,
                language,
                speaker,
                instruct,
                ref_audio_path,
                ref_text,
                seed,
                max_tokens,
                do_sample,
                speed,
                stt_model,
                x_vector_only_mode,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ) = self._resolve_settings(req)
            self._apply_seed(seed)

            language = self._normalize_language(language)
            if mode in ("voiceCloning", "voiceDesign"):
                speaker = None
            speaker = self._normalize_speaker(speaker, mode)
            if speaker == "INVALID":
                return {
                    "error": (
                        f"Speaker '{req.get('speaker')}' not supported. "
                        f"Available: {self.supported_speakers}"
                    )
                }
            (
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ) = self._resolve_sampling(temperature, top_k, top_p, repetition_penalty)
            max_tokens = self._normalize_optional_int(max_tokens)
            do_sample = self._normalize_optional_bool(do_sample)
            speed = self._normalize_optional_float(speed, min_value=0.1)
            x_vector_only_mode = self._normalize_optional_bool(x_vector_only_mode)
            stt_model = stt_model if isinstance(stt_model, str) and stt_model.strip() else None

            if mode == "voiceCloning":
                if not ref_audio_path:
                    return {"error": "Voice cloning requires a reference audio path."}
                if not os.path.isfile(ref_audio_path):
                    return {"error": f"Reference audio file not found: {ref_audio_path}"}
                if not ref_text and not x_vector_only_mode and not stt_model:
                    return {
                        "error": (
                            "Voice cloning requires a reference transcript, or enable "
                            "x-vector-only mode, or set an STT model for auto-transcription."
                        )
                    }

            if mode == "customVoice":
                params, accepts_kwargs = self._signature_info(
                    self.model.generate_custom_voice
                )
                kwargs = {"text": text, "instruct": instruct}
                self._add_speaker_kw(params, accepts_kwargs, kwargs, speaker, "speaker")
                self._add_language_kw(
                    params, accepts_kwargs, kwargs, language, "language"
                )
                self._add_kw(params, accepts_kwargs, kwargs, "temperature", temperature)
                self._add_kw(params, accepts_kwargs, kwargs, "top_k", top_k)
                self._add_kw(params, accepts_kwargs, kwargs, "top_p", top_p)
                self._add_kw(
                    params,
                    accepts_kwargs,
                    kwargs,
                    "repetition_penalty",
                    repetition_penalty,
                )
                self._add_max_tokens_kw(params, accepts_kwargs, kwargs, max_tokens)
                self._add_kw(params, accepts_kwargs, kwargs, "do_sample", do_sample)
                self._add_kw(params, accepts_kwargs, kwargs, "speed", speed)
                iterator = self.model.generate_custom_voice(
                    **self._filter_kwargs(params, accepts_kwargs, kwargs)
                )
            elif mode == "voiceDesign":
                params, accepts_kwargs = self._signature_info(
                    self.model.generate_voice_design
                )
                kwargs = {"text": text, "instruct": instruct}
                self._add_language_kw(
                    params, accepts_kwargs, kwargs, language, "language"
                )
                self._add_kw(params, accepts_kwargs, kwargs, "temperature", temperature)
                self._add_kw(params, accepts_kwargs, kwargs, "top_k", top_k)
                self._add_kw(params, accepts_kwargs, kwargs, "top_p", top_p)
                self._add_kw(
                    params,
                    accepts_kwargs,
                    kwargs,
                    "repetition_penalty",
                    repetition_penalty,
                )
                self._add_max_tokens_kw(params, accepts_kwargs, kwargs, max_tokens)
                self._add_kw(params, accepts_kwargs, kwargs, "do_sample", do_sample)
                self._add_kw(params, accepts_kwargs, kwargs, "speed", speed)
                iterator = self.model.generate_voice_design(
                    **self._filter_kwargs(params, accepts_kwargs, kwargs)
                )
            elif mode == "voiceCloning":
                ref_audio = ref_audio_path if ref_audio_path else None
                params, accepts_kwargs = self._signature_info(self.model.generate)
                if (
                    "ref_audio" not in params
                    and "ref_audio_path" not in params
                    and not accepts_kwargs
                ):
                    return {
                        "error": (
                            "This mlx-audio build does not accept reference audio for "
                            "voice cloning. Update mlx-audio to a newer version."
                        )
                    }
                kwargs = {"text": text}
                if ref_audio and "ref_audio" in params and "ref_audio_path" not in params:
                    # Pass the path string; mlx-audio will load the file internally.
                    ref_audio = str(ref_audio)
                self._add_ref_audio_kw(params, accepts_kwargs, kwargs, ref_audio)
                self._add_ref_text_kw(params, accepts_kwargs, kwargs, ref_text)
                self._add_language_kw(
                    params, accepts_kwargs, kwargs, language, "lang_code"
                )
                self._add_kw(params, accepts_kwargs, kwargs, "temperature", temperature)
                self._add_kw(params, accepts_kwargs, kwargs, "top_k", top_k)
                self._add_kw(params, accepts_kwargs, kwargs, "top_p", top_p)
                self._add_kw(
                    params,
                    accepts_kwargs,
                    kwargs,
                    "repetition_penalty",
                    repetition_penalty,
                )
                self._add_max_tokens_kw(params, accepts_kwargs, kwargs, max_tokens)
                self._add_kw(params, accepts_kwargs, kwargs, "do_sample", do_sample)
                self._add_kw(params, accepts_kwargs, kwargs, "speed", speed)
                self._add_kw(
                    params,
                    accepts_kwargs,
                    kwargs,
                    "x_vector_only_mode",
                    x_vector_only_mode,
                )
                self._add_kw(params, accepts_kwargs, kwargs, "stt_model", stt_model)
                iterator = self.model.generate(
                    **self._filter_kwargs(params, accepts_kwargs, kwargs)
                )
            else:
                params, accepts_kwargs = self._signature_info(self.model.generate)
                kwargs = {"text": text}
                self._add_speaker_kw(params, accepts_kwargs, kwargs, speaker, "voice")
                self._add_language_kw(
                    params, accepts_kwargs, kwargs, language, "lang_code"
                )
                self._add_kw(params, accepts_kwargs, kwargs, "temperature", temperature)
                self._add_kw(params, accepts_kwargs, kwargs, "top_k", top_k)
                self._add_kw(params, accepts_kwargs, kwargs, "top_p", top_p)
                self._add_kw(
                    params,
                    accepts_kwargs,
                    kwargs,
                    "repetition_penalty",
                    repetition_penalty,
                )
                self._add_max_tokens_kw(params, accepts_kwargs, kwargs, max_tokens)
                self._add_kw(params, accepts_kwargs, kwargs, "do_sample", do_sample)
                self._add_kw(params, accepts_kwargs, kwargs, "speed", speed)
                iterator = self.model.generate(
                    **self._filter_kwargs(params, accepts_kwargs, kwargs)
                )

            segments = []
            for result in iterator:
                audio_data = np.array(result.audio, copy=True)
                segments.append(audio_data)

            if not segments:
                return {"error": "No audio"}

            audio = segments[0] if len(segments) == 1 else np.concatenate(segments, axis=0)

            if np.max(np.abs(audio)) < 1e-6:
                return {"error": "Generated audio is silent"}

            audio_int16 = (audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
            return {"success": True, "audio": audio_b64}
        except Exception as e:
            import traceback

            return {"error": f"{str(e)}\n{traceback.format_exc()}"}


def main():
    worker = Worker()

    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            cmd = req.get("cmd")
            if cmd == "init":
                resp = worker.initialize(
                    req["model"],
                    mode=req.get("mode"),
                    language=req.get("language"),
                    speaker=req.get("speaker"),
                    seed=req.get("seed"),
                    max_tokens=req.get("maxTokens"),
                    do_sample=req.get("doSample"),
                    speed=req.get("speed"),
                    stt_model=req.get("sttModel"),
                    x_vector_only_mode=req.get("xVectorOnlyMode"),
                )
            elif cmd == "configure":
                resp = worker.configure(req)
            elif cmd == "generate":
                resp = worker.generate(req)
            else:
                resp = {"error": "Unknown command"}
            print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
