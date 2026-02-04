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
        self.supported_speakers = None
        self.supported_languages = None

    def initialize(self, model_name, mode=None, language=None, speaker=None, seed=None):
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
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ) = self._resolve_settings(req)
            self._apply_seed(seed)

            language = self._normalize_language(language)
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

            if mode == "customVoice":
                iterator = self.model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            elif mode == "voiceDesign":
                iterator = self.model.generate_voice_design(
                    text=text,
                    language=language,
                    instruct=instruct,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            elif mode == "voiceCloning":
                ref_audio = ref_audio_path if ref_audio_path else None
                iterator = self.model.generate(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    lang_code=language,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            else:
                iterator = self.model.generate(
                    text=text,
                    voice=speaker,
                    lang_code=language,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
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
