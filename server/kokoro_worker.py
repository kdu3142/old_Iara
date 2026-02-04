#!/usr/bin/env python3
"""
Standalone Kokoro TTS worker process.

This worker runs in complete isolation to avoid Metal threading conflicts.
It communicates via JSON over stdin/stdout.

Usage:
    python kokoro_worker.py

Commands:
    {"cmd": "init", "model": "mlx-community/Kokoro-82M-bf16", "voice": "af_heart"}
    {"cmd": "generate", "text": "Hello world"}
"""

import sys
import json
import base64
import traceback
import inspect
import numpy as np

# Add logging to worker
import logging
logging.basicConfig(level=logging.INFO, format='WORKER: %(message)s')

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class Worker:
    def __init__(self):
        self.model = None
        self.voice = None
        self.language = None
        self.available_voices = None
        self.inferred_language = None
        self.lang_code = None
        
    def initialize(self, model_name, voice, language=None):
        if not MLX_AVAILABLE:
            return {"error": "MLX not available"}
        try:
            self.model = load_model(model_name)
            self.voice = voice
            self.language = language
            self.inferred_language = self._infer_language_from_voice(voice)
            self.lang_code = self._resolve_lang_code()
            self.available_voices = self._get_available_voices()
            if self.available_voices is not None:
                logging.info(f"Available voices: {self.available_voices}")
                if voice not in self.available_voices:
                    return {
                        "error": f"Voice '{voice}' not available in model.",
                        "availableVoices": self.available_voices,
                    }
            # Test generation to ensure everything works (try with lang_code, fallback without)
            try:
                list(self.model.generate(**self._build_generate_kwargs(text="test")))
            except TypeError:
                list(self.model.generate(text="test", voice=self.voice, speed=1.0))
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

    def _get_available_voices(self):
        if not self.model:
            return None
        for attr in ("voices", "speakers", "speaker_ids", "speaker_list", "voice_map", "speaker_map"):
            if hasattr(self.model, attr):
                value = getattr(self.model, attr)
                if isinstance(value, dict):
                    return sorted(value.keys())
                if isinstance(value, (list, tuple, set)):
                    return sorted(list(value))
        return None

    def _infer_language_from_voice(self, voice: str):
        if not voice or not isinstance(voice, str):
            return None
        if voice.startswith(("af_", "am_")):
            return "a"
        if voice.startswith(("bf_", "bm_")):
            return "b"
        if voice.startswith(("jf_", "jm_")):
            return "j"
        if voice.startswith(("zf_", "zm_")):
            return "z"
        if voice.startswith(("ef_", "em_")):
            return "e"
        if voice.startswith("ff_"):
            return "f"
        if voice.startswith(("hf_", "hm_")):
            return "h"
        if voice.startswith(("if_", "im_")):
            return "i"
        if voice.startswith(("pf_", "pm_")):
            return "p"
        return None

    def _normalize_language_to_code(self, language: str):
        if not language or not isinstance(language, str):
            return None
        normalized = language.strip().lower()
        if normalized in {"a", "b", "j", "z", "e", "f", "h", "i", "p"}:
            return normalized
        mapping = {
            "en-us": "a",
            "en": "a",
            "en-gb": "b",
            "ja": "j",
            "zh": "z",
            "es": "e",
            "fr": "f",
            "hi": "h",
            "it": "i",
            "pt-br": "p",
            "pt": "p",
            "pt_br": "p",
            "ptbr": "p",
        }
        return mapping.get(normalized)

    def _resolve_lang_code(self):
        # Hard-force Brazilian Portuguese for pf_/pm_ voices.
        if self.voice and self.voice.startswith(("pf_", "pm_")):
            return "p"
        return (
            self._normalize_language_to_code(self.language)
            or self.inferred_language
        )

    def _build_generate_kwargs(self, *, text):
        kwargs = {"text": text, "voice": self.voice, "speed": 1.0}
        lang_code = self.lang_code or self._resolve_lang_code()
        if lang_code:
            kwargs["lang_code"] = lang_code
        return kwargs
    
    def generate(self, text):
        try:
            if not self.model:
                return {"error": "Not initialized"}
            
            segments = []
            lang_code = self.lang_code or self._resolve_lang_code()
            logging.info(f"Using lang_code={lang_code} for voice={self.voice}")
            try:
                iterator = self.model.generate(**self._build_generate_kwargs(text=text))
            except TypeError:
                iterator = self.model.generate(text=text, voice=self.voice, speed=1.0)
            for result in iterator:
                # Convert MLX array to numpy immediately
                audio_data = np.array(result.audio, copy=True)
                print(f"Generated segment shape: {audio_data.shape}, min: {audio_data.min():.4f}, max: {audio_data.max():.4f}", file=sys.stderr)
                segments.append(audio_data)
            
            if not segments:
                return {"error": "No audio"}
                
            # Concatenate all segments
            if len(segments) == 1:
                audio = segments[0]
            else:
                audio = np.concatenate(segments, axis=0)
            
            print(f"Final audio shape: {audio.shape}, min: {audio.min():.4f}, max: {audio.max():.4f}", file=sys.stderr)
            
            # Check if audio is silent
            if np.max(np.abs(audio)) < 1e-6:
                return {"error": "Generated audio is silent"}
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode()
            
            return {"success": True, "audio": audio_b64}
        except Exception as e:
            import traceback
            return {"error": f"{str(e)}\n{traceback.format_exc()}"}


def main():
    """Main worker loop - reads commands from stdin, writes responses to stdout."""
    worker = Worker()
    
    for line in sys.stdin:
        try:
            req = json.loads(line.strip())
            if req["cmd"] == "init":
                resp = worker.initialize(
                    req["model"],
                    req["voice"],
                    req.get("language"),
                )
            elif req["cmd"] == "generate":
                resp = worker.generate(req["text"])
            else:
                resp = {"error": "Unknown command"}
            print(json.dumps(resp), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()