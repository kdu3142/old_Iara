#
# Process-isolated Kokoro TTS service
# Uses a separate process to avoid Metal threading conflicts on Apple Silicon
#

import asyncio
import subprocess
import json
import base64
import sys
import os
import re
import threading
import time
from typing import AsyncGenerator, Optional, List, Dict, Tuple, Any
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class TTSMLXIsolated(TTSService):
    """Completely isolated Kokoro TTS using subprocess to avoid Metal issues."""
    _shared_state: Dict[Tuple[object, ...], Dict[str, Any]] = {}

    def __init__(
        self,
        *,
        model: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        language: Optional[str] = None,
        qwen_settings: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        sample_rate: int = 24000,
        sentence_streaming_enabled: bool = False,
        sentence_min_chars: int = 20,
        sentence_min_words: int = 3,
        sentence_max_chars: int = 220,
        sentence_max_words: int = 40,
        **kwargs,
    ):
        """Initialize the isolated Kokoro TTS service."""
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._language = language
        self._qwen_settings = qwen_settings or {}
        self._device = device

        self._process = None
        self._initialized = False
        self._sentence_streaming_enabled = sentence_streaming_enabled
        self._sentence_min_chars = sentence_min_chars
        self._sentence_min_words = sentence_min_words
        self._sentence_max_chars = sentence_max_chars
        self._sentence_max_words = sentence_max_words
        self._interrupt_id = 0

        # Get path to worker script
        self._worker_script = self._get_worker_script_path()

        self._settings = {
            "model": model,
            "voice": voice,
            "language": language,
            "sample_rate": sample_rate,
            "qwen_settings": self._qwen_settings,
        }
        self._signature = self._build_signature(
            model=model,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            qwen_settings=self._qwen_settings,
        )

    def _build_signature(
        self,
        *,
        model: str,
        voice: Optional[str],
        language: Optional[str],
        sample_rate: int,
        qwen_settings: Dict[str, Any],
    ) -> Tuple[object, ...]:
        if model.startswith("mlx-community/Qwen3-TTS-"):
            # Reuse a single worker per Qwen model to avoid reloads when
            # non-model settings (e.g., speaker/instruct/ref audio) change.
            return (model, sample_rate, "qwen")
        return (
            model,
            voice,
            language,
            sample_rate,
            json.dumps(qwen_settings, sort_keys=True),
        )

    def _get_shared_state(self) -> Dict[str, Any]:
        state = self.__class__._shared_state.get(self._signature)
        if not state:
            state = {"process": None, "initialized": False, "lock": threading.Lock()}
            self.__class__._shared_state[self._signature] = state
        return state

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _is_too_short(self, text: str) -> bool:
        return len(text.split()) < self._sentence_min_words

    def _is_too_long(self, text: str) -> bool:
        if len(text) > self._sentence_max_chars:
            return True
        return len(text.split()) > self._sentence_max_words

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences: List[str] = []
        start = 0
        for match in re.finditer(r"[.!?]+(?:['\")\]]+)?\s+", text):
            end = match.end()
            segment = text[start:end].strip()
            if segment:
                sentences.append(segment)
            start = end
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)
        return sentences

    def _merge_short_segments(self, segments: List[str]) -> List[str]:
        merged: List[str] = []
        buffer = ""
        for segment in segments:
            if not buffer:
                buffer = segment
                continue
            if self._is_too_short(buffer):
                buffer = f"{buffer} {segment}".strip()
            else:
                merged.append(buffer)
                buffer = segment
        if buffer:
            merged.append(buffer)
        return merged

    def _chunk_by_words(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks: List[str] = []
        current: List[str] = []
        for word in words:
            if not current:
                current.append(word)
                continue
            candidate = " ".join(current + [word])
            if (
                len(candidate) > self._sentence_max_chars
                or len(current) + 1 > self._sentence_max_words
            ):
                chunks.append(" ".join(current).strip())
                current = [word]
                continue
            current.append(word)
            if (
                len(" ".join(current)) >= self._sentence_min_chars
                and word.endswith((";", ":", "—", "-"))
            ):
                chunks.append(" ".join(current).strip())
                current = []
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _split_text_for_tts(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        segments = self._split_into_sentences(normalized)
        if not segments:
            return [normalized]
        segments = self._merge_short_segments(segments)
        expanded: List[str] = []
        for segment in segments:
            if self._is_too_long(segment):
                expanded.extend(self._chunk_by_words(segment))
            else:
                expanded.append(segment)
        expanded = [segment.strip() for segment in expanded if segment.strip()]
        return self._merge_short_segments(expanded) or [normalized]

    def _get_worker_script_path(self) -> str:
        """Get the path to the standalone worker script."""
        # Look for kokoro_worker.py in the same directory as this file
        current_dir = Path(__file__).parent
        if self._model_name.startswith("Marvis-AI"):
            worker_path = current_dir / "marvis_worker.py"
        elif self._model_name.startswith("mlx-community/Qwen3-TTS-"):
            worker_path = current_dir / "qwen3_worker.py"
        else:
            worker_path = current_dir / "kokoro_worker.py"

        logger.info(f"Using worker script: {worker_path}")

        if not worker_path.exists():
            raise FileNotFoundError(
                f"Worker script not found at {worker_path}. "
                "Make sure worker script is in the same directory as tts_mlx_isolated.py"
            )

        return str(worker_path)

    def _start_worker(self):
        """Start the worker process."""
        try:
            state = self._get_shared_state()
            existing = state.get("process")
            if existing and existing.poll() is None:
                self._process = existing
                logger.info(f"Reusing {self._model_name} worker process: {self._process.pid}")
                return True
            python_exec = os.environ.get("TTS_WORKER_PYTHON") or sys.executable
            if self._model_name.startswith("mlx-community/Qwen3-TTS-"):
                qwen_python = os.environ.get("QWEN_TTS_PYTHON")
                if qwen_python:
                    python_exec = qwen_python
                else:
                    logger.warning(
                        "QWEN_TTS_PYTHON not set. Using default Python for Qwen worker."
                    )
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            env.setdefault("TRANSFORMERS_VERBOSITY", "error")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._process = subprocess.Popen(
                [python_exec, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env=env,
            )
            state["process"] = self._process
            state["initialized"] = False
            logger.info(f"Started {self._model_name} worker process: {self._process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            return False

    def _reset_worker_state(self, state: Dict[str, Any], reason: str) -> None:
        """Terminate the worker process and clear shared state."""
        try:
            if self._process and self._process.poll() is None:
                logger.warning(f"Resetting worker due to: {reason}")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except Exception:
                    self._process.kill()
        except Exception as exc:
            logger.error(f"Failed to terminate worker cleanly: {exc}")
        finally:
            self._process = None
            state["process"] = None
            state["initialized"] = False
            self._initialized = False

    def _send_command(self, command: dict) -> dict:
        """Send command to worker and get response."""
        try:
            state = self._get_shared_state()
            with state["lock"]:
                if not self._process or self._process.poll() is not None:
                    logger.debug("Starting worker process...")
                    if not self._start_worker():
                        return {"error": "Failed to start worker"}

                # Send command
                cmd_json = json.dumps(command) + "\n"
                logger.debug(f"Sending command: {command}")
                self._process.stdin.write(cmd_json)
                self._process.stdin.flush()

                # Read response with timeout, skipping non-JSON worker output.
                import select

                timeout = 10.0
                if command.get("cmd") == "init":
                    timeout = 240.0
                    if str(command.get("model", "")).startswith("mlx-community/Qwen3-TTS-"):
                        timeout = 600.0
                elif (
                    command.get("cmd") == "generate"
                    and command.get("mode") is not None
                    and command.get("language") is not None
                ):
                    # Qwen generate can take longer on first run (model warmup / caching).
                    timeout = 240.0

                deadline = time.monotonic() + timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._reset_worker_state(state, "response timeout")
                        return {"error": "Worker response timeout"}

                    ready, _, _ = select.select([self._process.stdout], [], [], min(0.5, remaining))
                    if not ready:
                        continue

                    response_line = self._process.stdout.readline()
                    if not response_line:
                        # Check if process died
                        if self._process.poll() is not None:
                            stderr_output = (
                                self._process.stderr.read() if self._process.stderr else ""
                            )
                            self._reset_worker_state(state, "worker process died")
                            return {"error": f"Worker process died. stderr: {stderr_output}"}
                        return {"error": "No response from worker"}

                    response_text = response_line.strip()
                    if not response_text:
                        continue

                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.debug(f"Ignoring non-JSON worker output: {response_text}")
                        continue

                    if (
                        command.get("cmd") == "generate"
                        and response_data.get("success")
                        and "audio" not in response_data
                    ):
                        logger.warning(
                            "Worker returned success without audio; waiting for audio response."
                        )
                        continue

                    # Don't log the full response if it contains audio data (too verbose)
                    if "audio" in response_data:
                        logger.debug(
                            f"Worker response: success with {len(response_data.get('audio', ''))} chars of audio data"
                        )
                    else:
                        logger.debug(f"Worker response: {response_text}")
                    return response_data

        except Exception as e:
            logger.error(f"Worker communication error: {e}")
            self._reset_worker_state(self._get_shared_state(), "communication error")
            # Get stderr if available
            if self._process and self._process.stderr:
                try:
                    stderr_output = self._process.stderr.read()
                    logger.error(f"Worker stderr: {stderr_output}")
                except:
                    pass
            return {"error": str(e)}

    async def _initialize_if_needed(self):
        """Initialize the worker if not already done."""
        state = self._get_shared_state()
        if state.get("initialized"):
            self._initialized = True
            return True

        init_payload = {
            "cmd": "init",
            "model": self._model_name,
            "voice": self._voice,
            "language": self._language,
        }
        if self._model_name.startswith("mlx-community/Qwen3-TTS-"):
            qwen_mode = self._qwen_settings.get("mode")
            init_payload.update(
                {
                    "voice": None,
                    "language": self._qwen_settings.get("language"),
                    "mode": qwen_mode,
                    "seed": self._qwen_settings.get("seed"),
                    "temperature": self._qwen_settings.get("temperature"),
                    "topK": self._qwen_settings.get("topK"),
                    "topP": self._qwen_settings.get("topP"),
                    "repetitionPenalty": self._qwen_settings.get("repetitionPenalty"),
                    "maxTokens": self._qwen_settings.get("maxTokens"),
                    "doSample": self._qwen_settings.get("doSample"),
                    "speed": self._qwen_settings.get("speed"),
                    "sttModel": self._qwen_settings.get("sttModel"),
                    "xVectorOnlyMode": self._qwen_settings.get("xVectorOnlyMode"),
                }
            )
            if qwen_mode in ("base", "customVoice"):
                init_payload["speaker"] = self._qwen_settings.get("speaker")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._send_command,
            init_payload,
        )

        if result.get("success"):
            self._initialized = True
            state["initialized"] = True
            logger.info("Kokoro worker initialized")
            return True
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Worker initialization failed: {error_msg}")

            # Also check if process died
            if self._process and self._process.poll() is not None:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                logger.error(f"Worker process stderr: {stderr_output}")

            return False

    async def _handle_interruption(self, frame, direction):
        await super()._handle_interruption(frame, direction)
        # Increment interrupt id so any in-flight generation is dropped.
        self._interrupt_id += 1

    async def warmup(self) -> bool:
        """Preload the TTS worker and model."""
        return await self._initialize_if_needed()

    def can_generate_metrics(self) -> bool:
        return True

    def _attach_sequence_metadata(
        self, frame: Frame, index: int, total: int, text: str
    ) -> None:
        metadata = {
            "tts_segment_index": index,
            "tts_segment_count": total,
            "tts_segment_text": text,
        }
        try:
            if hasattr(frame, "set_metadata"):
                frame.set_metadata(metadata)
                return
            existing = getattr(frame, "metadata", None)
            if isinstance(existing, dict):
                existing.update(metadata)
            else:
                setattr(frame, "metadata", metadata)
        except Exception:
            pass

    async def _generate_audio_bytes(self, text: str) -> bytes:
        payload = {"cmd": "generate", "text": text}
        if self._model_name.startswith("mlx-community/Qwen3-TTS-"):
            qwen_mode = self._qwen_settings.get("mode")
            payload.update(
                {
                    "mode": qwen_mode,
                    "language": self._qwen_settings.get("language"),
                    "refAudioPath": self._qwen_settings.get("refAudioPath"),
                    "refText": self._qwen_settings.get("refText"),
                    "seed": self._qwen_settings.get("seed"),
                    "temperature": self._qwen_settings.get("temperature"),
                    "topK": self._qwen_settings.get("topK"),
                    "topP": self._qwen_settings.get("topP"),
                    "repetitionPenalty": self._qwen_settings.get("repetitionPenalty"),
                    "maxTokens": self._qwen_settings.get("maxTokens"),
                    "doSample": self._qwen_settings.get("doSample"),
                    "speed": self._qwen_settings.get("speed"),
                    "sttModel": self._qwen_settings.get("sttModel"),
                    "xVectorOnlyMode": self._qwen_settings.get("xVectorOnlyMode"),
                }
            )
            if qwen_mode in ("customVoice", "voiceDesign"):
                payload["instruct"] = self._qwen_settings.get("instruct")
            if qwen_mode in ("base", "customVoice"):
                payload["speaker"] = self._qwen_settings.get("speaker")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._send_command, payload
        )
        if not result.get("success"):
            raise RuntimeError(f"Audio generation failed: {result.get('error')}")
        audio_b64 = result["audio"]
        return base64.b64decode(audio_b64)

    async def warmup_generate(self, text: str = "Hello") -> bool:
        """Warm up model by generating a short audio sample."""
        if not await self._initialize_if_needed():
            return False
        await self._generate_audio_bytes(text)
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech using isolated worker process."""
        logger.debug(f"{self}: Generating TTS [{text}]")
        start_interrupt_id = self._interrupt_id

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Initialize worker if needed
            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro worker")
            if self._interrupt_id != start_interrupt_id:
                return

            segments = (
                self._split_text_for_tts(text)
                if self._sentence_streaming_enabled
                else [text]
            )
            if not segments:
                segments = [text]

            ttfb_stopped = False
            total_segments = len(segments)
            CHUNK_SIZE = self.chunk_size or 0

            for index, segment in enumerate(segments):
                if not segment:
                    continue
                if self._interrupt_id != start_interrupt_id:
                    return
                audio_bytes = await self._generate_audio_bytes(segment)
                if self._interrupt_id != start_interrupt_id:
                    return
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

                if CHUNK_SIZE <= 0:
                    CHUNK_SIZE = len(audio_bytes) if len(audio_bytes) > 0 else 1
                for i in range(0, len(audio_bytes), CHUNK_SIZE):
                    if self._interrupt_id != start_interrupt_id:
                        return
                    chunk = audio_bytes[i : i + CHUNK_SIZE]
                    if len(chunk) > 0:
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        self._attach_sequence_metadata(
                            frame, index=index, total=total_segments, text=segment
                        )
                        yield frame
                        await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    def _cleanup(self):
        """Clean up worker process."""
        self._process = None

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        await self._initialize_if_needed()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean shutdown."""
        self._cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)
