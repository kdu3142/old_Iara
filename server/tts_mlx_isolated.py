#
# Process-isolated Kokoro TTS service
# Uses a separate process to avoid Metal threading conflicts on Apple Silicon
#

import asyncio
import subprocess
import json
import base64
import sys
import re
import threading
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
        self._device = device

        self._process = None
        self._initialized = False
        self._sentence_streaming_enabled = sentence_streaming_enabled
        self._sentence_min_chars = sentence_min_chars
        self._sentence_min_words = sentence_min_words
        self._sentence_max_chars = sentence_max_chars
        self._sentence_max_words = sentence_max_words

        # Get path to worker script
        self._worker_script = self._get_worker_script_path()

        self._settings = {
            "model": model,
            "voice": voice,
            "language": language,
            "sample_rate": sample_rate,
        }
        self._signature = (model, voice, language, sample_rate)

    def _get_shared_state(self) -> Dict[str, Any]:
        state = self.__class__._shared_state.get(self._signature)
        if not state:
            state = {"process": None, "initialized": False, "lock": threading.Lock()}
            self.__class__._shared_state[self._signature] = state
        return state

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _is_too_short(self, text: str) -> bool:
        if len(text) < self._sentence_min_chars:
            return True
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
        if len(merged) > 1 and self._is_too_short(merged[-1]):
            merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
            merged.pop()
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
                and word.endswith((",", ";", ":", "—", "-"))
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
            self._process = subprocess.Popen(
                [sys.executable, self._worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )
            state["process"] = self._process
            state["initialized"] = False
            logger.info(f"Started {self._model_name} worker process: {self._process.pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            return False

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

                # Read response with timeout
                import select

                ready, _, _ = select.select([self._process.stdout], [], [], 10.0)  # 10 second timeout

                if not ready:
                    return {"error": "Worker response timeout"}

                response_line = self._process.stdout.readline()
                if not response_line:
                    # Check if process died
                    if self._process.poll() is not None:
                        stderr_output = self._process.stderr.read() if self._process.stderr else ""
                        return {"error": f"Worker process died. stderr: {stderr_output}"}
                    return {"error": "No response from worker"}

                response_data = json.loads(response_line.strip())
                # Don't log the full response if it contains audio data (too verbose)
                if "audio" in response_data:
                    logger.debug(
                        f"Worker response: success with {len(response_data.get('audio', ''))} chars of audio data"
                    )
                else:
                    logger.debug(f"Worker response: {response_line.strip()}")
                return response_data

        except Exception as e:
            logger.error(f"Worker communication error: {e}")
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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._send_command,
            {
                "cmd": "init",
                "model": self._model_name,
                "voice": self._voice,
                "language": self._language,
            },
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._send_command, {"cmd": "generate", "text": text}
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

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Initialize worker if needed
            if not await self._initialize_if_needed():
                raise RuntimeError("Failed to initialize Kokoro worker")

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
                audio_bytes = await self._generate_audio_bytes(segment)
                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

                if CHUNK_SIZE <= 0:
                    CHUNK_SIZE = len(audio_bytes) if len(audio_bytes) > 0 else 1
                for i in range(0, len(audio_bytes), CHUNK_SIZE):
                    chunk = audio_bytes[i : i + CHUNK_SIZE]
                    if len(chunk) > 0:
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                        self._attach_sequence_metadata(
                            frame, index=index, total=total_segments, text=segment
                        )
                        yield frame
                        await asyncio.sleep(0.001)

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
