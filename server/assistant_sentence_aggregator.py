#
# Assistant sentence aggregation for TTS streaming control.
#

from __future__ import annotations

from typing import List

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartInterruptionFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import match_endofsentence


class AssistantSentenceAggregator(FrameProcessor):
    """Aggregate LLM text into TTS chunks with custom sentence rules.

    Rules:
    - Do not split on commas (uses sentence-ending punctuation only).
    - If a sentence has fewer than `min_words`, buffer and annex it to the
      following sentence.
    - If the response ends and a short buffer remains, emit it as-is.
    """

    def __init__(self, *, enabled: bool, min_words: int = 3):
        super().__init__()
        self._enabled = enabled
        self._min_words = max(1, int(min_words))
        self._buffer = ""
        self._carry = ""
        self._started = False

    def _reset(self) -> None:
        self._buffer = ""
        self._carry = ""
        self._started = False

    def _is_short(self, text: str) -> bool:
        return len(text.split()) < self._min_words

    def _append_text(self, text: str) -> None:
        if text:
            self._buffer += text

    def _pop_sentence(self) -> str | None:
        end = match_endofsentence(self._buffer)
        if not end:
            return None
        sentence = self._buffer[:end]
        self._buffer = self._buffer[end:]
        return sentence.strip()

    def _merge_or_buffer(self, sentence: str) -> str | None:
        if not sentence:
            return None

        if self._carry:
            sentence = f"{self._carry} {sentence}".strip()
            self._carry = ""

        if self._is_short(sentence):
            self._carry = sentence
            return None

        return sentence

    async def _emit_ready_sentences(self) -> None:
        ready: List[str] = []
        while True:
            sentence = self._pop_sentence()
            if sentence is None:
                break
            merged = self._merge_or_buffer(sentence)
            if merged:
                ready.append(merged)

        for sentence in ready:
            await self.push_frame(LLMTextFrame(sentence))

    async def _flush_remaining(self) -> None:
        remaining = f"{self._carry} {self._buffer}".strip() if self._carry else self._buffer.strip()
        self._buffer = ""
        self._carry = ""
        if remaining:
            await self.push_frame(LLMTextFrame(remaining))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, CancelFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            self._carry = ""
            self._started = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            if self._enabled:
                await self._emit_ready_sentences()
                await self._flush_remaining()
            else:
                await self._flush_remaining()
            self._started = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TextFrame):
            if not self._started:
                await self.push_frame(frame, direction)
                return

            self._append_text(frame.text)

            if self._enabled:
                await self._emit_ready_sentences()
            # When disabled, we intentionally do not emit text until response end.
            return

        await self.push_frame(frame, direction)
