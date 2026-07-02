"""Default TTS provider: edge-tts (Microsoft Edge neural voices, free, no API key)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paper2pod.logging_setup import TTSError


def _is_retryable(exc: BaseException) -> bool:
    type_name = type(exc).__name__
    return any(marker in type_name for marker in ("Connection", "Timeout", "NoAudioReceived"))


class EdgeTTSProvider:
    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+8%"):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        def _run() -> None:
            asyncio.run(self._synthesize_async(text, out_path))

        try:
            _run()
        except Exception as e:
            raise TTSError(f"edge-tts synthesis failed: {e}") from e
        return out_path

    async def _synthesize_async(self, text: str, out_path: Path) -> None:
        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        await communicate.save(str(out_path))
