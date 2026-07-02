"""Optional keyed TTS provider: ElevenLabs, via its REST API (no SDK dependency)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paper2pod.logging_setup import TTSError

API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
# ElevenLabs' default public voice ("Rachel"), used when tts.voice isn't a
# recognizable ElevenLabs voice ID (e.g. still set to an edge-tts voice name).
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, urllib.error.URLError)


class ElevenLabsTTSProvider:
    def __init__(self, api_key: str, voice: str = DEFAULT_VOICE_ID):
        self.api_key = api_key
        self.voice = voice

    def synthesize(self, text: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{API_BASE}/{self.voice}",
            data=json.dumps({"text": text, "model_id": "eleven_monolingual_v1"}).encode("utf-8"),
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        def _do() -> bytes:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()

        try:
            audio_bytes = _do()
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise TTSError(
                    "ElevenLabs authentication failed (401). "
                    "Check ELEVENLABS_API_KEY in .env is correct."
                ) from e
            raise TTSError(f"ElevenLabs synthesis failed: {e}") from e
        except Exception as e:
            raise TTSError(f"ElevenLabs synthesis failed: {e}") from e

        out_path.write_bytes(audio_bytes)
        return out_path
