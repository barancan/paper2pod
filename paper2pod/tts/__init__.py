"""TTS provider factory."""

from __future__ import annotations

from typing import Any

from paper2pod.logging_setup import TTSError
from paper2pod.tts.base import TTSProvider
from paper2pod.tts.edge import EdgeTTSProvider
from paper2pod.tts.elevenlabs import ElevenLabsTTSProvider


def get_provider(tts_config: Any, secrets: Any) -> TTSProvider:
    if tts_config.provider == "edge":
        return EdgeTTSProvider(voice=tts_config.voice, rate=tts_config.rate)
    if tts_config.provider == "elevenlabs":
        if not secrets.elevenlabs_api_key:
            raise TTSError("ELEVENLABS_API_KEY is required when tts.provider=elevenlabs.")
        return ElevenLabsTTSProvider(api_key=secrets.elevenlabs_api_key, voice=tts_config.voice)
    raise TTSError(f"Unknown TTS provider: {tts_config.provider}")


__all__ = ["TTSProvider", "get_provider"]
