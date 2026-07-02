"""TTSProvider protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TTSProvider(Protocol):
    def synthesize(self, text: str, out_path: Path) -> Path:
        """Render text to speech, writing an MP3 to out_path, and return out_path."""
        ...
