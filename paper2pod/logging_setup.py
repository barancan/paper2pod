"""Console + rotating file logger, and typed pipeline-stage exceptions."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

LOGGER_NAME = "paper2pod"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
REDACTED = "***REDACTED***"


class Paper2PodError(Exception):
    """Base for typed pipeline errors. Carries the failing stage and input.

    `input_file` holds whatever input this pipeline was working on -- a
    local file path for most stages, or a URL for SourceError.
    """

    stage: str = "unknown"

    def __init__(self, message: str, input_file: str | None = None):
        self.input_file = input_file
        super().__init__(message)

    def __str__(self) -> str:
        suffix = f" (input: {self.input_file})" if self.input_file else ""
        return f"{super().__str__()}{suffix}"


class ParseError(Paper2PodError):
    stage = "parse"


class TranscriptError(Paper2PodError):
    stage = "transcript"


class TTSError(Paper2PodError):
    stage = "tts"


class StorageError(Paper2PodError):
    stage = "storage"


class SourceError(Paper2PodError):
    """Raised when fetching an external content source (e.g. OpenLabs) fails."""

    stage = "source"


def redact_secrets(text: str, secrets: Iterable[str]) -> str:
    """Replace any occurrence of a known secret value with a redaction marker."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, REDACTED)
    return text


class SecretRedactingFilter(logging.Filter):
    """Scrubs known secret values out of log messages before they're emitted."""

    def __init__(self, secrets: Iterable[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg, self._secrets)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_secrets(v, self._secrets) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    redact_secrets(a, self._secrets) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


def setup_logging(
    log_file: str | Path = "logs/paper2pod.log",
    level: str = "INFO",  # noqa: ARG001 - reserved; see README "Decisions"
    secrets: Iterable[str] | None = None,
) -> logging.Logger:
    """Configure the paper2pod logger: DEBUG to a rotating file, WARNING+ to console.

    Per spec §7 the file handler is always DEBUG+ and the console handler is
    always WARNING+; `level` (from config.yaml `logging.level`) is accepted for
    forward compatibility but does not currently change either threshold.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    console_handler = RichHandler(show_time=False, show_path=False, markup=False)
    console_handler.setLevel(logging.WARNING)

    if secrets:
        redact_filter = SecretRedactingFilter(secrets)
        file_handler.addFilter(redact_filter)
        console_handler.addFilter(redact_filter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
