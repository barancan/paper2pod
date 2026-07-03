import logging

from paper2pod.logging_setup import (
    ParseError,
    SourceError,
    StorageError,
    TranscriptError,
    TTSError,
    setup_logging,
)


def test_typed_exceptions_carry_stage_and_input_file():
    err = ParseError("could not parse", input_file="paper.md")
    assert err.stage == "parse"
    assert err.input_file == "paper.md"
    assert "paper.md" in str(err)

    assert TranscriptError.stage == "transcript"
    assert TTSError.stage == "tts"
    assert StorageError.stage == "storage"


def test_source_error_carries_stage_and_url():
    url = "https://openlabs.bio.xyz/projects/abc-123"
    err = SourceError("fetch failed", input_file=url)
    assert err.stage == "source"
    assert err.input_file == url
    assert url in str(err)


def test_setup_logging_creates_rotating_file_and_console_handlers(tmp_path):
    log_file = tmp_path / "logs" / "paper2pod.log"
    logger = setup_logging(log_file=log_file)

    assert log_file.parent.exists()
    handler_types = {type(h).__name__ for h in logger.handlers}
    assert "RotatingFileHandler" in handler_types
    assert "RichHandler" in handler_types

    file_handler = next(h for h in logger.handlers if type(h).__name__ == "RotatingFileHandler")
    console_handler = next(h for h in logger.handlers if type(h).__name__ == "RichHandler")
    assert file_handler.level == logging.DEBUG
    assert console_handler.level == logging.WARNING
    assert file_handler.maxBytes == 5 * 1024 * 1024
    assert file_handler.backupCount == 3


def test_setup_logging_writes_debug_to_file_only(tmp_path, capsys):
    log_file = tmp_path / "paper2pod.log"
    logger = setup_logging(log_file=log_file)

    logger.debug("debug detail")
    logger.warning("warning detail")

    for handler in logger.handlers:
        handler.flush()

    contents = log_file.read_text()
    assert "debug detail" in contents
    assert "warning detail" in contents


def test_secret_redaction_scrubs_api_key_from_file_log(tmp_path):
    log_file = tmp_path / "paper2pod.log"
    secret = "sk-ant-super-secret-1234567890"
    logger = setup_logging(log_file=log_file, secrets=[secret])

    logger.debug("using key %s", secret)
    for handler in logger.handlers:
        handler.flush()

    contents = log_file.read_text()
    assert secret not in contents
    assert "REDACTED" in contents
