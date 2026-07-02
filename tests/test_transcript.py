from types import SimpleNamespace

import pytest

from paper2pod.logging_setup import TranscriptError
from paper2pod.parser import PaperMetadata
from paper2pod.transcript import generate

GOOD_TEXT = "word " * 350
LONG_TEXT = "word " * 500
SHORT_REVISED_TEXT = "word " * 360


def _style_config(min_words=320, max_words=420, model="claude-sonnet-4-6"):
    return SimpleNamespace(provider="anthropic", model=model, target_words=(min_words, max_words))


class FakeAPIError(Exception):
    def __init__(self, status_code, message="api error"):
        super().__init__(message)
        self.status_code = status_code


def test_generate_returns_transcript_with_word_count_and_duration():
    def call_fn(model, system, messages):
        return GOOD_TEXT

    metadata = PaperMetadata(title="Test Paper", authors=["A. Author"])
    transcript = generate("body text", metadata, _style_config(), call_fn=call_fn)

    assert transcript.title == "Test Paper"
    assert transcript.authors == ["A. Author"]
    assert transcript.word_count == 350
    assert transcript.estimated_duration_s == pytest.approx(350 / 150 * 60, rel=1e-3)


def test_generate_strips_markdown_artifacts():
    def call_fn(model, system, messages):
        return "**Hold on** to your papers! # Big News\n- point one\n" + GOOD_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    transcript = generate("body", metadata, _style_config(), call_fn=call_fn)

    assert "*" not in transcript.text
    assert "#" not in transcript.text
    assert not transcript.text.lstrip().startswith("- ")


def test_generate_triggers_one_compression_pass_when_over_cap():
    calls = []

    def call_fn(model, system, messages):
        calls.append(messages)
        if len(calls) == 1:
            return LONG_TEXT
        return SHORT_REVISED_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    transcript = generate("body", metadata, _style_config(), call_fn=call_fn)

    assert len(calls) == 2
    assert transcript.word_count == 360
    # Second call's message history includes the over-length draft and a compression ask.
    assert calls[1][-2]["role"] == "assistant"
    assert calls[1][-1]["role"] == "user"
    compression_ask = calls[1][-1]["content"].lower()
    assert "compress" in compression_ask or "revise" in compression_ask


def test_generate_does_not_compress_when_under_cap():
    calls = []

    def call_fn(model, system, messages):
        calls.append(messages)
        return GOOD_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    generate("body", metadata, _style_config(), call_fn=call_fn)

    assert len(calls) == 1


def test_retries_on_429_then_succeeds():
    attempts = {"count": 0}

    def call_fn(model, system, messages):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeAPIError(429)
        return GOOD_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    transcript = generate("body", metadata, _style_config(), call_fn=call_fn)

    assert attempts["count"] == 3
    assert transcript.word_count == 350


def test_retries_on_500_then_succeeds():
    attempts = {"count": 0}

    def call_fn(model, system, messages):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise FakeAPIError(503)
        return GOOD_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    generate("body", metadata, _style_config(), call_fn=call_fn)
    assert attempts["count"] == 2


def test_401_fails_immediately_without_retry():
    attempts = {"count": 0}

    def call_fn(model, system, messages):
        attempts["count"] += 1
        raise FakeAPIError(401, "invalid api key")

    metadata = PaperMetadata(title="Test", authors=[])
    with pytest.raises(TranscriptError, match="Authentication failed"):
        generate("body", metadata, _style_config(), call_fn=call_fn)

    assert attempts["count"] == 1


def test_exhausted_retries_raise_transcript_error():
    def call_fn(model, system, messages):
        raise FakeAPIError(429)

    metadata = PaperMetadata(title="Test", authors=[])
    with pytest.raises(TranscriptError):
        generate("body", metadata, _style_config(), call_fn=call_fn)


def test_unknown_provider_raises_transcript_error():
    def call_fn(model, system, messages):
        return GOOD_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    style = SimpleNamespace(provider="not-a-provider", model="x", target_words=(320, 420))
    with pytest.raises(TranscriptError, match="Unknown transcript provider"):
        generate("body", metadata, style, secrets=None)
