from types import SimpleNamespace

import pytest

from paper2pod.parser import PaperMetadata
from paper2pod.transcript import generate

GOOD_TEXT = "word " * 350


def _style_config(min_words=320, max_words=420, model="claude-sonnet-4-6"):
    return SimpleNamespace(provider="anthropic", model=model, target_words=(min_words, max_words))


def _cta_config(enabled=True, text="Go check out OpenLabs and share your research."):
    return SimpleNamespace(enabled=enabled, text=text)


def _call_fn(model, system, messages):
    return GOOD_TEXT


def test_cta_appended_verbatim_when_enabled():
    cta_text = "If this got you curious, head over to OpenLabs and explore the projects."
    metadata = PaperMetadata(title="Test", authors=[])
    transcript = generate(
        "body", metadata, _style_config(), call_fn=_call_fn, cta_config=_cta_config(text=cta_text)
    )
    assert transcript.text.endswith(cta_text)
    # Byte-equal: the appended suffix matches the configured text exactly.
    assert transcript.text[-len(cta_text) :] == cta_text


def test_cta_disabled_leaves_body_unchanged():
    metadata = PaperMetadata(title="Test", authors=[])
    transcript = generate(
        "body", metadata, _style_config(), call_fn=_call_fn, cta_config=_cta_config(enabled=False)
    )
    assert transcript.text == GOOD_TEXT.strip()


def test_cta_config_none_matches_disabled():
    metadata = PaperMetadata(title="Test", authors=[])
    transcript = generate("body", metadata, _style_config(), call_fn=_call_fn, cta_config=None)
    assert transcript.text == GOOD_TEXT.strip()


def test_cta_text_never_sent_to_llm():
    cta_text = "UNIQUE_CTA_MARKER_TEXT_12345"
    seen_messages: list[dict[str, str]] = []

    def spy_call_fn(model, system, messages):
        seen_messages.extend(messages)
        return GOOD_TEXT

    metadata = PaperMetadata(title="Test", authors=[])
    generate(
        "body",
        metadata,
        _style_config(),
        call_fn=spy_call_fn,
        cta_config=_cta_config(text=cta_text),
    )

    for message in seen_messages:
        assert cta_text not in message["content"]


def test_cta_body_budget_unaffected_by_cta_length():
    """The 320-420 word compression trigger is based on body words only."""
    calls = []

    def call_fn(model, system, messages):
        calls.append(messages)
        return GOOD_TEXT  # 350 words, under the 420 cap on its own

    metadata = PaperMetadata(title="Test", authors=[])
    long_cta = " ".join(["cta"] * 70)  # well over 420 words if miscounted with body
    generate(
        "body", metadata, _style_config(), call_fn=call_fn, cta_config=_cta_config(text=long_cta)
    )
    # No compression pass should have been triggered by the CTA's length.
    assert len(calls) == 1


def test_duration_estimate_includes_cta_words():
    cta_text = "word " * 20
    cta_text = cta_text.strip()
    metadata = PaperMetadata(title="Test", authors=[])

    without_cta = generate(
        "body", metadata, _style_config(), call_fn=_call_fn, cta_config=_cta_config(enabled=False)
    )
    with_cta = generate(
        "body", metadata, _style_config(), call_fn=_call_fn, cta_config=_cta_config(text=cta_text)
    )

    assert with_cta.word_count == without_cta.word_count + 20
    expected_extra_seconds = 20 / 150 * 60
    actual_extra_seconds = with_cta.estimated_duration_s - without_cta.estimated_duration_s
    assert actual_extra_seconds == pytest.approx(expected_extra_seconds, abs=0.1)
