"""Provider-agnostic Two Minute Papers style transcript generation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paper2pod.logging_setup import TranscriptError
from paper2pod.parser import PaperMetadata

WORDS_PER_MINUTE = 150

SYSTEM_PROMPT_TEMPLATE = """You are the narrator for a Two Minute Papers-style science \
video. You turn dense research paper text into an enthusiastic, ear-friendly narration \
script meant to be read aloud.

Voice and tone:
- Enthusiastic and accessible, like you can't wait to share this discovery.
- Address the listener directly in second person (e.g. "Now, hold on to your papers, \
because...").
- Warm, curious, a little playful -- never dry or academic.

Structure, in this order:
1. Hook: open with a vivid, curiosity-grabbing line about the problem or promise.
2. What the researchers did: plainly describe the approach.
3. Why it's hard: give the listener a sense of the challenge being solved.
4. Key result: state the headline finding with one concrete number.
5. Limitation: briefly and honestly note a limitation or caveat.
6. Closing: end with a natural, satisfying closing sentence that wraps up the key result. \
Do not add a sign-off catchphrase or call to action of your own -- a call to action will \
be appended after your narration, so just land the story cleanly.

Formatting rules (strict):
- Plain speech only. No markdown, no bullet points, no headers, no asterisks, no backticks.
- No citations, no URLs, no reference numbers.
- Write every number the way it should be spoken aloud (e.g. "thirty-five thousand", not \
"35,000"; "two and a half times", not "2.5x").
- Target length: {min_words} to {max_words} words -- aim for the middle of that range.

Output only the narration text itself, with no preamble, labels, or commentary."""

PROJECT_BRIEF_SYSTEM_PROMPT_TEMPLATE = """You are the narrator for an OpenLabs project \
spotlight. You turn a research project's page content into an enthusiastic, ear-friendly \
spoken brief meant to be read aloud.

Voice and tone:
- Enthusiastic and accessible, like you can't wait to share what this project is doing.
- Address the listener directly in second person.
- Warm, curious, a little playful -- never dry or academic.

Structure, in this order:
1. Hook: open with a vivid, curiosity-grabbing line about the project's mission.
2. What the project is trying to solve.
3. Why it matters.
4. Approach and current status.
5. Who is behind it -- the team or creator name(s), if known.
6. One concrete detail: a specific number, milestone, or method drawn directly from the \
content.
7. Closing: end with a natural, satisfying closing sentence. Do not add a sign-off \
catchphrase or call to action of your own -- a call to action will be appended after your \
narration, so just land it cleanly.

Formatting rules (strict):
- Plain speech only. No markdown, no bullet points, no headers, no asterisks, no backticks.
- No citations, no URLs read aloud.
- Write every number the way it should be spoken aloud (e.g. "thirty-five thousand", not \
"35,000"; "two and a half times", not "2.5x").
- Target length: {min_words} to {max_words} words -- aim for the middle of that range.

Factual accuracy (strict):
- Only state claims that are explicitly present in the provided project content. Do not \
invent results, data, funding, dates, or outcomes.
- If the content describes a goal or an open question rather than a completed result, \
present it as a goal or an open question -- do not imply it has already been achieved.

Output only the narration text itself, with no preamble, labels, or commentary."""

SYSTEM_PROMPTS: dict[str, str] = {
    "paper": SYSTEM_PROMPT_TEMPLATE,
    "project_brief": PROJECT_BRIEF_SYSTEM_PROMPT_TEMPLATE,
}

USER_PROMPT_LABELS: dict[str, tuple[str, str, str]] = {
    "paper": ("Paper title", "Authors", "Paper content"),
    "project_brief": ("Project title", "Team", "Project content"),
}

MARKDOWN_ARTIFACT_RE = re.compile(r"[*_`#]+")
MARKDOWN_BULLET_RE = re.compile(r"^\s*[-•]\s+", re.MULTILINE)


@dataclass
class Transcript:
    text: str
    title: str
    authors: list[str]
    word_count: int
    estimated_duration_s: float


# Message content may be a plain string (markdown/OpenLabs) or a list of
# content blocks (PDF: a document block + a text block), so it's typed loosely.
ProviderCall = Callable[[str, str, list[dict[str, Any]]], str]


def _anthropic_messages_call(
    model: str, system: str, messages: list[dict[str, Any]], secrets: Any
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
    resp = client.messages.create(model=model, max_tokens=1024, system=system, messages=messages)
    return resp.content[0].text


def _openai_messages_call(
    model: str, system: str, messages: list[dict[str, Any]], secrets: Any
) -> str:
    import openai

    client = openai.OpenAI(api_key=secrets.openai_api_key)
    full_messages = [{"role": "system", "content": system}, *messages]
    resp = client.chat.completions.create(model=model, max_tokens=1024, messages=full_messages)
    return resp.choices[0].message.content or ""


def _bind_provider_call(provider: str, secrets: Any) -> ProviderCall:
    if provider == "anthropic":
        return lambda model, system, messages: _anthropic_messages_call(
            model, system, messages, secrets
        )
    if provider == "openai":
        return lambda model, system, messages: _openai_messages_call(
            model, system, messages, secrets
        )
    raise TranscriptError(f"Unknown transcript provider: {provider}")


def _is_retryable(exc: BaseException) -> bool:
    """Retry only on 429/5xx/network errors; 401 and other client errors fail fast."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code == 429 or status_code >= 500
    return "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__


def _call_with_retry(
    call: ProviderCall, model: str, system: str, messages: list[dict[str, Any]]
) -> str:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _do() -> str:
        return call(model, system, messages)

    try:
        return _do()
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        if status_code == 401:
            raise TranscriptError(
                "Authentication failed (401). Check your API key in .env is correct."
            ) from e
        raise TranscriptError(f"Transcript generation failed: {e}") from e


def _build_user_prompt(
    metadata: PaperMetadata, paper_text: str, min_words: int, max_words: int, style: str
) -> str:
    subject_label, who_label, content_label = USER_PROMPT_LABELS.get(
        style, USER_PROMPT_LABELS["paper"]
    )
    who = ", ".join(metadata.authors) if metadata.authors else "the researchers"
    # When paper_text is empty the content is supplied out-of-band as an
    # attached PDF document block, so point the model at that instead of
    # emitting an empty "Paper content:" section.
    content_section = (
        f"{content_label}:\n{paper_text}"
        if paper_text.strip()
        else f"{content_label}: the full paper is attached as a PDF document."
    )
    return (
        f"{subject_label}: {metadata.title}\n"
        f"{who_label}: {who}\n\n"
        f"{content_section}\n\n"
        f"Write the narration script now, {min_words}-{max_words} words."
    )


def _compression_prompt(word_count: int, min_words: int, max_words: int) -> str:
    return (
        f"Your narration was {word_count} words, over the {max_words}-word cap. Revise it "
        f"to be between {min_words} and {max_words} words while keeping the same structure, "
        "energy, and key result. Return only the revised narration."
    )


def _clean_plain_speech(text: str) -> str:
    text = MARKDOWN_ARTIFACT_RE.sub("", text)
    text = MARKDOWN_BULLET_RE.sub("", text)
    return text.strip()


def _count_words(text: str) -> int:
    return len(text.split())


def generate(
    paper_text: str,
    metadata: PaperMetadata,
    style_config: Any,
    secrets: Any = None,
    call_fn: ProviderCall | None = None,
    cta_config: Any = None,
    style: str = "paper",
    pdf_document: dict[str, Any] | None = None,
) -> Transcript:
    """Generate an enthusiastic narration transcript for the given content.

    style selects the system prompt / user-prompt labels: "paper" (default,
    Two Minute Papers style) or "project_brief" (OpenLabs project spotlight).

    When pdf_document is provided (a native Anthropic document block), the paper
    is sent to the model as the PDF itself rather than as paper_text, which is
    then ignored. This path is Anthropic-only. The word-budget compression
    follow-up stays text and does not resend the PDF.

    The LLM's 320-420 word budget applies to the body only. If cta_config is
    enabled, its text is appended verbatim afterward -- it is never sent to
    the LLM, so the configured wording is guaranteed to be what gets spoken.
    """
    if pdf_document is not None and style_config.provider != "anthropic":
        raise TranscriptError(
            "PDF ingestion requires the anthropic transcript provider "
            f"(configured provider is '{style_config.provider}')."
        )

    call = call_fn or _bind_provider_call(style_config.provider, secrets)
    min_words, max_words = style_config.target_words

    template = SYSTEM_PROMPTS.get(style)
    if template is None:
        raise TranscriptError(f"Unknown transcript style: {style}")
    system_prompt = template.format(min_words=min_words, max_words=max_words)
    user_prompt = _build_user_prompt(metadata, paper_text, min_words, max_words, style)
    content: str | list[dict[str, Any]] = (
        [pdf_document, {"type": "text", "text": user_prompt}]
        if pdf_document is not None
        else user_prompt
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

    body_text = _clean_plain_speech(
        _call_with_retry(call, style_config.model, system_prompt, messages)
    )
    body_word_count = _count_words(body_text)

    if body_word_count > max_words:
        messages.append({"role": "assistant", "content": body_text})
        messages.append(
            {"role": "user", "content": _compression_prompt(body_word_count, min_words, max_words)}
        )
        body_text = _clean_plain_speech(
            _call_with_retry(call, style_config.model, system_prompt, messages)
        )

    final_text = body_text.rstrip()
    if cta_config is not None and getattr(cta_config, "enabled", False):
        final_text = f"{final_text} {cta_config.text}"
    word_count = _count_words(final_text)

    return Transcript(
        text=final_text,
        title=metadata.title,
        authors=metadata.authors,
        word_count=word_count,
        estimated_duration_s=round(word_count / WORDS_PER_MINUTE * 60, 1),
    )
