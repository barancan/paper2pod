"""Markdown paper parsing: frontmatter/LLM metadata extraction, token-budget truncation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from paper2pod.logging_setup import ParseError

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)

# Rough token estimate used consistently for both the LLM-fallback excerpt size
# and the max_input_tokens truncation budget. No tokenizer dependency is listed
# in the spec's deps, so this uses the common ~4-chars-per-token heuristic for
# English text (see README "Decisions").
CHARS_PER_TOKEN = 4
LLM_FALLBACK_TOKEN_BUDGET = 2000

REMEDIATION_MESSAGE = (
    "Could not determine a title for this paper. Add a `# Title` heading or "
    "YAML frontmatter with a `title:` field to the top of the file."
)

PREFERRED_SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "intro",
    "results",
    "result",
    "conclusion",
]

MetadataExtractor = Callable[[str], dict[str, Any]]


@dataclass
class PaperMetadata:
    title: str
    authors: list[str] = field(default_factory=list)


def load_markdown(path: Path) -> str:
    """Read a UTF-8 .md file, raising ParseError with a clear message otherwise."""
    if not path.exists():
        raise ParseError(f"File not found: {path}", input_file=str(path))
    if path.is_dir():
        raise ParseError(f"Expected a file, got a directory: {path}", input_file=str(path))
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ParseError(f"Could not read file: {e}", input_file=str(path)) from e
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ParseError(
            "File is not valid UTF-8 text (is it binary?)", input_file=str(path)
        ) from e


def _extract_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Return (frontmatter_dict_or_None, body_with_frontmatter_stripped)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None, text
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None, text
    if not isinstance(data, dict):
        return None, text
    return data, text[match.end() :]


def _normalize_authors(authors: Any) -> list[str]:
    if not authors:
        return []
    if isinstance(authors, str):
        return [a.strip() for a in authors.split(",") if a.strip()]
    return [str(a).strip() for a in authors if str(a).strip()]


def _metadata_from_frontmatter(frontmatter: dict[str, Any]) -> PaperMetadata | None:
    title = frontmatter.get("title")
    if not title or not str(title).strip():
        return None
    authors = _normalize_authors(frontmatter.get("authors"))
    return PaperMetadata(title=str(title).strip(), authors=authors)


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def default_llm_extractor(provider: str, model: str, secrets: Any) -> MetadataExtractor:
    """Build a metadata extractor backed by the configured anthropic/openai provider."""

    def extractor(text: str) -> dict[str, Any]:
        prompt = (
            "Extract the title and author names from this research paper excerpt. "
            'Respond with ONLY compact JSON: {"title": string, "authors": [string, ...]}. '
            "If no clear title exists, use an empty string for title.\n\n" + text
        )
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
        elif provider == "openai":
            import openai

            client = openai.OpenAI(api_key=secrets.openai_api_key)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or ""
        else:
            raise ParseError(f"Unknown transcript provider: {provider}")
        return _extract_json_object(raw)

    return extractor


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split body into (heading, content) chunks on markdown ATX headings."""
    heading_re = re.compile(r"^#{1,6}\s+(.*)")
    sections: list[tuple[str, str]] = []
    heading = ""
    chunk: list[str] = []
    for line in body.splitlines(keepends=True):
        m = heading_re.match(line)
        if m:
            if chunk:
                sections.append((heading, "".join(chunk)))
            heading = m.group(1).strip()
            chunk = [line]
        else:
            chunk.append(line)
    if chunk:
        sections.append((heading, "".join(chunk)))
    return sections


def _section_priority(heading: str) -> int:
    lowered = heading.lower()
    for i, keyword in enumerate(PREFERRED_SECTION_KEYWORDS):
        if keyword in lowered:
            return i
    return len(PREFERRED_SECTION_KEYWORDS)


def truncate_to_token_budget(body: str, max_tokens: int) -> str:
    """Truncate to ~max_tokens, preferring abstract/intro/results sections when detectable."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(body) <= max_chars:
        return body

    sections = _split_sections(body)
    if not sections:
        return body[:max_chars]

    order = sorted(range(len(sections)), key=lambda i: (_section_priority(sections[i][0]), i))

    kept: list[str | None] = [None] * len(sections)
    used = 0
    for idx in order:
        if used >= max_chars:
            break
        _, content = sections[idx]
        remaining = max_chars - used
        if len(content) <= remaining:
            kept[idx] = content
            used += len(content)
        else:
            kept[idx] = content[:remaining]
            used += remaining
            break

    return "".join(c for c in kept if c is not None)


def parse_markdown(
    path: Path,
    max_input_tokens: int = 12000,
    llm_extractor: MetadataExtractor | None = None,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    secrets: Any = None,
) -> tuple[PaperMetadata, str]:
    """Parse a paper .md file into (metadata, truncated_body)."""
    text = load_markdown(path)
    frontmatter, body = _extract_frontmatter(text)

    metadata = _metadata_from_frontmatter(frontmatter) if frontmatter else None

    if metadata is None:
        extractor = llm_extractor or default_llm_extractor(provider, model, secrets)
        excerpt = body[: LLM_FALLBACK_TOKEN_BUDGET * CHARS_PER_TOKEN]
        try:
            result = extractor(excerpt) or {}
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Metadata extraction failed: {e}", input_file=str(path)) from e

        title = str(result.get("title") or "").strip()
        if title:
            metadata = PaperMetadata(title=title, authors=_normalize_authors(result.get("authors")))

    if metadata is None or not metadata.title:
        raise ParseError(REMEDIATION_MESSAGE, input_file=str(path))

    truncated_body = truncate_to_token_budget(body, max_input_tokens)
    return metadata, truncated_body
