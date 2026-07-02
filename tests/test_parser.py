from pathlib import Path

import pytest

from paper2pod.logging_setup import ParseError
from paper2pod.parser import (
    REMEDIATION_MESSAGE,
    parse_markdown,
    truncate_to_token_budget,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_frontmatter_metadata_used_without_calling_llm():
    calls = []

    def fail_extractor(text: str) -> dict:
        calls.append(text)
        raise AssertionError("LLM extractor should not be called when frontmatter exists")

    metadata, body = parse_markdown(FIXTURES / "frontmatter.md", llm_extractor=fail_extractor)

    assert metadata.title == "Diffusion Models Beat GANs on Image Synthesis"
    assert metadata.authors == ["Prafulla Dhariwal", "Alex Nichol"]
    assert not calls
    assert "---" not in body.split("\n")[0]


def test_heading_only_falls_back_to_llm_extractor():
    def mock_extractor(text: str) -> dict:
        assert "Attention Is All You Need" in text
        return {"title": "Attention Is All You Need", "authors": ["Ashish Vaswani", "Noam Shazeer"]}

    metadata, body = parse_markdown(FIXTURES / "heading_only.md", llm_extractor=mock_extractor)

    assert metadata.title == "Attention Is All You Need"
    assert metadata.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert "Transformer" in body


def test_no_title_raises_parse_error_with_remediation():
    def empty_extractor(text: str) -> dict:
        return {"title": "", "authors": []}

    with pytest.raises(ParseError) as exc_info:
        parse_markdown(FIXTURES / "no_title.md", llm_extractor=empty_extractor)

    assert REMEDIATION_MESSAGE in str(exc_info.value)
    assert exc_info.value.stage == "parse"
    assert exc_info.value.input_file == str(FIXTURES / "no_title.md")


def test_missing_file_raises_parse_error():
    with pytest.raises(ParseError, match="not found"):
        parse_markdown(FIXTURES / "does_not_exist.md", llm_extractor=lambda t: {})


def test_binary_file_raises_parse_error(tmp_path):
    binary_path = tmp_path / "binary.md"
    binary_path.write_bytes(b"\xff\xfe\x00\x01binary garbage\x80\x81")

    with pytest.raises(ParseError, match="UTF-8"):
        parse_markdown(binary_path, llm_extractor=lambda t: {})


def test_truncate_under_budget_returns_unchanged():
    body = "## Abstract\nShort text.\n"
    assert truncate_to_token_budget(body, max_tokens=1000) == body


def test_truncate_prefers_abstract_and_results_sections():
    filler = "word " * 2000
    body = (
        f"## Related Work\n{filler}\n"
        f"## Abstract\nCONCISE ABSTRACT CONTENT.\n"
        f"## Results\nCONCISE RESULTS CONTENT.\n"
    )
    truncated = truncate_to_token_budget(body, max_tokens=20)

    assert "CONCISE ABSTRACT CONTENT" in truncated
    assert "CONCISE RESULTS CONTENT" in truncated
    assert len(truncated) < len(body)


def test_parse_markdown_truncates_body_to_token_budget():
    def mock_extractor(text: str) -> dict:
        return {"title": "Long Paper", "authors": []}

    metadata, body = parse_markdown(
        FIXTURES / "heading_only.md", max_input_tokens=5, llm_extractor=mock_extractor
    )
    assert len(body) <= 5 * 4
