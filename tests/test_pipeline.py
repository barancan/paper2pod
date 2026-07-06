"""Unit tests for the shared pipeline module: stage-event sequencing and outcomes.

Both the CLI and the (future) API job worker call run_markdown_pipeline /
run_openlabs_pipeline directly, passing in whatever on_event callback they
want. These tests mock everything below the pipeline (parse/fetch, LLM,
TTS, storage, db) and assert on the StageEvent sequence and the returned
PipelineResult -- proving the event system itself, independent of any
particular consumer (Rich console, job store, or a bare test collector).
"""

from pathlib import Path

import pytest

from paper2pod import pipeline as pipeline_module
from paper2pod.config import Secrets, load_config
from paper2pod.logging_setup import (
    ParseError,
    RecordError,
    SourceError,
    StorageError,
    TranscriptError,
    TTSError,
)
from paper2pod.parser import PaperMetadata
from paper2pod.pipeline import run_markdown_pipeline, run_openlabs_pipeline
from paper2pod.sources.openlabs import ProjectContent
from paper2pod.storage import UploadResult
from paper2pod.transcript import Transcript

FIXTURES = Path(__file__).parent / "fixtures"
TRANSCRIPT_TEXT = ("word " * 350).strip()
PROJECT_URL = "https://openlabs.bio.xyz/projects/4b7c0a52-708e-4f5c-a319-38ed5104c964"


def _app_config():
    return load_config()


def _secrets():
    return Secrets(_env_file=None)


def _fake_metadata():
    return PaperMetadata(title="Test Paper", authors=["Author One"])


def _fake_parse_markdown(*args, **kwargs):
    return _fake_metadata(), "body text"


def _fake_project():
    return ProjectContent(
        title="Test Project",
        team_or_authors=["Team Member"],
        summary="summary",
        body_text="word " * 250,
        url=PROJECT_URL,
    )


def _fake_generate_transcript(paper_text, metadata, *args, **kwargs):
    return Transcript(
        text=TRANSCRIPT_TEXT,
        title=metadata.title,
        authors=metadata.authors,
        word_count=350,
        estimated_duration_s=140.0,
    )


class FakeTTSProvider:
    def synthesize(self, text, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"FAKE_MP3_BYTES")
        return out_path


def _fake_upload_recording(client, bucket, object_name, local_path, upsert=False):
    return UploadResult(
        object_path=object_name, url="https://fake.supabase.co/x.mp3", is_public=True
    )


def _fake_record_episode(client, record):
    return "episode-id-123"


def _mock_full_success(monkeypatch):
    monkeypatch.setattr(pipeline_module, "parse_markdown", _fake_parse_markdown)
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr(pipeline_module, "generate_transcript", _fake_generate_transcript)
    monkeypatch.setattr(
        pipeline_module, "get_provider", lambda tts_config, secrets: FakeTTSProvider()
    )
    monkeypatch.setattr("supabase.create_client", lambda url, key: object())
    monkeypatch.setattr(pipeline_module, "upload_recording", _fake_upload_recording)
    monkeypatch.setattr(pipeline_module, "record_episode", _fake_record_episode)


def _collector():
    events: list[tuple[str, str]] = []

    def on_event(event):
        events.append((event.stage, event.status))

    return events, on_event


SUCCESS_SEQUENCE_MARKDOWN = [
    ("parse", "started"),
    ("parse", "completed"),
    ("transcript", "started"),
    ("transcript", "completed"),
    ("tts", "started"),
    ("tts", "completed"),
    ("upload", "started"),
    ("upload", "completed"),
    ("record", "started"),
    ("record", "completed"),
]


def test_successful_markdown_pipeline_emits_expected_stage_sequence(monkeypatch):
    _mock_full_success(monkeypatch)
    events, on_event = _collector()

    result = run_markdown_pipeline(FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event)

    assert events == SUCCESS_SEQUENCE_MARKDOWN
    assert result.episode_id == "episode-id-123"
    assert result.record_error is None
    assert result.upload_result.url == "https://fake.supabase.co/x.mp3"


def test_successful_openlabs_pipeline_uses_fetch_stage_not_parse(monkeypatch):
    _mock_full_success(monkeypatch)
    events, on_event = _collector()

    run_openlabs_pipeline(PROJECT_URL, _app_config(), _secrets(), on_event)

    assert events[0] == ("fetch", "started")
    assert events[1] == ("fetch", "completed")
    assert ("parse", "started") not in events
    assert [e[0] for e in events] == [
        "fetch",
        "fetch",
        "transcript",
        "transcript",
        "tts",
        "tts",
        "upload",
        "upload",
        "record",
        "record",
    ]


def test_dry_run_stops_after_transcript_and_skips_publish_stages(monkeypatch):
    _mock_full_success(monkeypatch)
    events, on_event = _collector()

    result = run_markdown_pipeline(
        FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event, dry_run=True
    )

    assert events == [
        ("parse", "started"),
        ("parse", "completed"),
        ("transcript", "started"),
        ("transcript", "completed"),
    ]
    assert result.upload_result is None
    assert result.episode_id is None
    assert result.transcript.text == TRANSCRIPT_TEXT


def test_parse_failure_emits_failed_event_and_reraises(monkeypatch):
    _mock_full_success(monkeypatch)

    def failing_parse(*args, **kwargs):
        raise ParseError("could not parse")

    monkeypatch.setattr(pipeline_module, "parse_markdown", failing_parse)
    events, on_event = _collector()

    with pytest.raises(ParseError):
        run_markdown_pipeline(FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event)

    assert events == [("parse", "started"), ("parse", "failed")]


def test_fetch_failure_emits_failed_event_and_reraises(monkeypatch):
    _mock_full_success(monkeypatch)

    def failing_fetch(url, **kwargs):
        raise SourceError("not found")

    monkeypatch.setattr(pipeline_module, "fetch_project", failing_fetch)
    events, on_event = _collector()

    with pytest.raises(SourceError):
        run_openlabs_pipeline(PROJECT_URL, _app_config(), _secrets(), on_event)

    assert events == [("fetch", "started"), ("fetch", "failed")]


def test_transcript_failure_emits_failed_event_and_reraises(monkeypatch):
    _mock_full_success(monkeypatch)

    def failing_generate(*args, **kwargs):
        raise TranscriptError("LLM down")

    monkeypatch.setattr(pipeline_module, "generate_transcript", failing_generate)
    events, on_event = _collector()

    with pytest.raises(TranscriptError):
        run_markdown_pipeline(FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event)

    assert events == [
        ("parse", "started"),
        ("parse", "completed"),
        ("transcript", "started"),
        ("transcript", "failed"),
    ]


def test_tts_failure_emits_failed_event_and_reraises(monkeypatch):
    _mock_full_success(monkeypatch)

    class FailingTTSProvider:
        def synthesize(self, text, out_path):
            raise TTSError("network down")

    monkeypatch.setattr(
        pipeline_module, "get_provider", lambda tts_config, secrets: FailingTTSProvider()
    )
    events, on_event = _collector()

    with pytest.raises(TTSError):
        run_markdown_pipeline(FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event)

    assert events[-2:] == [("tts", "started"), ("tts", "failed")]


def test_upload_failure_emits_failed_event_and_reraises(monkeypatch):
    _mock_full_success(monkeypatch)

    def failing_upload(*args, **kwargs):
        raise StorageError("bucket not found")

    monkeypatch.setattr(pipeline_module, "upload_recording", failing_upload)
    events, on_event = _collector()

    with pytest.raises(StorageError):
        run_markdown_pipeline(FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event)

    assert events[-2:] == [("upload", "started"), ("upload", "failed")]


def test_record_error_does_not_reraise_and_is_captured_on_result(monkeypatch):
    _mock_full_success(monkeypatch)

    def failing_record(client, record):
        raise RecordError("Supabase unreachable")

    monkeypatch.setattr(pipeline_module, "record_episode", failing_record)
    events, on_event = _collector()

    # Must not raise -- RecordError is recoverable, the run already succeeded.
    result = run_markdown_pipeline(FIXTURES / "frontmatter.md", _app_config(), _secrets(), on_event)

    assert events[-2:] == [("record", "started"), ("record", "failed")]
    assert result.episode_id is None
    assert result.record_error is not None
    assert "Supabase unreachable" in result.record_error
    # The audio is still there -- upload succeeded before record was attempted.
    assert result.upload_result.url == "https://fake.supabase.co/x.mp3"


def test_two_independent_listeners_receive_identical_event_sequence(monkeypatch):
    """The crux of 'one event system, two consumers': any callback passed to
    the same pipeline call sees the same stage/status sequence, whether it
    renders to a terminal (CLI) or updates a job store (API)."""
    _mock_full_success(monkeypatch)
    cli_style_events, cli_style_on_event = _collector()
    api_style_events, api_style_on_event = _collector()

    config = _app_config()
    secrets = _secrets()
    run_markdown_pipeline(FIXTURES / "frontmatter.md", config, secrets, cli_style_on_event)
    run_markdown_pipeline(FIXTURES / "frontmatter.md", config, secrets, api_style_on_event)

    assert cli_style_events == api_style_events == SUCCESS_SEQUENCE_MARKDOWN
