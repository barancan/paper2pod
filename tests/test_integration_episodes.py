"""Integration test: episode recording after a successful pipeline run.

Covers both the run (markdown) and openlabs commands with mocked
LLM/TTS/Supabase clients, asserting record_episode is called exactly once
after the upload step with field values pulled from the run's actual
config, and that a record_episode failure never fails the pipeline.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paper2pod import cli as cli_module
from paper2pod.config import Secrets, load_config
from paper2pod.db import EpisodeRecord
from paper2pod.logging_setup import RecordError
from paper2pod.sources.openlabs import ProjectContent
from paper2pod.storage import UploadResult
from paper2pod.transcript import Transcript

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"

TRANSCRIPT_TEXT = ("word " * 350).strip()
PROJECT_URL = "https://openlabs.bio.xyz/projects/4b7c0a52-708e-4f5c-a319-38ed5104c964"


def _fake_generate(
    paper_text, metadata, style_config, secrets=None, call_fn=None, cta_config=None, style="paper"
):
    return Transcript(
        text=TRANSCRIPT_TEXT,
        title=metadata.title,
        authors=metadata.authors,
        word_count=350,
        estimated_duration_s=140.0,
    )


def _fake_project():
    return ProjectContent(
        title="Steam Collective: Where Heat Meets Science",
        team_or_authors=["STEAM SPIRIT"],
        summary="A project about heat and science.",
        body_text="word " * 1240,
        url=PROJECT_URL,
        fetched_at=datetime.now(UTC),
    )


class FakeTTSProvider:
    def synthesize(self, text, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"FAKE_MP3_BYTES")
        return out_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PAPER2POD_") or key in {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
        }:
            monkeypatch.delenv(key, raising=False)
    if os.environ.get("PAPER2POD_LIVE") != "1":
        monkeypatch.setattr(cli_module, "load_secrets", lambda: Secrets(_env_file=None))


def _setup_common_mocks(monkeypatch, is_public=True):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
    monkeypatch.setattr(cli_module, "generate_transcript", _fake_generate)
    monkeypatch.setattr(cli_module, "get_provider", lambda tts_config, secrets: FakeTTSProvider())
    fake_client = object()
    monkeypatch.setattr("supabase.create_client", lambda url, key: fake_client)

    def fake_upload(client, bucket, object_name, local_path, upsert=False):
        return UploadResult(
            object_path=object_name,
            url="https://fake.supabase.co/storage/v1/object/public/recordings/episode.mp3",
            is_public=is_public,
        )

    monkeypatch.setattr(cli_module, "upload_recording", fake_upload)
    return fake_client


def test_run_records_episode_once_after_upload_with_correct_fields(monkeypatch):
    fake_client = _setup_common_mocks(monkeypatch, is_public=True)
    calls = []

    def fake_record_episode(client, record):
        calls.append((client, record))
        return "new-episode-id"

    monkeypatch.setattr(cli_module, "record_episode", fake_record_episode)

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])

    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1
    client, record = calls[0]
    assert client is fake_client
    assert isinstance(record, EpisodeRecord)

    expected_config = load_config()
    assert record.source_type == "markdown"
    assert record.source_reference == str(FIXTURES / "frontmatter.md")
    assert record.title == "Diffusion Models Beat GANs on Image Synthesis"
    assert record.authors_or_team == "Prafulla Dhariwal, Alex Nichol"
    assert record.transcript_text == TRANSCRIPT_TEXT
    assert record.word_count == 350
    assert record.estimated_duration_seconds == 140
    assert record.transcript_provider == expected_config.transcript.provider
    assert record.transcript_model == expected_config.transcript.model
    assert record.tts_voice == expected_config.tts.voice
    assert record.audio_bucket == expected_config.storage.bucket
    assert record.audio_object_path == (
        "Diffusion Models Beat GANs on Image Synthesis - "
        "Prafulla Dhariwal, Alex Nichol.mp3"
    )
    assert record.episode_name == (
        "Diffusion Models Beat GANs on Image Synthesis - Prafulla Dhariwal, Alex Nichol"
    )
    assert record.audio_public_url == (
        "https://fake.supabase.co/storage/v1/object/public/recordings/episode.mp3"
    )
    if expected_config.cta.enabled:
        assert record.cta_text == expected_config.cta.text
    assert "✔ Episode recorded (id: new-episode-id)" in result.stdout


def test_record_episode_called_after_upload_mock(monkeypatch):
    _setup_common_mocks(monkeypatch, is_public=True)
    call_order = []

    def fake_upload(client, bucket, object_name, local_path, upsert=False):
        call_order.append("upload")
        return UploadResult(
            object_path=object_name, url="https://fake.supabase.co/x.mp3", is_public=True
        )

    def fake_record_episode(client, record):
        call_order.append("record")
        return "id"

    monkeypatch.setattr(cli_module, "upload_recording", fake_upload)
    monkeypatch.setattr(cli_module, "record_episode", fake_record_episode)

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])

    assert result.exit_code == 0, result.stdout
    assert call_order == ["upload", "record"]


def test_audio_public_url_none_when_bucket_private(monkeypatch):
    _setup_common_mocks(monkeypatch, is_public=False)
    calls = []
    monkeypatch.setattr(
        cli_module, "record_episode", lambda client, record: calls.append(record) or "id"
    )

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])

    assert result.exit_code == 0, result.stdout
    assert calls[0].audio_public_url is None


def test_openlabs_records_episode_with_source_type_openlabs(monkeypatch):
    fake_client = _setup_common_mocks(monkeypatch, is_public=True)
    monkeypatch.setattr(cli_module, "fetch_project", lambda url, **kwargs: _fake_project())
    calls = []

    def fake_record_episode(client, record):
        calls.append((client, record))
        return "id"

    monkeypatch.setattr(cli_module, "record_episode", fake_record_episode)

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL])

    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1
    client, record = calls[0]
    assert client is fake_client
    assert record.source_type == "openlabs"
    assert record.source_reference == PROJECT_URL
    assert record.title == "Steam Collective: Where Heat Meets Science"


def test_record_episode_failure_does_not_fail_pipeline_and_prints_audio_url(monkeypatch):
    _setup_common_mocks(monkeypatch, is_public=True)

    def failing_record_episode(client, record):
        raise RecordError("Supabase unreachable")

    monkeypatch.setattr(cli_module, "record_episode", failing_record_episode)

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])

    assert result.exit_code == 0, result.stdout
    assert "Episode record failed" in result.stdout
    assert "audio uploaded successfully" in result.stdout.lower()
    audio_url = "https://fake.supabase.co/storage/v1/object/public/recordings/episode.mp3"
    assert audio_url in result.stdout


def test_dry_run_never_calls_record_episode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(cli_module, "generate_transcript", _fake_generate)
    calls = []
    monkeypatch.setattr(
        cli_module, "record_episode", lambda client, record: calls.append(record) or "id"
    )

    result = runner.invoke(
        cli_module.app, ["run", str(FIXTURES / "frontmatter.md"), "--dry-run"]
    )

    assert result.exit_code == 0, result.stdout
    assert calls == []
