"""Integration test: full openlabs pipeline with mocked fetch/LLM/TTS/Supabase clients."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paper2pod import cli as cli_module
from paper2pod import pipeline as pipeline_module
from paper2pod.config import DEFAULT_CTA_TEXT, Secrets
from paper2pod.logging_setup import SourceError
from paper2pod.sources.openlabs import ProjectContent
from paper2pod.storage import UploadResult
from paper2pod.transcript import Transcript

runner = CliRunner()

FAKE_BRIEF_TEXT = "word " * 350
PROJECT_URL = "https://openlabs.bio.xyz/projects/4b7c0a52-708e-4f5c-a319-38ed5104c964"


def _fake_project(title="Steam Collective: Where Heat Meets Science", team=None):
    return ProjectContent(
        title=title,
        team_or_authors=team or ["STEAM SPIRIT"],
        summary="A project about heat and science.",
        body_text="word " * 1240,
        url=PROJECT_URL,
        fetched_at=datetime.now(UTC),
    )


def _fake_generate(
    paper_text, metadata, style_config, secrets=None, call_fn=None, cta_config=None, style="paper"
):
    return Transcript(
        text=FAKE_BRIEF_TEXT.strip(),
        title=metadata.title,
        authors=metadata.authors,
        word_count=350,
        estimated_duration_s=140.0,
    )


def _fake_bind_provider_call(provider, secrets):
    def call_fn(model, system, messages):
        return "word " * 350

    return call_fn


class FakeTTSProvider:
    def __init__(self):
        self.calls: list[tuple[str, Path]] = []

    def synthesize(self, text, out_path):
        self.calls.append((text, out_path))
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


def test_openlabs_dry_run_prints_fetch_stage_and_transcript(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr(pipeline_module, "generate_transcript", _fake_generate)

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL, "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert "Fetched OpenLabs project" in result.stdout
    assert "Steam Collective" in result.stdout
    assert "1,240 words" in result.stdout
    assert "Transcript generated" in result.stdout


def test_openlabs_full_pipeline_uploads_title_team_filename(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr(pipeline_module, "generate_transcript", _fake_generate)

    fake_tts = FakeTTSProvider()
    monkeypatch.setattr(pipeline_module, "get_provider", lambda tts_config, secrets: fake_tts)
    monkeypatch.setattr("supabase.create_client", lambda url, key: object())

    upload_calls = []

    def fake_upload(client, bucket, object_name, local_path, upsert=False):
        upload_calls.append((bucket, object_name, local_path, upsert))
        assert local_path.exists()
        return UploadResult(
            object_path=object_name,
            url="https://fake.supabase.co/storage/v1/object/public/recordings/brief.mp3",
            is_public=True,
        )

    monkeypatch.setattr(pipeline_module, "upload_recording", fake_upload)

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL])

    assert result.exit_code == 0, result.stdout
    assert len(upload_calls) == 1
    bucket, object_name, local_path, upsert = upload_calls[0]
    assert bucket == "recordings"
    assert object_name == "Steam Collective Where Heat Meets Science - STEAM SPIRIT.mp3"
    assert "https://fake.supabase.co" in result.stdout
    assert len(fake_tts.calls) == 1


def test_openlabs_dry_run_ends_with_default_cta_text(monkeypatch):
    """Exercises the real generate_transcript() to prove openlabs wires cta through."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL, "--dry-run"])

    assert result.exit_code == 0, result.stdout
    normalized = " ".join(result.stdout.split())
    assert normalized.endswith(" ".join(DEFAULT_CTA_TEXT.split()))


def test_openlabs_dry_run_cta_disabled_via_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    config_path = tmp_path / "config.yaml"
    config_path.write_text('cta:\n  enabled: false\n  text: ""\n')

    result = runner.invoke(
        cli_module.app,
        ["openlabs", PROJECT_URL, "--config", str(config_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.stdout
    for word in DEFAULT_CTA_TEXT.split()[:6]:
        assert word not in result.stdout


def test_openlabs_dry_run_custom_cta_text_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    custom_cta = "Go explore OpenLabs right now and post your own research there."
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'cta:\n  enabled: true\n  text: "{custom_cta}"\n')

    result = runner.invoke(
        cli_module.app,
        ["openlabs", PROJECT_URL, "--config", str(config_path), "--dry-run"],
    )
    assert result.exit_code == 0, result.stdout
    normalized = " ".join(result.stdout.split())
    assert normalized.endswith(custom_cta)


def test_openlabs_full_run_synthesizes_text_ending_with_cta(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
    monkeypatch.setattr(pipeline_module, "fetch_project", lambda url, **kwargs: _fake_project())
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    fake_tts = FakeTTSProvider()
    monkeypatch.setattr(pipeline_module, "get_provider", lambda tts_config, secrets: fake_tts)
    monkeypatch.setattr("supabase.create_client", lambda url, key: object())
    monkeypatch.setattr(
        pipeline_module,
        "upload_recording",
        lambda client, bucket, object_name, local_path, upsert=False: UploadResult(
            object_path=object_name, url="https://fake.supabase.co/x.mp3", is_public=True
        ),
    )

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL])

    assert result.exit_code == 0, result.stdout
    assert len(fake_tts.calls) == 1
    synthesized_text = fake_tts.calls[0][0]
    assert synthesized_text.endswith(DEFAULT_CTA_TEXT)


def test_openlabs_fetch_failure_exits_7(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")

    def failing_fetch(url, **kwargs):
        raise SourceError("OpenLabs project page has insufficient content", input_file=url)

    monkeypatch.setattr(pipeline_module, "fetch_project", failing_fetch)

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL])
    assert result.exit_code == 7
    assert "insufficient content" in result.stdout


def test_no_cache_flag_passes_use_cache_false_to_fetch_project(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured = {}

    def fake_fetch(url, **kwargs):
        captured.update(kwargs)
        return _fake_project()

    monkeypatch.setattr(pipeline_module, "fetch_project", fake_fetch)
    monkeypatch.setattr(pipeline_module, "generate_transcript", _fake_generate)

    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL, "--no-cache", "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert captured["use_cache"] is False


def test_missing_anthropic_key_exits_2_for_openlabs_too(monkeypatch):
    result = runner.invoke(cli_module.app, ["openlabs", PROJECT_URL])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.stdout
