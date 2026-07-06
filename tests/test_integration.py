"""Integration test: full pipeline with mocked LLM, TTS, and Supabase clients.

A single live smoke test at the bottom exercises the real providers end to
end and is skipped unless PAPER2POD_LIVE=1 is set (and a valid .env exists).
"""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paper2pod import cli as cli_module
from paper2pod import pipeline as pipeline_module
from paper2pod.config import DEFAULT_CTA_TEXT, Secrets
from paper2pod.storage import UploadResult
from paper2pod.transcript import Transcript

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"

FAKE_TRANSCRIPT_TEXT = "word " * 350


class FakeTTSProvider:
    def __init__(self):
        self.calls: list[tuple[str, Path]] = []

    def synthesize(self, text: str, out_path: Path) -> Path:
        self.calls.append((text, out_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"FAKE_MP3_BYTES")
        return out_path


def _fake_generate(
    paper_text,
    metadata,
    style_config,
    secrets=None,
    call_fn=None,
    cta_config=None,
    style="paper",
    pdf_document=None,
):
    return Transcript(
        text=FAKE_TRANSCRIPT_TEXT.strip(),
        title=metadata.title,
        authors=metadata.authors,
        word_count=350,
        estimated_duration_s=140.0,
    )


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
    # Isolate from a real .env file on disk so mocked tests only see secrets
    # set explicitly via monkeypatch.setenv. The live smoke test (gated on
    # PAPER2POD_LIVE=1) needs the real .env, so it's exempted below.
    if os.environ.get("PAPER2POD_LIVE") != "1":
        monkeypatch.setattr(cli_module, "load_secrets", lambda: Secrets(_env_file=None))


def test_dry_run_stops_before_tts_and_prints_transcript(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline_module, "generate_transcript", _fake_generate)

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md"), "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert "Parsed" in result.stdout
    assert "Transcript generated" in result.stdout
    # Rich wraps long lines to terminal width, so check content, not exact whitespace.
    assert result.stdout.count("word") >= 350


def _fake_bind_provider_call(provider, secrets):
    def call_fn(model, system, messages):
        return "word " * 350

    return call_fn


def test_run_dry_run_ends_with_default_cta_text(monkeypatch):
    """Exercises the real generate_transcript() (not mocked) to prove cli.py
    actually wires app_config.cta through, end to end."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md"), "--dry-run"])

    assert result.exit_code == 0, result.stdout
    normalized = " ".join(result.stdout.split())
    assert normalized.endswith(" ".join(DEFAULT_CTA_TEXT.split()))


def test_run_dry_run_cta_disabled_via_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    config_path = tmp_path / "config.yaml"
    config_path.write_text('cta:\n  enabled: false\n  text: ""\n')

    result = runner.invoke(
        cli_module.app,
        ["run", str(FIXTURES / "frontmatter.md"), "--config", str(config_path), "--dry-run"],
    )

    assert result.exit_code == 0, result.stdout
    for word in DEFAULT_CTA_TEXT.split()[:6]:
        assert word not in result.stdout


def test_run_dry_run_custom_cta_text_appears_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    custom_cta = "Go explore OpenLabs right now and post your own research there."
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'cta:\n  enabled: true\n  text: "{custom_cta}"\n')

    result = runner.invoke(
        cli_module.app,
        ["run", str(FIXTURES / "frontmatter.md"), "--config", str(config_path), "--dry-run"],
    )

    assert result.exit_code == 0, result.stdout
    normalized = " ".join(result.stdout.split())
    assert normalized.endswith(custom_cta)


def test_run_warns_when_cta_text_exceeds_60_words(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("paper2pod.transcript._bind_provider_call", _fake_bind_provider_call)

    long_cta = " ".join(["word"] * 65)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'cta:\n  enabled: true\n  text: "{long_cta}"\n')

    result = runner.invoke(
        cli_module.app,
        ["run", str(FIXTURES / "frontmatter.md"), "--config", str(config_path), "--dry-run"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Warning" in result.stdout
    assert "65 words" in result.stdout


def test_full_pipeline_uploads_and_prints_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
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
            url="https://fake.supabase.co/storage/v1/object/public/recordings/example.mp3",
            is_public=True,
        )

    monkeypatch.setattr(pipeline_module, "upload_recording", fake_upload)

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])

    assert result.exit_code == 0, result.stdout
    assert len(upload_calls) == 1
    bucket, object_name, local_path, upsert = upload_calls[0]
    assert bucket == "recordings"
    assert object_name == (
        "Diffusion Models Beat GANs on Image Synthesis - "
        "Prafulla Dhariwal, Alex Nichol.mp3"
    )
    assert upsert is False
    assert "https://fake.supabase.co" in result.stdout
    assert len(fake_tts.calls) == 1


def test_missing_anthropic_key_exits_2_with_variable_name(monkeypatch):
    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.stdout


def test_missing_file_exits_3(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    result = runner.invoke(
        cli_module.app, ["run", str(FIXTURES / "does_not_exist.md"), "--dry-run"]
    )
    assert result.exit_code == 3


def test_tts_failure_exits_5(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
    monkeypatch.setattr(pipeline_module, "generate_transcript", _fake_generate)

    class FailingTTSProvider:
        def synthesize(self, text, out_path):
            from paper2pod.logging_setup import TTSError

            raise TTSError("network unreachable")

    monkeypatch.setattr(
        pipeline_module, "get_provider", lambda tts_config, secrets: FailingTTSProvider()
    )

    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "frontmatter.md")])
    assert result.exit_code == 5

    log_path = Path("logs/paper2pod.log")
    assert log_path.exists()
    assert "network unreachable" in log_path.read_text()


@pytest.mark.skipif(
    os.environ.get("PAPER2POD_LIVE") != "1",
    reason="live smoke test disabled; set PAPER2POD_LIVE=1 with a valid .env to run",
)
def test_live_smoke_dry_run():
    result = runner.invoke(cli_module.app, ["run", str(FIXTURES / "sample_paper.md"), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert 250 <= len(result.stdout.split()) <= 550
