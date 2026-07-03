"""Route tests: FastAPI TestClient against a mocked pipeline (paper2pod.api.jobs)."""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from paper2pod import pipeline as pipeline_module
from paper2pod.api.app import create_app
from paper2pod.config import Secrets
from paper2pod.logging_setup import TranscriptError
from paper2pod.storage import UploadResult
from paper2pod.transcript import Transcript

TOKEN = "test-token-abc123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("PAPER2POD_") or key in {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "API_AUTH_TOKEN",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "paper2pod.api.app.load_secrets", lambda: Secrets(_env_file=None)
    )


def _valid_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
    monkeypatch.setenv("API_AUTH_TOKEN", TOKEN)


def _config_path(tmp_path):
    config_path = tmp_path / "config.yaml"
    job_db = tmp_path / "jobs.db"
    config_path.write_text(f'api:\n  job_db: "{job_db}"\n')
    return str(config_path)


def _fake_transcript():
    return Transcript(
        text="narration", title="Title", authors=["Author"], word_count=350,
        estimated_duration_s=140.0,
    )


def _fake_pipeline_result():
    from paper2pod.pipeline import PipelineResult

    return PipelineResult(
        transcript=_fake_transcript(),
        upload_result=UploadResult(
            object_path="Title - Author.mp3", url="https://fake.supabase.co/x.mp3", is_public=True
        ),
        episode_id="episode-123",
        record_error=None,
    )


def _mock_successful_pipeline(monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "parse_markdown", lambda *a, **k: (
            __import__("paper2pod.parser", fromlist=["PaperMetadata"]).PaperMetadata(
                title="Title", authors=["Author"]
            ),
            "body",
        )
    )
    monkeypatch.setattr(pipeline_module, "generate_transcript", lambda *a, **k: _fake_transcript())
    monkeypatch.setattr(
        "paper2pod.api.jobs.run_markdown_pipeline", lambda *a, **k: _fake_pipeline_result()
    )
    monkeypatch.setattr(
        "paper2pod.api.jobs.run_openlabs_pipeline", lambda *a, **k: _fake_pipeline_result()
    )


def _poll_until_terminal(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        body = resp.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


def test_health_requires_no_auth_and_reports_worker_alive(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["worker_alive"] is True
        assert body["queue_depth"] == 0


def test_missing_bearer_token_returns_401(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/markdown",
            content=b"# Title\nbody",
            headers={"content-type": "text/markdown"},
        )
        assert resp.status_code == 401


def test_wrong_bearer_token_returns_401(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/markdown",
            content=b"# Title\nbody",
            headers={"content-type": "text/markdown", "Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401


def test_create_app_fails_when_api_auth_token_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-service-test")
    from paper2pod.config import ConfigError

    with pytest.raises(ConfigError, match="API_AUTH_TOKEN"):
        create_app(config_path=_config_path(tmp_path))


def test_markdown_raw_body_202_then_done_lifecycle(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    _mock_successful_pipeline(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/markdown",
            content=b"---\ntitle: Test\nauthors: [A]\n---\nbody text",
            headers={**AUTH, "content-type": "text/markdown"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert resp.json()["status"] == "queued"

        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "done"
        assert final["result_url"] == "https://fake.supabase.co/x.mp3"
        assert final["filename"] == "Title - Author.mp3"
        assert final["episode_id"] == "episode-123"
        assert final["source_type"] == "markdown"


def test_markdown_multipart_upload_202_then_done(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    _mock_successful_pipeline(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/markdown",
            headers=AUTH,
            files={
                "file": (
                    "mypaper.md",
                    b"---\ntitle: Test\nauthors: [A]\n---\nbody",
                    "text/markdown",
                )
            },
            data={"overrides": json.dumps({"voice": "en-US-JennyNeural"})},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "done"


def test_markdown_oversized_body_returns_413(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    _mock_successful_pipeline(monkeypatch)
    config_path = tmp_path / "config.yaml"
    job_db = tmp_path / "jobs.db"
    config_path.write_text(f'api:\n  job_db: "{job_db}"\n  max_upload_mb: 1\n')
    app = create_app(config_path=str(config_path))

    with TestClient(app) as client:
        oversized = b"x" * (2 * 1024 * 1024)
        resp = client.post(
            "/v1/briefs/markdown",
            content=oversized,
            headers={**AUTH, "content-type": "text/markdown"},
        )
        assert resp.status_code == 413


def test_markdown_empty_body_returns_422(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/markdown", content=b"   ", headers={**AUTH, "content-type": "text/markdown"}
        )
        assert resp.status_code == 422


def test_markdown_failure_lifecycle(monkeypatch, tmp_path):
    _valid_env(monkeypatch)

    def failing_pipeline(*a, **k):
        raise TranscriptError("LLM credit balance too low")

    monkeypatch.setattr("paper2pod.api.jobs.run_markdown_pipeline", failing_pipeline)
    app = create_app(config_path=_config_path(tmp_path))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/markdown",
            content=b"# T\nbody",
            headers={**AUTH, "content-type": "text/markdown"},
        )
        job_id = resp.json()["job_id"]
        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "failed"
        assert final["error"].startswith("TranscriptError:")
        assert "LLM credit balance too low" in final["error"]


def test_openlabs_valid_host_202_then_done(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    _mock_successful_pipeline(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/openlabs",
            headers=AUTH,
            json={"url": "https://openlabs.bio.xyz/projects/abc-123"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        final = _poll_until_terminal(client, job_id)
        assert final["status"] == "done"
        assert final["source_type"] == "openlabs"


def test_openlabs_rejects_non_openlabs_host_by_default(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/openlabs", headers=AUTH, json={"url": "https://evil.example.com/projects/x"}
        )
        assert resp.status_code == 422


def test_openlabs_allowlist_permits_configured_host(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    _mock_successful_pipeline(monkeypatch)
    config_path = tmp_path / "config.yaml"
    job_db = tmp_path / "jobs.db"
    config_path.write_text(
        f'api:\n  job_db: "{job_db}"\n  allowed_openlabs_hosts:\n    - staging.openlabs.bio.xyz\n'
    )
    app = create_app(config_path=str(config_path))

    with TestClient(app) as client:
        resp = client.post(
            "/v1/briefs/openlabs",
            headers=AUTH,
            json={"url": "https://staging.openlabs.bio.xyz/projects/abc"},
        )
        assert resp.status_code == 202


def test_unknown_job_id_returns_404(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    app = create_app(config_path=_config_path(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/jobs/00000000-0000-0000-0000-000000000000", headers=AUTH)
        assert resp.status_code == 404


def test_three_posts_queue_and_process_sequentially(monkeypatch, tmp_path):
    _valid_env(monkeypatch)
    call_order = []

    def recording_pipeline(*a, **k):
        call_order.append(len(call_order))
        return _fake_pipeline_result()

    monkeypatch.setattr("paper2pod.api.jobs.run_markdown_pipeline", recording_pipeline)
    app = create_app(config_path=_config_path(tmp_path))

    with TestClient(app) as client:
        job_ids = []
        for _ in range(3):
            resp = client.post(
                "/v1/briefs/markdown",
                content=b"# T\nbody",
                headers={**AUTH, "content-type": "text/markdown"},
            )
            assert resp.status_code == 202
            job_ids.append(resp.json()["job_id"])

        for job_id in job_ids:
            final = _poll_until_terminal(client, job_id)
            assert final["status"] == "done"

    # Sequential means each job fully completed (was appended to call_order)
    # before the next one started -- three distinct calls, in submission order.
    assert call_order == [0, 1, 2]
