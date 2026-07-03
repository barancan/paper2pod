"""Unit tests for the SQLite job store and the async worker's job processing."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from paper2pod.api.jobs import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    JobStoreListener,
    _process_job,
    create_job,
    get_job,
    init_db,
    purge_old_jobs,
    update_job,
    worker_loop,
)
from paper2pod.config import Secrets, load_config
from paper2pod.logging_setup import TranscriptError
from paper2pod.pipeline import PipelineResult, StageEvent
from paper2pod.storage import UploadResult
from paper2pod.transcript import Transcript


class FakeLogger:
    def __init__(self):
        self.info_lines: list[str] = []
        self.error_lines: list[str] = []

    def info(self, msg):
        self.info_lines.append(msg)

    def error(self, msg):
        self.error_lines.append(msg)

    def exception(self, msg):
        self.error_lines.append(msg)


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    init_db(db_path)  # must not raise on the second call


def test_create_job_returns_valid_uuid_and_queued_status(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)

    job_id = create_job(db_path, "markdown")

    assert uuid.UUID(job_id)  # parses without error
    row = get_job(db_path, job_id)
    assert row is not None
    assert row.status == STATUS_QUEUED
    assert row.source_type == "markdown"
    assert row.stage is None


def test_update_job_sets_fields_and_bumps_updated_at(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "openlabs")
    original = get_job(db_path, job_id)

    update_job(db_path, job_id, status=STATUS_RUNNING, stage="transcript")

    updated = get_job(db_path, job_id)
    assert updated.status == STATUS_RUNNING
    assert updated.stage == "transcript"
    assert updated.updated_at >= original.updated_at


def test_get_job_returns_none_for_unknown_id(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    assert get_job(db_path, str(uuid.uuid4())) is None


def test_purge_old_jobs_deletes_only_stale_rows(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    old_id = create_job(db_path, "markdown")
    new_id = create_job(db_path, "markdown")

    stale_created_at = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (stale_created_at, old_id))

    deleted = purge_old_jobs(db_path, retention_days=30)

    assert deleted == 1
    assert get_job(db_path, old_id) is None
    assert get_job(db_path, new_id) is not None


def test_job_store_listener_updates_stage_and_logs(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "markdown")
    logger = FakeLogger()
    listener = JobStoreListener(db_path, job_id, logger)

    listener(StageEvent(stage="transcript", status="started", message="Generating..."))

    row = get_job(db_path, job_id)
    assert row.stage == "transcript"
    assert row.status == STATUS_QUEUED  # listener never touches status
    assert any("transcript started" in line for line in logger.info_lines)


def test_job_store_listener_failed_event_logs_but_never_sets_status(tmp_path):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "markdown")
    logger = FakeLogger()
    listener = JobStoreListener(db_path, job_id, logger)

    listener(
        StageEvent(
            stage="record",
            status="failed",
            message="record failed",
            data={"error": "Supabase unreachable"},
        )
    )

    row = get_job(db_path, job_id)
    assert row.status == STATUS_QUEUED  # not flipped to failed
    assert any(
        f"[job {job_id}]" in line and "Supabase unreachable" in line
        for line in logger.error_lines
    )


def _fake_pipeline_result(episode_id="episode-123"):
    return PipelineResult(
        transcript=Transcript(
            text="narration", title="Title", authors=["Author"], word_count=350,
            estimated_duration_s=140.0,
        ),
        upload_result=UploadResult(
            object_path="Title - Author.mp3", url="https://fake.supabase.co/x.mp3", is_public=True
        ),
        episode_id=episode_id,
        record_error=None,
    )


def test_process_job_success_marks_done_with_result_fields(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "markdown")

    monkeypatch.setattr(
        "paper2pod.api.jobs.run_markdown_pipeline",
        lambda *a, **k: _fake_pipeline_result(),
    )

    payload = {
        "file_path": tmp_path / "paper.md",
        "app_config": load_config(),
        "source_reference": "paper.md",
    }
    _process_job(job_id, "markdown", payload, Secrets(_env_file=None), db_path, FakeLogger())

    row = get_job(db_path, job_id)
    assert row.status == STATUS_DONE
    assert row.result_url == "https://fake.supabase.co/x.mp3"
    assert row.filename == "Title - Author.mp3"
    assert row.episode_id == "episode-123"
    assert row.error is None


def test_process_job_success_with_null_episode_id_is_still_done(tmp_path, monkeypatch):
    """A RecordError inside the pipeline yields episode_id=None but the job is still done."""
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "markdown")

    monkeypatch.setattr(
        "paper2pod.api.jobs.run_markdown_pipeline",
        lambda *a, **k: _fake_pipeline_result(episode_id=None),
    )

    payload = {"file_path": tmp_path / "paper.md", "app_config": load_config()}
    _process_job(job_id, "markdown", payload, Secrets(_env_file=None), db_path, FakeLogger())

    row = get_job(db_path, job_id)
    assert row.status == STATUS_DONE
    assert row.episode_id is None
    assert row.result_url == "https://fake.supabase.co/x.mp3"


def test_process_job_typed_failure_marks_failed_with_error_name(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "markdown")

    def failing_pipeline(*a, **k):
        raise TranscriptError("LLM credit balance too low")

    monkeypatch.setattr("paper2pod.api.jobs.run_markdown_pipeline", failing_pipeline)

    payload = {"file_path": tmp_path / "paper.md", "app_config": load_config()}
    _process_job(job_id, "markdown", payload, Secrets(_env_file=None), db_path, FakeLogger())

    row = get_job(db_path, job_id)
    assert row.status == STATUS_FAILED
    assert row.error.startswith("TranscriptError:")
    assert "LLM credit balance too low" in row.error


def test_process_job_openlabs_kind_calls_openlabs_pipeline(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_id = create_job(db_path, "openlabs")

    called = {}

    def fake_openlabs_pipeline(url, app_config, secrets, listener, **kwargs):
        called["url"] = url
        return _fake_pipeline_result()

    monkeypatch.setattr("paper2pod.api.jobs.run_openlabs_pipeline", fake_openlabs_pipeline)

    payload = {"url": "https://openlabs.bio.xyz/projects/abc", "app_config": load_config()}
    _process_job(job_id, "openlabs", payload, Secrets(_env_file=None), db_path, FakeLogger())

    assert called["url"] == "https://openlabs.bio.xyz/projects/abc"
    assert get_job(db_path, job_id).status == STATUS_DONE


@pytest.mark.asyncio
async def test_worker_loop_processes_jobs_sequentially_and_continues_after_failure(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "jobs.db"
    init_db(db_path)
    job_ids = [create_job(db_path, "markdown") for _ in range(3)]

    call_order = []

    def fake_pipeline(*a, **k):
        job_index = len(call_order)
        call_order.append(job_index)
        if job_index == 1:
            raise TranscriptError("boom")
        return _fake_pipeline_result(episode_id=f"episode-{job_index}")

    monkeypatch.setattr("paper2pod.api.jobs.run_markdown_pipeline", fake_pipeline)

    queue: asyncio.Queue = asyncio.Queue()
    for job_id in job_ids:
        payload = {"file_path": tmp_path / "paper.md", "app_config": load_config()}
        await queue.put((job_id, "markdown", payload))

    worker_task = asyncio.create_task(
        worker_loop(queue, Secrets(_env_file=None), db_path, FakeLogger())
    )
    await asyncio.wait_for(queue.join(), timeout=5)
    worker_task.cancel()

    assert call_order == [0, 1, 2]  # processed one at a time, in order
    assert get_job(db_path, job_ids[0]).status == STATUS_DONE
    assert get_job(db_path, job_ids[1]).status == STATUS_FAILED
    assert get_job(db_path, job_ids[2]).status == STATUS_DONE  # worker kept going after the failure
