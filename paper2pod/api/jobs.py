"""SQLite job store + single-worker async queue for the FastAPI job model.

jobs.db tracks ephemeral request/job status (queued/running/done/failed)
for polling. This is entirely separate from the Supabase `episodes` table,
which is the durable content record written by record_episode() as the
last step of the shared pipeline (paper2pod/pipeline.py) -- the same call
the CLI already triggers. Do not conflate the two or route episode data
through this store.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from paper2pod.config import AppConfig, Secrets
from paper2pod.logging_setup import (
    ParseError,
    SourceError,
    StorageError,
    TranscriptError,
    TTSError,
)
from paper2pod.pipeline import (
    StageEvent,
    run_markdown_pipeline,
    run_openlabs_pipeline,
    run_pdf_pipeline,
)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_TYPED_PIPELINE_EXCEPTIONS = (
    ParseError,
    SourceError,
    TranscriptError,
    TTSError,
    StorageError,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    result_url TEXT,
    filename TEXT,
    episode_id TEXT,
    error TEXT
)
"""


@dataclass
class JobRow:
    id: str
    source_type: str
    status: str
    stage: str | None
    created_at: str
    updated_at: str
    result_url: str | None = None
    filename: str | None = None
    episode_id: str | None = None
    error: str | None = None


def init_db(db_path: str | Path) -> None:
    """Create the jobs table if it doesn't exist yet. Call once at API startup."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_job(db_path: str | Path, source_type: str) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs (id, source_type, status, stage, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, source_type, STATUS_QUEUED, None, now, now),
        )
    return job_id


def update_job(db_path: str | Path, job_id: str, **fields: Any) -> None:
    """Update arbitrary job columns (status, stage, result_url, filename, episode_id, error).

    Column names always come from our own code, never from request input,
    so building the SET clause from fields.keys() is safe here.
    """
    if not fields:
        return
    fields = {**fields, "updated_at": _now()}
    columns = ", ".join(f"{key} = ?" for key in fields)
    values = [*fields.values(), job_id]
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)


def get_job(db_path: str | Path, job_id: str) -> JobRow | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return JobRow(**dict(row))


def purge_old_jobs(db_path: str | Path, retention_days: int) -> int:
    """Delete jobs created before the retention window. Returns rows deleted."""
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
        return cursor.rowcount


class JobStoreListener:
    """StageEvent consumer for the API: logs with job_id context, tracks stage progress.

    Deliberately never sets job `status` -- status transitions (running ->
    done/failed) are owned solely by _process_job below, avoiding two code
    paths racing to write status. This also means a stage="record",
    status="failed" event (a RecordError) never flips the job to failed --
    it just logs -- enforced structurally rather than as a special case.
    """

    def __init__(self, db_path: str | Path, job_id: str, logger: logging.Logger):
        self.db_path = db_path
        self.job_id = job_id
        self.logger = logger

    def __call__(self, event: StageEvent) -> None:
        update_job(self.db_path, self.job_id, stage=event.stage)
        if event.status == "failed":
            error = event.data.get("error", event.message)
            self.logger.error(f"[job {self.job_id}] {event.stage} failed: {error}")
        else:
            self.logger.info(f"[job {self.job_id}] {event.stage} {event.status}")


def _process_job(
    job_id: str,
    job_kind: str,
    payload: dict[str, Any],
    secrets: Secrets,
    db_path: str | Path,
    logger: logging.Logger,
) -> None:
    """Run one job's pipeline synchronously (invoked via asyncio.to_thread)."""
    update_job(db_path, job_id, status=STATUS_RUNNING)
    listener = JobStoreListener(db_path, job_id, logger)
    app_config: AppConfig = payload["app_config"]

    try:
        if job_kind == "markdown":
            result = run_markdown_pipeline(
                payload["file_path"],
                app_config,
                secrets,
                listener,
                source_reference=payload.get("source_reference"),
            )
        elif job_kind == "pdf":
            result = run_pdf_pipeline(
                payload["file_path"],
                app_config,
                secrets,
                listener,
                source_reference=payload.get("source_reference"),
            )
        else:
            result = run_openlabs_pipeline(payload["url"], app_config, secrets, listener)
    except _TYPED_PIPELINE_EXCEPTIONS as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"[job {job_id}] failed: {error}")
        update_job(db_path, job_id, status=STATUS_FAILED, error=error)
        return

    filename = Path(result.upload_result.object_path).name
    update_job(
        db_path,
        job_id,
        status=STATUS_DONE,
        result_url=result.upload_result.url,
        filename=filename,
        episode_id=result.episode_id,
    )


async def worker_loop(
    queue: asyncio.Queue,
    secrets: Secrets,
    db_path: str | Path,
    logger: logging.Logger,
) -> None:
    """Single sequential worker: awaits each job before dequeuing the next.

    The outer try/except is the "job failure must never crash the worker"
    safety net -- independent of _process_job's own typed-exception
    handling, which already covers the expected pipeline failure modes.
    """
    while True:
        job_id, job_kind, payload = await queue.get()
        try:
            await asyncio.to_thread(
                _process_job, job_id, job_kind, payload, secrets, db_path, logger
            )
        except Exception:
            logger.exception(f"[job {job_id}] unexpected worker error")
        finally:
            queue.task_done()
