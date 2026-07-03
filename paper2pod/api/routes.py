"""FastAPI routes: validate, enqueue, and report -- zero pipeline logic here.

Both POST endpoints only ever build a per-request AppConfig (via the same
build_overrides()/load_config() the CLI uses) and a job payload, then hand
off to the shared async queue. All actual pipeline work happens in the
worker (paper2pod/api/jobs.py), which calls the exact same
run_markdown_pipeline()/run_openlabs_pipeline() functions the CLI calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from paper2pod import __version__
from paper2pod.api.auth import require_auth
from paper2pod.api.jobs import create_job, get_job
from paper2pod.config import AppConfig, build_overrides, load_config

router = APIRouter(
    prefix="/v1",
    tags=["paper2pod"],
)


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    source_type: str
    created_at: str
    updated_at: str
    result_url: str | None = None
    filename: str | None = None
    episode_id: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    worker_alive: bool
    queue_depth: int


class OpenLabsBriefRequest(BaseModel):
    url: str
    voice: str | None = None
    model: str | None = None
    cta_enabled: bool | None = None


def _resolve_request_config(request: Request, voice, model, cta_enabled) -> AppConfig:
    """Same precedence as CLI flags: this request's overrides > env > yaml > defaults."""
    overrides = build_overrides(voice=voice, model=model, cta_enabled=cta_enabled)
    return load_config(config_path=request.app.state.config_path, cli_overrides=overrides)


def _parse_overrides_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"overrides: invalid JSON ({e})") from e
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="overrides: must be a JSON object")
    allowed = {"voice", "model", "cta_enabled"}
    unknown = set(data) - allowed
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"overrides: unknown field(s) {sorted(unknown)}"
        )
    return data


@router.post(
    "/briefs/markdown",
    status_code=202,
    dependencies=[Depends(require_auth)],
    summary="Submit a markdown paper for narration",
    response_model=JobCreatedResponse,
)
async def submit_markdown_brief(request: Request) -> JobCreatedResponse:
    """Accepts a `.md` paper as either a raw `text/markdown` body or a
    `multipart/form-data` upload (field name `file`), enqueues the same
    parse -> transcript -> TTS -> upload -> record pipeline the CLI's `run`
    command uses, and returns immediately with a job id to poll.

    Optional overrides (`voice`, `model`, `cta_enabled`) can be sent as a
    JSON `overrides` form field for multipart uploads, or as query
    parameters (`?voice=...&model=...&cta_enabled=true`) for raw-body
    uploads, since a plain markdown body has no room for an embedded field.
    """
    app_config: AppConfig = request.app.state.app_config
    max_bytes = app_config.api.max_upload_mb * 1024 * 1024
    content_type = request.headers.get("content-type", "")

    overrides_data: dict = {}
    source_reference = "api-upload"

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not isinstance(upload, UploadFile):
            raise HTTPException(status_code=422, detail="file: missing multipart file part")
        raw_bytes = await upload.read()
        source_reference = upload.filename or source_reference
        overrides_part = form.get("overrides")
        if overrides_part:
            overrides_text = (
                overrides_part
                if isinstance(overrides_part, str)
                else (await overrides_part.read()).decode("utf-8")
            )
            overrides_data = _parse_overrides_json(overrides_text)
    else:
        raw_bytes = await request.body()

    if len(raw_bytes) > max_bytes:
        limit = app_config.api.max_upload_mb
        raise HTTPException(status_code=413, detail=f"Markdown body exceeds the {limit} MB limit")

    markdown_text = raw_bytes.decode("utf-8", errors="replace")
    if not markdown_text.strip():
        raise HTTPException(status_code=422, detail="body: markdown content must not be empty")

    # Query params are the override channel for the raw-body submission path,
    # where there's no room for a second JSON field inside plain markdown text.
    voice = overrides_data.get("voice") or request.query_params.get("voice")
    model = overrides_data.get("model") or request.query_params.get("model")
    cta_enabled_raw = overrides_data.get("cta_enabled")
    if cta_enabled_raw is None and "cta_enabled" in request.query_params:
        cta_enabled_raw = request.query_params["cta_enabled"].lower() in ("1", "true", "yes")
    cta_enabled = cta_enabled_raw if isinstance(cta_enabled_raw, bool) else None

    per_request_config = _resolve_request_config(request, voice, model, cta_enabled)

    tmp_dir = Path(tempfile.mkdtemp(prefix="paper2pod-api-"))
    file_path = tmp_dir / (source_reference if source_reference != "api-upload" else "paper.md")
    file_path.write_text(markdown_text, encoding="utf-8")

    job_id = create_job(app_config.api.job_db, "markdown")
    payload = {
        "file_path": file_path,
        "app_config": per_request_config,
        "source_reference": source_reference,
    }
    await request.app.state.queue.put((job_id, "markdown", payload))
    return JobCreatedResponse(job_id=job_id, status="queued")


def _validate_openlabs_url(url: str, app_config: AppConfig) -> None:
    submitted_host = urlsplit(url).netloc.lower()
    allowed_hosts = {h.lower() for h in app_config.api.allowed_openlabs_hosts}
    if not allowed_hosts:
        allowed_hosts = {urlsplit(app_config.openlabs.base_url).netloc.lower()}
    if not submitted_host or submitted_host not in allowed_hosts:
        raise HTTPException(
            status_code=422,
            detail=f"url: host '{submitted_host or url}' is not an allowed OpenLabs host",
        )


@router.post(
    "/briefs/openlabs",
    status_code=202,
    dependencies=[Depends(require_auth)],
    summary="Submit an OpenLabs project URL for narration",
    response_model=JobCreatedResponse,
)
async def submit_openlabs_brief(
    request: Request, body: OpenLabsBriefRequest
) -> JobCreatedResponse:
    """Fetches the given OpenLabs project and enqueues the same
    fetch -> transcript -> TTS -> upload -> record pipeline the CLI's
    `openlabs` command uses. The URL's host must match `openlabs.base_url`
    (or be listed in `api.allowed_openlabs_hosts`) or the request is
    rejected with 422. Optional overrides: `voice`, `model`, `cta_enabled`.
    """
    app_config: AppConfig = request.app.state.app_config
    _validate_openlabs_url(body.url, app_config)

    per_request_config = _resolve_request_config(request, body.voice, body.model, body.cta_enabled)

    job_id = create_job(app_config.api.job_db, "openlabs")
    payload = {"url": body.url, "app_config": per_request_config}
    await request.app.state.queue.put((job_id, "openlabs", payload))
    return JobCreatedResponse(job_id=job_id, status="queued")


@router.get(
    "/jobs/{job_id}",
    dependencies=[Depends(require_auth)],
    summary="Poll a job's status",
    response_model=JobStatusResponse,
)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """Returns the job's current `status` (queued/running/done/failed) and
    `stage` (parse/fetch/transcript/tts/upload/record). On `done`,
    `result_url`/`filename`/`episode_id` are populated (`episode_id` may be
    null if the Supabase episode record write failed -- the audio itself
    is still safely uploaded, `result_url` is what confirms that). On
    `failed`, `error` holds the typed exception name and a safe message.
    404 if the job id is unknown.
    """
    app_config: AppConfig = request.app.state.app_config
    row = get_job(app_config.api.job_db, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return JobStatusResponse(
        job_id=row.id,
        status=row.status,
        stage=row.stage,
        source_type=row.source_type,
        created_at=row.created_at,
        updated_at=row.updated_at,
        result_url=row.result_url,
        filename=row.filename,
        episode_id=row.episode_id,
        error=row.error,
    )


@router.get("/health", summary="Health check (no auth required)", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """No auth required. `worker_alive: false` means the background job
    worker task has died and jobs will never progress past `queued` --
    restart the server."""
    worker_task = getattr(request.app.state, "worker_task", None)
    queue = getattr(request.app.state, "queue", None)
    return HealthResponse(
        status="ok",
        version=__version__,
        worker_alive=bool(worker_task is not None and not worker_task.done()),
        queue_depth=queue.qsize() if queue is not None else 0,
    )
