"""FastAPI app factory: config/secrets loading, startup validation, worker lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from paper2pod.api.jobs import init_db, purge_old_jobs, worker_loop
from paper2pod.config import (
    ConfigError,
    load_config,
    load_secrets,
    validate_api_secrets,
    validate_secrets,
)
from paper2pod.logging_setup import setup_logging


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Build the FastAPI app.

    Validates config/secrets eagerly (raises ConfigError on failure -- the
    caller, `paper2pod serve`, converts that to the same fail-fast exit 2 as
    every other command). No accidental open server: a missing
    API_AUTH_TOKEN is a hard startup failure, not a runtime 401 surprise.
    """
    app_config = load_config(config_path=config_path)
    secrets = load_secrets()
    validate_api_secrets(secrets)
    try:
        validate_secrets(secrets, app_config, dry_run=False)
    except ConfigError as e:
        raise ConfigError(f"{e} (required for `paper2pod serve` to process any job)") from e

    secret_values = [
        v
        for v in (
            secrets.anthropic_api_key,
            secrets.openai_api_key,
            secrets.elevenlabs_api_key,
            secrets.supabase_service_role_key,
            secrets.api_auth_token,
        )
        if v
    ]
    logger = setup_logging(
        log_file=app_config.logging.file, level=app_config.logging.level, secrets=secret_values
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(app_config.api.job_db)
        purge_old_jobs(app_config.api.job_db, app_config.api.job_retention_days)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        app.state.queue = queue
        worker_task = asyncio.create_task(
            worker_loop(queue, secrets, app_config.api.job_db, logger)
        )
        app.state.worker_task = worker_task
        try:
            yield
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="paper2pod",
        description=(
            "Convert research papers and OpenLabs projects into narrated, "
            "uploaded audio via async jobs."
        ),
        openapi_tags=[
            {
                "name": "paper2pod",
                "description": (
                    "Submit a markdown paper or an OpenLabs project URL and get back a "
                    "narrated, uploaded audio brief. POST endpoints enqueue an async job "
                    "and return 202 immediately; poll GET /v1/jobs/{job_id} for progress "
                    "and the final result. All routes require a bearer token except "
                    "/v1/health."
                ),
            }
        ],
        lifespan=lifespan,
    )
    app.state.app_config = app_config
    app.state.secrets = secrets
    app.state.logger = logger
    app.state.config_path = config_path
    app.state.api_auth_token = secrets.api_auth_token

    from paper2pod.api.routes import router

    app.include_router(router)
    return app
