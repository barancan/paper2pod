"""Episodes persistence layer: the `episodes` table in Supabase Postgres.

Uses the same supabase-py client already used for storage uploads, via
`client.table("episodes")` -- no new dependency, no client construction of
its own. Functions take an explicit `client` (dependency injection, same
pattern as parser.py's llm_extractor / transcript.py's call_fn) so this
module has no CLI-specific coupling and can be called from any future
caller (e.g. an HTTP API) exactly as-is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paper2pod.logging_setup import RecordError

TABLE_NAME = "episodes"


@dataclass
class EpisodeRecord:
    episode_name: str
    source_type: str  # "markdown" | "openlabs" | "pdf"
    source_reference: str
    title: str
    authors_or_team: str
    transcript_text: str
    word_count: int
    estimated_duration_seconds: int
    transcript_provider: str
    transcript_model: str
    tts_voice: str
    audio_bucket: str
    audio_object_path: str
    cta_text: str | None = None
    audio_public_url: str | None = None
    id: str | None = None  # server-generated; None on insert
    created_at: str | None = None  # server-generated; None on insert


def _is_retryable(exc: BaseException) -> bool:
    """Retry only on 429/5xx/network errors; other failures fail fast."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code == 429 or status_code >= 500
    return "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__


def _call_with_retry(fn: Callable[[], Any]) -> Any:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _do() -> Any:
        return fn()

    return _do()


def _row_to_record(row: dict[str, Any]) -> EpisodeRecord:
    return EpisodeRecord(
        episode_name=row["episode_name"],
        source_type=row["source_type"],
        source_reference=row["source_reference"],
        title=row["title"],
        authors_or_team=row["authors_or_team"],
        transcript_text=row["transcript_text"],
        word_count=row["word_count"],
        estimated_duration_seconds=row["estimated_duration_seconds"],
        transcript_provider=row["transcript_provider"],
        transcript_model=row["transcript_model"],
        tts_voice=row["tts_voice"],
        audio_bucket=row["audio_bucket"],
        audio_object_path=row["audio_object_path"],
        cta_text=row.get("cta_text"),
        audio_public_url=row.get("audio_public_url"),
        id=row.get("id"),
        created_at=row.get("created_at"),
    )


def record_episode(client: Any, record: EpisodeRecord) -> str:
    """Insert a new episode row, returning its id. Raises RecordError on failure."""
    payload = {k: v for k, v in asdict(record).items() if k not in ("id", "created_at")}

    try:
        response = _call_with_retry(
            lambda: client.table(TABLE_NAME).insert(payload).execute()
        )
    except Exception as e:
        raise RecordError(f"Failed to record episode: {e}") from e

    data = getattr(response, "data", None) or []
    if not data:
        raise RecordError("Episode insert succeeded but no row was returned")
    return data[0]["id"]


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except ValueError:
        return False


def get_episode(client: Any, id_or_name: str) -> EpisodeRecord | None:
    """Look up an episode by id, exact episode_name, or partial episode_name.

    If id_or_name parses as a UUID, look up by id. Otherwise try an exact
    episode_name match first, then fall back to a case-insensitive partial
    match, returning the most recently created match in either case.
    """
    try:
        if _is_uuid(id_or_name):
            response = _call_with_retry(
                lambda: client.table(TABLE_NAME).select("*").eq("id", id_or_name).limit(1).execute()
            )
            data = getattr(response, "data", None) or []
            return _row_to_record(data[0]) if data else None

        response = _call_with_retry(
            lambda: client.table(TABLE_NAME)
            .select("*")
            .eq("episode_name", id_or_name)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        data = getattr(response, "data", None) or []
        if data:
            return _row_to_record(data[0])

        response = _call_with_retry(
            lambda: client.table(TABLE_NAME)
            .select("*")
            .ilike("episode_name", f"%{id_or_name}%")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        data = getattr(response, "data", None) or []
        return _row_to_record(data[0]) if data else None
    except Exception as e:
        raise RecordError(f"Failed to look up episode '{id_or_name}': {e}") from e


def list_episodes(client: Any, limit: int = 20) -> list[EpisodeRecord]:
    """List the most recently recorded episodes first."""
    try:
        response = _call_with_retry(
            lambda: client.table(TABLE_NAME)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as e:
        raise RecordError(f"Failed to list episodes: {e}") from e

    data = getattr(response, "data", None) or []
    return [_row_to_record(row) for row in data]
