"""OpenLabs project fetching: JSON API client, extraction, and local caching.

The OpenLabs app (openlabs.bio.xyz) is a client-rendered Next.js app with no
server-embedded project data. Its frontend calls a separate public JSON API
at api.openlabs.bio.xyz (no auth required), discovered by inspecting the
app's JS bundles -- see README for details. This module talks to that API
directly rather than scraping rendered HTML or driving a headless browser.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from paper2pod.logging_setup import SourceError

DEFAULT_CACHE_DIR = Path("cache/openlabs")
FetchFn = Callable[[str], Any]


@dataclass
class ProjectContent:
    title: str
    team_or_authors: list[str] = field(default_factory=lambda: ["OpenLabs"])
    summary: str = ""
    body_text: str = ""
    url: str = ""
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _derive_api_base(base_url: str) -> str:
    """openlabs.bio.xyz (the app) -> api.openlabs.bio.xyz (the JSON API)."""
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.netloc
    if not host.startswith("api."):
        host = f"api.{host}"
    return urllib.parse.urlunsplit((parsed.scheme, host, "", "", ""))


def _extract_project_id(url: str) -> str:
    path = urllib.parse.urlsplit(url).path
    segments = [s for s in path.split("/") if s]
    if not segments:
        raise SourceError(f"Could not extract a project id from URL: {url}", input_file=url)
    return segments[-1]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, urllib.error.URLError)


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "paper2pod"}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def _call_with_retry(getter: FetchFn, url: str) -> Any:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _do() -> Any:
        return getter(url)

    return _do()


def _cache_path(cache_dir: Path, project_id: str) -> Path:
    return cache_dir / f"{project_id}.json"


def _read_cache(path: Path, ttl_hours: int) -> ProjectContent | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(data["fetched_at"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None
    if datetime.now(UTC) - fetched_at > timedelta(hours=ttl_hours):
        return None
    return ProjectContent(
        title=data["title"],
        team_or_authors=data["team_or_authors"],
        summary=data["summary"],
        body_text=data["body_text"],
        url=data["url"],
        fetched_at=fetched_at,
    )


def _write_cache(path: Path, project: ProjectContent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(project)
    payload["fetched_at"] = project.fetched_at.isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_team(detail: dict[str, Any], collaborators: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for collab in collaborators:
        profile = collab.get("profile") or {}
        name = profile.get("display_name") or profile.get("handle")
        if name and name not in names:
            names.append(name)
    if names:
        return names

    creator = detail.get("creator") or {}
    name = creator.get("display_name") or creator.get("handle")
    return [name] if name else ["OpenLabs"]


def _build_body_text(detail: dict[str, Any], updates: list[dict[str, Any]]) -> str:
    parts = []
    summary = (detail.get("summary") or "").strip()
    if summary:
        parts.append(summary)
    for update in updates:
        body = (update.get("body") or "").strip()
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def fetch_project(
    url: str,
    *,
    base_url: str = "https://openlabs.bio.xyz",
    min_content_words: int = 200,
    cache_ttl_hours: int = 24,
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    fetch_fn: FetchFn | None = None,
) -> ProjectContent:
    """Fetch and extract an OpenLabs project's content for narration.

    Only the primary project-detail call is fatal on failure. Collaborators
    and updates are best-effort: on failure they fall back to the project's
    creator name and the summary alone, respectively, keeping the pipeline
    resilient to partial API degradation.
    """
    project_id = _extract_project_id(url)
    cache_path = _cache_path(cache_dir, project_id)

    if use_cache:
        cached = _read_cache(cache_path, cache_ttl_hours)
        if cached is not None:
            return cached

    getter = fetch_fn or _get_json
    api_base = _derive_api_base(base_url)

    try:
        detail = _call_with_retry(getter, f"{api_base}/api/v1/projects/{project_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SourceError(f"Project not found: {url}", input_file=url) from e
        raise SourceError(f"Failed to fetch OpenLabs project: {e}", input_file=url) from e
    except Exception as e:
        raise SourceError(f"Failed to fetch OpenLabs project: {e}", input_file=url) from e

    title = str(detail.get("title") or "").strip()
    if not title:
        raise SourceError(f"OpenLabs project has no title: {url}", input_file=url)

    try:
        collaborators = _call_with_retry(
            getter, f"{api_base}/api/v1/projects/{project_id}/collaborators"
        )
        if not isinstance(collaborators, list):
            collaborators = []
    except Exception:
        collaborators = []

    try:
        updates = _call_with_retry(getter, f"{api_base}/api/v1/projects/{project_id}/updates")
        if not isinstance(updates, list):
            updates = []
    except Exception:
        updates = []

    team_or_authors = _extract_team(detail, collaborators)
    body_text = _build_body_text(detail, updates)
    word_count = len(body_text.split())
    if word_count < min_content_words:
        raise SourceError(
            f"OpenLabs project page has insufficient content "
            f"({word_count} words, need at least {min_content_words}): {url}",
            input_file=url,
        )

    project = ProjectContent(
        title=title,
        team_or_authors=team_or_authors,
        summary=(detail.get("summary") or "").strip(),
        body_text=body_text,
        url=url,
        fetched_at=datetime.now(UTC),
    )
    _write_cache(cache_path, project)
    return project
