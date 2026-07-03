import json
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from paper2pod.logging_setup import SourceError
from paper2pod.sources.openlabs import (
    ProjectContent,
    _derive_api_base,
    _extract_project_id,
    fetch_project,
)

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT_URL = "https://openlabs.bio.xyz/projects/4b7c0a52-708e-4f5c-a319-38ed5104c964"


def _load_fixture() -> dict:
    return json.loads((FIXTURES / "openlabs_project.json").read_text())


def _make_fetch_fn(fixture: dict, calls: list[str] | None = None):
    def fetch_fn(url: str):
        if calls is not None:
            calls.append(url)
        if url.endswith("/collaborators"):
            return fixture["collaborators"]
        if url.endswith("/updates"):
            return fixture["updates"]
        return fixture["detail"]

    return fetch_fn


def test_derive_api_base_swaps_host():
    assert _derive_api_base("https://openlabs.bio.xyz") == "https://api.openlabs.bio.xyz"


def test_extract_project_id_from_url():
    assert _extract_project_id(PROJECT_URL) == "4b7c0a52-708e-4f5c-a319-38ed5104c964"
    assert _extract_project_id(PROJECT_URL + "/") == "4b7c0a52-708e-4f5c-a319-38ed5104c964"


def test_fetch_project_extracts_from_fixture(tmp_path):
    fixture = _load_fixture()
    project = fetch_project(
        PROJECT_URL,
        use_cache=False,
        cache_dir=tmp_path / "cache",
        fetch_fn=_make_fetch_fn(fixture),
    )
    assert project.title == "Steam Collective: Where Heat Meets Science"
    assert project.team_or_authors == ["STEAM SPIRIT"]
    assert "Heat has been medicine" in project.summary
    # body_text combines summary + updates.
    assert "Heat has been medicine" in project.body_text
    assert "Growth hormone" in project.body_text
    assert project.url == PROJECT_URL
    assert len(project.body_text.split()) >= 200


def test_team_falls_back_to_creator_when_no_collaborators(tmp_path):
    fixture = _load_fixture()
    fixture["collaborators"] = []
    fixture["detail"] = {
        **fixture["detail"],
        "creator": {"display_name": "Sunny", "handle": "sunny"},
    }
    project = fetch_project(
        PROJECT_URL,
        use_cache=False,
        cache_dir=tmp_path / "cache",
        fetch_fn=_make_fetch_fn(fixture),
    )
    assert project.team_or_authors == ["Sunny"]


def test_team_falls_back_to_openlabs_when_nothing_present(tmp_path):
    fixture = _load_fixture()
    fixture["collaborators"] = []
    fixture["detail"] = {**fixture["detail"], "creator": {}}
    project = fetch_project(
        PROJECT_URL,
        use_cache=False,
        cache_dir=tmp_path / "cache",
        fetch_fn=_make_fetch_fn(fixture),
    )
    assert project.team_or_authors == ["OpenLabs"]


def test_insufficient_content_raises_source_error(tmp_path):
    fixture = _load_fixture()
    fixture["detail"] = {**fixture["detail"], "summary": "Too short."}
    fixture["updates"] = []
    with pytest.raises(SourceError, match="insufficient content"):
        fetch_project(
            PROJECT_URL,
            use_cache=False,
            cache_dir=tmp_path / "cache",
            fetch_fn=_make_fetch_fn(fixture),
        )


def test_cache_hit_skips_fetch_fn(tmp_path):
    fixture = _load_fixture()
    cache_dir = tmp_path / "cache"
    calls: list[str] = []

    fetch_project(
        PROJECT_URL, use_cache=True, cache_dir=cache_dir, fetch_fn=_make_fetch_fn(fixture, calls)
    )
    assert len(calls) == 3  # detail, collaborators, updates

    calls.clear()
    project = fetch_project(
        PROJECT_URL, use_cache=True, cache_dir=cache_dir, fetch_fn=_make_fetch_fn(fixture, calls)
    )
    assert calls == []  # served entirely from cache
    assert project.title == "Steam Collective: Where Heat Meets Science"


def test_no_cache_flag_always_refetches(tmp_path):
    fixture = _load_fixture()
    cache_dir = tmp_path / "cache"
    calls: list[str] = []

    fetch_project(
        PROJECT_URL, use_cache=True, cache_dir=cache_dir, fetch_fn=_make_fetch_fn(fixture, calls)
    )
    calls.clear()
    fetch_project(
        PROJECT_URL, use_cache=False, cache_dir=cache_dir, fetch_fn=_make_fetch_fn(fixture, calls)
    )
    assert len(calls) == 3


def test_stale_cache_triggers_refetch(tmp_path):
    fixture = _load_fixture()
    cache_dir = tmp_path / "cache"
    project_id = "4b7c0a52-708e-4f5c-a319-38ed5104c964"
    cache_dir.mkdir(parents=True)
    stale_payload = {
        "title": "Stale Title",
        "team_or_authors": ["OpenLabs"],
        "summary": "stale",
        "body_text": "stale " * 250,
        "url": PROJECT_URL,
        "fetched_at": (datetime.now(UTC) - timedelta(hours=48)).isoformat(),
    }
    (cache_dir / f"{project_id}.json").write_text(json.dumps(stale_payload))

    calls: list[str] = []
    project = fetch_project(
        PROJECT_URL,
        use_cache=True,
        cache_ttl_hours=24,
        cache_dir=cache_dir,
        fetch_fn=_make_fetch_fn(fixture, calls),
    )
    assert len(calls) == 3
    assert project.title == "Steam Collective: Where Heat Meets Science"


def test_404_is_not_retried(tmp_path):
    calls = {"count": 0}

    def failing_fetch(url: str):
        calls["count"] += 1
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with pytest.raises(SourceError, match="Project not found"):
        fetch_project(
            PROJECT_URL, use_cache=False, cache_dir=tmp_path / "cache", fetch_fn=failing_fetch
        )
    assert calls["count"] == 1


def test_transient_failure_retried_three_times_then_source_error(tmp_path):
    calls = {"count": 0}

    def failing_fetch(url: str):
        calls["count"] += 1
        raise urllib.error.URLError("network unreachable")

    with pytest.raises(SourceError) as exc_info:
        fetch_project(
            PROJECT_URL, use_cache=False, cache_dir=tmp_path / "cache", fetch_fn=failing_fetch
        )
    assert calls["count"] == 3
    assert PROJECT_URL in str(exc_info.value)


def test_project_content_cache_round_trip_via_write_and_read(tmp_path):
    from paper2pod.sources.openlabs import _cache_path, _read_cache, _write_cache

    project = ProjectContent(
        title="X",
        team_or_authors=["A", "B"],
        summary="s",
        body_text="body " * 250,
        url=PROJECT_URL,
        fetched_at=datetime.now(UTC),
    )
    path = _cache_path(tmp_path, "some-id")
    _write_cache(path, project)
    loaded = _read_cache(path, ttl_hours=24)
    assert loaded is not None
    assert loaded.title == "X"
    assert loaded.team_or_authors == ["A", "B"]
