import pytest

from paper2pod.db import EpisodeRecord, get_episode, list_episodes, record_episode
from paper2pod.logging_setup import RecordError


def _make_record(**overrides):
    defaults = dict(
        episode_name="Test Episode - Author",
        source_type="markdown",
        source_reference="paper.md",
        title="Test Episode",
        authors_or_team="Author",
        transcript_text="Full transcript text here.",
        word_count=350,
        estimated_duration_seconds=140,
        transcript_provider="anthropic",
        transcript_model="claude-sonnet-4-6",
        tts_voice="en-US-GuyNeural",
        audio_bucket="recordings",
        audio_object_path="Test Episode - Author.mp3",
        cta_text="Go check out OpenLabs.",
        audio_public_url="https://fake.supabase.co/public/x.mp3",
    )
    defaults.update(overrides)
    return EpisodeRecord(**defaults)


def test_episode_record_defaults_id_and_created_at_to_none():
    record = _make_record()
    assert record.id is None
    assert record.created_at is None


def test_episode_record_cta_and_public_url_default_to_none():
    record = EpisodeRecord(
        episode_name="X",
        source_type="markdown",
        source_reference="x.md",
        title="X",
        authors_or_team="Y",
        transcript_text="text",
        word_count=1,
        estimated_duration_seconds=1,
        transcript_provider="anthropic",
        transcript_model="claude-sonnet-4-6",
        tts_voice="en-US-GuyNeural",
        audio_bucket="recordings",
        audio_object_path="X.mp3",
    )
    assert record.cta_text is None
    assert record.audio_public_url is None


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table, mode, payload=None):
        self.table = table
        self.mode = mode
        self.payload = payload
        self.filters = []
        self.order_col = None
        self.order_desc = False
        self.limit_n = None

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def ilike(self, column, pattern):
        self.filters.append(("ilike", column, pattern))
        return self

    def order(self, column, desc=False):
        self.order_col = column
        self.order_desc = desc
        return self

    def limit(self, size, **_kwargs):
        self.limit_n = size
        return self

    def execute(self):
        self.table.execute_call_count += 1
        if self.table.fail_times > 0:
            self.table.fail_times -= 1
            raise self.table.fail_exc_factory()

        if self.mode == "insert":
            new_row = dict(self.payload)
            new_row.setdefault("id", f"generated-id-{len(self.table.rows) + 1}")
            new_row.setdefault("created_at", "2026-01-01T00:00:00+00:00")
            self.table.rows.append(new_row)
            return FakeResponse([new_row])

        rows = list(self.table.rows)
        for op, column, value in self.filters:
            if op == "eq":
                rows = [r for r in rows if r.get(column) == value]
            elif op == "ilike":
                pattern = value.strip("%").lower()
                rows = [r for r in rows if pattern in str(r.get(column, "")).lower()]
        if self.order_col:
            rows = sorted(rows, key=lambda r: r.get(self.order_col, ""), reverse=self.order_desc)
        if self.limit_n is not None:
            rows = rows[: self.limit_n]
        return FakeResponse(rows)


class FakeTable:
    def __init__(self, rows=None, fail_times=0, fail_exc_factory=None):
        self.rows = rows or []
        self.fail_times = fail_times
        self.fail_exc_factory = fail_exc_factory or (lambda: ConnectionError("boom"))
        self.insert_calls: list[dict] = []
        self.execute_call_count = 0

    def insert(self, payload):
        self.insert_calls.append(payload)
        return FakeQuery(self, mode="insert", payload=payload)

    def select(self, *_args, **_kwargs):
        return FakeQuery(self, mode="select")


class FakeSupabaseClient:
    def __init__(self, rows=None, fail_times=0, fail_exc_factory=None):
        self._table = FakeTable(rows=rows, fail_times=fail_times, fail_exc_factory=fail_exc_factory)

    def table(self, name):
        assert name == "episodes"
        return self._table


def test_record_episode_returns_new_id():
    client = FakeSupabaseClient()
    new_id = record_episode(client, _make_record())
    assert new_id == "generated-id-1"
    assert len(client._table.rows) == 1


def test_record_episode_excludes_id_and_created_at_from_insert_payload():
    client = FakeSupabaseClient()
    record_episode(client, _make_record())
    payload = client._table.insert_calls[0]
    assert "id" not in payload
    assert "created_at" not in payload
    assert payload["episode_name"] == "Test Episode - Author"


def test_record_episode_retries_transient_failure_then_succeeds():
    client = FakeSupabaseClient(fail_times=2)
    new_id = record_episode(client, _make_record())
    assert new_id == "generated-id-1"
    assert client._table.execute_call_count == 3


def test_record_episode_raises_record_error_after_exhausted_retries():
    client = FakeSupabaseClient(fail_times=10)
    with pytest.raises(RecordError, match="Failed to record episode"):
        record_episode(client, _make_record())
    assert client._table.execute_call_count == 3


def test_get_episode_by_uuid():
    client = FakeSupabaseClient(
        rows=[
            {
                "id": "7ed1e5eb-20d6-4943-9fd7-5548f45e8bf4",
                "episode_name": "Sun-Human Interface - Sunny",
                "source_type": "openlabs",
                "source_reference": "https://openlabs.bio.xyz/projects/7ed1e5eb-20d6-4943-9fd7-5548f45e8bf4",
                "title": "Sun-Human Interface",
                "authors_or_team": "Sunny",
                "transcript_text": "text",
                "word_count": 350,
                "estimated_duration_seconds": 140,
                "transcript_provider": "anthropic",
                "transcript_model": "claude-sonnet-4-6",
                "tts_voice": "en-US-GuyNeural",
                "audio_bucket": "recordings",
                "audio_object_path": "Sun-Human Interface - Sunny.mp3",
                "created_at": "2026-06-01T00:00:00+00:00",
            }
        ]
    )
    found = get_episode(client, "7ed1e5eb-20d6-4943-9fd7-5548f45e8bf4")
    assert found is not None
    assert found.title == "Sun-Human Interface"


def test_get_episode_by_exact_name():
    client = FakeSupabaseClient(
        rows=[
            _row_for(
                "Diffusion Models Beat GANs - Dhariwal, Nichol", "2026-06-01T00:00:00+00:00"
            )
        ]
    )
    found = get_episode(client, "Diffusion Models Beat GANs - Dhariwal, Nichol")
    assert found is not None
    assert found.episode_name == "Diffusion Models Beat GANs - Dhariwal, Nichol"


def test_get_episode_partial_match_returns_most_recent():
    client = FakeSupabaseClient(
        rows=[
            _row_for("Diffusion Models Beat GANs - Dhariwal, Nichol", "2026-06-01T00:00:00+00:00"),
            _row_for("Diffusion Models Beat GANs v2 - Dhariwal", "2026-06-15T00:00:00+00:00"),
        ]
    )
    found = get_episode(client, "diffusion models")
    assert found is not None
    assert found.episode_name == "Diffusion Models Beat GANs v2 - Dhariwal"


def test_get_episode_not_found_returns_none():
    client = FakeSupabaseClient(rows=[])
    assert get_episode(client, "nonexistent-episode") is None
    assert get_episode(client, "00000000-0000-0000-0000-000000000000") is None


def test_list_episodes_most_recent_first_and_respects_limit():
    client = FakeSupabaseClient(
        rows=[
            _row_for("Episode A", "2026-06-01T00:00:00+00:00"),
            _row_for("Episode B", "2026-06-03T00:00:00+00:00"),
            _row_for("Episode C", "2026-06-02T00:00:00+00:00"),
        ]
    )
    episodes = list_episodes(client, limit=2)
    assert [e.episode_name for e in episodes] == ["Episode B", "Episode C"]


def _row_for(episode_name: str, created_at: str) -> dict:
    return {
        "id": f"id-{episode_name}",
        "episode_name": episode_name,
        "source_type": "markdown",
        "source_reference": "paper.md",
        "title": episode_name,
        "authors_or_team": "Author",
        "transcript_text": "text",
        "word_count": 350,
        "estimated_duration_seconds": 140,
        "transcript_provider": "anthropic",
        "transcript_model": "claude-sonnet-4-6",
        "tts_voice": "en-US-GuyNeural",
        "audio_bucket": "recordings",
        "audio_object_path": f"{episode_name}.mp3",
        "created_at": created_at,
    }
