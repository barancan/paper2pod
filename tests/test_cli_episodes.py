"""CLI tests for `paper2pod list` and `paper2pod show` against a fake Supabase client."""

import os

import pytest
from typer.testing import CliRunner

from paper2pod import cli as cli_module
from paper2pod.config import Secrets

runner = CliRunner()


def _row(episode_name: str, created_at: str, **overrides) -> dict:
    row = {
        "id": f"id-{episode_name}",
        "episode_name": episode_name,
        "source_type": "markdown",
        "source_reference": "paper.md",
        "title": episode_name,
        "authors_or_team": "Author",
        "transcript_text": f"Full transcript for {episode_name}.",
        "word_count": 350,
        "estimated_duration_seconds": 140,
        "transcript_provider": "anthropic",
        "transcript_model": "claude-sonnet-4-6",
        "tts_voice": "en-US-GuyNeural",
        "audio_bucket": "recordings",
        "audio_object_path": f"{episode_name}.mp3",
        "audio_public_url": f"https://fake.supabase.co/public/{episode_name}.mp3",
        "created_at": created_at,
    }
    row.update(overrides)
    return row


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
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args, **_kwargs):
        return FakeQuery(self, mode="select")


class FakeSupabaseClient:
    def __init__(self, rows):
        self._table = FakeTable(rows)

    def table(self, name):
        assert name == "episodes"
        return self._table


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
    monkeypatch.setattr(cli_module, "load_secrets", lambda: Secrets(_env_file=None))


def _connect_with_fake_client(monkeypatch, rows):
    fake_client = FakeSupabaseClient(rows)
    monkeypatch.setattr(cli_module, "_connect_supabase", lambda config: (fake_client, None))
    return fake_client


def test_list_shows_episodes_most_recent_first(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(
        monkeypatch,
        rows=[
            _row("Episode A", "2026-06-01T10:00:00+00:00"),
            _row("Episode B", "2026-06-03T10:00:00+00:00"),
        ],
    )

    result = runner.invoke(cli_module.app, ["list"])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.index("Episode B") < result.stdout.index("Episode A")


def test_list_respects_limit(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(
        monkeypatch,
        rows=[
            _row("Episode A", "2026-06-01T10:00:00+00:00"),
            _row("Episode B", "2026-06-02T10:00:00+00:00"),
            _row("Episode C", "2026-06-03T10:00:00+00:00"),
        ],
    )

    result = runner.invoke(cli_module.app, ["list", "--limit", "1"])

    assert result.exit_code == 0, result.stdout
    assert "Episode C" in result.stdout
    assert "Episode B" not in result.stdout
    assert "Episode A" not in result.stdout


def test_list_empty_prints_friendly_message(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(monkeypatch, rows=[])

    result = runner.invoke(cli_module.app, ["list"])

    assert result.exit_code == 0, result.stdout
    assert "No episodes recorded yet" in result.stdout


def test_list_missing_supabase_url_exits_2(monkeypatch):
    result = runner.invoke(cli_module.app, ["list"])
    assert result.exit_code == 2
    assert "SUPABASE_URL" in result.stdout


def test_show_prints_transcript_and_audio_sections(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(
        monkeypatch,
        rows=[_row("Diffusion Models Beat GANs - Dhariwal, Nichol", "2026-06-01T10:00:00+00:00")],
    )

    result = runner.invoke(
        cli_module.app, ["show", "Diffusion Models Beat GANs - Dhariwal, Nichol"]
    )

    assert result.exit_code == 0, result.stdout
    assert "--- TRANSCRIPT ---" in result.stdout
    assert "Full transcript for Diffusion Models Beat GANs - Dhariwal, Nichol." in result.stdout
    assert "--- AUDIO ---" in result.stdout
    assert "https://fake.supabase.co/public/" in result.stdout


def test_show_partial_name_match_finds_closest_episode(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(
        monkeypatch,
        rows=[_row("Diffusion Models Beat GANs - Dhariwal, Nichol", "2026-06-01T10:00:00+00:00")],
    )

    result = runner.invoke(cli_module.app, ["show", "diffusion models"])

    assert result.exit_code == 0, result.stdout
    assert "Diffusion Models Beat GANs" in result.stdout


def test_show_not_found_exits_1(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(monkeypatch, rows=[])

    result = runner.invoke(cli_module.app, ["show", "nonexistent"])

    assert result.exit_code == 1
    assert "No episode found matching" in result.stdout


def test_show_generates_signed_url_when_no_public_url_stored(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sk-test")
    _connect_with_fake_client(
        monkeypatch,
        rows=[
            _row(
                "Private Episode - Author",
                "2026-06-01T10:00:00+00:00",
                audio_public_url=None,
            )
        ],
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_audio_url",
        lambda client, bucket, path: (f"https://fake.supabase.co/signed/{path}", False),
    )

    result = runner.invoke(cli_module.app, ["show", "Private Episode - Author"])

    assert result.exit_code == 0, result.stdout
    assert "https://fake.supabase.co/signed/Private Episode - Author.mp3" in result.stdout
