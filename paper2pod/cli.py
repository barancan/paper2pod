"""Typer entrypoint: CLI commands driving the shared pipeline, with Rich progress."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from paper2pod.config import (
    CTA_WARN_WORDS,
    AppConfig,
    ConfigError,
    Secrets,
    build_overrides,
    cta_word_count,
    load_config,
    load_secrets,
    validate_secrets,
    validate_supabase_secrets,
)
from paper2pod.db import get_episode, list_episodes
from paper2pod.logging_setup import (
    ParseError,
    RecordError,
    SourceError,
    StorageError,
    TranscriptError,
    TTSError,
    setup_logging,
)
from paper2pod.pipeline import (
    StageEvent,
    format_duration,
    run_markdown_pipeline,
    run_openlabs_pipeline,
)
from paper2pod.storage import resolve_url as resolve_audio_url

EXIT_COMMAND_FAILED = 1

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def _main() -> None:
    """Paper2Pod: convert a research paper .md file into an uploaded audio recording."""

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_PARSE_ERROR = 3
EXIT_TRANSCRIPT_ERROR = 4
EXIT_TTS_ERROR = 5
EXIT_STORAGE_ERROR = 6
EXIT_SOURCE_ERROR = 7


def _load_and_validate_config(
    config: Path, voice: str | None, model: str | None, dry_run: bool
) -> tuple[AppConfig, Secrets]:
    try:
        app_config = load_config(config_path=config, cli_overrides=build_overrides(voice, model))
        secrets = load_secrets()
        validate_secrets(secrets, app_config, dry_run=dry_run)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise typer.Exit(EXIT_CONFIG_ERROR) from e

    if app_config.cta.enabled:
        cta_words = cta_word_count(app_config.cta.text)
        if cta_words > CTA_WARN_WORDS:
            console.print(
                f"[yellow]Warning:[/] cta.text is {cta_words} words; "
                "long CTAs add to the spoken runtime."
            )
    return app_config, secrets


def _setup_logger(app_config: AppConfig, secrets: Secrets) -> logging.Logger:
    secret_values = [
        v
        for v in (
            secrets.anthropic_api_key,
            secrets.openai_api_key,
            secrets.elevenlabs_api_key,
            secrets.supabase_service_role_key,
        )
        if v
    ]
    return setup_logging(
        log_file=app_config.logging.file, level=app_config.logging.level, secrets=secret_values
    )


def _connect_supabase(config_path: Path) -> tuple[Any, logging.Logger]:
    """Minimal setup for read-only commands that only need Supabase (list, show)."""
    try:
        app_config = load_config(config_path=config_path)
        secrets = load_secrets()
        validate_supabase_secrets(secrets)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise typer.Exit(EXIT_CONFIG_ERROR) from e

    logger = _setup_logger(app_config, secrets)
    from supabase import create_client

    client = create_client(secrets.supabase_url, secrets.supabase_service_role_key)
    return client, logger


class RichPipelineListener:
    """Renders pipeline StageEvents via Rich, reproducing the CLI's exact output."""

    def __init__(self, console: Console, logger: logging.Logger):
        self.console = console
        self.logger = logger
        self._status: Any = None

    def __call__(self, event: StageEvent) -> None:
        if event.status == "started":
            self._status = self.console.status(event.message)
            self._status.__enter__()
        elif event.status == "completed":
            self._exit_status()
            self.console.print(event.message)
        elif event.status == "failed":
            self._exit_status()
            self.logger.error(event.data.get("error", event.message))
            self.console.print(event.message)

    def _exit_status(self) -> None:
        if self._status is not None:
            self._status.__exit__(None, None, None)
            self._status = None


@app.command()
def run(
    file: Annotated[Path, typer.Argument(help="Path to the paper .md file")],
    config: Annotated[Path, typer.Option("--config", help="Path to config.yaml")] = Path(
        "config.yaml"
    ),
    voice: Annotated[str | None, typer.Option("--voice", help="Override tts.voice")] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override transcript.model")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Stop after transcript generation")
    ] = False,
) -> None:
    """Convert a paper .md file into a narrated, uploaded audio recording."""
    app_config, secrets = _load_and_validate_config(config, voice, model, dry_run)
    logger = _setup_logger(app_config, secrets)
    listener = RichPipelineListener(console, logger)

    try:
        result = run_markdown_pipeline(
            file, app_config, secrets, listener, dry_run=dry_run, source_reference=str(file)
        )
    except ParseError as e:
        raise typer.Exit(EXIT_PARSE_ERROR) from e
    except TranscriptError as e:
        raise typer.Exit(EXIT_TRANSCRIPT_ERROR) from e
    except TTSError as e:
        raise typer.Exit(EXIT_TTS_ERROR) from e
    except StorageError as e:
        raise typer.Exit(EXIT_STORAGE_ERROR) from e

    if dry_run:
        console.print()
        console.print(result.transcript.text)
    raise typer.Exit(EXIT_OK)


@app.command()
def openlabs(
    project_url: Annotated[str, typer.Argument(help="OpenLabs project URL")],
    config: Annotated[Path, typer.Option("--config", help="Path to config.yaml")] = Path(
        "config.yaml"
    ),
    voice: Annotated[str | None, typer.Option("--voice", help="Override tts.voice")] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Override transcript.model")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Stop after transcript generation")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the local OpenLabs fetch cache")
    ] = False,
) -> None:
    """Fetch an OpenLabs project and publish a narrated, uploaded audio brief."""
    app_config, secrets = _load_and_validate_config(config, voice, model, dry_run)
    logger = _setup_logger(app_config, secrets)
    listener = RichPipelineListener(console, logger)

    try:
        result = run_openlabs_pipeline(
            project_url, app_config, secrets, listener, dry_run=dry_run, no_cache=no_cache
        )
    except SourceError as e:
        raise typer.Exit(EXIT_SOURCE_ERROR) from e
    except TranscriptError as e:
        raise typer.Exit(EXIT_TRANSCRIPT_ERROR) from e
    except TTSError as e:
        raise typer.Exit(EXIT_TTS_ERROR) from e
    except StorageError as e:
        raise typer.Exit(EXIT_STORAGE_ERROR) from e

    if dry_run:
        console.print()
        console.print(result.transcript.text)
    raise typer.Exit(EXIT_OK)


@app.command(name="list")
def list_command(
    limit: Annotated[int, typer.Option("--limit", help="Max episodes to show")] = 20,
    config: Annotated[Path, typer.Option("--config", help="Path to config.yaml")] = Path(
        "config.yaml"
    ),
) -> None:
    """List recorded episodes, most recent first."""
    client, _logger = _connect_supabase(config)
    try:
        episodes = list_episodes(client, limit=limit)
    except RecordError as e:
        console.print(f"[bold red]Failed to list episodes:[/] {e}")
        raise typer.Exit(EXIT_COMMAND_FAILED) from e

    if not episodes:
        console.print("No episodes recorded yet.")
        raise typer.Exit(EXIT_OK)

    table = Table()
    table.add_column("Created")
    table.add_column("Episode")
    table.add_column("Source")
    table.add_column("Duration")
    for episode in episodes:
        created = (episode.created_at or "")[:19].replace("T", " ")
        table.add_row(
            created,
            episode.episode_name,
            episode.source_type,
            format_duration(episode.estimated_duration_seconds),
        )
    console.print(table)
    raise typer.Exit(EXIT_OK)


@app.command()
def show(
    id_or_name: Annotated[str, typer.Argument(help="Episode id or (partial) episode name")],
    config: Annotated[Path, typer.Option("--config", help="Path to config.yaml")] = Path(
        "config.yaml"
    ),
) -> None:
    """Print a recorded episode's full transcript and audio link."""
    client, logger = _connect_supabase(config)
    try:
        episode = get_episode(client, id_or_name)
    except RecordError as e:
        logger.error(str(e))
        console.print(f"[bold red]Failed to look up episode:[/] {e}")
        raise typer.Exit(EXIT_COMMAND_FAILED) from e

    if episode is None:
        console.print(f"[bold red]No episode found matching:[/] {id_or_name}")
        raise typer.Exit(EXIT_COMMAND_FAILED)

    audio_url = episode.audio_public_url
    if not audio_url:
        try:
            audio_url, _is_public = resolve_audio_url(
                client, episode.audio_bucket, episode.audio_object_path
            )
        except Exception as e:
            logger.error(str(e))
            audio_url = None

    console.print(f"[bold]{episode.episode_name}[/]")
    console.print()
    console.print("--- TRANSCRIPT ---")
    console.print(episode.transcript_text)
    console.print()
    console.print("--- AUDIO ---")
    console.print(audio_url or "[yellow]Could not generate an audio URL.[/]")
    raise typer.Exit(EXIT_OK)


if __name__ == "__main__":
    app()
