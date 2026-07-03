"""Typer entrypoint: orchestrates parse -> transcript -> TTS -> upload with Rich progress."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from paper2pod.config import (
    CTA_WARN_WORDS,
    ConfigError,
    cta_word_count,
    load_config,
    load_secrets,
    validate_secrets,
)
from paper2pod.logging_setup import (
    ParseError,
    StorageError,
    TranscriptError,
    TTSError,
    setup_logging,
)
from paper2pod.parser import parse_markdown
from paper2pod.storage import build_object_name
from paper2pod.storage import upload as upload_recording
from paper2pod.transcript import generate as generate_transcript
from paper2pod.tts import get_provider

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


def _format_duration(total_seconds: float) -> str:
    minutes, seconds = divmod(int(round(total_seconds)), 60)
    return f"{minutes}m {seconds}s"


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
    cli_overrides: dict[str, dict[str, str]] = {}
    if voice:
        cli_overrides["tts"] = {"voice": voice}
    if model:
        cli_overrides["transcript"] = {"model": model}

    try:
        app_config = load_config(config_path=config, cli_overrides=cli_overrides)
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
    logger = setup_logging(
        log_file=app_config.logging.file, level=app_config.logging.level, secrets=secret_values
    )

    # Stage 1: Parse
    t0 = time.monotonic()
    try:
        with console.status(f"Parsing {file.name}..."):
            metadata, body = parse_markdown(
                file,
                max_input_tokens=app_config.transcript.max_input_tokens,
                provider=app_config.transcript.provider,
                model=app_config.transcript.model,
                secrets=secrets,
            )
    except ParseError as e:
        logger.error(str(e))
        console.print(f"[bold red]✘ Parse failed:[/] {e}")
        raise typer.Exit(EXIT_PARSE_ERROR) from e
    elapsed = time.monotonic() - t0
    author_count = len(metadata.authors)
    author_word = "author" if author_count == 1 else "authors"
    console.print(
        f'✔ Parsed {file.name} (title: "{metadata.title}", {author_count} {author_word})'
        f"   {elapsed:.1f}s"
    )

    # Stage 2: Transcript generation
    t0 = time.monotonic()
    try:
        with console.status("Generating transcript..."):
            transcript = generate_transcript(
                body, metadata, app_config.transcript, secrets=secrets, cta_config=app_config.cta
            )
    except TranscriptError as e:
        logger.error(str(e))
        console.print(f"[bold red]✘ Transcript generation failed:[/] {e}")
        raise typer.Exit(EXIT_TRANSCRIPT_ERROR) from e
    elapsed = time.monotonic() - t0
    console.print(
        f"✔ Transcript generated ({transcript.word_count} words, "
        f"~{_format_duration(transcript.estimated_duration_s)})   {elapsed:.1f}s"
    )

    if dry_run:
        console.print()
        console.print(transcript.text)
        raise typer.Exit(EXIT_OK)

    # Stage 3: TTS synthesis
    t0 = time.monotonic()
    out_path = Path(tempfile.mkdtemp(prefix="paper2pod-")) / build_object_name(
        metadata.title, metadata.authors
    )
    try:
        with console.status(
            f"Synthesizing audio ({app_config.tts.provider}-tts, {app_config.tts.voice})..."
        ):
            tts_provider = get_provider(app_config.tts, secrets)
            tts_provider.synthesize(transcript.text, out_path)
    except TTSError as e:
        logger.error(str(e))
        console.print(f"[bold red]✘ TTS synthesis failed:[/] {e}")
        raise typer.Exit(EXIT_TTS_ERROR) from e
    elapsed = time.monotonic() - t0
    console.print(
        f"✔ Synthesized audio ({app_config.tts.provider}-tts, {app_config.tts.voice})"
        f"   {elapsed:.1f}s"
    )

    # Stage 4: Upload to Supabase Storage
    t0 = time.monotonic()
    try:
        with console.status(f"Uploading to bucket '{app_config.storage.bucket}'..."):
            from supabase import create_client

            client = create_client(secrets.supabase_url, secrets.supabase_service_role_key)
            object_name = build_object_name(metadata.title, metadata.authors)
            url = upload_recording(
                client,
                app_config.storage.bucket,
                object_name,
                out_path,
                upsert=app_config.storage.upsert,
            )
    except StorageError as e:
        logger.error(str(e))
        console.print(f"[bold red]✘ Upload failed:[/] {e}")
        raise typer.Exit(EXIT_STORAGE_ERROR) from e
    elapsed = time.monotonic() - t0
    console.print(f"✔ Uploaded to '{app_config.storage.bucket}'   {elapsed:.1f}s")
    console.print(url)

    out_path.unlink(missing_ok=True)
    raise typer.Exit(EXIT_OK)


if __name__ == "__main__":
    app()
