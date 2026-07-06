"""Shared publish pipeline: parse/fetch -> transcript -> CTA -> TTS -> upload -> record.

Both the CLI and the API call exactly these two entry points
(run_markdown_pipeline / run_openlabs_pipeline). Progress and outcomes are
reported via StageEvent objects passed to an injected on_event callback --
this module never touches a console or a logger directly, so it works
identically whether the caller is an interactive terminal (a Rich renderer)
or a background API job worker (a job-store updater). One event system,
two consumers.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper2pod.config import AppConfig, Secrets
from paper2pod.db import EpisodeRecord, record_episode
from paper2pod.logging_setup import (
    ParseError,
    RecordError,
    SourceError,
    StorageError,
    TranscriptError,
    TTSError,
)
from paper2pod.parser import PaperMetadata, parse_markdown
from paper2pod.pdf import build_pdf_document_block, extract_pdf_metadata, load_pdf
from paper2pod.sources.openlabs import fetch_project
from paper2pod.storage import UploadResult, build_object_name, format_authors
from paper2pod.storage import upload as upload_recording
from paper2pod.transcript import Transcript
from paper2pod.transcript import generate as generate_transcript
from paper2pod.tts import get_provider


@dataclass
class StageEvent:
    stage: str  # "parse" | "fetch" | "transcript" | "tts" | "upload" | "record"
    status: str  # "started" | "completed" | "failed"
    message: str  # ready-to-print text, exact wording the CLI has always used
    data: dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[StageEvent], None]


@dataclass
class PipelineResult:
    transcript: Transcript
    upload_result: UploadResult | None = None  # None only when dry_run
    episode_id: str | None = None  # None if record_episode failed
    record_error: str | None = None  # human message if it failed, else None


def format_duration(total_seconds: float) -> str:
    minutes, seconds = divmod(int(round(total_seconds)), 60)
    return f"{minutes}m {seconds}s"


def run_markdown_pipeline(
    file: Path,
    app_config: AppConfig,
    secrets: Secrets,
    on_event: EventCallback,
    dry_run: bool = False,
    source_reference: str | None = None,
) -> PipelineResult:
    """Parse a local .md file, then run the shared publish pipeline."""
    t0 = time.monotonic()
    on_event(StageEvent(stage="parse", status="started", message=f"Parsing {file.name}..."))
    try:
        metadata, body = parse_markdown(
            file,
            max_input_tokens=app_config.transcript.max_input_tokens,
            provider=app_config.transcript.provider,
            model=app_config.transcript.model,
            secrets=secrets,
        )
    except ParseError as e:
        on_event(
            StageEvent(
                stage="parse",
                status="failed",
                message=f"[bold red]✘ Parse failed:[/] {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
        )
        raise
    elapsed = time.monotonic() - t0
    author_count = len(metadata.authors)
    author_word = "author" if author_count == 1 else "authors"
    on_event(
        StageEvent(
            stage="parse",
            status="completed",
            message=(
                f'✔ Parsed {file.name} (title: "{metadata.title}", '
                f"{author_count} {author_word})   {elapsed:.1f}s"
            ),
            data={"title": metadata.title, "author_count": author_count},
        )
    )

    return _run_core_pipeline(
        metadata,
        body,
        "paper",
        app_config,
        secrets,
        on_event,
        dry_run=dry_run,
        source_type="markdown",
        source_reference=source_reference or str(file),
    )


def run_pdf_pipeline(
    file: Path,
    app_config: AppConfig,
    secrets: Secrets,
    on_event: EventCallback,
    dry_run: bool = False,
    source_reference: str | None = None,
) -> PipelineResult:
    """Load a local .pdf, then run the shared publish pipeline with the PDF sent
    natively to Claude (Anthropic-only)."""
    t0 = time.monotonic()
    on_event(StageEvent(stage="parse", status="started", message=f"Parsing {file.name}..."))
    try:
        pdf_bytes = load_pdf(file, max_mb=app_config.api.max_pdf_mb)
        pdf_document = build_pdf_document_block(pdf_bytes)
        metadata = extract_pdf_metadata(
            pdf_document,
            provider=app_config.transcript.provider,
            model=app_config.transcript.model,
            secrets=secrets,
            fallback_title=file.stem,
        )
    except ParseError as e:
        on_event(
            StageEvent(
                stage="parse",
                status="failed",
                message=f"[bold red]✘ Parse failed:[/] {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
        )
        raise
    elapsed = time.monotonic() - t0
    author_count = len(metadata.authors)
    author_word = "author" if author_count == 1 else "authors"
    on_event(
        StageEvent(
            stage="parse",
            status="completed",
            message=(
                f'✔ Parsed {file.name} (title: "{metadata.title}", '
                f"{author_count} {author_word})   {elapsed:.1f}s"
            ),
            data={"title": metadata.title, "author_count": author_count},
        )
    )

    return _run_core_pipeline(
        metadata,
        "",
        "paper",
        app_config,
        secrets,
        on_event,
        dry_run=dry_run,
        source_type="pdf",
        source_reference=source_reference or str(file),
        pdf_document=pdf_document,
    )


def run_openlabs_pipeline(
    project_url: str,
    app_config: AppConfig,
    secrets: Secrets,
    on_event: EventCallback,
    dry_run: bool = False,
    no_cache: bool = False,
) -> PipelineResult:
    """Fetch an OpenLabs project, then run the shared publish pipeline."""
    t0 = time.monotonic()
    on_event(StageEvent(stage="fetch", status="started", message="Fetching OpenLabs project..."))
    try:
        project = fetch_project(
            project_url,
            base_url=app_config.openlabs.base_url,
            min_content_words=app_config.openlabs.min_content_words,
            cache_ttl_hours=app_config.openlabs.cache_ttl_hours,
            use_cache=not no_cache,
        )
    except SourceError as e:
        on_event(
            StageEvent(
                stage="fetch",
                status="failed",
                message=f"[bold red]✘ Fetch failed:[/] {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
        )
        raise
    elapsed = time.monotonic() - t0
    word_count = len(project.body_text.split())
    on_event(
        StageEvent(
            stage="fetch",
            status="completed",
            message=(
                f'✔ Fetched OpenLabs project (title: "{project.title}", {word_count:,} words)'
                f"   {elapsed:.1f}s"
            ),
            data={"title": project.title, "word_count": word_count},
        )
    )

    metadata = PaperMetadata(title=project.title, authors=project.team_or_authors)
    return _run_core_pipeline(
        metadata,
        project.body_text,
        "project_brief",
        app_config,
        secrets,
        on_event,
        dry_run=dry_run,
        source_type="openlabs",
        source_reference=project_url,
    )


def _run_core_pipeline(
    metadata: PaperMetadata,
    body_text: str,
    style: str,
    app_config: AppConfig,
    secrets: Secrets,
    on_event: EventCallback,
    dry_run: bool,
    source_type: str,
    source_reference: str,
    pdf_document: dict[str, Any] | None = None,
) -> PipelineResult:
    """transcript -> CTA -> TTS -> upload -> record, shared by all source types."""
    # Transcript generation
    t0 = time.monotonic()
    on_event(
        StageEvent(stage="transcript", status="started", message="Generating transcript...")
    )
    try:
        transcript = generate_transcript(
            body_text,
            metadata,
            app_config.transcript,
            secrets=secrets,
            cta_config=app_config.cta,
            style=style,
            pdf_document=pdf_document,
        )
    except TranscriptError as e:
        on_event(
            StageEvent(
                stage="transcript",
                status="failed",
                message=f"[bold red]✘ Transcript generation failed:[/] {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
        )
        raise
    elapsed = time.monotonic() - t0
    on_event(
        StageEvent(
            stage="transcript",
            status="completed",
            message=(
                f"✔ Transcript generated ({transcript.word_count} words, "
                f"~{format_duration(transcript.estimated_duration_s)})   {elapsed:.1f}s"
            ),
            data={"word_count": transcript.word_count},
        )
    )

    if dry_run:
        return PipelineResult(transcript=transcript)

    # TTS synthesis
    t0 = time.monotonic()
    out_path = Path(tempfile.mkdtemp(prefix="paper2pod-")) / build_object_name(
        metadata.title, metadata.authors
    )
    on_event(
        StageEvent(
            stage="tts",
            status="started",
            message=(
                f"Synthesizing audio ({app_config.tts.provider}-tts, {app_config.tts.voice})..."
            ),
        )
    )
    try:
        tts_provider = get_provider(app_config.tts, secrets)
        tts_provider.synthesize(transcript.text, out_path)
    except TTSError as e:
        on_event(
            StageEvent(
                stage="tts",
                status="failed",
                message=f"[bold red]✘ TTS synthesis failed:[/] {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
        )
        raise
    elapsed = time.monotonic() - t0
    on_event(
        StageEvent(
            stage="tts",
            status="completed",
            message=(
                f"✔ Synthesized audio ({app_config.tts.provider}-tts, {app_config.tts.voice})"
                f"   {elapsed:.1f}s"
            ),
        )
    )

    # Upload to Supabase Storage
    t0 = time.monotonic()
    on_event(
        StageEvent(
            stage="upload",
            status="started",
            message=f"Uploading to bucket '{app_config.storage.bucket}'...",
        )
    )
    try:
        from supabase import create_client

        client = create_client(secrets.supabase_url, secrets.supabase_service_role_key)
        object_name = build_object_name(metadata.title, metadata.authors)
        upload_result = upload_recording(
            client,
            app_config.storage.bucket,
            object_name,
            out_path,
            upsert=app_config.storage.upsert,
        )
    except StorageError as e:
        on_event(
            StageEvent(
                stage="upload",
                status="failed",
                message=f"[bold red]✘ Upload failed:[/] {e}",
                data={"error": str(e), "error_type": type(e).__name__},
            )
        )
        raise
    elapsed = time.monotonic() - t0
    on_event(
        StageEvent(
            stage="upload",
            status="completed",
            # Embeds the URL as a second line -- matches today's two
            # separate console.print() calls byte-for-byte (verified).
            message=(
                f"✔ Uploaded to '{app_config.storage.bucket}'   {elapsed:.1f}s\n"
                f"{upload_result.url}"
            ),
            data={"url": upload_result.url, "object_path": upload_result.object_path},
        )
    )

    # Record the episode (best-effort: must never fail an otherwise-successful run)
    t0 = time.monotonic()
    on_event(StageEvent(stage="record", status="started", message="Recording episode..."))
    record = EpisodeRecord(
        episode_name=Path(upload_result.object_path).stem,
        source_type=source_type,
        source_reference=source_reference,
        title=metadata.title,
        authors_or_team=format_authors(metadata.authors),
        transcript_text=transcript.text,
        cta_text=app_config.cta.text if app_config.cta.enabled else None,
        word_count=transcript.word_count,
        estimated_duration_seconds=int(round(transcript.estimated_duration_s)),
        transcript_provider=app_config.transcript.provider,
        transcript_model=app_config.transcript.model,
        tts_voice=app_config.tts.voice,
        audio_bucket=app_config.storage.bucket,
        audio_object_path=upload_result.object_path,
        audio_public_url=upload_result.url if upload_result.is_public else None,
    )
    episode_id: str | None = None
    record_error: str | None = None
    try:
        episode_id = record_episode(client, record)
        elapsed = time.monotonic() - t0
        on_event(
            StageEvent(
                stage="record",
                status="completed",
                message=f"✔ Episode recorded (id: {episode_id})   {elapsed:.1f}s",
                data={"episode_id": episode_id},
            )
        )
    except RecordError as e:
        record_error = str(e)
        on_event(
            StageEvent(
                stage="record",
                status="failed",
                message=(
                    f"[yellow]⚠ Episode record failed:[/] {e}\n"
                    f"The audio uploaded successfully; nothing was lost. "
                    f"URL: {upload_result.url}"
                ),
                data={"error": record_error, "recoverable": True},
            )
        )

    out_path.unlink(missing_ok=True)
    return PipelineResult(
        transcript=transcript,
        upload_result=upload_result,
        episode_id=episode_id,
        record_error=record_error,
    )
