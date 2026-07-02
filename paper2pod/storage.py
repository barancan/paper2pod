"""Supabase Storage upload: filename sanitization per spec §4.4 and object upload."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from paper2pod.logging_setup import StorageError

ILLEGAL_CHARS_RE = re.compile(r'[/\\:*?"<>|]')
WHITESPACE_RE = re.compile(r"\s+")
MAX_STEM_LENGTH = 180
MAX_AUTHORS_SHOWN = 3


def sanitize_component(text: str) -> str:
    """Strip object-key-illegal characters and collapse whitespace."""
    text = ILLEGAL_CHARS_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def format_authors(authors: list[str]) -> str:
    """Join author names with ', ', truncating to the first 3 + 'et al.' if longer."""
    cleaned = [sanitize_component(a) for a in authors if sanitize_component(a)]
    if len(cleaned) > MAX_AUTHORS_SHOWN:
        return ", ".join(cleaned[:MAX_AUTHORS_SHOWN]) + " et al."
    return ", ".join(cleaned)


def build_object_name(title: str, authors: list[str], extension: str = "mp3") -> str:
    """Build the '{sanitized_title} - {sanitized_authors}.mp3' object name, capped at 180 chars."""
    clean_title = sanitize_component(title)
    author_str = format_authors(authors)
    stem = f"{clean_title} - {author_str}" if author_str else clean_title
    if len(stem) > MAX_STEM_LENGTH:
        stem = stem[:MAX_STEM_LENGTH].rstrip()
    return f"{stem}.{extension}"


def _next_collision_name(object_name: str, attempt: int) -> str:
    stem, _, ext = object_name.rpartition(".")
    return f"{stem} ({attempt}).{ext}"


def _is_collision_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "duplicate" in message


def upload(
    client: Any,
    bucket: str,
    object_name: str,
    local_path: Path,
    upsert: bool = False,
    max_collision_attempts: int = 20,
) -> str:
    """Upload local_path to Supabase Storage, returning the public/signed URL."""
    storage = client.storage.from_(bucket)
    candidate = object_name
    file_options: dict[str, str] = {"content-type": "audio/mpeg"}
    if upsert:
        file_options["upsert"] = "true"

    for attempt in range(2, 2 + max_collision_attempts):
        try:
            storage.upload(path=candidate, file=local_path, file_options=file_options)
            break
        except Exception as e:
            if upsert or not _is_collision_error(e):
                raise StorageError(f"Supabase upload failed: {e}") from e
            candidate = _next_collision_name(object_name, attempt)
    else:
        raise StorageError(f"Could not find a non-colliding object name for {object_name}")

    try:
        return _resolve_url(client, bucket, candidate)
    except Exception as e:
        raise StorageError(f"Uploaded but failed to resolve URL: {e}") from e


def _resolve_url(client: Any, bucket: str, object_name: str) -> str:
    """Print a public URL if the bucket is public, else a signed URL."""
    storage = client.storage.from_(bucket)
    is_public = False
    try:
        bucket_info = client.storage.get_bucket(bucket)
        is_public = bool(getattr(bucket_info, "public", False))
    except Exception:
        is_public = False

    if is_public:
        result = storage.get_public_url(object_name)
        return str(result)

    result = storage.create_signed_url(object_name, 3600)
    if isinstance(result, dict):
        return result.get("signedURL") or result.get("signedUrl") or ""
    return str(result)
