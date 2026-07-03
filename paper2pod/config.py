"""Configuration loading: merges config.yaml + .env + CLI overrides.

Precedence (highest to lowest): CLI flags > env vars > config.yaml > defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_ENV_PATH = Path(".env")

DEFAULT_CTA_TEXT = (
    "If this got you curious, head over to OpenLabs and explore the projects "
    "other researchers are working on. And if you have research of your own, "
    "share it there. The next breakthrough might be yours."
)
MAX_CTA_WORDS = 80
CTA_WARN_WORDS = 60


def cta_word_count(text: str) -> int:
    return len(text.split())

# Mirrors config.yaml defaults from spec §5. Only leaf keys listed here are
# eligible for the PAPER2POD_<SECTION>__<KEY> environment variable override.
DEFAULTS: dict[str, dict[str, Any]] = {
    "transcript": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        # Empty by default -- shipped config.yaml provides the real per-provider
        # defaults, same as every other section. Only non-empty here if the
        # user opts in via config.yaml/env/CLI; see _resolve_provider_model().
        "models": {},
        "max_input_tokens": 12000,
        "target_words": [320, 420],
    },
    "tts": {
        "provider": "edge",
        "voice": "en-US-GuyNeural",
        "rate": "+8%",
    },
    "storage": {
        "bucket": "recordings",
        "upsert": False,
    },
    "logging": {
        "level": "INFO",
        "file": "logs/paper2pod.log",
    },
    "cta": {
        "enabled": True,
        "text": DEFAULT_CTA_TEXT,
    },
    "openlabs": {
        "base_url": "https://openlabs.bio.xyz",
        "cache_ttl_hours": 24,
        "min_content_words": 200,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "max_upload_mb": 2,
        "job_db": "jobs.db",
        "job_retention_days": 30,
        "allowed_openlabs_hosts": [],
    },
}


class ConfigError(Exception):
    """Raised for missing/invalid configuration or secrets. Maps to exit code 2."""


def _missing_var_error(name: str) -> ConfigError:
    return ConfigError(
        f"Missing required environment variable: {name}. "
        f"Set it in your .env file (see .env.example)."
    )


class TranscriptConfig(BaseModel):
    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    # Optional per-provider model overrides, e.g. {"anthropic": "...", "openai": "..."}.
    # When the active provider has an entry here, it's used instead of `model`
    # (unless --model was passed on the CLI) -- see _resolve_provider_model().
    models: dict[str, str] = {}
    max_input_tokens: int = 12000
    target_words: tuple[int, int] = (320, 420)


class TTSConfig(BaseModel):
    provider: Literal["edge", "elevenlabs"] = "edge"
    voice: str = "en-US-GuyNeural"
    rate: str = "+8%"


class StorageConfig(BaseModel):
    bucket: str = "recordings"
    upsert: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/paper2pod.log"


class CTAConfig(BaseModel):
    enabled: bool = True
    text: str = DEFAULT_CTA_TEXT

    @field_validator("text")
    @classmethod
    def _strip_text(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _validate_when_enabled(self) -> CTAConfig:
        if self.enabled:
            if not self.text:
                raise ValueError("cta.text must not be empty when cta.enabled is true")
            word_count = cta_word_count(self.text)
            if word_count > MAX_CTA_WORDS:
                raise ValueError(
                    f"cta.text is {word_count} words; must be {MAX_CTA_WORDS} or fewer"
                )
        return self


class OpenLabsConfig(BaseModel):
    base_url: str = "https://openlabs.bio.xyz"
    cache_ttl_hours: int = 24
    min_content_words: int = 200


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 2
    job_db: str = "jobs.db"
    job_retention_days: int = 30
    # Empty means: only the host of openlabs.base_url is accepted.
    allowed_openlabs_hosts: list[str] = []


class AppConfig(BaseModel):
    transcript: TranscriptConfig = TranscriptConfig()
    tts: TTSConfig = TTSConfig()
    storage: StorageConfig = StorageConfig()
    logging: LoggingConfig = LoggingConfig()
    cta: CTAConfig = CTAConfig()
    openlabs: OpenLabsConfig = OpenLabsConfig()
    api: ApiConfig = ApiConfig()


class Secrets(BaseSettings):
    """Secrets sourced from the environment / .env file. Never logged or printed."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    api_auth_token: str | None = None


def build_overrides(
    voice: str | None = None,
    model: str | None = None,
    cta_enabled: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a cli_overrides-shaped dict for load_config(), shared by the CLI and API.

    Both front ends produce overrides the identical way, so "same precedence
    rules as CLI flags" (CLI/request overrides > env > yaml > defaults) holds
    by construction rather than by two separately-maintained implementations.
    """
    overrides: dict[str, dict[str, Any]] = {}
    if voice:
        overrides["tts"] = {"voice": voice}
    if model:
        overrides["transcript"] = {"model": model}
    if cta_enabled is not None:
        overrides["cta"] = {"enabled": cta_enabled}
    return overrides


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides() -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for section, keys in DEFAULTS.items():
        section_overrides = {}
        for key in keys:
            env_name = f"PAPER2POD_{section.upper()}__{key.upper()}"
            if env_name in os.environ:
                section_overrides[key] = os.environ[env_name]
        if section_overrides:
            overrides[section] = section_overrides
    return overrides


def _resolve_provider_model(
    merged: dict[str, Any], cli_overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Apply transcript.models[provider] to transcript.model, unless --model was passed."""
    transcript_cfg = merged.get("transcript")
    if not isinstance(transcript_cfg, dict):
        return merged

    models_map = transcript_cfg.get("models")
    if not isinstance(models_map, dict):
        return merged

    cli_set_model = bool(((cli_overrides or {}).get("transcript") or {}).get("model"))
    provider = transcript_cfg.get("provider")
    if not cli_set_model and provider in models_map:
        transcript_cfg["model"] = models_map[provider]
    return merged


def load_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load and merge config.yaml + env vars + CLI overrides into a validated AppConfig."""
    path = Path(config_path)
    yaml_data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULTS, yaml_data)
    merged = _deep_merge(merged, _env_overrides())
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    merged = _resolve_provider_model(merged, cli_overrides)

    try:
        return AppConfig(**merged)
    except ValidationError as e:
        raise ConfigError(f"Invalid configuration: {e}") from e


def load_secrets(env_file: Path | str = DEFAULT_ENV_PATH) -> Secrets:
    return Secrets(_env_file=env_file)  # type: ignore[call-arg]


def validate_secrets(secrets: Secrets, app_config: AppConfig, dry_run: bool = False) -> None:
    """Fail fast with the exact missing variable name for keys required by this run."""
    if app_config.transcript.provider == "anthropic" and not secrets.anthropic_api_key:
        raise _missing_var_error("ANTHROPIC_API_KEY")
    if app_config.transcript.provider == "openai" and not secrets.openai_api_key:
        raise _missing_var_error("OPENAI_API_KEY")

    if dry_run:
        return

    if app_config.tts.provider == "elevenlabs" and not secrets.elevenlabs_api_key:
        raise _missing_var_error("ELEVENLABS_API_KEY")
    if not secrets.supabase_url:
        raise _missing_var_error("SUPABASE_URL")
    if not secrets.supabase_service_role_key:
        raise _missing_var_error("SUPABASE_SERVICE_ROLE_KEY")


def validate_supabase_secrets(secrets: Secrets) -> None:
    """Fail fast for commands that only need Supabase connectivity (list, show)."""
    if not secrets.supabase_url:
        raise _missing_var_error("SUPABASE_URL")
    if not secrets.supabase_service_role_key:
        raise _missing_var_error("SUPABASE_SERVICE_ROLE_KEY")


def validate_api_secrets(secrets: Secrets) -> None:
    """Fail fast for `paper2pod serve`: no accidental open server."""
    if not secrets.api_auth_token:
        raise _missing_var_error("API_AUTH_TOKEN")
