"""Configuration loading: merges config.yaml + .env + CLI overrides.

Precedence (highest to lowest): CLI flags > env vars > config.yaml > defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_ENV_PATH = Path(".env")

# Mirrors config.yaml defaults from spec §5. Only leaf keys listed here are
# eligible for the PAPER2POD_<SECTION>__<KEY> environment variable override.
DEFAULTS: dict[str, dict[str, Any]] = {
    "transcript": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
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


class AppConfig(BaseModel):
    transcript: TranscriptConfig = TranscriptConfig()
    tts: TTSConfig = TTSConfig()
    storage: StorageConfig = StorageConfig()
    logging: LoggingConfig = LoggingConfig()


class Secrets(BaseSettings):
    """Secrets sourced from the environment / .env file. Never logged or printed."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None


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
