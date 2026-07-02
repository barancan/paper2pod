import pytest

from paper2pod.config import (
    ConfigError,
    Secrets,
    load_config,
    validate_secrets,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no PAPER2POD_* or secret env vars leak between tests."""
    for key in list(__import__("os").environ):
        if key.startswith("PAPER2POD_") or key in {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
        }:
            monkeypatch.delenv(key, raising=False)


def test_defaults_used_when_no_yaml(tmp_path):
    missing_path = tmp_path / "does_not_exist.yaml"
    cfg = load_config(config_path=missing_path)
    assert cfg.transcript.provider == "anthropic"
    assert cfg.transcript.model == "claude-sonnet-4-6"
    assert cfg.tts.voice == "en-US-GuyNeural"
    assert cfg.storage.bucket == "recordings"
    assert cfg.storage.upsert is False


def test_yaml_overrides_defaults(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("tts:\n  voice: en-US-JennyNeural\n")
    cfg = load_config(config_path=yaml_path)
    assert cfg.tts.voice == "en-US-JennyNeural"
    # Untouched keys keep their defaults.
    assert cfg.tts.provider == "edge"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("tts:\n  voice: en-US-JennyNeural\n")
    monkeypatch.setenv("PAPER2POD_TTS__VOICE", "en-US-AriaNeural")
    cfg = load_config(config_path=yaml_path)
    assert cfg.tts.voice == "en-US-AriaNeural"


def test_cli_overrides_env(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("tts:\n  voice: en-US-JennyNeural\n")
    monkeypatch.setenv("PAPER2POD_TTS__VOICE", "en-US-AriaNeural")
    cfg = load_config(
        config_path=yaml_path, cli_overrides={"tts": {"voice": "en-US-GuyNeural"}}
    )
    assert cfg.tts.voice == "en-US-GuyNeural"


def test_full_precedence_chain(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("transcript:\n  model: yaml-model\n")
    monkeypatch.setenv("PAPER2POD_TRANSCRIPT__MODEL", "env-model")
    # No CLI override: env should win over yaml.
    cfg = load_config(config_path=yaml_path)
    assert cfg.transcript.model == "env-model"
    # CLI override should win over env.
    cfg = load_config(
        config_path=yaml_path, cli_overrides={"transcript": {"model": "cli-model"}}
    )
    assert cfg.transcript.model == "cli-model"


def test_invalid_provider_raises_config_error(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("transcript:\n  provider: not-a-real-provider\n")
    with pytest.raises(ConfigError):
        load_config(config_path=yaml_path)


def test_missing_anthropic_key_names_exact_variable(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    secrets = Secrets(_env_file=None)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        validate_secrets(secrets, cfg)


def test_missing_openai_key_named_when_provider_openai(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("transcript:\n  provider: openai\n")
    cfg = load_config(config_path=yaml_path)
    secrets = Secrets(_env_file=None)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        validate_secrets(secrets, cfg)


def test_missing_supabase_url_after_transcript_key_present(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    secrets = Secrets(_env_file=None, anthropic_api_key="sk-test")
    with pytest.raises(ConfigError, match="SUPABASE_URL"):
        validate_secrets(secrets, cfg)


def test_missing_elevenlabs_key_when_tts_provider_elevenlabs(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("tts:\n  provider: elevenlabs\n")
    cfg = load_config(config_path=yaml_path)
    secrets = Secrets(_env_file=None, anthropic_api_key="sk-test")
    with pytest.raises(ConfigError, match="ELEVENLABS_API_KEY"):
        validate_secrets(secrets, cfg)


def test_dry_run_skips_tts_and_storage_secret_checks(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("tts:\n  provider: elevenlabs\n")
    cfg = load_config(config_path=yaml_path)
    secrets = Secrets(_env_file=None, anthropic_api_key="sk-test")
    # Should not raise even though ELEVENLABS_API_KEY / SUPABASE_* are missing.
    validate_secrets(secrets, cfg, dry_run=True)


def test_secrets_never_repr_leaks_via_missing_error_message(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    secrets = Secrets(_env_file=None, anthropic_api_key="super-secret-value")
    with pytest.raises(ConfigError) as exc_info:
        validate_secrets(secrets, cfg)
    assert "super-secret-value" not in str(exc_info.value)
