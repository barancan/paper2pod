import pytest

from paper2pod.config import (
    DEFAULT_CTA_TEXT,
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


def test_cta_defaults_enabled_with_default_text(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    assert cfg.cta.enabled is True
    assert cfg.cta.text == DEFAULT_CTA_TEXT


def test_cta_disabled_skips_validation_entirely(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("cta:\n  enabled: false\n  text: \"\"\n")
    cfg = load_config(config_path=yaml_path)
    assert cfg.cta.enabled is False
    assert cfg.cta.text == ""


def test_cta_empty_text_while_enabled_raises_config_error(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("cta:\n  enabled: true\n  text: \"   \"\n")
    with pytest.raises(ConfigError):
        load_config(config_path=yaml_path)


def test_cta_over_80_words_while_enabled_raises_config_error(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    long_text = " ".join(["word"] * 81)
    yaml_path.write_text(f'cta:\n  enabled: true\n  text: "{long_text}"\n')
    with pytest.raises(ConfigError, match="81 words"):
        load_config(config_path=yaml_path)


def test_cta_between_61_and_80_words_does_not_raise(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    text_70_words = " ".join(["word"] * 70)
    yaml_path.write_text(f'cta:\n  enabled: true\n  text: "{text_70_words}"\n')
    cfg = load_config(config_path=yaml_path)
    assert cfg.cta.enabled is True


def test_cta_text_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER2POD_CTA__TEXT", "Go check out OpenLabs today.")
    cfg = load_config(config_path=tmp_path / "none.yaml")
    assert cfg.cta.text == "Go check out OpenLabs today."


def test_openlabs_defaults(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    assert cfg.openlabs.base_url == "https://openlabs.bio.xyz"
    assert cfg.openlabs.cache_ttl_hours == 24
    assert cfg.openlabs.min_content_words == 200


def test_openlabs_yaml_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "openlabs:\n  base_url: https://staging.openlabs.bio.xyz\n  cache_ttl_hours: 6\n"
    )
    cfg = load_config(config_path=yaml_path)
    assert cfg.openlabs.base_url == "https://staging.openlabs.bio.xyz"
    assert cfg.openlabs.cache_ttl_hours == 6
    assert cfg.openlabs.min_content_words == 200


def test_default_models_by_provider(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    assert cfg.transcript.provider == "anthropic"
    assert cfg.transcript.model == "claude-sonnet-4-6"


def test_per_provider_model_used_for_active_provider(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "transcript:\n"
        "  provider: openai\n"
        "  models:\n"
        "    anthropic: claude-custom\n"
        "    openai: gpt-custom\n"
    )
    cfg = load_config(config_path=yaml_path)
    assert cfg.transcript.model == "gpt-custom"


def test_switching_provider_switches_resolved_model(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "transcript:\n"
        "  models:\n"
        "    anthropic: claude-custom\n"
        "    openai: gpt-custom\n"
    )
    anthropic_cfg = load_config(config_path=yaml_path)
    assert anthropic_cfg.transcript.model == "claude-custom"

    yaml_path.write_text(
        "transcript:\n"
        "  provider: openai\n"
        "  models:\n"
        "    anthropic: claude-custom\n"
        "    openai: gpt-custom\n"
    )
    openai_cfg = load_config(config_path=yaml_path)
    assert openai_cfg.transcript.model == "gpt-custom"


def test_flat_model_used_when_no_per_provider_entry(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "transcript:\n  provider: openai\n  model: gpt-flat-fallback\n  models: {}\n"
    )
    cfg = load_config(config_path=yaml_path)
    assert cfg.transcript.model == "gpt-flat-fallback"


def test_cli_model_override_wins_over_per_provider_models(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "transcript:\n"
        "  provider: anthropic\n"
        "  models:\n"
        "    anthropic: claude-from-yaml\n"
    )
    cfg = load_config(
        config_path=yaml_path, cli_overrides={"transcript": {"model": "claude-from-cli"}}
    )
    assert cfg.transcript.model == "claude-from-cli"


def test_api_config_defaults(tmp_path):
    cfg = load_config(config_path=tmp_path / "none.yaml")
    assert cfg.api.host == "127.0.0.1"
    assert cfg.api.port == 8000
    assert cfg.api.max_upload_mb == 2
    assert cfg.api.job_db == "jobs.db"
    assert cfg.api.job_retention_days == 30
    assert cfg.api.allowed_openlabs_hosts == []


def test_api_config_yaml_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "api:\n  port: 9001\n  allowed_openlabs_hosts:\n    - staging.openlabs.bio.xyz\n"
    )
    cfg = load_config(config_path=yaml_path)
    assert cfg.api.port == 9001
    assert cfg.api.allowed_openlabs_hosts == ["staging.openlabs.bio.xyz"]
    assert cfg.api.host == "127.0.0.1"


def test_secrets_api_auth_token_field():
    secrets = Secrets(_env_file=None, api_auth_token="secret-token")
    assert secrets.api_auth_token == "secret-token"
