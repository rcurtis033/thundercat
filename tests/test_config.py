from pathlib import Path

import pytest

from thundercat.config import (
    DEFAULT_MAX_TURNS,
    XAI_BASE_URL,
    ConfigError,
    load_settings,
)


def test_defaults_use_each_providers_top_model_at_highest_reasoning():
    settings = load_settings({})
    gpt, grok = settings.agents["gpt"], settings.agents["grok"]

    assert (gpt.model, gpt.reasoning_effort, gpt.reasoning_mode) == ("gpt-6-astra", "max", None)
    assert (grok.model, grok.reasoning_effort, grok.reasoning_mode) == ("grok-4.7", "xhigh", None)
    assert (gpt.base_url, gpt.api_key_env) == (None, "OPENAI_API_KEY")
    assert (grok.base_url, grok.api_key_env) == (XAI_BASE_URL, "XAI_API_KEY")
    assert not gpt.configured and not grok.configured
    assert settings.workspace == Path.cwd().resolve()
    assert settings.max_turns == DEFAULT_MAX_TURNS
    assert settings.tracing is False


def test_environment_overrides(tmp_path):
    settings = load_settings(
        {
            "OPENAI_API_KEY": "sk-1",
            "XAI_API_KEY": "xai-1",
            "OPENAI_BASE_URL": "https://proxy.example/v1",
            "XAI_BASE_URL": "https://xai.example/v1",
            "THUNDERCAT_GPT_MODEL": "gpt-custom",
            "THUNDERCAT_GROK_MODEL": "grok-custom",
            "THUNDERCAT_GPT_REASONING_EFFORT": "HIGH",
            "THUNDERCAT_GPT_REASONING_MODE": "Pro",
            "THUNDERCAT_GROK_REASONING_EFFORT": "low",
            "THUNDERCAT_WORKSPACE": str(tmp_path),
            "THUNDERCAT_MAX_TURNS": "7",
            "THUNDERCAT_TRACING": "true",
        }
    )
    gpt, grok = settings.agents["gpt"], settings.agents["grok"]

    assert gpt.configured and grok.configured
    assert (gpt.model, gpt.base_url, gpt.reasoning_effort, gpt.reasoning_mode) == (
        "gpt-custom",
        "https://proxy.example/v1",
        "high",
        "pro",
    )
    assert (grok.model, grok.base_url, grok.reasoning_effort) == (
        "grok-custom",
        "https://xai.example/v1",
        "low",
    )
    assert settings.workspace == tmp_path.resolve()
    assert settings.max_turns == 7
    assert settings.tracing is True


def test_blank_values_count_as_unset():
    settings = load_settings({"XAI_API_KEY": "  ", "THUNDERCAT_GROK_MODEL": ""})
    assert not settings.agents["grok"].configured
    assert settings.agents["grok"].model == "grok-4.7"


def test_api_key_is_not_in_repr():
    spec = load_settings({"XAI_API_KEY": "xai-very-secret"}).agents["grok"]
    assert "xai-very-secret" not in repr(spec)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("THUNDERCAT_MAX_TURNS", "lots"),
        ("THUNDERCAT_MAX_TURNS", "0"),
        ("THUNDERCAT_GPT_REASONING_EFFORT", "turbo"),
        ("THUNDERCAT_GROK_REASONING_EFFORT", "extreme"),
        ("THUNDERCAT_GPT_REASONING_MODE", "turbo"),
    ],
)
def test_invalid_values_raise_config_error(key, value):
    with pytest.raises(ConfigError, match=key):
        load_settings({key: value})
