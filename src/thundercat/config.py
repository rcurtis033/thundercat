"""Configuration for the GPT and Grok agents, read from environment variables.

The CLI and MCP server also load a `.env` file (searched from the current directory
upwards), so keys can live there instead of in your shell profile. Variables that are
already set in the environment win over `.env`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from dotenv import find_dotenv, load_dotenv
from openai.types.shared import Reasoning
from pydantic import ValidationError

# Each provider's most capable model at its highest reasoning setting:
# gpt-6-astra accepts low..max and grok-4.7 accepts low..xhigh.
DEFAULT_GPT_MODEL = "gpt-6-astra"
DEFAULT_GPT_REASONING_EFFORT = "max"
DEFAULT_GROK_MODEL = "grok-4.7"
DEFAULT_GROK_REASONING_EFFORT = "xhigh"
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MAX_TURNS = 25
# Maximum-effort reasoning can think for many minutes before answering.
REQUEST_TIMEOUT_SECONDS = 3600.0

REASONING_MODES = ("standard", "pro")
Backend = Literal["auto", "api", "cli"]
BACKENDS: tuple[Backend, ...] = ("auto", "api", "cli")
_TRUTHY = {"1", "true", "yes", "on"}


class ConfigError(ValueError):
    """An environment variable has an invalid value."""


@dataclass(frozen=True)
class AgentSpec:
    """How to reach one model and run it as an agent."""

    name: str
    """Short id used by the CLI and in MCP tool names, e.g. "gpt"."""
    label: str
    """Human-readable name, e.g. "GPT (OpenAI)"."""
    model: str
    api_key_env: str
    """Environment variable that holds the API key, shown in error messages."""
    api_key: str | None = field(repr=False)
    base_url: str | None
    reasoning_effort: str
    cli_program: str
    """The vendor's agent CLI, used when there is no API key: codex or grok."""
    backend: Backend = "auto"
    """"api" (needs the key), "cli" (needs the vendor CLI signed in), or "auto": api if keyed."""
    reasoning_mode: str | None = None
    """OpenAI API only: "pro" makes the model do more work per answer (and cost more)."""

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Settings:
    agents: Mapping[str, AgentSpec]
    workspace: Path
    """Directory the agents' read-only file tools are confined to."""
    max_turns: int
    """Upper bound on model calls per task, so a looping agent cannot run forever."""
    tracing: bool
    """Send run traces to the OpenAI dashboard (off by default: Grok runs would be included)."""


def load_dotenv_file() -> None:
    """Load `.env` from the current directory or the nearest parent that has one."""
    load_dotenv(find_dotenv(usecwd=True), override=False)


def _get(env: Mapping[str, str], key: str) -> str | None:
    """The stripped value of `key`, treating empty strings as unset."""
    return env.get(key, "").strip() or None


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ConfigError(f"{key} must be at least 1, got {value}")
    return value


def _get_reasoning_effort(env: Mapping[str, str], key: str, default: str) -> str:
    value = (_get(env, key) or default).lower()
    try:
        Reasoning(effort=value)  # validate against the values the OpenAI SDK accepts
    except ValidationError:
        raise ConfigError(
            f"{key} must be a reasoning effort such as low, medium, high, xhigh or max; "
            f"got {value!r}"
        ) from None
    return value


def _get_reasoning_mode(env: Mapping[str, str], key: str) -> str | None:
    value = (_get(env, key) or "").lower() or None
    if value is not None and value not in REASONING_MODES:
        raise ConfigError(f"{key} must be one of {', '.join(REASONING_MODES)}; got {value!r}")
    return value


def _get_backend(env: Mapping[str, str], key: str) -> Backend:
    value = (_get(env, key) or "auto").lower()
    if value not in BACKENDS:
        raise ConfigError(f"{key} must be one of {', '.join(BACKENDS)}; got {value!r}")
    return cast(Backend, value)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from `env` (defaults to the process environment)."""
    env = os.environ if env is None else env
    agents = {
        "gpt": AgentSpec(
            name="gpt",
            label="GPT (OpenAI)",
            model=_get(env, "THUNDERCAT_GPT_MODEL") or DEFAULT_GPT_MODEL,
            api_key_env="OPENAI_API_KEY",
            api_key=_get(env, "OPENAI_API_KEY"),
            base_url=_get(env, "OPENAI_BASE_URL"),
            reasoning_effort=_get_reasoning_effort(
                env, "THUNDERCAT_GPT_REASONING_EFFORT", DEFAULT_GPT_REASONING_EFFORT
            ),
            reasoning_mode=_get_reasoning_mode(env, "THUNDERCAT_GPT_REASONING_MODE"),
            backend=_get_backend(env, "THUNDERCAT_GPT_BACKEND"),
            cli_program=_get(env, "THUNDERCAT_GPT_CLI") or "codex",
        ),
        "grok": AgentSpec(
            name="grok",
            label="Grok (xAI)",
            model=_get(env, "THUNDERCAT_GROK_MODEL") or DEFAULT_GROK_MODEL,
            api_key_env="XAI_API_KEY",
            api_key=_get(env, "XAI_API_KEY"),
            base_url=_get(env, "XAI_BASE_URL") or XAI_BASE_URL,
            reasoning_effort=_get_reasoning_effort(
                env, "THUNDERCAT_GROK_REASONING_EFFORT", DEFAULT_GROK_REASONING_EFFORT
            ),
            backend=_get_backend(env, "THUNDERCAT_GROK_BACKEND"),
            cli_program=_get(env, "THUNDERCAT_GROK_CLI") or "grok",
        ),
    }
    return Settings(
        agents=agents,
        workspace=Path(_get(env, "THUNDERCAT_WORKSPACE") or ".").expanduser().resolve(),
        max_turns=_get_int(env, "THUNDERCAT_MAX_TURNS", DEFAULT_MAX_TURNS),
        tracing=(_get(env, "THUNDERCAT_TRACING") or "").lower() in _TRUTHY,
    )
