"""Real runs against OpenAI and xAI. Skipped by default; run with `uv run pytest -m live`.

Each agent runs on whatever it is set up with: its API key, or its signed-in vendor CLI.
"""

import os

import pytest

from thundercat import vendor_cli
from thundercat.config import load_settings
from thundercat.team import Team

pytestmark = pytest.mark.live


@pytest.mark.parametrize("name", ["gpt", "grok"])
async def test_agent_reads_a_file_to_answer(name, workspace):
    team = Team(load_settings({**os.environ, "THUNDERCAT_WORKSPACE": str(workspace)}))
    spec = team.spec(name)
    ready = spec.configured if team.backend(name) == "api" else vendor_cli.is_signed_in(spec)
    if not ready:
        pytest.skip(f"{name}: set {spec.api_key_env} or run `thundercat login {name}`")

    answer = await team.ask(
        name, "Read notes.txt in the workspace and reply with only its second line."
    )

    assert "beta" in answer.lower()
