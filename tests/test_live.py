"""Real calls to OpenAI and xAI. Skipped by default; run with `uv run pytest -m live`."""

import os

import pytest

from thundercat.config import load_settings
from thundercat.team import Team

pytestmark = pytest.mark.live


@pytest.mark.parametrize(("name", "key"), [("gpt", "OPENAI_API_KEY"), ("grok", "XAI_API_KEY")])
async def test_agent_uses_a_tool_to_answer(name, key, workspace):
    if not os.environ.get(key):
        pytest.skip(f"{key} is not set")
    settings = load_settings({**os.environ, "THUNDERCAT_WORKSPACE": str(workspace)})

    answer = await Team(settings).ask(
        name, "Use your tools to read notes.txt, then reply with only its second line."
    )

    assert "beta" in answer.lower()
