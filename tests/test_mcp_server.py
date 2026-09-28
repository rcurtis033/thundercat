import os
import sys

from agents.testing import ScriptedModel, assistant_message
from mcp import Client, StdioServerParameters

from thundercat.config import load_settings
from thundercat.mcp_server import create_server
from thundercat.team import Team


async def test_exposes_one_tool_per_agent(make_team):
    async with Client(create_server(make_team())) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"ask_gpt", "ask_grok"}
    grok = tools["ask_grok"]
    assert grok.input_schema["required"] == ["task"]
    assert "grok-4.7" in grok.description
    assert grok.annotations.read_only_hint is True
    assert "gpt-6-astra" in tools["ask_gpt"].description


async def test_tool_call_runs_the_agent_and_returns_its_answer(make_team):
    grok = ScriptedModel([[assistant_message("Grok says hi.")]])
    async with Client(create_server(make_team(grok=grok))) as client:
        result = await client.call_tool("ask_grok", {"task": "Say hi"})

    assert not result.is_error
    assert [block.text for block in result.content] == ["Grok says hi."]
    assert "Say hi" in str(grok.calls[0].input)


async def test_failures_reach_the_client_with_their_cause(workspace):
    team = Team(load_settings({"THUNDERCAT_WORKSPACE": str(workspace)}))  # no API keys
    async with Client(create_server(team)) as client:
        result = await client.call_tool("ask_grok", {"task": "hi"})

    assert result.is_error
    assert "XAI_API_KEY" in result.content[0].text


async def test_stdio_entry_point_speaks_mcp(workspace):
    # The same command Claude Code runs; anything stray on stdout would break the protocol.
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "thundercat", "mcp"],
        env={"PATH": os.environ["PATH"], "THUNDERCAT_WORKSPACE": str(workspace)},
        cwd=str(workspace),
    )
    async with Client(server) as client:
        names = sorted(tool.name for tool in (await client.list_tools()).tools)
    assert names == ["ask_gpt", "ask_grok"]
