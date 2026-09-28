import pytest
from agents import OpenAIResponsesModel
from agents.testing import ScriptedModel, assistant_message, function_call

from thundercat.config import load_settings
from thundercat.team import (
    AgentNotConfiguredError,
    Team,
    UnknownAgentError,
    build_model,
    model_settings,
)

WORKSPACE_TOOLS = ["current_time", "list_directory", "read_file", "search_files"]


def test_build_model_points_each_agent_at_its_provider(settings):
    gpt = build_model(settings.agents["gpt"])
    grok = build_model(settings.agents["grok"])

    assert isinstance(gpt, OpenAIResponsesModel) and isinstance(grok, OpenAIResponsesModel)
    assert (gpt.model, grok.model) == ("gpt-6-astra", "grok-4.7")
    assert str(gpt._client.base_url) == "https://api.openai.com/v1/"
    assert str(grok._client.base_url) == "https://api.x.ai/v1/"
    assert grok._client.api_key == "xai-test-key"
    assert grok._client.timeout == 3600.0


def test_build_model_requires_the_api_key():
    with pytest.raises(AgentNotConfiguredError, match="XAI_API_KEY"):
        build_model(load_settings({}).agents["grok"])


def test_model_settings_carry_only_reasoning():
    settings = load_settings({"THUNDERCAT_GPT_REASONING_MODE": "pro"})
    gpt = model_settings(settings.agents["gpt"])
    grok = model_settings(settings.agents["grok"])

    assert gpt.reasoning.model_dump(exclude_unset=True) == {"effort": "max", "mode": "pro"}
    assert grok.reasoning.model_dump(exclude_unset=True) == {"effort": "xhigh"}
    assert grok.presence_penalty is None and grok.temperature is None


def test_agent_gets_workspace_tools_and_optional_peers(make_team):
    team = make_team(gpt=ScriptedModel(), grok=ScriptedModel())

    assert [t.name for t in team.agent("gpt").tools] == WORKSPACE_TOOLS
    duo = team.agent("gpt", consult=["grok", "grok", "gpt"])  # duplicates and self are ignored
    assert [t.name for t in duo.tools] == [*WORKSPACE_TOOLS, "ask_grok"]
    assert "ask_grok" in duo.instructions


def test_unknown_agents_are_rejected(make_team):
    team = make_team(gpt=ScriptedModel())
    with pytest.raises(UnknownAgentError, match="gpt, grok"):
        team.agent("claude")
    with pytest.raises(UnknownAgentError):
        team.agent("gpt", consult=["claude"])


async def test_agent_calls_tools_then_answers(make_team):
    grok = ScriptedModel(
        [
            [
                function_call(
                    "read_file",
                    {"path": "notes.txt", "start_line": 1, "end_line": None},
                    call_id="call_1",
                )
            ],
            [assistant_message("The notes list alpha, beta and gamma.")],
        ]
    )
    team = make_team(grok=grok)

    answer = await team.ask("grok", "What is in notes.txt?")

    assert answer == "The notes list alpha, beta and gamma."
    first, second = grok.calls
    assert first.system_instructions.startswith("You are Grok (xAI)")
    assert first.model_settings.reasoning.effort == "xhigh"
    tool_output = next(i for i in second.input if i.get("type") == "function_call_output")
    assert "     2\tbeta" in tool_output["output"]
    grok.assert_complete()


async def test_agent_can_consult_a_peer_agent(make_team):
    gpt = ScriptedModel(
        [
            [function_call("ask_grok", {"task": "Is 7919 prime?"}, call_id="call_1")],
            [assistant_message("Yes, and Grok agrees: 7919 is prime.")],
        ]
    )
    grok = ScriptedModel([[assistant_message("7919 is prime.")]])
    team = make_team(gpt=gpt, grok=grok)

    answer = await team.ask("gpt", "Is 7919 prime? Double-check with Grok.", consult=["grok"])

    assert answer == "Yes, and Grok agrees: 7919 is prime."
    assert "Is 7919 prime?" in str(grok.calls[0].input)
    tool_output = next(i for i in gpt.calls[1].input if i.get("type") == "function_call_output")
    assert tool_output["output"] == "7919 is prime."


def test_tracing_is_opt_in(make_team, workspace):
    assert make_team().run_config().tracing_disabled is True
    traced = load_settings({"THUNDERCAT_TRACING": "1", "THUNDERCAT_WORKSPACE": str(workspace)})
    assert Team(traced).run_config().tracing_disabled is False
