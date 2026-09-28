import builtins
import io
import os
import sys

import pytest
from agents.testing import ScriptedModel, assistant_message, function_call

from thundercat import cli
from thundercat.team import Team

PROVIDER_VARS = {"OPENAI_API_KEY", "XAI_API_KEY", "OPENAI_BASE_URL", "XAI_BASE_URL"}


@pytest.fixture(autouse=True)
def cli_env(monkeypatch, workspace):
    """Run the CLI against the test workspace, isolated from the real environment and .env."""
    for key in list(os.environ):
        if key.startswith("THUNDERCAT_") or key in PROVIDER_VARS:
            monkeypatch.delenv(key)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setenv("THUNDERCAT_WORKSPACE", str(workspace))
    monkeypatch.setattr(cli, "load_dotenv_file", lambda: None)


@pytest.fixture
def use_models(monkeypatch):
    """Make the CLI's agents run on ScriptedModels (keyed by agent name)."""

    def use(**models: ScriptedModel) -> None:
        monkeypatch.setattr(
            cli, "Team", lambda settings: Team(settings, model_factory=lambda s: models[s.name])
        )

    return use


def test_agents_shows_models_reasoning_and_key_status(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY")
    assert cli.main(["agents"]) == 0
    gpt, grok = capsys.readouterr().out.splitlines()
    assert gpt.split()[:5] == ["gpt", "GPT", "(OpenAI)", "gpt-6-astra", "reasoning=max"]
    assert gpt.endswith("via API key")
    assert grok.split()[:5] == ["grok", "Grok", "(xAI)", "grok-4.7", "reasoning=xhigh"]
    assert grok.endswith("not set up: set XAI_API_KEY, or install Grok Build and sign in")


def test_ask_prints_the_final_answer(use_models, capsys):
    use_models(grok=ScriptedModel([[assistant_message("Hello from Grok.")]]))
    assert cli.main(["ask", "grok", "--no-stream", "say", "hello"]) == 0
    assert capsys.readouterr().out == "Hello from Grok.\n"


def test_ask_streams_text_and_reports_tool_calls(use_models, capsys):
    args = {"path": "notes.txt", "start_line": 1, "end_line": None}
    use_models(
        gpt=ScriptedModel(
            [
                [function_call("read_file", args, call_id="call_1")],
                [assistant_message("It lists three words.")],
            ]
        )
    )
    assert cli.main(["ask", "gpt", "What is in notes.txt?"]) == 0
    out, err = capsys.readouterr()
    assert "It lists three words." in out
    assert '[gpt -> read_file {"path":"notes.txt"' in err


def test_ask_reads_the_task_from_stdin(use_models, monkeypatch):
    model = ScriptedModel([[assistant_message("ok")]])
    use_models(gpt=model)
    monkeypatch.setattr(sys, "stdin", io.StringIO("task from stdin\n"))
    assert cli.main(["ask", "gpt", "--no-stream"]) == 0
    assert "task from stdin" in str(model.calls[0].input)


def test_chat_keeps_the_conversation(use_models, monkeypatch, capsys):
    model = ScriptedModel([[assistant_message("answer one")], [assistant_message("answer two")]])
    use_models(grok=model)
    lines = iter(["first question", "second question"])

    def fake_input(prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)
    assert cli.main(["chat", "grok", "--no-stream"]) == 0

    out = capsys.readouterr().out
    assert "grok> answer one" in out and "grok> answer two" in out
    second_turn = str(model.calls[1].input)
    assert "first question" in second_turn
    assert "answer one" in second_turn
    assert "second question" in second_turn


def test_missing_key_is_reported_clearly(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY")
    assert cli.main(["ask", "grok", "hi"]) == 2
    assert "XAI_API_KEY" in capsys.readouterr().err


def test_unknown_agent_is_reported_clearly(capsys):
    assert cli.main(["ask", "claude", "hi"]) == 2
    assert "choose from: gpt, grok" in capsys.readouterr().err
