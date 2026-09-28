"""The vendor-CLI backend, exercised against fake `codex` and `grok` executables."""

import json
import sys
from types import SimpleNamespace

import pytest
from agents.testing import ScriptedModel, assistant_message, function_call
from mcp import Client

from thundercat import cli
from thundercat.config import load_settings
from thundercat.mcp_server import create_server
from thundercat.team import BackendError, Team
from thundercat.vendor_cli import VendorCliError, _grok_answer, find_executable

# Each fake records its argv (and task) to $FAKE_CLI_LOG, one JSON object per call.
FAKE_CODEX = """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
record = {"program": "codex", "argv": args}
if args[:2] == ["login", "status"]:
    sys.exit(0 if Path(os.environ["FAKE_CODEX_SIGNED_IN"]).exists() else 1)
if args[:1] == ["exec"]:
    record["task"] = sys.stdin.read()
    answer = Path(args[args.index("--output-last-message") + 1])
    answer.write_text("codex says: " + record["task"])
with open(os.environ["FAKE_CLI_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
if os.environ.get("FAKE_FAIL"):
    sys.stderr.write("model overloaded")
    sys.exit(3)
"""

FAKE_GROK = """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
record = {"program": "grok", "argv": args}
if "--prompt-file" in args:
    record["task"] = Path(args[args.index("--prompt-file") + 1]).read_text()
    # The shape Grok Build 1.0.41 prints for --output-format json.
    reply = {"text": "grok says: " + record["task"], "stopReason": "end_turn",
             "sessionId": "s-1", "thought": "(reasoning summary)", "usage": {}}
    print(json.dumps(reply, indent=2))
with open(os.environ["FAKE_CLI_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
"""


@pytest.fixture
def clis(tmp_path, monkeypatch, workspace):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    paths = {}
    for name, source in {"codex": FAKE_CODEX, "grok": FAKE_GROK}.items():
        path = bin_dir / f"fake-{name}"
        path.write_text(f"#!{sys.executable}\n{source}")
        path.chmod(0o755)
        paths[name] = path
    log = tmp_path / "calls.jsonl"
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    codex_marker = tmp_path / "codex-signed-in"
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))
    monkeypatch.setenv("FAKE_CODEX_SIGNED_IN", str(codex_marker))
    monkeypatch.setenv("GROK_HOME", str(grok_home))
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    def sign_in():
        codex_marker.touch()
        (grok_home / "auth.json").write_text("{}")

    def calls(program=None):
        entries = (
            [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        )
        return [e for e in entries if program in (None, e["program"])]

    env = {
        "THUNDERCAT_WORKSPACE": str(workspace),
        "THUNDERCAT_GPT_CLI": str(paths["codex"]),
        "THUNDERCAT_GROK_CLI": str(paths["grok"]),
    }
    return SimpleNamespace(env=env, sign_in=sign_in, calls=calls, workspace=workspace)


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_backend_prefers_the_api_key_then_the_cli(clis):
    team = Team(load_settings(clis.env))
    assert (team.backend("gpt"), team.backend("grok")) == ("cli", "cli")

    keyed = Team(load_settings({**clis.env, "OPENAI_API_KEY": "sk-1"}))
    assert keyed.backend("gpt") == "api"

    no_cli = Team(load_settings({**clis.env, "THUNDERCAT_GROK_CLI": "no-such-program"}))
    assert no_cli.backend("grok") == "api"  # which then explains how to set either up

    forced = Team(
        load_settings({**clis.env, "OPENAI_API_KEY": "sk-1", "THUNDERCAT_GPT_BACKEND": "cli"})
    )
    assert forced.backend("gpt") == "cli"


async def test_gpt_runs_through_codex_read_only_at_max_effort(clis):
    clis.sign_in()
    team = Team(load_settings(clis.env))

    answer = await team.ask("gpt", "Review notes.txt\nand be thorough")

    assert answer == "codex says: Review notes.txt\nand be thorough"
    (call,) = clis.calls("codex")
    argv = call["argv"]
    assert argv[0] == "exec" and argv[-1] == "-"
    assert flag(argv, "--model") == "gpt-6-astra"
    assert flag(argv, "--config") == 'model_reasoning_effort="max"'
    assert flag(argv, "--sandbox") == "read-only"
    assert flag(argv, "--cd") == str(clis.workspace)
    assert "--ephemeral" in argv
    assert call["task"] == "Review notes.txt\nand be thorough"


async def test_grok_runs_through_grok_build_read_only_at_xhigh(clis):
    clis.sign_in()
    team = Team(load_settings(clis.env))

    answer = await team.ask("grok", "Summarize the repo")

    assert answer == "grok says: Summarize the repo"
    (call,) = clis.calls("grok")
    argv = call["argv"]
    assert flag(argv, "--model") == "grok-4.7"
    assert flag(argv, "--reasoning-effort") == "xhigh"
    assert flag(argv, "--output-format") == "json"
    assert flag(argv, "--permission-mode") == "dontAsk"
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--allow"] == ["Read", "Grep"]
    assert flag(argv, "--sandbox") == "read-only"
    assert flag(argv, "--cwd") == str(clis.workspace)
    assert call["task"] == "Summarize the repo"


async def test_signed_out_cli_explains_how_to_sign_in(clis):
    team = Team(load_settings(clis.env))
    with pytest.raises(VendorCliError, match="thundercat login grok"):
        await team.ask("grok", "hi")
    with pytest.raises(VendorCliError, match="thundercat login gpt"):
        await team.ask("gpt", "hi")


async def test_cli_failures_carry_the_cli_output(clis, monkeypatch):
    clis.sign_in()
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(VendorCliError, match="status 3: model overloaded"):
        await Team(load_settings(clis.env)).ask("gpt", "hi")


async def test_cli_backed_agents_cannot_consult(clis):
    clis.sign_in()
    with pytest.raises(BackendError, match="--consult"):
        await Team(load_settings(clis.env)).ask("grok", "hi", consult=["gpt"])


async def test_api_agent_can_consult_a_cli_backed_peer(clis):
    clis.sign_in()
    gpt = ScriptedModel(
        [
            [function_call("ask_grok", {"task": "What does notes.txt say?"}, call_id="call_1")],
            [assistant_message("Grok checked: alpha, beta, gamma.")],
        ]
    )
    settings = load_settings({**clis.env, "OPENAI_API_KEY": "sk-1"})
    team = Team(settings, model_factory=lambda spec: gpt)

    answer = await team.ask("gpt", "Ask Grok about notes.txt", consult=["grok"])

    assert answer == "Grok checked: alpha, beta, gamma."
    tool_output = next(i for i in gpt.calls[1].input if i.get("type") == "function_call_output")
    assert tool_output["output"] == "grok says: What does notes.txt say?"


async def test_mcp_tools_work_through_the_cli_backend(clis):
    clis.sign_in()
    async with Client(create_server(Team(load_settings(clis.env)))) as client:
        result = await client.call_tool("ask_grok", {"task": "ping"})
    assert not result.is_error
    assert result.content[0].text == "grok says: ping"


def test_login_runs_the_vendor_device_sign_in(clis, monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv_file", lambda: None)
    for key, value in clis.env.items():
        monkeypatch.setenv(key, value)

    assert cli.main(["login", "gpt"]) == 0
    assert cli.main(["login", "grok"]) == 0

    argvs = [call["argv"] for call in clis.calls()]
    assert argvs == [["login", "--device-auth"], ["--no-auto-update", "login", "--device-auth"]]


def test_agents_command_reports_cli_sign_in_state(clis, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv_file", lambda: None)
    for key in ("OPENAI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in clis.env.items():
        monkeypatch.setenv(key, value)

    cli.main(["agents"])
    assert "via Codex CLI: not signed in, run `thundercat login gpt`" in capsys.readouterr().out
    clis.sign_in()
    cli.main(["agents"])
    out = capsys.readouterr().out
    assert "via Codex CLI (signed in)" in out and "via Grok Build (signed in)" in out


@pytest.mark.parametrize(
    ("output", "answer"),
    [
        ('{\n  "text": "the answer",\n  "thought": "not this"\n}', "the answer"),
        ('{"type": "result", "result": "the answer"}', "the answer"),
        ('progress...\n{"response": "last-line json"}', "last-line json"),
        ("plain text answer\n", "plain text answer"),
        ("\x1b[1mbold\x1b[0m text", "bold text"),
    ],
)
def test_grok_answer_extraction(output, answer):
    assert _grok_answer(output) == answer


def test_grok_reported_errors_raise():
    with pytest.raises(VendorCliError, match="reported an error"):
        _grok_answer('{"is_error": true, "result": "quota exceeded"}')


def test_find_executable_falls_back_to_the_grok_install_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    grok = tmp_path / ".grok" / "bin" / "grok"
    grok.parent.mkdir(parents=True)
    grok.write_text("#!/bin/sh\n")
    grok.chmod(0o755)
    # `find_executable` is the real function, imported before conftest hides installed CLIs.
    assert find_executable(load_settings({}).agents["grok"]) == str(grok)
    assert find_executable(load_settings({}).agents["gpt"]) is None
