"""`thundercat` command line: run the GPT and Grok agents, or serve them over MCP."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from agents import (
    Agent,
    AgentsException,
    RunItemStreamEvent,
    RunResultStreaming,
    TResponseInputItem,
)
from openai import APIError
from openai.types.responses import ResponseTextDeltaEvent

from . import vendor_cli
from .config import ConfigError, load_dotenv_file, load_settings
from .team import AgentNotConfiguredError, BackendError, Team, UnknownAgentError
from .vendor_cli import VendorCliError

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thundercat",
        description="Use GPT (OpenAI) and Grok (xAI) as tool-using agents.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    commands.add_parser("agents", help="show the agents and how each one will run")

    login = commands.add_parser(
        "login", help="sign in to an agent's vendor CLI with your account (no API key needed)"
    )
    login.add_argument("agent", help="agent to sign in for: gpt (ChatGPT) or grok (xAI)")

    ask = commands.add_parser("ask", help="give an agent one task and print its answer")
    ask.add_argument("agent", help="agent to run: gpt or grok")
    ask.add_argument("prompt", nargs="*", help="the task (read from stdin when omitted)")
    _add_run_options(ask)

    chat = commands.add_parser("chat", help="interactive multi-turn session with an agent")
    chat.add_argument("agent", help="agent to chat with: gpt or grok")
    _add_run_options(chat)

    commands.add_parser("mcp", help="serve the agents as MCP tools over stdio (for Claude Code)")
    return parser


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--consult",
        action="append",
        default=[],
        metavar="AGENT",
        help="let the agent call another agent as a tool (repeatable), e.g. --consult grok",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="print only the final answer, when it is done"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    # argparse hands `ask` an empty prompt when the task comes after an option
    # (`ask grok --no-stream explain this`); those words arrive here as leftovers.
    if extra and (args.command != "ask" or any(word.startswith("-") for word in extra)):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if extra:
        args.prompt += extra
    load_dotenv_file()
    try:
        team = Team(load_settings())
        if args.command == "agents":
            return show_agents(team)
        if args.command == "login":
            return vendor_cli.login(team.spec(args.agent))
        if args.command == "mcp":
            from .mcp_server import create_server

            create_server(team).run("stdio")
            return 0
        if args.command == "ask":
            prompt = " ".join(args.prompt).strip() or _read_stdin()
            if not prompt:
                print("error: give the task as an argument or on stdin", file=sys.stderr)
                return 2
            return asyncio.run(ask(team, args.agent, prompt, args.consult, not args.no_stream))
        return asyncio.run(chat(team, args.agent, args.consult, not args.no_stream))
    except (ConfigError, AgentNotConfiguredError, UnknownAgentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (APIError, AgentsException, BackendError, VendorCliError) as exc:
        print(f"\nerror: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def show_agents(team: Team) -> int:
    for name in team.names:
        spec = team.spec(name)
        reasoning = spec.reasoning_effort
        if spec.reasoning_mode:
            reasoning += f" ({spec.reasoning_mode} mode)"
        runs_via = _runs_via(team, name)
        print(f"{name:<5} {spec.label:<13} {spec.model:<12} reasoning={reasoning:<7} {runs_via}")
    return 0


def _runs_via(team: Team, name: str) -> str:
    spec = team.spec(name)
    product = vendor_cli.vendor(spec).product
    if team.backend(name) == "api":
        if spec.configured:
            return "via API key"
        if spec.backend == "api":
            return f"not set up: set {spec.api_key_env}"
        return f"not set up: set {spec.api_key_env}, or install {product} and sign in"
    if vendor_cli.find_executable(spec) is None:
        return f"not set up: {product} is not installed"
    if vendor_cli.is_signed_in(spec):
        return f"via {product} (signed in)"
    return f"via {product}: not signed in, run `thundercat login {name}`"


async def ask(team: Team, name: str, prompt: str, consult: list[str], stream: bool) -> int:
    if team.backend(name) == "cli":  # the vendor CLI returns its answer in one piece
        print(await team.ask(name, prompt, consult=consult))
        return 0
    agent = team.agent(name, consult=consult)
    if stream:
        await _stream(team, agent, prompt)
    else:
        result = await team.run(agent, prompt)
        print(result.final_output)
    return 0


async def chat(team: Team, name: str, consult: list[str], stream: bool) -> int:
    spec = team.spec(name)
    if team.backend(name) == "cli":
        program = vendor_cli.find_executable(spec) or spec.cli_program
        raise BackendError(
            f"chat needs {spec.api_key_env}; for an interactive session on your account, "
            f"run {program} directly."
        )
    agent = team.agent(name, consult=consult)
    print(f"Chatting with {spec.label} ({spec.model}). /exit or Ctrl-D to quit.", file=sys.stderr)
    history: list[TResponseInputItem] = []
    while True:
        try:
            line = (await asyncio.to_thread(input, "you> ")).strip()
        except EOFError:
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line in EXIT_COMMANDS:
            return 0
        history.append({"role": "user", "content": line})
        if stream:
            print(f"{name}> ", end="", flush=True)
            result = await _stream(team, agent, history)
        else:
            result = await team.run(agent, history)
            print(f"{name}> {result.final_output}")
        history = result.to_input_list()


async def _stream(
    team: Team, agent: Agent, input: str | list[TResponseInputItem]
) -> RunResultStreaming:
    """Print text as it is generated, and each tool call (on stderr) as it happens."""
    result = team.run_streamed(agent, input)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
        elif isinstance(event, RunItemStreamEvent) and event.name == "tool_called":
            print(
                f"\n  [{agent.name} -> {_describe_tool_call(event.item.raw_item)}]", file=sys.stderr
            )
    print()
    return result


def _describe_tool_call(raw: object) -> str:
    def field(key: str) -> str:
        value = raw.get(key) if isinstance(raw, dict) else getattr(raw, key, None)
        return value if isinstance(value, str) else ""

    arguments = field("arguments")
    if len(arguments) > 120:
        arguments = arguments[:117] + "..."
    return f"{field('name') or 'tool'} {arguments}".rstrip()


def _read_stdin() -> str:
    return "" if sys.stdin.isatty() else sys.stdin.read().strip()


if __name__ == "__main__":
    sys.exit(main())
