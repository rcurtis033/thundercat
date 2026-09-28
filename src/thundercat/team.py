"""Build the GPT and Grok agents on the OpenAI Agents SDK and run them.

OpenAI and xAI both serve the Responses API, so each agent is an SDK `Agent` whose model
is an `AsyncOpenAI` client pointed at its provider's base URL. Everything else — the
tool loop, tools, turn limits — is shared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agents import (
    Agent,
    ModelSettings,
    OpenAIResponsesModel,
    RunConfig,
    Runner,
    RunResult,
    RunResultStreaming,
    Tool,
    TResponseInputItem,
)
from agents.models.interface import Model
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

from .config import REQUEST_TIMEOUT_SECONDS, AgentSpec, Settings
from .tools import Workspace, workspace_tools

INSTRUCTIONS = """\
You are {label}, working as an autonomous agent (model: {model}).

Carry the task you are given through to a complete answer, then reply with that answer.
- Read-only workspace tools (list_directory, search_files, read_file) and a clock
  (current_time) are available. Use them to ground your answer in the actual files
  instead of guessing, and cite files as path:line.
- You cannot modify files or run commands. If the task needs that, say exactly what
  should be changed.
- If the task is ambiguous, state your assumptions and proceed.
- Be direct and concise; lead with the answer.
"""

PEER_INSTRUCTIONS = """\
- You can consult other agents with the {tools} tool(s) when an independent second
  opinion would materially improve your answer. Give them a self-contained question.
"""

ModelFactory = Callable[[AgentSpec], Model]


class AgentNotConfiguredError(RuntimeError):
    """The agent's API key is missing."""


class UnknownAgentError(ValueError):
    """No agent with that name exists."""


def build_model(spec: AgentSpec) -> Model:
    """An Agents SDK model that calls `spec`'s provider."""
    if not spec.api_key:
        raise AgentNotConfiguredError(
            f"The {spec.name} agent needs an API key: set {spec.api_key_env} "
            "in your environment or in a .env file."
        )
    client = AsyncOpenAI(
        api_key=spec.api_key, base_url=spec.base_url, timeout=REQUEST_TIMEOUT_SECONDS
    )
    return OpenAIResponsesModel(model=spec.model, openai_client=client)


def model_settings(spec: AgentSpec) -> ModelSettings:
    # Send nothing beyond the reasoning settings: Grok's reasoning models reject sampling
    # parameters such as presence_penalty and stop, and xAI has no reasoning "mode".
    reasoning: dict[str, str] = {}
    if spec.reasoning_effort:
        reasoning["effort"] = spec.reasoning_effort
    if spec.reasoning_mode:
        reasoning["mode"] = spec.reasoning_mode
    return ModelSettings(reasoning=Reasoning(**reasoning) if reasoning else None)


class Team:
    """The configured agents, ready to build and run."""

    def __init__(self, settings: Settings, *, model_factory: ModelFactory = build_model) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._tools: list[Tool] = list(workspace_tools(Workspace(settings.workspace)))

    @property
    def names(self) -> list[str]:
        return list(self.settings.agents)

    def spec(self, name: str) -> AgentSpec:
        try:
            return self.settings.agents[name]
        except KeyError:
            known = ", ".join(self.settings.agents)
            raise UnknownAgentError(f"Unknown agent {name!r}; choose from: {known}") from None

    def agent(self, name: str, *, consult: Sequence[str] = ()) -> Agent:
        """Build agent `name`. Agents listed in `consult` become tools it can call."""
        spec = self.spec(name)
        tools = list(self._tools)
        peers = [peer for peer in dict.fromkeys(consult) if peer != name]
        for peer in peers:
            peer_spec = self.spec(peer)
            tools.append(
                self.agent(peer).as_tool(
                    tool_name=f"ask_{peer}",
                    tool_description=(
                        f"Ask the {peer_spec.label} agent ({peer_spec.model}) to work on a "
                        "self-contained task or question and return its answer."
                    ),
                    run_config=self.run_config(),
                    max_turns=self.settings.max_turns,
                )
            )
        instructions = INSTRUCTIONS.format(label=spec.label, model=spec.model)
        if peers:
            instructions += PEER_INSTRUCTIONS.format(tools=", ".join(f"ask_{p}" for p in peers))
        return Agent(
            name=spec.name,
            instructions=instructions,
            model=self._model_factory(spec),
            model_settings=model_settings(spec),
            tools=tools,
        )

    def run_config(self) -> RunConfig:
        return RunConfig(workflow_name="thundercat", tracing_disabled=not self.settings.tracing)

    async def run(self, agent: Agent, input: str | list[TResponseInputItem]) -> RunResult:
        return await Runner.run(
            agent, input, max_turns=self.settings.max_turns, run_config=self.run_config()
        )

    def run_streamed(
        self, agent: Agent, input: str | list[TResponseInputItem]
    ) -> RunResultStreaming:
        return Runner.run_streamed(
            agent, input, max_turns=self.settings.max_turns, run_config=self.run_config()
        )

    async def ask(self, name: str, task: str, *, consult: Sequence[str] = ()) -> str:
        """Run agent `name` on `task` and return its final answer."""
        result = await self.run(self.agent(name, consult=consult), task)
        return str(result.final_output)
