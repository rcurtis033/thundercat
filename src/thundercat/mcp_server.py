"""MCP server that exposes each agent as a tool: `ask_gpt`, `ask_grok`.

This is how Claude Code, or any other MCP client, uses GPT and Grok as agents. The client
calls a tool with a task, that agent runs its own tool loop against its provider, and
its final answer comes back as the tool result. `.mcp.json` in the repository root
registers the server with Claude Code.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .team import Team

SERVER_INSTRUCTIONS = """\
Delegates work to independent AI agents running on other model providers: {agents}.
Every call is a fresh agent run. The agent sees only the task text you send, can read
(but not modify) files under {workspace}, and returns its final answer. Use these tools
for second opinions, independent reviews, or parallel investigation, and put all the
context the agent needs into the task.
"""

Task = Annotated[
    str,
    Field(
        description=(
            "A complete, self-contained task or question. The agent sees nothing else, so "
            "include the relevant context, file paths, and what a good answer looks like."
        )
    ),
]


def create_server(team: Team) -> MCPServer:
    agents = "; ".join(
        f"ask_{name} ({team.spec(name).label}, model {team.spec(name).model})"
        for name in team.names
    )
    server = MCPServer(
        name="thundercat",
        instructions=SERVER_INSTRUCTIONS.format(agents=agents, workspace=team.settings.workspace),
    )
    for name in team.names:
        spec = team.spec(name)
        server.add_tool(
            _ask_tool(team, name),
            name=f"ask_{name}",
            title=f"Ask {spec.label}",
            description=(
                f"Delegate a task to the {spec.label} agent (model {spec.model}). It works "
                "autonomously with read-only access to the workspace files and returns its "
                "final answer."
            ),
            annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
            structured_output=False,
        )
    return server


def _ask_tool(team: Team, name: str) -> Callable[[str], Awaitable[str]]:
    async def ask(task: Task) -> str:
        try:
            return await team.ask(name, task)
        except Exception as exc:
            # Only ToolError messages reach the client; anything else is reported as a bare
            # "Error executing tool", hiding causes like a missing key or a rate limit.
            raise ToolError(f"The {name} agent failed: {type(exc).__name__}: {exc}") from exc

    return ask
