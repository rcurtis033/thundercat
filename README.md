# thundercat

Use **GPT** (OpenAI) and **Grok** (xAI) as tool-using agents — from the command line,
from Python, or from Claude Code, which can hand either one a task over MCP.

| Agent  | Provider | Default model  | Reasoning (default = highest) | API key          |
| ------ | -------- | -------------- | ----------------------------- | ---------------- |
| `gpt`  | OpenAI   | `gpt-6-astra`  | `max`                         | `OPENAI_API_KEY` |
| `grok` | xAI      | `grok-4.7`     | `xhigh`                       | `XAI_API_KEY`    |

Each agent runs its own tool loop (built on the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/))
against its provider's Responses API. Both share the same tools: a clock and read-only
access to the workspace (`list_directory`, `search_files`, `read_file`). An agent can
also consult the other one as a tool (`--consult`).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then paste in your API keys
uv run thundercat agents
```

Keys come from https://platform.openai.com/api-keys and https://console.x.ai. They can
live in `.env` (gitignored) or in your environment; you only need a key for the agent
you use. For Claude Code on the web, add `OPENAI_API_KEY` / `XAI_API_KEY` in the cloud
environment's settings instead, then start a new session.

## Command line

```bash
uv run thundercat ask gpt "Review src/thundercat/tools.py for path-traversal bugs"
uv run thundercat ask grok --consult gpt "Is our retry logic safe? Get a second opinion."
git diff | uv run thundercat ask grok        # the task can come from stdin
uv run thundercat chat gpt                   # multi-turn session; /exit to quit
```

Text streams as it is generated and tool calls are shown on stderr; `--no-stream` prints
only the final answer.

## In Claude Code

`.mcp.json` registers a `thundercat` MCP server with two tools, `ask_gpt` and `ask_grok`.
Open the repo in Claude Code, approve the server when prompted (or check `/mcp`), and ask
for them by name — for example *"ask_grok to review this diff"* or *"get GPT and Grok's
takes on this design, then compare"*. Each call is an independent agent run: it sees only
the task text Claude sends, reads files in this repo as needed, and returns its answer.

Maximum-effort reasoning can take several minutes per call. If Claude Code gives up
waiting, raise its MCP tool timeout (the `MCP_TOOL_TIMEOUT` environment variable, in
milliseconds) or lower the effort.

## From Python

```python
import asyncio
from thundercat import Team, load_settings

team = Team(load_settings())
print(asyncio.run(team.ask("grok", "Summarize README.md", consult=["gpt"])))
```

## Configuration

All settings are environment variables (or `.env` entries):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY`, `XAI_API_KEY` | — | Provider keys |
| `THUNDERCAT_GPT_MODEL` | `gpt-6-astra` | OpenAI model |
| `THUNDERCAT_GPT_REASONING_EFFORT` | `max` | `low` · `medium` · `high` · `xhigh` · `max` |
| `THUNDERCAT_GPT_REASONING_MODE` | standard | `pro` makes GPT do more work per answer, at a higher token cost |
| `THUNDERCAT_GROK_MODEL` | `grok-4.7` | xAI model |
| `THUNDERCAT_GROK_REASONING_EFFORT` | `xhigh` | `low` · `medium` · `high` · `xhigh` |
| `THUNDERCAT_WORKSPACE` | current directory | Directory the agents may read |
| `THUNDERCAT_MAX_TURNS` | `25` | Maximum model calls per task |
| `THUNDERCAT_TRACING` | off | `1` sends run traces to the OpenAI dashboard (including Grok runs) |
| `OPENAI_BASE_URL`, `XAI_BASE_URL` | provider default | Point at a proxy or gateway |

Highest reasoning is the default because that is what the agents are for; it is also the
slowest and most expensive setting, so lower the effort for quick questions.

## What the agents can see

Everything a tool returns is sent to OpenAI or xAI. The file tools are read-only, cannot
leave the workspace (symlinks included), and refuse files that usually hold secrets:
`.env*` (except `.env.example`), private keys and certificates, `.git/`, `.ssh/`, cloud
credential directories, `.netrc`/`.npmrc`/`.pypirc`, and Terraform state. That list is a
best-effort guard, not a sandbox — point `THUNDERCAT_WORKSPACE` at a directory you are
comfortable sharing.

## Development

```bash
uv run pytest               # offline: scripted models and a local fake of the Responses API
uv run pytest -m live       # real API calls; needs the keys
uv run ruff check . && uv run ruff format --check .
```
