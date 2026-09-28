# thundercat

Use **GPT** (OpenAI) and **Grok** (xAI) as tool-using agents — from the command line,
from Python, or from Claude Code, which can hand either one a task over MCP.

| Agent  | Provider | Default model  | Reasoning (default = highest) | API key          | Or sign in with  |
| ------ | -------- | -------------- | ----------------------------- | ---------------- | ---------------- |
| `gpt`  | OpenAI   | `gpt-6-astra`  | `max`                         | `OPENAI_API_KEY` | ChatGPT (Codex CLI) |
| `grok` | xAI      | `grok-4.7`     | `xhigh`                       | `XAI_API_KEY`    | xAI account (Grok Build) |

Each agent runs one of two ways:

- **API key** — the agent runs its own tool loop (built on the
  [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)) against its
  provider's Responses API, with read-only workspace tools (`list_directory`,
  `search_files`, `read_file`) and a clock. Agents can consult each other (`--consult`).
- **Your account, no API key** — the agent runs through its vendor's own agent CLI,
  signed in with your ChatGPT or xAI account: OpenAI's Codex CLI for `gpt`, xAI's Grok
  Build for `grok`. Same model and reasoning effort, headless and read-only.

With a key set, the API is used; otherwise the signed-in CLI (see `THUNDERCAT_*_BACKEND`).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run thundercat agents   # shows how each agent will run
```

Then, for each agent, either:

- **Sign in with your account.** Install the CLI once, then sign in with a device code
  (open the printed link on any device and enter the code):

  ```bash
  npm install -g @openai/codex                     # GPT
  curl -fsSL https://x.ai/cli/install.sh | bash    # Grok
  uv run thundercat login gpt
  uv run thundercat login grok
  ```

- **Or use an API key** from https://platform.openai.com/api-keys or
  https://console.x.ai: `cp .env.example .env` and fill it in (it is gitignored), or
  export the variables. For Claude Code on the web, add them in the cloud environment's
  settings instead, then start a new session.

## Command line

```bash
uv run thundercat ask gpt "Review src/thundercat/tools.py for path-traversal bugs"
uv run thundercat ask grok --consult gpt "Is our retry logic safe? Get a second opinion."
git diff | uv run thundercat ask grok        # the task can come from stdin
uv run thundercat chat gpt                   # multi-turn session; /exit to quit
```

Text streams as it is generated and tool calls are shown on stderr; `--no-stream` prints
only the final answer. Streaming, `chat` and `--consult` need the API backend; through a
signed-in CLI, `ask` prints the answer when it is done (and for an interactive session,
run `codex` or `grok` directly).

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
| `THUNDERCAT_GPT_REASONING_MODE` | standard | `pro` makes GPT do more work per answer, at a higher token cost (API only) |
| `THUNDERCAT_GROK_MODEL` | `grok-4.7` | xAI model |
| `THUNDERCAT_GROK_REASONING_EFFORT` | `xhigh` | `low` · `medium` · `high` · `xhigh` |
| `THUNDERCAT_GPT_BACKEND`, `THUNDERCAT_GROK_BACKEND` | `auto` | `api`, `cli` (signed-in vendor CLI), or `auto`: API when a key is set |
| `THUNDERCAT_GPT_CLI`, `THUNDERCAT_GROK_CLI` | `codex`, `grok` | Path or name of the vendor CLI |
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

Through a signed-in CLI, the vendor's own agent reads the files instead: Codex runs with
`--sandbox read-only`, Grok Build with `--sandbox read-only` and only its Read and Grep
tools allowed. Neither applies thundercat's secret-file list, so keep secrets out of the
workspace when using that route.

## Development

```bash
uv run pytest               # offline: scripted models, a fake Responses API, fake vendor CLIs
uv run pytest -m live       # real runs; each agent needs its key or a signed-in CLI
uv run ruff check . && uv run ruff format --check .
```
