# thundercat

GPT (OpenAI, `gpt-6-astra` at reasoning `max`) and Grok (xAI, `grok-4.7` at `xhigh`) set up
as read-only agents. See README.md for the full picture.

## Use the GPT and Grok agents whenever they would help

Standing instruction from the user: in every chat, use these agents when needed — don't
wait to be asked. They are the MCP tools `ask_gpt` and `ask_grok` (server `thundercat`,
from `.mcp.json`), already allowed in `.claude/settings.json`.

Good times to use them:

- A second opinion on a plan, design, or non-obvious decision before committing to it.
- An independent review of your diff before you commit or push anything non-trivial.
- Debugging that is not converging: ask for hypotheses you have not considered.
- High-stakes questions: ask both in parallel and compare their answers.

Skip them for trivial or mechanical work: calls at maximum reasoning take a while and use
the user's accounts or API credits.

How to call them:

- Every call is a fresh, stateless run. The agent sees only the `task` text, so include the
  goal, the relevant context, file paths, and what a good answer looks like.
- They can read files in this repo (read-only), so point them at paths instead of pasting
  large files.
- Treat answers as advice: verify claims against the code before acting on them, and tell
  the user briefly what you asked and what came back.

## If an agent is not ready

The SessionStart hook prints each agent's status at the start of the session; check again
with `uv run thundercat agents`.

- "not signed in": sign in with the user's account. Run `uv run thundercat login gpt` (or
  `grok`) as a background command, read the link and code it prints, and give both to the
  user: they open the link in any browser and enter the code (there is no pop-up). Codes
  expire after about 15 minutes. The running MCP server picks up the sign-in by itself.
- "not set up": the vendor CLI is missing (the SessionStart hook installs it on the web)
  and there is no API key. `OPENAI_API_KEY` / `XAI_API_KEY` in the environment settings
  also work, with no sign-in needed.
- From the shell, `uv run thundercat ask gpt "..."` runs an agent directly.

## Development

- `uv sync`, then `uv run pytest` (offline) and `uv run ruff check . && uv run ruff format --check .`
- `uv run pytest -m live` makes real calls through each agent's key or signed-in CLI.
- Code: `src/thundercat/` — `config.py` (env settings), `team.py` (API backend on the
  OpenAI Agents SDK), `vendor_cli.py` (Codex / Grok Build backend), `tools.py` (read-only
  workspace tools), `cli.py`, `mcp_server.py`.
- Tests never touch real vendor CLIs (see the autouse fixture in `tests/conftest.py`).
