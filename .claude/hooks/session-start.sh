#!/bin/bash
# SessionStart hook for Claude Code on the web: installs what the GPT and Grok agents
# need, then reports whether they are ready. Installs are idempotent and cached with the
# container. Sign-ins are not, so each new session signs in again (`thundercat login`)
# unless OPENAI_API_KEY / XAI_API_KEY are set in the environment settings.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

CODEX_VERSION="0.158.0"
GROK_VERSION="1.0.41"

cd "$CLAUDE_PROJECT_DIR"
warn() { echo "session-start: $*" >&2; }

# Python environment for thundercat (required).
if ! command -v uv >/dev/null 2>&1; then
  pip install --quiet uv >&2
fi
uv sync --quiet >&2

# OpenAI Codex CLI: runs the GPT agent on a ChatGPT sign-in.
if [ "$(codex --version 2>/dev/null)" != "codex-cli $CODEX_VERSION" ]; then
  npm install -g --silent "@openai/codex@$CODEX_VERSION" >&2 || warn "could not install Codex CLI"
fi

# xAI Grok Build CLI: runs the Grok agent on an xAI sign-in.
if ! "$HOME/.grok/bin/grok" --no-auto-update version 2>/dev/null | grep -q "^grok $GROK_VERSION "; then
  curl -fsSL https://x.ai/cli/install.sh | bash -s "$GROK_VERSION" >&2 || warn "could not install Grok Build"
fi

# bubblewrap: Grok Build's read-only sandbox refuses to start on Linux without it.
if ! command -v bwrap >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  { apt-get install -y -q bubblewrap || { apt-get update -q && apt-get install -y -q bubblewrap; }; } >&2 \
    || warn "could not install bubblewrap"
fi

# Stdout becomes session context: say whether the agents are ready (see CLAUDE.md).
echo "GPT/Grok agents (thundercat) — how each will run this session:"
uv run --quiet thundercat agents || warn "could not read agent status"
