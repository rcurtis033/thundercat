"""Run an agent through its vendor's own agent CLI, signed in with your account.

This is how the agents work without API keys: `gpt` runs through OpenAI's Codex CLI
(sign in with ChatGPT) and `grok` through xAI's Grok Build CLI (sign in with your xAI
account). Both run headless and read-only, with the same model and reasoning effort
as the API backend. Unlike the API backend's file tools, the vendor CLIs do not apply
thundercat's secret-file filter; they rely on their own read-only sandboxes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import REQUEST_TIMEOUT_SECONDS, AgentSpec, Settings

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class VendorCliError(RuntimeError):
    """The vendor CLI is missing, not signed in, or failed."""


@dataclass(frozen=True)
class VendorCli:
    product: str
    install: str
    login_args: tuple[str, ...]


VENDOR_CLIS = {
    "gpt": VendorCli(
        product="Codex CLI",
        install="npm install -g @openai/codex",
        login_args=("login", "--device-auth"),
    ),
    "grok": VendorCli(
        product="Grok Build",
        install="curl -fsSL https://x.ai/cli/install.sh | bash",
        login_args=("--no-auto-update", "login", "--device-auth"),
    ),
}


def vendor(spec: AgentSpec) -> VendorCli:
    return VENDOR_CLIS[spec.name]


def find_executable(spec: AgentSpec) -> str | None:
    """The CLI's path: on PATH, or where the Grok installer puts it (~/.grok/bin)."""
    found = shutil.which(spec.cli_program)
    if found:
        return found
    installed = Path.home() / ".grok" / "bin" / spec.cli_program
    return str(installed) if spec.name == "grok" and os.access(installed, os.X_OK) else None


def is_signed_in(spec: AgentSpec) -> bool:
    executable = find_executable(spec)
    if executable is None:
        return False
    if spec.name == "gpt":
        try:
            status = subprocess.run(
                [executable, "login", "status"], capture_output=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return status.returncode == 0
    grok_home = Path(os.environ.get("GROK_HOME") or Path.home() / ".grok")
    return (grok_home / "auth.json").is_file() or bool(os.environ.get("XAI_API_KEY"))


def login(spec: AgentSpec) -> int:
    """Run the vendor's device-code sign-in in this terminal; returns its exit status."""
    executable = _require(spec)
    return subprocess.run([executable, *vendor(spec).login_args], check=False).returncode


async def run(spec: AgentSpec, settings: Settings, task: str) -> str:
    """Give `task` to the vendor CLI and return its final answer."""
    executable = _require(spec)
    if not await asyncio.to_thread(is_signed_in, spec):
        raise VendorCliError(
            f"{vendor(spec).product} is not signed in: run `thundercat login {spec.name}` "
            f"(or set {spec.api_key_env} to use the API instead)."
        )
    with tempfile.TemporaryDirectory(prefix="thundercat-") as scratch:
        if spec.name == "gpt":
            answer_file = Path(scratch, "answer.md")
            await _execute(spec, _codex_command(executable, spec, settings, answer_file), task)
            answer = answer_file.read_text().strip() if answer_file.exists() else ""
        else:
            prompt_file = Path(scratch, "task.md")
            prompt_file.write_text(task)
            output = await _execute(spec, _grok_command(executable, spec, settings, prompt_file))
            answer = _grok_answer(output)
    if not answer:
        raise VendorCliError(f"{vendor(spec).product} finished without an answer")
    return answer


def _codex_command(
    executable: str, spec: AgentSpec, settings: Settings, answer_file: Path
) -> list[str]:
    return [
        executable,
        "exec",
        "--model",
        spec.model,
        "--config",
        f'model_reasoning_effort="{spec.reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--cd",
        str(settings.workspace),
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "--output-last-message",
        str(answer_file),
        "-",  # read the task from stdin, which has no length limit
    ]


def _grok_command(
    executable: str, spec: AgentSpec, settings: Settings, prompt_file: Path
) -> list[str]:
    return [
        executable,
        "--no-auto-update",
        "--prompt-file",
        str(prompt_file),
        "--model",
        spec.model,
        "--reasoning-effort",
        spec.reasoning_effort,
        "--output-format",
        "json",
        "--cwd",
        str(settings.workspace),
        # Headless read-only: deny every tool call except reading and searching files.
        "--permission-mode",
        "dontAsk",
        "--allow",
        "Read",
        "--allow",
        "Grep",
        "--sandbox",
        "read-only",
        "--max-turns",
        str(settings.max_turns),
    ]


def _grok_answer(output: str) -> str:
    """Pull the answer out of `--output-format json`, falling back to the raw text."""
    text = _ANSI.sub("", output).strip()
    for candidate in (text, text.rsplit("\n", 1)[-1]):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if data.get("is_error") or data.get("error"):
                raise VendorCliError(f"Grok Build reported an error: {candidate[:2000]}")
            for key in ("result", "response", "text", "content", "output"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return text


def _require(spec: AgentSpec) -> str:
    executable = find_executable(spec)
    if executable is None:
        cli = vendor(spec)
        raise VendorCliError(
            f"The {spec.name} agent has no API key and {cli.product} is not installed. "
            f"Set {spec.api_key_env}, or install it (`{cli.install}`) and run "
            f"`thundercat login {spec.name}`."
        )
    return executable


async def _execute(spec: AgentSpec, command: list[str], stdin: str | None = None) -> str:
    product = vendor(spec).product
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin.encode() if stdin is not None else None),
            REQUEST_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise VendorCliError(
            f"{product} did not finish within {REQUEST_TIMEOUT_SECONDS:.0f}s"
        ) from None
    finally:
        # Covers timeouts and cancellation (e.g. the MCP client gave up): don't leave it running.
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode != 0:
        detail = _ANSI.sub("", (stderr or stdout).decode(errors="replace")).strip()[-2000:]
        raise VendorCliError(f"{product} exited with status {process.returncode}: {detail}")
    return stdout.decode(errors="replace")
