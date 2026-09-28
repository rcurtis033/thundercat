"""Read-only workspace tools shared by every thundercat agent.

Agents can look around a project (list, search, read) but never modify it, and never
leave the workspace root. Anything a tool returns is sent to the model provider, so
files that usually hold secrets (.env files, private keys, credential stores) are
refused. That deny-list is a best-effort guard, not a sandbox: point the workspace at
a directory you are comfortable sharing with OpenAI / xAI.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import fnmatch
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

from agents import FunctionTool, function_tool

MAX_READ_CHARS = 100_000
MAX_LIST_ENTRIES = 500
MAX_SEARCH_MATCHES = 200
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_MATCH_LINE_CHARS = 300

# Never listed, searched, or read — these commonly contain credentials.
DENIED_DIR_NAMES = frozenset({".git", ".hg", ".svn", ".ssh", ".gnupg", ".aws", ".azure", ".kube"})
DENIED_FILE_GLOBS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "*.tfstate",
    "*.tfstate.*",
)
ALLOWED_FILE_NAMES = frozenset({".env.example", ".env.sample", ".env.template"})

# Readable on request, but skipped when searching recursively (large and rarely relevant).
SEARCH_SKIP_DIR_NAMES = frozenset(
    {"node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


class ToolError(Exception):
    """A problem the agent should be told about (bad path, denied file, ...)."""


def is_denied(relative: Path) -> bool:
    """Whether a workspace-relative path points at (or into) something secret-looking."""
    if any(part in DENIED_DIR_NAMES for part in relative.parts):
        return True
    name = relative.name.lower()
    if name in ALLOWED_FILE_NAMES:
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in DENIED_FILE_GLOBS)


def _is_binary(path: Path) -> bool:
    with path.open("rb") as f:
        return b"\0" in f.read(8192)


class Workspace:
    """The directory tree agents are allowed to read."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        """Map an agent-supplied path to a real path inside the workspace."""
        if is_denied(Path(path)):
            raise ToolError(f"{path!r} may contain secrets and is not available to agents")
        target = (self.root / path).resolve()
        return self._checked(target, shown=path)

    def _checked(self, target: Path, *, shown: str) -> Path:
        # `target` is already resolved, so symlinks pointing outside the root are caught here.
        if not target.is_relative_to(self.root):
            raise ToolError(f"{shown!r} is outside the workspace")
        if is_denied(target.relative_to(self.root)):
            raise ToolError(f"{shown!r} may contain secrets and is not available to agents")
        return target

    def display(self, target: Path) -> str:
        return target.relative_to(self.root).as_posix() or "."

    def list_directory(self, path: str = ".") -> str:
        directory = self.resolve(path)
        if not directory.is_dir():
            raise ToolError(f"{path!r} is not a directory")
        entries: list[str] = []
        for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if is_denied(child.relative_to(directory)):
                continue
            entries.append(f"{child.name}/" if child.is_dir() else child.name)
        header = f"Contents of {self.display(directory)}/"
        if not entries:
            return f"{header}\n(empty)"
        shown = entries[:MAX_LIST_ENTRIES]
        if len(entries) > MAX_LIST_ENTRIES:
            shown.append(f"... and {len(entries) - MAX_LIST_ENTRIES} more entries")
        return "\n".join([header, *shown])

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        file = self.resolve(path)
        if not file.is_file():
            raise ToolError(f"{path!r} is not a file")
        if _is_binary(file):
            raise ToolError(f"{path!r} looks like a binary file")
        start_line = max(start_line, 1)
        if end_line is not None and end_line < start_line:
            raise ToolError("end_line must be greater than or equal to start_line")

        lines: list[str] = []
        used = 0
        with file.open(encoding="utf-8", errors="replace") as f:
            for number, line in enumerate(f, start=1):
                if number < start_line:
                    continue
                if end_line is not None and number > end_line:
                    break
                numbered = f"{number:>6}\t{line.rstrip()}"
                if used + len(numbered) > MAX_READ_CHARS:
                    lines.append(
                        f"... [truncated at line {number - 1}; call read_file again with "
                        f"start_line={number} to continue]"
                    )
                    break
                lines.append(numbered)
                used += len(numbered) + 1
        if not lines:
            return f"{self.display(file)}: no lines in the requested range"
        return "\n".join(lines)

    def search(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        base = self.resolve(path)
        matches: list[str] = []
        for file in self._searchable_files(base):
            try:
                if file.stat().st_size > MAX_SEARCH_FILE_BYTES or _is_binary(file):
                    continue
                with file.open(encoding="utf-8", errors="replace") as f:
                    for number, line in enumerate(f, start=1):
                        if regex.search(line):
                            text = line.strip()[:MAX_MATCH_LINE_CHARS]
                            matches.append(f"{self.display(file)}:{number}: {text}")
                            if len(matches) >= MAX_SEARCH_MATCHES:
                                matches.append(f"... stopped after {MAX_SEARCH_MATCHES} matches")
                                return "\n".join(matches)
            except OSError:
                continue
        return "\n".join(matches) if matches else f"No matches for {pattern!r}"

    def _searchable_files(self, base: Path) -> Iterator[Path]:
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):  # does not follow symlinked dirs
            dirnames[:] = sorted(
                d for d in dirnames if d not in DENIED_DIR_NAMES and d not in SEARCH_SKIP_DIR_NAMES
            )
            for name in sorted(filenames):
                try:
                    yield self._checked(Path(dirpath, name).resolve(), shown=name)
                except ToolError:
                    continue


async def _run(func: Callable[..., str], *args: object) -> str:
    """Run blocking file work off the event loop and turn expected failures into text."""
    try:
        return await asyncio.to_thread(func, *args)
    except (ToolError, OSError) as exc:
        return f"Error: {exc}"


def workspace_tools(workspace: Workspace) -> list[FunctionTool]:
    """The default tool set: a clock plus read-only access to `workspace`."""

    @function_tool
    def current_time() -> str:
        """Get the current date and time, in UTC and in the machine's local time zone."""
        now = dt.datetime.now(dt.UTC)
        local = now.astimezone().isoformat(timespec="seconds")
        return f"UTC: {now.isoformat(timespec='seconds')}\nLocal: {local}"

    @function_tool
    async def list_directory(path: str) -> str:
        """List the files and subdirectories of a directory in the workspace.

        Args:
            path: Directory path relative to the workspace root. Use "." for the root.
        """
        return await _run(workspace.list_directory, path)

    @function_tool
    async def read_file(path: str, start_line: int, end_line: int | None) -> str:
        """Read a text file from the workspace. Lines are returned with line numbers.

        Args:
            path: File path relative to the workspace root.
            start_line: First line to return (1-based). Use 1 to start at the top.
            end_line: Last line to return (inclusive), or null to read to the end of the file.
        """
        return await _run(workspace.read_file, path, start_line, end_line)

    @function_tool
    async def search_files(pattern: str, path: str) -> str:
        """Search text files for a regular expression, like `grep -rn`.

        Args:
            pattern: Python regular expression to look for, matched line by line.
            path: File or directory to search, relative to the workspace root. Use "." for all.
        """
        return await _run(workspace.search, pattern, path)

    return [current_time, list_directory, read_file, search_files]
