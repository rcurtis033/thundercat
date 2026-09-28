from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from agents.testing import ScriptedModel

from thundercat import vendor_cli
from thundercat.config import Settings, load_settings
from thundercat.team import Team

TEST_ENV = {"OPENAI_API_KEY": "sk-test-openai", "XAI_API_KEY": "xai-test-key"}


@pytest.fixture(autouse=True)
def hide_installed_vendor_clis(request, monkeypatch):
    """Keep offline tests off any real codex/grok on this machine; they pass fakes by path."""
    if request.node.get_closest_marker("live"):
        return
    real = vendor_cli.find_executable
    monkeypatch.setattr(
        vendor_cli,
        "find_executable",
        lambda spec: real(spec) if os.path.isabs(spec.cli_program) else None,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    (root / "src" / "app.py").write_text("def greet():\n    return 'hello thundercat'\n")
    (root / ".env").write_text("OPENAI_API_KEY=sk-super-secret\n")
    (root / ".env.example").write_text("OPENAI_API_KEY=\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[remote]\nurl = https://token@example.com\n")
    return root


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return load_settings({**TEST_ENV, "THUNDERCAT_WORKSPACE": str(workspace)})


@pytest.fixture
def make_team(settings: Settings) -> Callable[..., Team]:
    """Build a Team whose agents use ScriptedModels (keyed by agent name), not real APIs."""

    def make(**models: ScriptedModel) -> Team:
        return Team(settings, model_factory=lambda spec: models[spec.name])

    return make


class FakeProvider:
    """A local OpenAI-compatible HTTP API that records requests and replays canned replies."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                headers = {key.lower(): value for key, value in self.headers.items()}
                provider.requests.append(
                    {"path": self.path, "headers": headers, "json": json.loads(body)}
                )
                if provider.replies:
                    status, reply = 200, provider.replies.pop(0)
                else:
                    status, reply = 500, {"error": {"message": "FakeProvider: no reply left"}}
                payload = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_provider() -> Iterator[Callable[[list[dict[str, Any]]], FakeProvider]]:
    """Start FakeProviders on demand; they are shut down after the test."""
    started: list[FakeProvider] = []

    def start(replies: list[dict[str, Any]]) -> FakeProvider:
        started.append(FakeProvider(replies))
        return started[-1]

    yield start
    for provider in started:
        provider.close()
