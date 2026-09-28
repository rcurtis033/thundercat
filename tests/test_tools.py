import pytest

from thundercat import tools
from thundercat.tools import ToolError, Workspace


@pytest.fixture
def ws(workspace):
    return Workspace(workspace)


def test_list_directory_hides_secrets(ws):
    listing = ws.list_directory(".").splitlines()
    assert listing[0] == "Contents of ./"
    assert listing[1:] == ["src/", ".env.example", "notes.txt"]


def test_read_file_numbers_lines_and_supports_ranges(ws):
    assert ws.read_file("notes.txt") == "     1\talpha\n     2\tbeta\n     3\tgamma"
    assert ws.read_file("notes.txt", 2, 2) == "     2\tbeta"
    assert "no lines" in ws.read_file("notes.txt", 10)


def test_read_file_truncates_long_output(ws, workspace, monkeypatch):
    monkeypatch.setattr(tools, "MAX_READ_CHARS", 20)
    out = ws.read_file("notes.txt")
    assert out.startswith("     1\talpha")
    assert "call read_file again with start_line=2" in out


@pytest.mark.parametrize(
    "path",
    [".env", "./.env", "src/../.env", ".git/config", "id_rsa", "server.pem", "deploy.key"],
)
def test_secret_paths_are_refused(ws, path):
    with pytest.raises(ToolError, match="secrets"):
        ws.resolve(path)


def test_env_templates_are_allowed(ws):
    assert ws.read_file(".env.example") == "     1\tOPENAI_API_KEY="


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd"])
def test_paths_outside_workspace_are_refused(ws, path):
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.resolve(path)


def test_symlinks_cannot_escape_or_reach_secrets(ws, workspace, tmp_path):
    (tmp_path / "outside.txt").write_text("private")
    (workspace / "escape.txt").symlink_to(tmp_path / "outside.txt")
    (workspace / "innocent.txt").symlink_to(workspace / ".env")

    with pytest.raises(ToolError, match="outside the workspace"):
        ws.read_file("escape.txt")
    with pytest.raises(ToolError, match="secrets"):
        ws.read_file("innocent.txt")
    assert ws.search("private", ".") == "No matches for 'private'"


def test_binary_files_are_refused(ws, workspace):
    (workspace / "blob.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        ws.read_file("blob.bin")


def test_search_reports_path_and_line(ws):
    assert ws.search(r"hello \w+", ".") == "src/app.py:2: return 'hello thundercat'"


def test_search_skips_secrets_and_noise_dirs(ws, workspace):
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    (workspace / "node_modules" / "pkg" / "index.js").write_text("sk-super-secret\n")
    assert ws.search("sk-super-secret", ".") == "No matches for 'sk-super-secret'"


def test_search_rejects_invalid_regex(ws):
    with pytest.raises(ToolError, match="invalid regular expression"):
        ws.search("(unclosed", ".")


async def test_tools_turn_errors_into_messages_for_the_model(ws):
    by_name = {tool.name: tool for tool in tools.workspace_tools(ws)}
    assert set(by_name) == {"current_time", "list_directory", "read_file", "search_files"}

    result = await tools._run(ws.read_file, ".env", 1, None)
    assert result.startswith("Error:") and "secrets" in result
