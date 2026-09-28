"""End-to-end runs over real HTTP against a local fake of the Responses API.

These pin down what actually goes over the wire to OpenAI and xAI: endpoint, auth,
model, reasoning settings, tool definitions, and the multi-turn tool loop.
"""

import json

from thundercat.config import load_settings
from thundercat.team import Team

WORKSPACE_TOOLS = ["current_time", "list_directory", "read_file", "search_files"]
# Grok's reasoning models return an error when these are present.
GROK_REJECTED_PARAMS = {"presence_penalty", "frequency_penalty", "stop"}


def response(*output):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "fake-model",
        "status": "completed",
        "output": list(output),
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def message(text):
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def function_call(name, arguments):
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-blob"}


async def test_grok_runs_a_tool_loop_against_xai(fake_provider, workspace):
    provider = fake_provider(
        [
            response(
                REASONING,
                function_call(
                    "read_file", {"path": "notes.txt", "start_line": 1, "end_line": None}
                ),
            ),
            response(message("notes.txt lists alpha, beta and gamma.")),
        ]
    )
    settings = load_settings(
        {
            "XAI_API_KEY": "xai-test-key",
            "XAI_BASE_URL": provider.base_url,
            "THUNDERCAT_WORKSPACE": str(workspace),
        }
    )

    answer = await Team(settings).ask("grok", "Summarize notes.txt")

    assert answer == "notes.txt lists alpha, beta and gamma."
    first, second = provider.requests
    assert first["path"] == "/v1/responses"
    assert first["headers"]["authorization"] == "Bearer xai-test-key"
    body = first["json"]
    assert body["model"] == "grok-4.7"
    assert body["reasoning"] == {"effort": "xhigh"}
    assert sorted(tool["name"] for tool in body["tools"]) == WORKSPACE_TOOLS
    assert body["instructions"].startswith("You are Grok (xAI)")
    assert not GROK_REJECTED_PARAMS & body.keys()

    # xAI asks for reasoning items to be passed back unchanged on the next turn.
    replayed = second["json"]["input"]
    reasoning = next(item for item in replayed if item.get("type") == "reasoning")
    assert reasoning["encrypted_content"] == "opaque-blob"
    tool_output = next(item for item in replayed if item.get("type") == "function_call_output")
    assert "     2\tbeta" in tool_output["output"]


async def test_gpt_sends_max_reasoning_to_openai(fake_provider, workspace):
    provider = fake_provider([response(message("Done."))])
    settings = load_settings(
        {
            "OPENAI_API_KEY": "sk-test-openai",
            "OPENAI_BASE_URL": provider.base_url,
            "THUNDERCAT_GPT_REASONING_MODE": "pro",
            "THUNDERCAT_WORKSPACE": str(workspace),
        }
    )

    assert await Team(settings).ask("gpt", "Say done") == "Done."

    (request,) = provider.requests
    assert request["path"] == "/v1/responses"
    assert request["headers"]["authorization"] == "Bearer sk-test-openai"
    assert request["json"]["model"] == "gpt-6-astra"
    assert request["json"]["reasoning"] == {"effort": "max", "mode": "pro"}
