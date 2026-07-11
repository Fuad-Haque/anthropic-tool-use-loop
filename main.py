import os
import subprocess
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-6"
SEARCH_ROOT = os.environ.get("SEARCH_ROOT", "./sample_project")

# ---- Section 2 model, made real: the tool the model is allowed to call ----
TOOLS = [
    {
        "name": "search_files",
        "description": (
            "Search for a text pattern inside files under the project's search "
            "root directory. Returns matching file paths and line numbers. Use "
            "this whenever the user asks what a file contains, whether a term "
            "or function exists in the codebase, or where something is defined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The text or code pattern to search for, case-insensitive.",
                }
            },
            "required": ["pattern"],
        },
    }
]


def search_files(pattern: str) -> str:
    """
    The actual tool implementation. This is Claude's grep, not Claude's answer.
    Claude never runs this itself -- our server runs it and hands back the result.
    """
    if not pattern or not pattern.strip():
        raise ValueError("pattern must be a non-empty string")

    if not os.path.isdir(SEARCH_ROOT):
        raise FileNotFoundError(f"search root does not exist: {SEARCH_ROOT}")

    result = subprocess.run(
        ["grep", "-rn", "-i", pattern, SEARCH_ROOT],
        capture_output=True,
        text=True,
        timeout=5,
    )

    if result.returncode not in (0, 1):
        # 0 = matches found, 1 = no matches, anything else is a real grep error
        raise RuntimeError(f"grep failed: {result.stderr.strip()}")

    output = result.stdout.strip()
    return output if output else "No matches found."


def execute_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """
    Dispatch a tool_use block to its implementation.
    Returns (content_string, is_error).
    """
    if name != "search_files":
        return f"Unknown tool: {name}", True

    try:
        return search_files(tool_input.get("pattern", "")), False
    except Exception as e:
        return f"{type(e).__name__}: {e}", True


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    messages: list[dict[str, Any]] = [{"role": "user", "content": req.question}]

    # ---- The loop: call, check stop_reason, execute, feed back, repeat ----
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # end_turn (or anything else): Claude has its final answer.
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": final_text, "stop_reason": response.stop_reason}

        # stop_reason == "tool_use": at least one tool_use block is present.
        # Append the assistant's turn (including the tool_use block) verbatim --
        # the API requires the full content array back, not just the tool result.
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            result_text, is_error = execute_tool(block.name, block.input)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                }
            )

        # tool_result blocks go back as a user turn. This is the "return
        # tool_result" step -- it's what lets the loop continue.
        messages.append({"role": "user", "content": tool_results})


@app.get("/health")
def health():
    return {"status": "ok", "search_root": SEARCH_ROOT}