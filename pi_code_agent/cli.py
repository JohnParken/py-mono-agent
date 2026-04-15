"""Simple interactive coding-agent CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from pi_ai import get_llm_config, get_model
from pi_agent_core import Agent, AgentOptions
from pi_agent_core.types import MessageUpdateEvent, ToolExecutionEndEvent, ToolExecutionStartEvent

from .tools import create_coding_tools


DEFAULT_SYSTEM_PROMPT = """You are a careful coding assistant working inside a local repository.

Use tools instead of guessing whenever the answer depends on the repository contents.
Prefer this workflow:
1. list_files / search_code to locate relevant files
2. read_file to inspect exact code
3. edit_file or write_file to make targeted changes
4. run_command to validate with tests or linters when helpful

Rules:
- Only operate inside the provided workspace.
- Be explicit about what you changed and why.
- Do not invent file contents you have not read.
- Prefer small, precise edits over whole-file rewrites unless necessary.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py-mono-agent",
        description="Run this project as a minimal coding assistant.",
    )
    parser.add_argument("prompt", nargs="?", help="Single prompt to run. Omit for interactive mode.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace root the agent is allowed to access. Defaults to the current directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an llm.yaml config file. Defaults to pi_ai/llm.yaml or llm.yaml.example fallback.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Model config name from the llm.yaml file. Defaults to use_llm.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Direct provider name, for example 'openai'.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Direct model id, for example 'gpt-4o-mini'. Requires --provider.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override API key for direct model mode.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL for OpenAI-compatible providers in direct model mode.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Override the default coding-agent system prompt.",
    )
    return parser


def resolve_model(args: argparse.Namespace):
    if args.provider or args.model:
        if not (args.provider and args.model):
            raise ValueError("--provider and --model must be provided together.")
        return get_model(
            provider=args.provider,
            model_id=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    config = get_llm_config(args.config)
    return config.get_model(args.model_config)


def build_agent(args: argparse.Namespace) -> Agent:
    workspace = Path(args.workspace).resolve()
    tools = create_coding_tools(workspace)
    model = resolve_model(args)
    return Agent(
        AgentOptions(
            initial_state={
                "system_prompt": args.system_prompt,
                "model": model,
                "tools": tools,
            }
        )
    )


def render_event(event) -> None:
    if isinstance(event, MessageUpdateEvent):
        assistant_event = event.assistant_message_event
        if getattr(assistant_event, "type", None) == "text_delta":
            print(assistant_event.delta, end="", flush=True)
    elif isinstance(event, ToolExecutionStartEvent):
        print(f"\n[tool:start] {event.tool_name} {event.args}", file=sys.stderr)
    elif isinstance(event, ToolExecutionEndEvent):
        status = "error" if event.is_error else "ok"
        print(f"[tool:end] {event.tool_name} ({status})", file=sys.stderr)


async def run_once(agent: Agent, prompt: str) -> None:
    print("> " + prompt)
    agent.subscribe(render_event)
    await agent.prompt(prompt)
    print()


async def run_repl(agent: Agent) -> None:
    print("Interactive coding agent. Type 'exit' or 'quit' to leave.")
    agent.subscribe(render_event)
    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return
        await agent.prompt(prompt)
        print()


async def async_main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        agent = build_agent(args)
    except Exception as exc:
        print(f"Failed to initialize agent: {exc}", file=sys.stderr)
        return 1

    if args.prompt:
        await run_once(agent, args.prompt)
    else:
        await run_repl(agent)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
