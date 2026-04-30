"""Simple interactive coding-agent CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from pi_ai import get_llm_config, get_model
from pi_agent_core import Agent
from pi_agent_core.types import (
    AgentEndEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from pi_logger import get_logger

from .session import AgentSession, SessionCreateOptions

logger = get_logger("pi_code_agent.cli")


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
- When writing multiline files with write_file, prefer content_lines over content.
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
        model = get_model(
            provider=args.provider,
            model_id=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        logger.debug(
            "[CODE-CLI] resolved direct model provider=%s model=%s base_url=%s has_api_key=%s",
            model.provider,
            model.id,
            model.base_url,
            bool(model.api_key),
        )
        return model

    config = get_llm_config(args.config)
    model_name = args.model_config or config.get_current_name()
    logger.debug(
        "[CODE-CLI] resolving model from config config_path=%s model_config=%s",
        args.config,
        model_name,
    )
    model = config.get_model(args.model_config)
    logger.debug(
        "[CODE-CLI] resolved configured model provider=%s model=%s base_url=%s has_api_key=%s",
        model.provider,
        model.id,
        model.base_url,
        bool(model.api_key),
    )
    return model


def build_session(args: argparse.Namespace) -> AgentSession:
    workspace = Path(args.workspace).resolve()
    model = resolve_model(args)
    logger.debug(
        "[CODE-CLI] building agent session workspace=%s provider=%s model=%s",
        workspace,
        model.provider,
        model.id,
    )
    return AgentSession.create(
        SessionCreateOptions(
            workspace=workspace,
            config_path=args.config,
            model_config=args.model_config,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            custom_system_prompt=args.system_prompt,
            model_instance=model,
        )
    )


def _message_text(message) -> str:
    parts: list[str] = []
    for content in getattr(message, "content", []) or []:
        content_type = getattr(content, "type", None)
        if content_type == "text":
            text = getattr(content, "text", "")
            if text:
                parts.append(text)
    return "".join(parts)


def _message_tool_call_count(message) -> int:
    count = 0
    for content in getattr(message, "content", []) or []:
        if getattr(content, "type", None) == "toolCall":
            count += 1
    return count


def _describe_message(message) -> str:
    if message is None:
        return "message=None"
    text = _message_text(message)
    return (
        f"role={getattr(message, 'role', None)} "
        f"stop_reason={getattr(message, 'stop_reason', None)} "
        f"error_message={getattr(message, 'error_message', None)!r} "
        f"text_len={len(text)} "
        f"tool_calls={_message_tool_call_count(message)}"
    )


def _report_prompt_outcome(agent: Agent, start_index: int, source: str) -> None:
    new_messages = agent.state.messages[start_index:]
    assistant_messages = [
        message
        for message in new_messages
        if getattr(message, "role", None) == "assistant"
    ]

    logger.debug(
        "%s outcome new_messages=%s assistant_messages=%s state_error=%r",
        source,
        len(new_messages),
        len(assistant_messages),
        agent.state.error,
    )

    if agent.state.error:
        logger.warning("%s finished with state.error=%s", source, agent.state.error)
        print(f"\n[agent:error] {agent.state.error}", file=sys.stderr)

    if not assistant_messages:
        logger.warning("%s finished without assistant message", source)
        print("\n[assistant:empty] no assistant message generated", file=sys.stderr)
        return

    last_assistant = assistant_messages[-1]
    logger.debug("%s last assistant %s", source, _describe_message(last_assistant))

    error_message = getattr(last_assistant, "error_message", None)
    if error_message:
        logger.warning("%s assistant error=%s", source, error_message)
        print(f"\n[assistant:error] {error_message}", file=sys.stderr)
        return

    text = _message_text(last_assistant)
    tool_calls = _message_tool_call_count(last_assistant)
    if text.strip():
        logger.debug(
            "%s assistant produced visible text text_len=%s preview=%r",
            source,
            len(text),
            text[:200],
        )
        return

    if tool_calls:
        logger.debug("%s assistant finished with tool calls only tool_calls=%s", source, tool_calls)
        return

    logger.warning("%s assistant finished without visible text or tool calls", source)
    print("\n[assistant:empty] completed without visible text output", file=sys.stderr)


def render_event(event) -> None:
    if isinstance(event, MessageUpdateEvent):
        assistant_event = event.assistant_message_event
        event_type = getattr(assistant_event, "type", None)
        if getattr(assistant_event, "type", None) == "text_delta":
            print(assistant_event.delta, end="", flush=True)
        elif event_type == "error":
            logger.warning("[CODE-CLI] assistant stream error event received")
    elif isinstance(event, MessageEndEvent):
        message = event.message
        logger.debug("[CODE-CLI] message end %s", _describe_message(message))
        if getattr(message, "role", None) == "assistant":
            error_message = getattr(message, "error_message", None)
            if error_message:
                print(f"\n[assistant:error] {error_message}", file=sys.stderr)
            else:
                text = _message_text(message)
                if text.strip():
                    return
                elif _message_tool_call_count(message):
                    logger.debug(
                        "[CODE-CLI] assistant message ended with tool calls only tool_calls=%s",
                        _message_tool_call_count(message),
                    )
                else:
                    logger.warning("[CODE-CLI] assistant message ended empty: %s", _describe_message(message))
                    print("\n[assistant:empty] completed without visible text output", file=sys.stderr)
    elif isinstance(event, ToolExecutionStartEvent):
        logger.debug("[CODE-CLI] tool execution start tool=%s args=%s", event.tool_name, event.args)
        print(f"\n[tool:start] {event.tool_name} {event.args}", file=sys.stderr)
    elif isinstance(event, ToolExecutionUpdateEvent):
        return
    elif isinstance(event, ToolExecutionEndEvent):
        status = "error" if event.is_error else "ok"
        logger.debug(
            "[CODE-CLI] tool execution end tool=%s status=%s result=%s",
            event.tool_name,
            status,
            event.result,
        )
        print(f"[tool:end] {event.tool_name} ({status})", file=sys.stderr)
    elif isinstance(event, AgentEndEvent):
        last_message = event.messages[-1] if event.messages else None
        logger.debug(
            "[CODE-CLI] agent end messages=%s last_message=%s",
            len(event.messages),
            _describe_message(last_message),
        )


async def run_once(session: AgentSession, prompt: str) -> None:
    agent = session.agent
    logger.debug("[CODE-CLI] run once prompt_len=%s prompt_preview=%r", len(prompt), prompt[:120])
    print("> " + prompt)
    session.subscribe(render_event)
    start_index = len(agent.state.messages)
    await session.prompt(prompt)
    _report_prompt_outcome(agent, start_index, "Run once")
    logger.debug("[CODE-CLI] run once completed")
    print()


async def run_repl(session: AgentSession) -> None:
    agent = session.agent
    logger.debug("[CODE-CLI] starting interactive REPL")
    print("Interactive coding agent. Type 'exit' or 'quit' to leave.")
    session.subscribe(render_event)
    while True:
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            logger.debug("[CODE-CLI] REPL exit requested")
            return
        logger.debug("[CODE-CLI] REPL prompt_len=%s prompt_preview=%r", len(prompt), prompt[:120])
        start_index = len(agent.state.messages)
        await session.prompt(prompt)
        _report_prompt_outcome(agent, start_index, "REPL prompt")
        logger.debug("[CODE-CLI] REPL prompt completed")
        print()


async def async_main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger.debug(
        "[CODE-CLI] start workspace=%s config=%s model_config=%s provider=%s model=%s",
        args.workspace,
        args.config,
        args.model_config,
        args.provider,
        args.model,
    )
    try:
        session = build_session(args)
    except Exception as exc:
        logger.exception("[CODE-CLI] failed to initialize agent: %s", exc)
        print(f"Failed to initialize agent: {exc}", file=sys.stderr)
        return 1

    if args.prompt:
        await run_once(session, args.prompt)
    else:
        await run_repl(session)
    logger.debug("[CODE-CLI] finished successfully")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
