"""Composable prompt builder for Qwen tool-use modes."""

from __future__ import annotations

import importlib.resources
import json
from typing import Any, Dict, Iterable, List

from .models import QwenToolPromptOptions


_PACKAGE = "pi_ai.prompts.qwen_tools"


def _load_text_resource(filename: str) -> str:
    return (
        importlib.resources.files(_PACKAGE)
        .joinpath(filename)
        .read_text(encoding="utf-8")
        .strip()
    )


def _join_sections(sections: Iterable[str]) -> str:
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_tools_block(api_tools: List[Dict[str, Any]]) -> str:
    if not api_tools:
        return ""
    tools_block = "\n".join(
        json.dumps(tool_schema, ensure_ascii=False, indent=2)
        for tool_schema in api_tools
    )
    return f"# Tools\n\n<tools>\n{tools_block}\n</tools>"


def build_qwen_tool_prompt(
    system_prompt: str,
    api_tools: List[Dict[str, Any]],
    *,
    mode: str,
    include_fewshots: bool = False,
) -> str:
    """Build the full tool prompt for Qwen native or text mode."""
    resolved_mode = "native" if str(mode).lower() == "native" else "text"
    options = QwenToolPromptOptions(
        system_prompt=system_prompt or "",
        mode=resolved_mode,
        include_fewshots=include_fewshots,
    )

    shared_rules = _load_text_resource("shared_rules.md")
    mode_rules = _load_text_resource(f"{options.mode}_rules.md")
    sections: List[str] = [options.system_prompt, shared_rules, mode_rules]

    tools_block = _render_tools_block(api_tools)
    if tools_block and options.mode == "text":
        sections.append(tools_block)

    if options.include_fewshots:
        fewshots = _load_text_resource(f"{options.mode}_fewshots.md")
        if fewshots:
            sections.append(fewshots)

    if options.mode == "text":
        sections.append(
            "If a tool is needed, output only one "
            "<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call> block."
        )

    return _join_sections(sections)
