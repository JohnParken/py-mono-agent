"""Typed options for Qwen tool prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QwenToolPromptMode = Literal["text"]


@dataclass(frozen=True)
class QwenToolPromptOptions:
    """Assembly options for Qwen tool prompts."""

    system_prompt: str
    mode: QwenToolPromptMode
    include_fewshots: bool = False
