"""Product-layer settings for the coding agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16_384
    keep_recent_tokens: int = 20_000


@dataclass
class ResourceSettings:
    enable_skills: bool = True
    skill_paths: list[str] = field(default_factory=list)
    context_file_patterns: list[str] = field(
        default_factory=lambda: ["AGENTS.md", ".pi/**/*.md"]
    )


@dataclass
class SessionSettings:
    auto_save: bool = True
    session_dir: str | None = None
    continue_last_session: bool = False


@dataclass
class AgentProductSettings:
    steering_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    follow_up_mode: Literal["all", "one-at-a-time"] = "one-at-a-time"
    compact_on_overflow: bool = True
    compact_on_threshold: bool = False


@dataclass
class CodeAgentSettings:
    session: SessionSettings = field(default_factory=SessionSettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    product: AgentProductSettings = field(default_factory=AgentProductSettings)
    compaction: CompactionSettings = field(default_factory=CompactionSettings)


def get_default_agent_dir() -> Path:
    return Path.home() / ".pi" / "agent"


def load_settings(
    workspace: Path,
    agent_dir: Path | None = None,
) -> CodeAgentSettings:
    agent_root = agent_dir or get_default_agent_dir()
    global_path = agent_root / "settings.json"
    project_path = workspace / ".pi" / "settings.json"

    data: dict[str, Any] = {}
    for path in (global_path, project_path):
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = _deep_merge(data, loaded)

    return _parse_settings(data)


def resolve_session_dir(
    settings: CodeAgentSettings,
    workspace: Path,
    agent_dir: Path | None = None,
) -> Path:
    raw = settings.session.session_dir
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        return candidate.resolve()
    _ = agent_dir
    return (workspace / ".pi" / "sessions").resolve()


def _parse_settings(data: dict[str, Any]) -> CodeAgentSettings:
    session_data = _get_dict(data, "session")
    resources_data = _get_dict(data, "resources")
    product_data = _get_dict(data, "product")
    compaction_data = _get_dict(data, "compaction")

    return CodeAgentSettings(
        session=SessionSettings(
            auto_save=bool(session_data.get("auto_save", True)),
            session_dir=_get_str_or_none(session_data.get("session_dir")),
            continue_last_session=bool(session_data.get("continue_last_session", False)),
        ),
        resources=ResourceSettings(
            enable_skills=bool(resources_data.get("enable_skills", True)),
            skill_paths=_coerce_str_list(resources_data.get("skill_paths", [])),
            context_file_patterns=_coerce_str_list(
                resources_data.get("context_file_patterns", ["AGENTS.md", ".pi/**/*.md"])
            ),
        ),
        product=AgentProductSettings(
            steering_mode=_coerce_mode(product_data.get("steering_mode"), "one-at-a-time"),
            follow_up_mode=_coerce_mode(product_data.get("follow_up_mode"), "one-at-a-time"),
            compact_on_overflow=bool(product_data.get("compact_on_overflow", True)),
            compact_on_threshold=bool(product_data.get("compact_on_threshold", False)),
        ),
        compaction=CompactionSettings(
            enabled=bool(compaction_data.get("enabled", True)),
            reserve_tokens=int(compaction_data.get("reserve_tokens", 16_384)),
            keep_recent_tokens=int(compaction_data.get("keep_recent_tokens", 20_000)),
        ),
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _get_str_or_none(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def _coerce_mode(value: Any, default: str) -> Literal["all", "one-at-a-time"]:
    if value in ("all", "one-at-a-time"):
        return value
    return default  # type: ignore[return-value]
