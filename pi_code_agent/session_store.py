"""Append-only session persistence for coding-agent sessions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_agent_core.types import AgentMessage

SessionEntryType = Literal[
    "session_meta",
    "system_snapshot",
    "user_message",
    "assistant_message",
    "tool_result",
    "thinking_level_change",
    "compaction",
]


@dataclass
class SessionEntry:
    id: str
    parent_id: str | None
    timestamp: float
    type: SessionEntryType
    payload: dict[str, Any]


@dataclass
class SessionMeta:
    session_id: str
    workspace: str
    created_at: float
    updated_at: float
    title: str | None = None


@dataclass
class SessionSnapshot:
    meta: SessionMeta
    entries: list[SessionEntry]


class SessionStore:
    def __init__(self, session_file: Path) -> None:
        self._session_file = session_file.resolve()

    @classmethod
    def create_new(cls, session_dir: Path, workspace: Path) -> "SessionStore":
        session_dir.mkdir(parents=True, exist_ok=True)
        session_id = str(uuid.uuid4())
        session_file = session_dir / f"{session_id}.jsonl"
        store = cls(session_file)
        now = time.time()
        meta = SessionEntry(
            id=store._new_entry_id(),
            parent_id=None,
            timestamp=now,
            type="session_meta",
            payload={
                "session_id": session_id,
                "workspace": str(workspace.resolve()),
                "created_at": now,
                "updated_at": now,
                "title": None,
            },
        )
        store.append_entry(meta)
        return store

    @classmethod
    def open_existing(cls, session_file: Path) -> "SessionStore":
        return cls(session_file)

    def load_snapshot(self) -> SessionSnapshot:
        entries = self.read_entries()
        if not entries or entries[0].type != "session_meta":
            raise ValueError(f"Invalid session file: {self._session_file}")
        meta_payload = entries[0].payload
        meta = SessionMeta(
            session_id=str(meta_payload["session_id"]),
            workspace=str(meta_payload["workspace"]),
            created_at=float(meta_payload["created_at"]),
            updated_at=float(meta_payload["updated_at"]),
            title=meta_payload.get("title"),
        )
        return SessionSnapshot(meta=meta, entries=entries)

    def append_entry(self, entry: SessionEntry) -> None:
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        with self._session_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._entry_to_dict(entry), ensure_ascii=False) + "\n")

    def append_entries(self, entries: list[SessionEntry]) -> None:
        for entry in entries:
            self.append_entry(entry)

    def read_entries(self) -> list[SessionEntry]:
        if not self._session_file.exists():
            return []
        entries: list[SessionEntry] = []
        with self._session_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(self._dict_to_entry(json.loads(line)))
        return entries

    def session_file(self) -> Path:
        return self._session_file

    def rebuild_agent_messages(self) -> list[AgentMessage]:
        messages: list[AgentMessage] = []
        for entry in self.read_entries():
            if entry.type not in ("user_message", "assistant_message", "tool_result"):
                continue
            messages.append(self._deserialize_agent_message(entry.payload))
        return messages

    def append_agent_messages(self, messages: list[AgentMessage]) -> list[SessionEntry]:
        entries: list[SessionEntry] = []
        parent_id = self._last_entry_id()
        for message in messages:
            entry = SessionEntry(
                id=self._new_entry_id(),
                parent_id=parent_id,
                timestamp=float(getattr(message, "timestamp", time.time())),
                type=self._entry_type_for_message(message),
                payload=self._serialize_agent_message(message),
            )
            self.append_entry(entry)
            entries.append(entry)
            parent_id = entry.id
        self._touch_meta()
        return entries

    def append_system_snapshot(
        self,
        system_prompt: str,
        tool_names: list[str],
        model_info: dict[str, Any],
    ) -> SessionEntry:
        entry = SessionEntry(
            id=self._new_entry_id(),
            parent_id=self._last_entry_id(),
            timestamp=time.time(),
            type="system_snapshot",
            payload={
                "system_prompt": system_prompt,
                "tool_names": list(tool_names),
                "model_info": model_info,
            },
        )
        self.append_entry(entry)
        self._touch_meta()
        return entry

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str | None,
    ) -> SessionEntry:
        entry = SessionEntry(
            id=self._new_entry_id(),
            parent_id=self._last_entry_id(),
            timestamp=time.time(),
            type="compaction",
            payload={
                "summary": summary,
                "first_kept_entry_id": first_kept_entry_id,
            },
        )
        self.append_entry(entry)
        self._touch_meta()
        return entry

    def _last_entry_id(self) -> str | None:
        entries = self.read_entries()
        return entries[-1].id if entries else None

    def _touch_meta(self) -> None:
        entries = self.read_entries()
        if not entries or entries[0].type != "session_meta":
            return
        entries[0].payload["updated_at"] = time.time()
        with self._session_file.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(self._entry_to_dict(entry), ensure_ascii=False) + "\n")

    def _entry_type_for_message(self, message: AgentMessage) -> SessionEntryType:
        role = getattr(message, "role", None)
        if role == "user":
            return "user_message"
        if role == "assistant":
            return "assistant_message"
        if role == "toolResult":
            return "tool_result"
        raise ValueError(f"Unsupported message role for session persistence: {role}")

    def _serialize_agent_message(self, message: AgentMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": getattr(message, "role", None),
            "timestamp": float(getattr(message, "timestamp", time.time())),
            "content": [self._serialize_content(item) for item in getattr(message, "content", [])],
        }
        role = payload["role"]
        if role == "assistant":
            payload.update(
                {
                    "api": getattr(message, "api", ""),
                    "provider": getattr(message, "provider", ""),
                    "model": getattr(message, "model", ""),
                    "stop_reason": getattr(message, "stop_reason", "stop"),
                    "error_message": getattr(message, "error_message", None),
                    "usage": getattr(message, "usage", None),
                }
            )
        elif role == "toolResult":
            payload.update(
                {
                    "tool_call_id": getattr(message, "tool_call_id", ""),
                    "tool_name": getattr(message, "tool_name", ""),
                    "is_error": bool(getattr(message, "is_error", False)),
                    "details": getattr(message, "details", None),
                }
            )
        return payload

    def _deserialize_agent_message(self, payload: dict[str, Any]) -> AgentMessage:
        role = payload.get("role")
        content = [self._deserialize_content(item) for item in payload.get("content", [])]
        timestamp = float(payload.get("timestamp", time.time()))
        if role == "user":
            return UserMessage(content=content, timestamp=timestamp)  # type: ignore[arg-type]
        if role == "assistant":
            return AssistantMessage(
                content=content,  # type: ignore[arg-type]
                api=str(payload.get("api", "")),
                provider=str(payload.get("provider", "")),
                model=str(payload.get("model", "")),
                stop_reason=str(payload.get("stop_reason", "stop")),
                error_message=payload.get("error_message"),
                usage=payload.get("usage"),
                timestamp=timestamp,
            )
        if role == "toolResult":
            return ToolResultMessage(
                tool_call_id=str(payload.get("tool_call_id", "")),
                tool_name=str(payload.get("tool_name", "")),
                content=content,  # type: ignore[arg-type]
                is_error=bool(payload.get("is_error", False)),
                details=payload.get("details"),
                timestamp=timestamp,
            )
        raise ValueError(f"Unsupported persisted message role: {role}")

    def _serialize_content(self, item: Any) -> dict[str, Any]:
        item_type = getattr(item, "type", None)
        if item_type == "text":
            return {"type": "text", "text": getattr(item, "text", "")}
        if item_type == "image":
            return {
                "type": "image",
                "data": getattr(item, "data", ""),
                "mime_type": getattr(item, "mime_type", ""),
            }
        if item_type == "thinking":
            return {"type": "thinking", "thinking": getattr(item, "thinking", "")}
        if item_type == "toolCall":
            return {
                "type": "toolCall",
                "id": getattr(item, "id", ""),
                "name": getattr(item, "name", ""),
                "arguments": getattr(item, "arguments", {}),
                "parse_error": getattr(item, "parse_error", None),
            }
        raise ValueError(f"Unsupported content type: {item_type}")

    def _deserialize_content(self, payload: dict[str, Any]) -> Any:
        item_type = payload.get("type")
        if item_type == "text":
            return TextContent(text=str(payload.get("text", "")))
        if item_type == "image":
            return ImageContent(
                data=str(payload.get("data", "")),
                mime_type=str(payload.get("mime_type", "")),
            )
        if item_type == "thinking":
            return ThinkingContent(thinking=str(payload.get("thinking", "")))
        if item_type == "toolCall":
            return ToolCall(
                id=str(payload.get("id", "")),
                name=str(payload.get("name", "")),
                arguments=payload.get("arguments", {}) or {},
                parse_error=payload.get("parse_error"),
            )
        raise ValueError(f"Unsupported persisted content type: {item_type}")

    def _entry_to_dict(self, entry: SessionEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "parent_id": entry.parent_id,
            "timestamp": entry.timestamp,
            "type": entry.type,
            "payload": entry.payload,
        }

    def _dict_to_entry(self, data: dict[str, Any]) -> SessionEntry:
        return SessionEntry(
            id=str(data["id"]),
            parent_id=data.get("parent_id"),
            timestamp=float(data["timestamp"]),
            type=data["type"],
            payload=data.get("payload", {}),
        )

    def _new_entry_id(self) -> str:
        return uuid.uuid4().hex
