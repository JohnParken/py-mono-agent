"""Minimal built-in coding tools inspired by pi-mono's coding agent."""

from __future__ import annotations

import asyncio
import difflib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, List

from pydantic import BaseModel, Field, model_validator

from pi_ai.types import TextContent
from pi_agent_core.types import AgentTool, AgentToolResult
from pi_logger import get_logger


DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}
MAX_READ_LINES = 400
MAX_OUTPUT_CHARS = 24_000
MAX_FILE_SIZE = 512 * 1024
MAX_COMMAND_OUTPUT_BYTES = 16 * 1024
MAX_COMMAND_OUTPUT_LINES = 200
MAX_COMMAND_BUFFER_BYTES = MAX_COMMAND_OUTPUT_BYTES * 2

_FILE_MUTATION_LOCKS: dict[str, asyncio.Lock] = {}

logger = get_logger("pi_code_agent.tools")


@dataclass(frozen=True)
class Workspace:
    root: Path

    def resolve_path(self, raw_path: str, *, allow_root: bool = True) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if not allow_root and candidate == self.root:
            raise ValueError("Path must point to a file or subdirectory inside the workspace.")
        if not _is_relative_to(candidate, self.root):
            raise ValueError(
                f"Path '{raw_path}' resolves outside the workspace root '{self.root}'."
            )
        return candidate

    def display_path(self, path: Path) -> str:
        if path == self.root:
            return "."
        return path.relative_to(self.root).as_posix()


class ReadArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    start_line: int = Field(default=1, ge=1, description="First line number to read.")
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Last line number to read (inclusive). Defaults to a bounded window.",
    )


class WriteArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    content: str | None = Field(
        default=None,
        description="Full file contents to write. For multiline files, prefer content_lines.",
    )
    content_lines: List[str] | None = Field(
        default=None,
        description=(
            "Alternative to content: file lines joined with newline characters. "
            "Use this for multiline content to avoid JSON string escaping issues."
        ),
    )
    append: bool = Field(default=False, description="Append instead of overwriting.")
    create_directories: bool = Field(
        default=True,
        description="Create parent directories automatically when needed.",
    )


class EditArgs(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")
    edits: List["EditBlock"] | None = Field(
        default=None,
        description=(
            "One or more exact, non-overlapping replacements. Each old_text must match "
            "exactly once in the original file."
        ),
    )
    old_text: str | None = Field(
        default=None,
        description="Legacy single-edit form: exact text to replace.",
    )
    new_text: str | None = Field(
        default=None,
        description="Legacy single-edit form: replacement text.",
    )
    replace_all: bool = Field(
        default=False,
        description="Legacy single-edit option. If true, replace all exact matches.",
    )

    @model_validator(mode="after")
    def _validate_edit_args(self) -> "EditArgs":
        has_edits = bool(self.edits)
        has_legacy = self.old_text is not None or self.new_text is not None
        if has_edits and has_legacy:
            raise ValueError("Provide either edits or old_text/new_text, not both.")
        if not has_edits and not has_legacy:
            raise ValueError("edit_file requires edits or old_text/new_text.")
        if has_legacy and (self.old_text is None or self.new_text is None):
            raise ValueError("Both old_text and new_text are required for legacy edit mode.")
        return self


class EditBlock(BaseModel):
    old_text: str = Field(description="Exact text for one targeted replacement.")
    new_text: str = Field(description="Replacement text for this targeted edit.")


class ListFilesArgs(BaseModel):
    path: str = Field(default=".", description="Directory path relative to the workspace root.")
    recursive: bool = Field(default=True, description="List files recursively.")
    max_entries: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of paths to return.",
    )


class SearchArgs(BaseModel):
    pattern: str = Field(description="Text or regular expression to search for.")
    path: str = Field(default=".", description="Directory or file path to search within.")
    literal: bool = Field(
        default=True,
        description="Treat the pattern as literal text instead of a regular expression.",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of matching lines to return.",
    )


class ShellArgs(BaseModel):
    command: str = Field(description="Shell command to execute inside the workspace.")
    cwd: str = Field(
        default=".",
        description="Working directory relative to the workspace root.",
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Timeout in seconds.",
    )


# Backward-compatible alias for earlier internal naming.
BashArgs = ShellArgs


def create_coding_tools(workspace_root: str | Path) -> List[AgentTool]:
    workspace = Workspace(Path(workspace_root).resolve())
    logger.debug("[CODE-TOOL] creating coding tools for workspace=%s", workspace.root)

    async def read_file(
        tool_call_id: str,
        args: ReadArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.debug(
            "[CODE-TOOL] read_file start tool_call_id=%s path=%s start_line=%s end_line=%s",
            tool_call_id,
            args.path,
            args.start_line,
            args.end_line,
        )
        path = workspace.resolve_path(args.path, allow_root=False)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {workspace.display_path(path)}")
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {workspace.display_path(path)}")
        text = _read_text_file(path)
        lines = text.splitlines()
        start = args.start_line
        end = args.end_line or min(len(lines), start + MAX_READ_LINES - 1)
        if end < start:
            raise ValueError("end_line must be greater than or equal to start_line.")
        selected = lines[start - 1:end]
        body = "\n".join(f"{index}: {line}" for index, line in enumerate(selected, start))
        if not body:
            body = "[empty selection]"
        summary = (
            f"FILE: {workspace.display_path(path)}\n"
            f"LINES: {start}-{start + max(len(selected) - 1, 0)} of {len(lines)}\n"
            f"{body}"
        )
        logger.debug(
            "[CODE-TOOL] read_file success path=%s selected_lines=%s",
            workspace.display_path(path),
            len(selected),
        )
        return _result(summary)

    async def write_file(
        tool_call_id: str,
        args: WriteArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.debug(
            "[CODE-TOOL] write_file start tool_call_id=%s path=%s append=%s content_len=%s",
            tool_call_id,
            args.path,
            args.append,
            len(_get_write_content(args)),
        )
        path = workspace.resolve_path(args.path, allow_root=False)
        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {workspace.display_path(path)}")
        if args.create_directories:
            path.parent.mkdir(parents=True, exist_ok=True)
        elif not path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {workspace.display_path(path.parent)}"
            )
        mode = "a" if args.append else "w"
        content = _get_write_content(args)
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        action = "Appended to" if args.append else "Wrote"
        logger.debug(
            "[CODE-TOOL] write_file success path=%s action=%s content_len=%s",
            workspace.display_path(path),
            action,
            len(content),
        )
        return _result(
            f"{action} {workspace.display_path(path)} ({len(content)} characters)."
        )

    async def edit_file(
        tool_call_id: str,
        args: EditArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.debug(
            "[CODE-TOOL] edit_file start tool_call_id=%s path=%s replace_all=%s old_len=%s new_len=%s",
            tool_call_id,
            args.path,
            args.replace_all,
            len(args.old_text or ""),
            len(args.new_text or ""),
        )
        path = workspace.resolve_path(args.path, allow_root=False)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {workspace.display_path(path)}")
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {workspace.display_path(path)}")

        edit_blocks = _resolve_edit_blocks(args)
        async with _file_mutation_lock(path):
            text = _read_text_file(path)
            bom, content = _strip_bom(text)
            original_line_ending = _detect_line_ending(content)
            normalized_content = _normalize_to_lf(content)
            base_content, updated_content = _apply_edits_to_normalized_content(
                normalized_content,
                edit_blocks,
                args.path,
                replace_all=args.replace_all and len(edit_blocks) == 1,
            )
            diff_text = _generate_diff(base_content, updated_content, workspace.display_path(path))
            first_changed_line = _first_changed_line(base_content, updated_content)
            final_content = bom + _restore_line_endings(updated_content, original_line_ending)
            path.write_text(final_content, encoding="utf-8")

        replaced = len(edit_blocks)
        logger.debug(
            "[CODE-TOOL] edit_file success path=%s replaced=%s",
            workspace.display_path(path),
            replaced,
        )
        return AgentToolResult(
            content=[
                TextContent(
                    text=(
                        f"Successfully replaced {replaced} block(s) in "
                        f"{workspace.display_path(path)}."
                    )
                )
            ],
            details={
                "diff": diff_text,
                "first_changed_line": first_changed_line,
                "replaced_blocks": replaced,
            },
        )

    async def list_files(
        tool_call_id: str,
        args: ListFilesArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.debug(
            "[CODE-TOOL] list_files start tool_call_id=%s path=%s recursive=%s max_entries=%s",
            tool_call_id,
            args.path,
            args.recursive,
            args.max_entries,
        )
        path = workspace.resolve_path(args.path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {workspace.display_path(path)}")
        if path.is_file():
            logger.debug("[CODE-TOOL] list_files target is file path=%s", workspace.display_path(path))
            return _result(workspace.display_path(path))

        entries = _walk_workspace(path, recursive=args.recursive, max_entries=args.max_entries)
        rendered = "\n".join(workspace.display_path(entry) for entry in entries)
        suffix = ""
        if len(entries) >= args.max_entries:
            suffix = f"\n[truncated to first {args.max_entries} entries]"
        logger.debug(
            "[CODE-TOOL] list_files success path=%s entries=%s truncated=%s",
            workspace.display_path(path),
            len(entries),
            len(entries) >= args.max_entries,
        )
        return _result(rendered + suffix if rendered else "[empty directory]")

    async def search_code(
        tool_call_id: str,
        args: SearchArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.debug(
            "[CODE-TOOL] search_code start tool_call_id=%s path=%s literal=%s max_results=%s pattern=%r",
            tool_call_id,
            args.path,
            args.literal,
            args.max_results,
            args.pattern[:120],
        )
        path = workspace.resolve_path(args.path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {workspace.display_path(path)}")

        regex = re.compile(args.pattern if not args.literal else re.escape(args.pattern))
        hits: list[str] = []
        files = [path] if path.is_file() else _walk_workspace(path, recursive=True, max_entries=10_000)
        for file_path in files:
            if len(hits) >= args.max_results:
                break
            if file_path.is_dir():
                continue
            try:
                text = _read_text_file(file_path)
            except (UnicodeDecodeError, ValueError):
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{workspace.display_path(file_path)}:{line_no}: {line}")
                    if len(hits) >= args.max_results:
                        break
        if not hits:
            logger.debug("[CODE-TOOL] search_code success path=%s hits=0", workspace.display_path(path))
            return _result("No matches found.")
        suffix = ""
        if len(hits) >= args.max_results:
            suffix = f"\n[truncated to first {args.max_results} matches]"
        logger.debug(
            "[CODE-TOOL] search_code success path=%s hits=%s truncated=%s",
            workspace.display_path(path),
            len(hits),
            len(hits) >= args.max_results,
        )
        return _result("\n".join(hits) + suffix)

    async def run_command(
        tool_call_id: str,
        args: ShellArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.debug(
            "[CODE-TOOL] run_command start tool_call_id=%s cwd=%s timeout=%s command=%r",
            tool_call_id,
            args.cwd,
            args.timeout,
            args.command[:200],
        )
        cwd = workspace.resolve_path(args.cwd)
        if not cwd.exists():
            raise FileNotFoundError(f"Working directory not found: {workspace.display_path(cwd)}")
        if not cwd.is_dir():
            raise NotADirectoryError(
                f"Working directory must be a directory: {workspace.display_path(cwd)}"
            )
        _guard_command(args.command)
        command_args, shell_name = _build_shell_command(args.command)

        if update_callback:
            update_callback(AgentToolResult(content=[], details=None))

        process = await asyncio.create_subprocess_exec(
            *command_args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        total_bytes = 0
        chunks: list[bytes] = []
        chunks_bytes = 0
        temp_output_path: str | None = None
        temp_handle = None

        def ensure_temp_output() -> None:
            nonlocal temp_output_path, temp_handle
            if temp_handle is not None:
                return
            temp_file = NamedTemporaryFile(
                mode="wb",
                prefix="pi-code-agent-",
                suffix=".log",
                delete=False,
            )
            temp_output_path = temp_file.name
            temp_handle = temp_file
            for chunk in chunks:
                temp_handle.write(chunk)
            temp_handle.flush()

        async def read_stream(stream) -> None:
            nonlocal total_bytes, chunks_bytes
            while True:
                data = await stream.read(4096)
                if not data:
                    break
                total_bytes += len(data)
                chunks.append(data)
                chunks_bytes += len(data)
                if total_bytes > MAX_COMMAND_OUTPUT_BYTES:
                    ensure_temp_output()
                if temp_handle is not None:
                    temp_handle.write(data)
                    temp_handle.flush()
                while chunks_bytes > MAX_COMMAND_BUFFER_BYTES and len(chunks) > 1:
                    removed = chunks.pop(0)
                    chunks_bytes -= len(removed)
                if update_callback:
                    partial_text = b"".join(chunks).decode("utf-8", errors="ignore")
                    truncation = _truncate_tail(
                        partial_text,
                        max_lines=MAX_COMMAND_OUTPUT_LINES,
                        max_bytes=MAX_COMMAND_OUTPUT_BYTES,
                    )
                    if truncation["truncated"]:
                        ensure_temp_output()
                    update_callback(
                        AgentToolResult(
                            content=[TextContent(text=truncation["content"])],
                            details={
                                "truncation": truncation if truncation["truncated"] else None,
                                "full_output_path": temp_output_path,
                            },
                        )
                    )

        stdout_task = asyncio.create_task(read_stream(process.stdout))
        stderr_task = asyncio.create_task(read_stream(process.stderr))
        wait_task = asyncio.create_task(process.wait())
        cancel_task = (
            asyncio.create_task(cancel_event.wait())
            if cancel_event is not None
            else None
        )

        cancelled = False
        timed_out = False
        try:
            pending = {wait_task}
            if cancel_task is not None:
                pending.add(cancel_task)
            done, _ = await asyncio.wait(
                pending,
                timeout=args.timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task not in done:
                if cancel_task is not None and cancel_task in done:
                    cancelled = True
                else:
                    timed_out = True
                process.kill()
                await process.wait()
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            if cancel_task is not None:
                cancel_task.cancel()
                await asyncio.gather(cancel_task, return_exceptions=True)
            if temp_handle is not None:
                temp_handle.close()

        exit_code = process.returncode
        combined = b"".join(chunks).decode("utf-8", errors="ignore")
        truncation = _truncate_tail(
            combined,
            max_lines=MAX_COMMAND_OUTPUT_LINES,
            max_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        if truncation["truncated"] and temp_output_path is None:
            ensure_temp_output()
            if temp_handle is not None:
                temp_handle.close()

        output_text = truncation["content"] or "[no output]"
        details = None
        if truncation["truncated"] or temp_output_path is not None:
            details = {
                "truncation": truncation if truncation["truncated"] else None,
                "full_output_path": temp_output_path,
            }
            if truncation["truncated"]:
                output_text += _format_truncation_notice(truncation, temp_output_path)

        summary = (
            f"Command: {args.command}\n"
            f"Shell: {shell_name}\n"
            f"Exit code: {exit_code}\n"
            f"Working directory: {workspace.display_path(cwd)}\n"
            f"Output:\n{_truncate(output_text.strip() or '[no output]', MAX_OUTPUT_CHARS)}"
        )
        logger.debug(
            "[CODE-TOOL] run_command success cwd=%s shell=%s exit_code=%s output_len=%s",
            workspace.display_path(cwd),
            shell_name,
            exit_code,
            len(output_text),
        )
        if cancelled:
            raise RuntimeError(summary + "\n\nCommand aborted.")
        if timed_out:
            raise RuntimeError(summary + f"\n\nCommand timed out after {args.timeout} seconds.")
        if exit_code not in (0, None):
            raise RuntimeError(summary + f"\n\nCommand exited with code {exit_code}.")
        return AgentToolResult(content=[TextContent(text=summary)], details=details)

    return [
        AgentTool(
            name="list_files",
            label="List Files",
            description="List files and directories inside the workspace.",
            parameters=ListFilesArgs,
            execute=list_files,
        ),
        AgentTool(
            name="read_file",
            label="Read File",
            description="Read a text file from the workspace with line numbers.",
            parameters=ReadArgs,
            execute=read_file,
        ),
        AgentTool(
            name="search_code",
            label="Search Code",
            description="Search for matching lines in workspace files.",
            parameters=SearchArgs,
            execute=search_code,
        ),
        AgentTool(
            name="write_file",
            label="Write File",
            description="Write or append text to a file inside the workspace.",
            parameters=WriteArgs,
            execute=write_file,
        ),
        AgentTool(
            name="edit_file",
            label="Edit File",
            description=(
                "Edit a single file using exact text replacement. Supports multiple "
                "non-overlapping edits matched against the original file."
            ),
            parameters=EditArgs,
            execute=edit_file,
        ),
        AgentTool(
            name="run_command",
            label="Run Command",
            description=(
                "Run a shell command inside the workspace using the native shell for "
                "the current OS. Streams partial output and truncates large output."
            ),
            parameters=ShellArgs,
            execute=run_command,
        ),
    ]


def _walk_workspace(path: Path, *, recursive: bool, max_entries: int) -> list[Path]:
    entries: list[Path] = []
    if recursive:
        iterator: Iterable[Path] = sorted(path.rglob("*"))
    else:
        iterator = sorted(path.iterdir())

    for entry in iterator:
        if _should_skip(entry):
            continue
        entries.append(entry)
        if len(entries) >= max_entries:
            break
    return entries


def _get_write_content(args: WriteArgs) -> str:
    has_content = args.content is not None
    has_content_lines = args.content_lines is not None
    if has_content and has_content_lines:
        raise ValueError("Provide either content or content_lines, not both.")
    if has_content_lines:
        return "\n".join(args.content_lines or [])
    if has_content:
        return args.content or ""
    raise ValueError("write_file requires content or content_lines.")


def _should_skip(path: Path) -> bool:
    return any(part in DEFAULT_IGNORED_DIRS for part in path.parts)


def _read_text_file(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File is too large to read safely: {path.name}")
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"Binary files are not supported: {path.name}")
    return data.decode("utf-8")


def _result(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=_truncate(text, MAX_OUTPUT_CHARS))])


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    remaining = len(text) - limit
    return text[:limit] + f"\n...[truncated {remaining} characters]"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _guard_command(command: str) -> None:
    lowered = command.lower()
    banned = (
        "rm -rf /",
        "sudo ",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "mkfs",
        ":(){:|:&};:",
    )
    if any(token in lowered for token in banned):
        raise ValueError("Refusing to run a clearly destructive shell command.")


def _resolve_edit_blocks(args: EditArgs) -> list[EditBlock]:
    if args.edits:
        return list(args.edits)
    return [EditBlock(old_text=args.old_text or "", new_text=args.new_text or "")]


def _file_mutation_lock(path: Path) -> asyncio.Lock:
    key = str(path)
    lock = _FILE_MUTATION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _FILE_MUTATION_LOCKS[key] = lock
    return lock


def _strip_bom(text: str) -> tuple[str, str]:
    if text.startswith("\ufeff"):
        return "\ufeff", text[1:]
    return "", text


def _detect_line_ending(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_line_endings(text: str, line_ending: str) -> str:
    if line_ending == "\n":
        return text
    return text.replace("\n", line_ending)


def _apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[EditBlock],
    path: str,
    *,
    replace_all: bool,
) -> tuple[str, str]:
    replacements: list[tuple[int, int, str, str]] = []
    for edit in edits:
        old_text = _normalize_to_lf(edit.old_text)
        new_text = _normalize_to_lf(edit.new_text)
        if old_text == "":
            raise ValueError("edit_file old_text must not be empty.")
        match_spans = _find_match_spans(normalized_content, old_text)
        if not match_spans:
            raise ValueError(f"Could not find old_text in {path}.")
        if not replace_all and len(match_spans) != 1:
            raise ValueError(
                f"old_text matched {len(match_spans)} times in {path}. "
                "Provide a more specific snippet or use replace_all for legacy single-edit mode."
            )
        spans_to_use = match_spans if replace_all else [match_spans[0]]
        for start, end in spans_to_use:
            replacements.append((start, end, old_text, new_text))

    _ensure_non_overlapping_replacements(replacements, path)

    updated = normalized_content
    for start, end, _old_text, new_text in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + new_text + updated[end:]
    return normalized_content, updated


def _find_match_spans(haystack: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            break
        spans.append((index, index + len(needle)))
        start = index + len(needle)
    return spans


def _ensure_non_overlapping_replacements(
    replacements: list[tuple[int, int, str, str]],
    path: str,
) -> None:
    ordered = sorted(replacements, key=lambda item: item[0])
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"edit_file received overlapping edits for {path}. Merge nearby changes into one edit."
            )


def _generate_diff(old_text: str, new_text: str, display_path: str) -> str:
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{display_path}",
        tofile=f"b/{display_path}",
        lineterm="",
    )
    return "".join(diff) or "[no diff]"


def _first_changed_line(old_text: str, new_text: str) -> int | None:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    for index, (old_line, new_line) in enumerate(zip(old_lines, new_lines), start=1):
        if old_line != new_line:
            return index
    if len(old_lines) != len(new_lines):
        return min(len(old_lines), len(new_lines)) + 1
    return None


def _truncate_tail(text: str, *, max_lines: int, max_bytes: int) -> dict[str, Any]:
    original_text = text
    lines = original_text.splitlines()
    total_lines = len(lines)
    truncated = False
    truncated_by = None
    last_line_partial = False

    if total_lines > max_lines:
        text = "\n".join(lines[-max_lines:])
        truncated = True
        truncated_by = "lines"

    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        tail = encoded[-max_bytes:]
        decoded = tail.decode("utf-8", errors="ignore")
        if "\n" in decoded:
            decoded = decoded.split("\n", 1)[1]
        else:
            last_line_partial = True
        text = decoded
        truncated = True
        truncated_by = "bytes"

    output_lines = len(text.splitlines()) if text else 0
    output_bytes = len(text.encode("utf-8"))
    return {
        "content": _truncate(text.strip() or "[no output]", MAX_OUTPUT_CHARS),
        "truncated": truncated,
        "truncated_by": truncated_by,
        "total_lines": total_lines,
        "output_lines": output_lines,
        "output_bytes": output_bytes,
        "max_bytes": max_bytes,
        "last_line_partial": last_line_partial,
    }


def _format_truncation_notice(truncation: dict[str, Any], full_output_path: str | None) -> str:
    if not truncation["truncated"]:
        return ""
    path_suffix = f" Full output: {full_output_path}." if full_output_path else ""
    end_line = truncation["total_lines"]
    start_line = max(1, end_line - truncation["output_lines"] + 1)
    if truncation["last_line_partial"]:
        return (
            f"\n\n[Showing last {truncation['output_bytes']} bytes of line {end_line}."
            f"{path_suffix}]"
        )
    if truncation["truncated_by"] == "lines":
        return (
            f"\n\n[Showing lines {start_line}-{end_line} of {truncation['total_lines']}."
            f"{path_suffix}]"
        )
    return (
        f"\n\n[Showing lines {start_line}-{end_line} of {truncation['total_lines']} "
        f"({MAX_COMMAND_OUTPUT_BYTES} byte limit).{path_suffix}]"
    )


def _build_shell_command(command: str) -> tuple[list[str], str]:
    if sys.platform == "win32":
        powershell = shutil.which("pwsh") or shutil.which("powershell.exe")
        if powershell:
            return [powershell, "-NoProfile", "-Command", command], Path(powershell).name

        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/d", "/s", "/c", command], Path(comspec).name

    shell = os.environ.get("SHELL", "/bin/zsh")
    return [shell, "-lc", command], Path(shell).name
