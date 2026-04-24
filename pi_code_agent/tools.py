"""Minimal built-in coding tools inspired by pi-mono's coding agent."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pydantic import BaseModel, Field

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
    old_text: str = Field(description="Exact text to replace.")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False,
        description="Replace all matches. If false, exactly one match is required.",
    )


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
    logger.info("[CODE-TOOL] creating coding tools for workspace=%s", workspace.root)

    async def read_file(
        tool_call_id: str,
        args: ReadArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.info(
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
        logger.info(
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
        logger.info(
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
        logger.info(
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
        logger.info(
            "[CODE-TOOL] edit_file start tool_call_id=%s path=%s replace_all=%s old_len=%s new_len=%s",
            tool_call_id,
            args.path,
            args.replace_all,
            len(args.old_text),
            len(args.new_text),
        )
        path = workspace.resolve_path(args.path, allow_root=False)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {workspace.display_path(path)}")
        text = _read_text_file(path)
        matches = text.count(args.old_text)
        if matches == 0:
            raise ValueError("old_text was not found in the target file.")
        if not args.replace_all and matches != 1:
            raise ValueError(
                f"old_text matched {matches} times. Use replace_all=true or provide a more specific snippet."
            )
        updated = text.replace(args.old_text, args.new_text) if args.replace_all else text.replace(args.old_text, args.new_text, 1)
        path.write_text(updated, encoding="utf-8")
        replaced = matches if args.replace_all else 1
        logger.info(
            "[CODE-TOOL] edit_file success path=%s replaced=%s",
            workspace.display_path(path),
            replaced,
        )
        return _result(
            f"Updated {workspace.display_path(path)} by replacing {replaced} occurrence(s)."
        )

    async def list_files(
        tool_call_id: str,
        args: ListFilesArgs,
        cancel_event,
        update_callback,
    ) -> AgentToolResult:
        logger.info(
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
            logger.info("[CODE-TOOL] list_files target is file path=%s", workspace.display_path(path))
            return _result(workspace.display_path(path))

        entries = _walk_workspace(path, recursive=args.recursive, max_entries=args.max_entries)
        rendered = "\n".join(workspace.display_path(entry) for entry in entries)
        suffix = ""
        if len(entries) >= args.max_entries:
            suffix = f"\n[truncated to first {args.max_entries} entries]"
        logger.info(
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
        logger.info(
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
            logger.info("[CODE-TOOL] search_code success path=%s hits=0", workspace.display_path(path))
            return _result("No matches found.")
        suffix = ""
        if len(hits) >= args.max_results:
            suffix = f"\n[truncated to first {args.max_results} matches]"
        logger.info(
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
        logger.info(
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
        completed = subprocess.run(
            command_args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        combined = completed.stdout
        if completed.stderr:
            if combined:
                combined += "\n"
            combined += completed.stderr
        combined = combined.strip() or "[no output]"
        combined = _truncate(combined, MAX_OUTPUT_CHARS)
        summary = (
            f"Command: {args.command}\n"
            f"Shell: {shell_name}\n"
            f"Exit code: {completed.returncode}\n"
            f"Working directory: {workspace.display_path(cwd)}\n"
            f"Output:\n{combined}"
        )
        logger.info(
            "[CODE-TOOL] run_command success cwd=%s shell=%s exit_code=%s output_len=%s",
            workspace.display_path(cwd),
            shell_name,
            completed.returncode,
            len(combined),
        )
        return _result(summary)

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
            description="Replace an exact text snippet inside a workspace file.",
            parameters=EditArgs,
            execute=edit_file,
        ),
        AgentTool(
            name="run_command",
            label="Run Command",
            description="Run a shell command inside the workspace using the native shell for the current OS.",
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


def _build_shell_command(command: str) -> tuple[list[str], str]:
    if sys.platform == "win32":
        powershell = shutil.which("pwsh") or shutil.which("powershell.exe")
        if powershell:
            return [powershell, "-NoProfile", "-Command", command], Path(powershell).name

        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/d", "/s", "/c", command], Path(comspec).name

    shell = os.environ.get("SHELL", "/bin/zsh")
    return [shell, "-lc", command], Path(shell).name
