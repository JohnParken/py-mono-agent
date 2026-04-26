"""Tool selection and prompt metadata for coding-agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pi_agent_core.types import AgentTool

from .tools import create_coding_tools


@dataclass
class ToolRegistryOptions:
    allow_tools: list[str] | None = None
    no_tools: bool = False
    include_builtin: bool = True
    custom_tools: list[AgentTool] = field(default_factory=list)


@dataclass
class ToolDescriptor:
    name: str
    label: str
    description: str
    prompt_snippet: str


class ToolRegistry:
    def __init__(
        self,
        workspace: Path,
        options: ToolRegistryOptions | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.options = options or ToolRegistryOptions()
        self._cached_tools: list[AgentTool] | None = None

    def build_tools(self) -> list[AgentTool]:
        if self._cached_tools is None:
            tools: list[AgentTool] = []
            if self.options.include_builtin and not self.options.no_tools:
                tools.extend(self._builtin_tools())
            tools.extend(self.options.custom_tools)
            self._cached_tools = self._filter_tools(tools)
        return list(self._cached_tools)

    def get_active_tool_names(self) -> list[str]:
        return [tool.name for tool in self.build_tools()]

    def get_tool_snippets(self) -> dict[str, str]:
        descriptors = self._descriptor_map()
        return {
            name: descriptors[name].prompt_snippet
            for name in self.get_active_tool_names()
            if name in descriptors
        }

    def _builtin_tools(self) -> list[AgentTool]:
        return create_coding_tools(self.workspace)

    def _filter_tools(self, tools: list[AgentTool]) -> list[AgentTool]:
        if self.options.allow_tools is None:
            return tools
        allowed = set(self.options.allow_tools)
        return [tool for tool in tools if tool.name in allowed]

    def _descriptor_map(self) -> dict[str, ToolDescriptor]:
        return {
            "list_files": ToolDescriptor(
                name="list_files",
                label="List Files",
                description="List files and directories in the workspace.",
                prompt_snippet="List files and directories in the workspace",
            ),
            "read_file": ToolDescriptor(
                name="read_file",
                label="Read File",
                description="Read exact file contents with line numbers.",
                prompt_snippet="Read exact file contents with line numbers",
            ),
            "search_code": ToolDescriptor(
                name="search_code",
                label="Search Code",
                description="Search for matching lines in workspace files.",
                prompt_snippet="Search code for matching text or regex lines",
            ),
            "write_file": ToolDescriptor(
                name="write_file",
                label="Write File",
                description="Write or append text to files in the workspace.",
                prompt_snippet="Write a full file or append text to a file",
            ),
            "edit_file": ToolDescriptor(
                name="edit_file",
                label="Edit File",
                description="Replace exact text snippets in a workspace file.",
                prompt_snippet="Make precise file edits with exact text replacement",
            ),
            "run_command": ToolDescriptor(
                name="run_command",
                label="Run Command",
                description="Run shell commands inside the workspace.",
                prompt_snippet="Run shell commands for validation or inspection",
            ),
        }
