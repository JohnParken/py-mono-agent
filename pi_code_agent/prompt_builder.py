"""System prompt construction for coding-agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pi_agent_core.skills import Skill, format_skills_for_prompt

from .resources import ContextFile


@dataclass
class PromptBuildInput:
    cwd: str
    selected_tools: list[str]
    tool_snippets: dict[str, str] = field(default_factory=dict)
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    prompt_guidelines: list[str] = field(default_factory=list)


@dataclass
class PromptBuildResult:
    system_prompt: str
    sections: list[str] = field(default_factory=list)


class PromptBuilder:
    def build(self, data: PromptBuildInput) -> PromptBuildResult:
        sections: list[str] = []
        prompt = self._build_default_prompt(data)
        sections.append("base")

        context_section = self._render_context_files(data.context_files)
        if context_section:
            prompt += context_section
            sections.append("context_files")

        skills_section = self._render_skills(data.skills, data.selected_tools)
        if skills_section:
            prompt += skills_section
            sections.append("skills")

        prompt += f"\nCurrent date: {self._today_str()}"
        prompt += f"\nCurrent working directory: {data.cwd.replace(chr(92), '/')}"
        sections.append("metadata")
        return PromptBuildResult(system_prompt=prompt, sections=sections)

    def _build_default_prompt(self, data: PromptBuildInput) -> str:
        if data.custom_prompt:
            prompt = data.custom_prompt
            if data.append_system_prompt:
                prompt += f"\n\n{data.append_system_prompt}"
            return prompt

        tools_section = self._render_tools_section(
            data.selected_tools,
            data.tool_snippets,
        )
        guidelines = [
            "Be concise in your responses.",
            "Show file paths clearly when working with files.",
            *[item for item in data.prompt_guidelines if item.strip()],
        ]
        prompt = (
            "You are a careful coding assistant working inside a local repository.\n\n"
            "Use tools instead of guessing whenever the answer depends on the repository contents.\n"
            "Prefer this workflow:\n"
            "1. Locate relevant files with listing or search tools.\n"
            "2. Read exact code before changing it.\n"
            "3. Prefer precise edits over whole-file rewrites.\n"
            "4. Run validation commands when helpful.\n\n"
            "Rules:\n"
            "- Only operate inside the provided workspace.\n"
            "- Be explicit about what you changed and why.\n"
            "- Do not invent file contents you have not read.\n"
            "- Prefer small, precise edits over broad rewrites.\n\n"
            f"Available tools:\n{tools_section}\n\n"
            "Guidelines:\n"
            + "\n".join(f"- {line}" for line in guidelines)
        )
        if data.append_system_prompt:
            prompt += f"\n\n{data.append_system_prompt}"
        return prompt

    def _render_tools_section(
        self,
        selected_tools: list[str],
        tool_snippets: dict[str, str],
    ) -> str:
        if not selected_tools:
            return "(none)"
        lines: list[str] = []
        for tool_name in selected_tools:
            snippet = tool_snippets.get(tool_name)
            if snippet:
                lines.append(f"- {tool_name}: {snippet}")
            else:
                lines.append(f"- {tool_name}")
        return "\n".join(lines)

    def _render_context_files(self, context_files: list[ContextFile]) -> str:
        if not context_files:
            return ""
        chunks = ["\n\n# Project Context\n"]
        for item in context_files:
            chunks.append(f"\n## {item.path}\n\n{item.content}\n")
        return "".join(chunks)

    def _render_skills(self, skills: list[Skill], selected_tools: list[str]) -> str:
        if "read_file" not in selected_tools and "read" not in selected_tools:
            return ""
        return format_skills_for_prompt(skills)

    def _today_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")
