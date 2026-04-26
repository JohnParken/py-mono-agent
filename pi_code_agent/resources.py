"""Resource loading for coding-agent sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pi_agent_core.skills import (
    LoadSkillsOptions,
    ResourceDiagnostic,
    Skill,
    load_skills,
)

from .settings import CodeAgentSettings


@dataclass
class ContextFile:
    path: str
    abs_path: str
    content: str


@dataclass
class LoadedResources:
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


class ResourceLoader:
    def __init__(
        self,
        workspace: Path,
        settings: CodeAgentSettings,
        agent_dir: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.settings = settings
        self.agent_dir = agent_dir

    def reload(self) -> LoadedResources:
        context_files = self.load_context_files()
        skills, diagnostics = self.load_skills()
        return LoadedResources(
            context_files=context_files,
            skills=skills,
            diagnostics=diagnostics,
        )

    def load_context_files(self) -> list[ContextFile]:
        files: list[ContextFile] = []
        for path in self._expand_context_patterns(
            self.settings.resources.context_file_patterns
        ):
            try:
                content = self._read_text_file(path)
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            files.append(
                ContextFile(
                    path=path.relative_to(self.workspace).as_posix(),
                    abs_path=str(path),
                    content=content,
                )
            )
        return files

    def load_skills(self) -> tuple[list[Skill], list[ResourceDiagnostic]]:
        if not self.settings.resources.enable_skills:
            return [], []
        result = load_skills(
            LoadSkillsOptions(
                cwd=str(self.workspace),
                agent_dir=str(self.agent_dir) if self.agent_dir else "",
                skill_paths=list(self.settings.resources.skill_paths),
                include_defaults=True,
            )
        )
        return result.skills, result.diagnostics

    def _expand_context_patterns(self, patterns: list[str]) -> list[Path]:
        found: list[Path] = []
        for pattern in patterns:
            matches = self.workspace.glob(pattern)
            for path in matches:
                if path.is_file():
                    found.append(path.resolve())
        return self._dedupe_paths(found)

    def _read_text_file(self, path: Path, max_bytes: int = 64_000) -> str:
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"Context file too large: {path}")
        if b"\x00" in data:
            raise ValueError(f"Binary context file is not supported: {path}")
        return data.decode("utf-8")

    def _dedupe_paths(self, paths: list[Path]) -> list[Path]:
        seen: set[Path] = set()
        ordered: list[Path] = []
        for path in sorted(paths):
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
        return ordered
