"""AgentSession runtime for the coding-agent product layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pi_ai import Model, get_llm_config, get_model
from pi_ai.types import ImageContent
from pi_agent_core import Agent, AgentOptions
from pi_agent_core.types import AgentEvent, AgentMessage, StreamFn

from .prompt_builder import PromptBuildInput, PromptBuilder
from .resources import LoadedResources, ResourceLoader
from .session_store import SessionStore
from .settings import CodeAgentSettings, load_settings, resolve_session_dir
from .tool_registry import ToolRegistry, ToolRegistryOptions


@dataclass
class SessionCreateOptions:
    workspace: Path
    config_path: str | None = None
    model_config: str | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    session_file: Path | None = None
    continue_last_session: bool = False
    allow_tools: list[str] | None = None
    no_tools: bool = False
    model_instance: Model | None = None
    stream_fn: StreamFn | None = None


@dataclass
class SessionStats:
    session_id: str
    session_file: str
    user_messages: int
    assistant_messages: int
    tool_results: int
    total_messages: int
    last_error: str | None = None


@dataclass
class SessionRuntimeState:
    session_id: str
    workspace: Path
    loaded_resources: LoadedResources
    current_system_prompt: str
    current_tool_names: list[str]
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    queued_steering: list[str] = field(default_factory=list)
    queued_follow_up: list[str] = field(default_factory=list)
    last_compaction_summary: str | None = None


class AgentSession:
    def __init__(
        self,
        agent: Agent,
        store: SessionStore,
        settings: CodeAgentSettings,
        resources: ResourceLoader,
        prompt_builder: PromptBuilder,
        tool_registry: ToolRegistry,
        runtime: SessionRuntimeState,
    ) -> None:
        self.agent = agent
        self.store = store
        self.settings = settings
        self.resources = resources
        self.prompt_builder = prompt_builder
        self.tool_registry = tool_registry
        self.runtime = runtime

    @classmethod
    def create(cls, options: SessionCreateOptions) -> "AgentSession":
        workspace = options.workspace.resolve()
        settings = load_settings(workspace)
        tool_registry = ToolRegistry(
            workspace,
            ToolRegistryOptions(
                allow_tools=options.allow_tools,
                no_tools=options.no_tools,
            ),
        )
        resources = ResourceLoader(workspace, settings)
        loaded_resources = resources.reload()
        prompt_builder = PromptBuilder()
        model = cls._resolve_model(options)
        store = cls._open_store(workspace, settings, options)

        prompt = prompt_builder.build(
            PromptBuildInput(
                cwd=str(workspace),
                selected_tools=tool_registry.get_active_tool_names(),
                tool_snippets=tool_registry.get_tool_snippets(),
                context_files=loaded_resources.context_files,
                skills=loaded_resources.skills,
                custom_prompt=options.custom_system_prompt,
                append_system_prompt=options.append_system_prompt,
            )
        )

        restored_messages = store.rebuild_agent_messages()
        agent = Agent(
            AgentOptions(
                initial_state={
                    "system_prompt": prompt.system_prompt,
                    "model": model,
                    "tools": tool_registry.build_tools(),
                    "messages": restored_messages,
                },
                steering_mode=settings.product.steering_mode,
                follow_up_mode=settings.product.follow_up_mode,
                stream_fn=options.stream_fn,
            )
        )

        runtime = SessionRuntimeState(
            session_id=store.load_snapshot().meta.session_id,
            workspace=workspace,
            loaded_resources=loaded_resources,
            current_system_prompt=prompt.system_prompt,
            current_tool_names=tool_registry.get_active_tool_names(),
            custom_system_prompt=options.custom_system_prompt,
            append_system_prompt=options.append_system_prompt,
        )

        session = cls(
            agent=agent,
            store=store,
            settings=settings,
            resources=resources,
            prompt_builder=prompt_builder,
            tool_registry=tool_registry,
            runtime=runtime,
        )

        if not restored_messages:
            session.store.append_system_snapshot(
                system_prompt=prompt.system_prompt,
                tool_names=runtime.current_tool_names,
                model_info={
                    "provider": model.provider,
                    "id": model.id,
                    "base_url": model.base_url,
                },
            )

        return session

    async def prompt(
        self,
        text: str,
        images: list[ImageContent] | None = None,
    ) -> None:
        start_index = len(self.agent.state.messages)
        await self.agent.prompt(text, images=images)
        self._persist_new_messages(start_index)

    async def continue_(self) -> None:
        start_index = len(self.agent.state.messages)
        await self.agent.continue_()
        self._persist_new_messages(start_index)

    async def abort(self) -> None:
        self.agent.abort()
        await self.agent.wait_for_idle()

    async def reload_resources(self) -> LoadedResources:
        loaded = self.resources.reload()
        self.runtime.loaded_resources = loaded
        await self.rebuild_system_prompt()
        return loaded

    async def rebuild_system_prompt(self) -> str:
        result = self.prompt_builder.build(
            PromptBuildInput(
                cwd=str(self.runtime.workspace),
                selected_tools=self.tool_registry.get_active_tool_names(),
                tool_snippets=self.tool_registry.get_tool_snippets(),
                context_files=self.runtime.loaded_resources.context_files,
                skills=self.runtime.loaded_resources.skills,
                custom_prompt=self.runtime.custom_system_prompt,
                append_system_prompt=self.runtime.append_system_prompt,
            )
        )
        self.agent.set_system_prompt(result.system_prompt)
        self.runtime.current_system_prompt = result.system_prompt
        self.runtime.current_tool_names = self.tool_registry.get_active_tool_names()
        model = self.agent.state.model
        self.store.append_system_snapshot(
            system_prompt=result.system_prompt,
            tool_names=self.runtime.current_tool_names,
            model_info={
                "provider": getattr(model, "provider", ""),
                "id": getattr(model, "id", ""),
                "base_url": getattr(model, "base_url", None),
            },
        )
        return result.system_prompt

    async def compact(self, instructions: str | None = None) -> str | None:
        _ = instructions
        return None

    def get_stats(self) -> SessionStats:
        messages = self.agent.state.messages
        user_messages = sum(1 for msg in messages if getattr(msg, "role", None) == "user")
        assistant_messages = sum(
            1 for msg in messages if getattr(msg, "role", None) == "assistant"
        )
        tool_results = sum(
            1 for msg in messages if getattr(msg, "role", None) == "toolResult"
        )
        return SessionStats(
            session_id=self.runtime.session_id,
            session_file=str(self.store.session_file()),
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tool_results=tool_results,
            total_messages=len(messages),
            last_error=self.agent.state.error,
        )

    def subscribe(self, fn) -> callable:
        return self.agent.subscribe(fn)

    def _persist_new_messages(self, start_index: int) -> None:
        if not self.settings.session.auto_save:
            return
        new_messages = self.agent.state.messages[start_index:]
        persistable = [
            message
            for message in new_messages
            if getattr(message, "role", None) in ("user", "assistant", "toolResult")
        ]
        if persistable:
            self.store.append_agent_messages(persistable)

    @staticmethod
    def _resolve_model(options: SessionCreateOptions) -> Model:
        if options.model_instance is not None:
            return options.model_instance
        if options.provider or options.model:
            if not (options.provider and options.model):
                raise ValueError("--provider and --model must be provided together.")
            return get_model(
                provider=options.provider,
                model_id=options.model,
                api_key=options.api_key,
                base_url=options.base_url,
            )
        config = get_llm_config(options.config_path)
        return config.get_model(options.model_config)

    @staticmethod
    def _open_store(
        workspace: Path,
        settings: CodeAgentSettings,
        options: SessionCreateOptions,
    ) -> SessionStore:
        if options.session_file:
            return SessionStore.open_existing(options.session_file)
        session_dir = resolve_session_dir(settings, workspace)
        if options.continue_last_session or settings.session.continue_last_session:
            latest = AgentSession._find_latest_session_file(session_dir)
            if latest is not None:
                return SessionStore.open_existing(latest)
        return SessionStore.create_new(session_dir, workspace)

    @staticmethod
    def _find_latest_session_file(session_dir: Path) -> Path | None:
        if not session_dir.exists():
            return None
        candidates = [path for path in session_dir.glob("*.jsonl") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)
