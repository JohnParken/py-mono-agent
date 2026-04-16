"""
LLM 抽象层

对应 TypeScript 版本的 @mariozechner/pi-ai 核心功能。
提供统一的 LLM 模型定义和流式调用接口。

支持的 Provider:
  - openai (OpenAI API 兼容)
  - anthropic (Anthropic Claude)
  - google (Google Gemini)

用户也可以自定义 Provider 适配器。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import requests

from .types import (
    AssistantMessage,
    Content,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    ToolDef,
)
from .exceptions import (
    LLMStreamError,
    LLMTimeoutError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMAuthenticationError,
    AgentValidationError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model
# =============================================================================


@dataclass
class Model:
    """
    LLM 模型定义。

    Attributes:
        provider: 提供商标识 (e.g. "openai", "anthropic", "google")
        id: 模型 ID (e.g. "gpt-4o", "claude-sonnet-4-20250514")
        api: API 类型 (e.g. "openai", "anthropic", "google")
        api_key: API 密钥（可选，也可通过环境变量设置）
        base_url: 自定义 API 基础 URL（可选）
    """

    provider: str
    id: str
    api: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self):
        if not self.api:
            self.api = self.provider


def get_model(provider: str, model_id: str, **kwargs) -> Model:
    """
    创建一个 Model 实例。

    Args:
        provider: 提供商标识
        model_id: 模型 ID
        **kwargs: 额外参数 (api_key, base_url 等)

    Returns:
        Model 实例
    """
    return Model(provider=provider, id=model_id, **kwargs)


# =============================================================================
# Assistant Message Event (流式事件)
# =============================================================================


@dataclass
class StreamStartEvent:
    """流开始"""

    partial: AssistantMessage
    type: str = "start"


@dataclass
class StreamTextStartEvent:
    """文本块开始"""

    content_index: int
    partial: AssistantMessage
    type: str = "text_start"


@dataclass
class StreamTextDeltaEvent:
    """文本增量"""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = "text_delta"


@dataclass
class StreamTextEndEvent:
    """文本块结束"""

    content_index: int
    content: str
    partial: AssistantMessage
    type: str = "text_end"


@dataclass
class StreamThinkingStartEvent:
    """思考块开始"""

    content_index: int
    partial: AssistantMessage
    type: str = "thinking_start"


@dataclass
class StreamThinkingDeltaEvent:
    """思考增量"""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = "thinking_delta"


@dataclass
class StreamThinkingEndEvent:
    """思考块结束"""

    content_index: int
    content: str
    partial: AssistantMessage
    type: str = "thinking_end"


@dataclass
class StreamToolCallStartEvent:
    """工具调用开始"""

    content_index: int
    partial: AssistantMessage
    type: str = "toolcall_start"


@dataclass
class StreamToolCallDeltaEvent:
    """工具调用增量"""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = "toolcall_delta"


@dataclass
class StreamToolCallEndEvent:
    """工具调用结束"""

    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: str = "toolcall_end"


@dataclass
class StreamDoneEvent:
    """流完成"""

    reason: str  # "stop" | "length" | "toolUse"
    message: AssistantMessage
    type: str = "done"


@dataclass
class StreamErrorEvent:
    """流错误"""

    reason: str  # "aborted" | "error"
    error: AssistantMessage
    type: str = "error"


# 流事件联合类型
AssistantMessageEvent = Union[
    StreamStartEvent,
    StreamTextStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamThinkingStartEvent,
    StreamThinkingDeltaEvent,
    StreamThinkingEndEvent,
    StreamToolCallStartEvent,
    StreamToolCallDeltaEvent,
    StreamToolCallEndEvent,
    StreamDoneEvent,
    StreamErrorEvent,
]


# =============================================================================
# Stream Response Wrapper
# =============================================================================


class StreamResponse:
    """
    流式响应包装器。

    包装 LLM 的流式响应，提供事件迭代和最终结果获取。
    """

    def __init__(self, events_gen: AsyncGenerator[AssistantMessageEvent, None]):
        self._events_gen = events_gen
        self._result: Optional[AssistantMessage] = None

    def __aiter__(self):
        return self._wrap_events()

    async def _wrap_events(self):
        async for event in self._events_gen:
            if event.type == "done":
                self._result = event.message
            elif event.type == "error":
                self._result = event.error
            yield event

    async def result(self) -> AssistantMessage:
        """获取最终的完整消息"""
        if self._result is None:
            # 消费所有事件
            async for _ in self._wrap_events():
                pass
        if self._result is None:
            raise LLMStreamError("流式响应未产生结果")
        return self._result


# =============================================================================
# Tool 参数验证
# =============================================================================


def validate_tool_arguments(tool: ToolDef, tool_call: ToolCall) -> Any:
    """
    使用 Pydantic 模型验证工具调用参数。

    Args:
        tool: 工具定义
        tool_call: LLM 的工具调用

    Returns:
        验证后的参数对象

    Raises:
        AgentValidationError: 参数验证失败
    """
    try:
        validated = tool.parameters(**tool_call.arguments)
        return validated
    except Exception as e:
        raise AgentValidationError(
            f"工具 '{tool.name}' 参数验证失败: {e}",
            field=list(tool_call.arguments.keys()) if tool_call.arguments else None
        ) from e


# =============================================================================
# Provider 适配器协议
# =============================================================================


@runtime_checkable
class ProviderAdapter(Protocol):
    """LLM Provider 适配器协议"""

    async def stream(
        self,
        model: Model,
        messages: List[Any],
        system_prompt: str,
        tools: Optional[List[ToolDef]] = None,
        **kwargs,
    ) -> AsyncGenerator[AssistantMessageEvent, None]: ...


# =============================================================================
# OpenAI 兼容 Provider
# =============================================================================


class OpenAIProvider:
    """
    OpenAI API 兼容的 Provider 适配器。

    支持 OpenAI 官方 API 和所有兼容的第三方 API（如 DeepSeek、
    Moonshot、Together AI 等）。
    """

    def __init__(self):
        self._client = None

    def _get_client(self, model: Model, api_key: Optional[str] = None):
        """延迟初始化 OpenAI 客户端"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai 包: pip install openai"
            )

        key = api_key or model.api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = model.base_url

        return AsyncOpenAI(api_key=key, base_url=base_url)

    async def stream(
        self,
        model: Model,
        messages: List[Any],
        system_prompt: str,
        tools: Optional[List[ToolDef]] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        """流式调用 OpenAI API"""
        client = self._get_client(model, api_key)

        # 构建消息
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "user":
                content_parts = []
                for c in msg.content:
                    if c.type == "text":
                        content_parts.append({"type": "text", "text": c.text})
                    elif c.type == "image":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{c.mime_type};base64,{c.data}"
                            },
                        })
                api_messages.append({"role": "user", "content": content_parts})

            elif msg.role == "assistant":
                content_text = ""
                tool_calls_list = []
                for c in msg.content:
                    if c.type == "text":
                        content_text += c.text
                    elif c.type == "toolCall":
                        tool_calls_list.append({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        })
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if content_text:
                    assistant_msg["content"] = content_text
                if tool_calls_list:
                    assistant_msg["tool_calls"] = tool_calls_list
                api_messages.append(assistant_msg)

            elif msg.role == "toolResult":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": "\n".join(
                        c.text for c in msg.content if c.type == "text"
                    ),
                })

        # 构建工具定义
        api_tools = None
        if tools:
            api_tools = []
            for tool in tools:
                schema = tool.parameters.model_json_schema()
                # 移除 Pydantic 自动添加的 title 字段
                schema.pop("title", None)
                api_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                })

        # 建立流式连接
        create_kwargs: Dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "stream": True,
        }
        if api_tools:
            create_kwargs["tools"] = api_tools

        # 合并额外参数（排除不应传递给 API 的参数）
        excluded_keys = ("signal", "api_key", "session_id", "user_id", "project_id")
        for k, v in kwargs.items():
            if k not in excluded_keys and v is not None:
                create_kwargs[k] = v

        partial = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage={
                "input": 0,
                "output": 0,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": 0,
                "cost": {"input": 0, "output": 0, "total": 0},
            },
        )

        yield StreamStartEvent(partial=partial)

        try:
            response = await client.chat.completions.create(**create_kwargs)

            current_tool_calls: Dict[int, Dict[str, Any]] = {}

            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

                # 处理文本内容
                if delta.content:
                    # 确保有文本内容块
                    if not partial.content or partial.content[-1].type != "text":
                        text_content = TextContent(text="")
                        partial.content.append(text_content)
                        ci = len(partial.content) - 1
                        yield StreamTextStartEvent(
                            content_index=ci, partial=partial
                        )

                    ci = len(partial.content) - 1
                    text_block = partial.content[ci]
                    if isinstance(text_block, TextContent):
                        text_block.text += delta.content
                    yield StreamTextDeltaEvent(
                        content_index=ci,
                        delta=delta.content,
                        partial=partial,
                    )

                # 处理工具调用
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function and tc.function.name else "",
                                "arguments": "",
                            }
                            tool_call_obj = ToolCall(
                                id=current_tool_calls[idx]["id"],
                                name=current_tool_calls[idx]["name"],
                                arguments={},
                            )
                            partial.content.append(tool_call_obj)
                            ci = len(partial.content) - 1
                            yield StreamToolCallStartEvent(
                                content_index=ci, partial=partial
                            )
                        else:
                            if tc.id:
                                current_tool_calls[idx]["id"] = tc.id
                            if tc.function and tc.function.name:
                                current_tool_calls[idx]["name"] = tc.function.name

                        if tc.function and tc.function.arguments:
                            current_tool_calls[idx]["arguments"] += tc.function.arguments
                            # 更新 partial
                            ci_offset = len(partial.content) - len(current_tool_calls) + idx
                            if 0 <= ci_offset < len(partial.content):
                                tc_content = partial.content[ci_offset]
                                if isinstance(tc_content, ToolCall):
                                    tc_content.id = current_tool_calls[idx]["id"]
                                    tc_content.name = current_tool_calls[idx]["name"]
                                    try:
                                        tc_content.arguments = json.loads(
                                            current_tool_calls[idx]["arguments"]
                                        )
                                    except json.JSONDecodeError:
                                        pass
                                yield StreamToolCallDeltaEvent(
                                    content_index=ci_offset,
                                    delta=tc.function.arguments,
                                    partial=partial,
                                )

                # 完成
                if finish_reason:
                    # 结束所有打开的文本块
                    for i, c in enumerate(partial.content):
                        if isinstance(c, TextContent):
                            yield StreamTextEndEvent(
                                content_index=i,
                                content=c.text,
                                partial=partial,
                            )

                    # 结束所有工具调用
                    for idx, tc_data in current_tool_calls.items():
                        ci_offset = len(partial.content) - len(current_tool_calls) + idx
                        if 0 <= ci_offset < len(partial.content):
                            tc_content = partial.content[ci_offset]
                            if isinstance(tc_content, ToolCall):
                                yield StreamToolCallEndEvent(
                                    content_index=ci_offset,
                                    tool_call=tc_content,
                                    partial=partial,
                                )

                    # 设置 usage
                    if chunk.usage:
                        # 提取缓存 token (OpenAI 特有)
                        cache_read_tokens = 0
                        if hasattr(chunk.usage, "prompt_tokens_details"):
                            details = chunk.usage.prompt_tokens_details
                            if details and hasattr(details, "cached_tokens"):
                                cache_read_tokens = details.cached_tokens or 0

                        partial.usage = {
                            "input": chunk.usage.prompt_tokens or 0,
                            "output": chunk.usage.completion_tokens or 0,
                            "cached_tokens": cache_read_tokens,  # 保持向后兼容
                            "cache_read": cache_read_tokens,
                            "cache_write": 0,  # OpenAI 不提供此字段
                            "total_tokens": chunk.usage.total_tokens or 0,
                            "cost": {"input": 0, "output": 0, "total": 0},
                        }

                    if finish_reason == "tool_calls":
                        partial.stop_reason = "toolUse"
                    elif finish_reason == "length":
                        partial.stop_reason = "length"
                    else:
                        partial.stop_reason = "stop"

                    yield StreamDoneEvent(
                        reason=partial.stop_reason, message=partial
                    )

        except Exception as e:
            error_str = str(e)
            
            # 检测速率限制错误 (429)
            if "429" in error_str or "rate limit" in error_str.lower() or "速率限制" in error_str:
                partial.stop_reason = "error"
                partial.error_message = error_str
                yield StreamErrorEvent(reason="error", error=partial)
                raise LLMRateLimitError(provider=model.provider)
            
            # 检测认证错误
            if "401" in error_str or "unauthorized" in error_str.lower():
                partial.stop_reason = "error"
                partial.error_message = error_str
                yield StreamErrorEvent(reason="error", error=partial)
                raise LLMAuthenticationError(provider=model.provider)
            
            # 其他错误
            partial.stop_reason = "error"
            partial.error_message = error_str
            yield StreamErrorEvent(reason="error", error=partial)


class QwenLLMProvider:
    """
    Qwen Provider 适配器。

    使用 requests 库请求 DashScope 原生文本生成接口。
    construct_request() 基于 JSON 模板字符串做深拷贝，再按 messages /
    parameters / input 等字段进行定制构造。
    """

    DEFAULT_BASE_URL = (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    )
    REQUEST_TEMPLATE_JSON = json.dumps(
        {
            "model": "",
            "input": {"messages": []},
            "parameters": {
                "result_format": "message",
                "incremental_output": True,
            },
        }
    )

    def construct_request(
        self,
        model: Model,
        messages: List[Any],
        system_prompt: str,
        tools: Optional[List[ToolDef]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        构造请求参数。

        基于 JSON 模板字符串做深拷贝，再注入 messages、parameters 等字段。
        """
        template_json = kwargs.pop("request_template_json", self.REQUEST_TEMPLATE_JSON)
        payload = self._clone_request_template(template_json)

        payload["model"] = model.id
        payload.setdefault("input", {})
        payload["input"]["messages"] = self._build_messages(messages, system_prompt)

        parameters = payload.setdefault("parameters", {})
        parameters.setdefault("result_format", "message")
        parameters.setdefault("incremental_output", True)

        # 将 reasoning 语义映射到 Qwen 的 enable_thinking。
        reasoning = kwargs.pop("reasoning", None)
        if reasoning is not None:
            parameters["enable_thinking"] = reasoning != "off"

        if tools:
            payload["tools"] = self._build_tools(tools)

        # 允许调用方直接定制原生 input / parameters 字段。
        input_overrides = kwargs.pop("input", None)
        if isinstance(input_overrides, dict):
            payload["input"].update(self._deep_copy_json_value(input_overrides))
            payload["input"]["messages"] = self._build_messages(messages, system_prompt)

        parameter_overrides = kwargs.pop("parameters", None)
        if isinstance(parameter_overrides, dict):
            parameters.update(self._deep_copy_json_value(parameter_overrides))

        request_overrides = kwargs.pop("request_overrides", None)
        if isinstance(request_overrides, dict):
            payload = self._deep_merge(payload, self._deep_copy_json_value(request_overrides))
            payload["model"] = model.id
            payload.setdefault("input", {})
            payload["input"]["messages"] = self._build_messages(messages, system_prompt)
            parameters = payload.setdefault("parameters", {})

        excluded_keys = ("signal", "api_key", "session_id", "user_id", "project_id", "timeout")
        for key, value in kwargs.items():
            if key not in excluded_keys and value is not None:
                parameters[key] = value

        return payload

    def _clone_request_template(self, template_json: str) -> Dict[str, Any]:
        if not isinstance(template_json, str):
            raise TypeError("request_template_json must be a JSON string.")
        return json.loads(template_json)

    def _deep_copy_json_value(self, value: Any) -> Any:
        return json.loads(json.dumps(value))

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._deep_copy_json_value(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _build_messages(
        self,
        messages: List[Any],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        api_messages: List[Dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "user":
                api_messages.append(
                    {
                        "role": "user",
                        "content": self._stringify_message_content(msg.content),
                    }
                )

            elif msg.role == "assistant":
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": self._stringify_message_content(msg.content),
                }
                tool_calls = []
                for content in msg.content:
                    if content.type == "toolCall":
                        tool_calls.append(
                            {
                                "id": content.id,
                                "type": "function",
                                "function": {
                                    "name": content.name,
                                    "arguments": json.dumps(content.arguments),
                                },
                            }
                        )
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                api_messages.append(assistant_msg)

            elif msg.role == "toolResult":
                api_messages.append(
                    {
                        "role": "tool",
                        "content": self._stringify_message_content(msg.content),
                        "tool_call_id": msg.tool_call_id,
                    }
                )

        return api_messages

    def _stringify_message_content(self, content_blocks: List[Any]) -> str:
        parts: List[str] = []
        for content in content_blocks:
            if content.type == "text":
                parts.append(content.text)
            elif content.type == "thinking":
                parts.append(content.thinking)
            elif content.type == "image":
                raise ValueError(
                    "QwenLLMProvider 当前使用文本生成接口，不支持 image content。"
                )
        return "".join(parts)

    def _build_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        api_tools = []
        for tool in tools:
            schema = tool.parameters.model_json_schema()
            schema.pop("title", None)
            api_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                }
            )
        return api_tools

    def _chunk_context(self, chunk_data: Dict[str, Any]) -> str:
        question_id = chunk_data.get("questionId")
        session_id = chunk_data.get("sessionId")
        parts = []
        if question_id:
            parts.append(f"questionId={question_id}")
        if session_id:
            parts.append(f"sessionId={session_id}")
        return ", ".join(parts)

    def _is_chunk_failed(self, chunk_data: Dict[str, Any]) -> bool:
        status = str(chunk_data.get("status") or "").lower()
        return status in {"failed", "error", "cancelled", "canceled", "aborted"}

    def _build_chunk_error(self, chunk_data: Dict[str, Any]) -> str:
        res_code = chunk_data.get("resCode", "")
        res_message = chunk_data.get("resMessage", "")
        status = chunk_data.get("status", "")
        context = self._chunk_context(chunk_data)
        pieces = [
            "Qwen provider chunk failed",
            f"status={status}" if status else "",
            f"resCode={res_code}" if res_code else "",
            res_message,
            context,
        ]
        return " | ".join(piece for piece in pieces if piece)

    async def stream(
        self,
        model: Model,
        messages: List[Any],
        system_prompt: str,
        tools: Optional[List[ToolDef]] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        key = api_key or model.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = model.base_url or self.DEFAULT_BASE_URL
        payload = self.construct_request(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )

        partial = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage={
                "input": 0,
                "output": 0,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": 0,
                "cost": {"input": 0, "output": 0, "total": 0},
            },
        )
        yield StreamStartEvent(partial=partial)

        try:
            text_index: Optional[int] = None
            thinking_index: Optional[int] = None
            completed = False

            async for chunk_data in self._stream_request(
                base_url=base_url,
                api_key=key,
                payload=payload,
                timeout=kwargs.get("timeout", 60),
            ):
                if not isinstance(chunk_data, dict):
                    continue

                chunk_context = self._chunk_context(chunk_data)

                res_code = chunk_data.get("resCode")
                if res_code and res_code != "PLA0000":
                    raise RuntimeError(self._build_chunk_error(chunk_data))

                if self._is_chunk_failed(chunk_data):
                    raise RuntimeError(self._build_chunk_error(chunk_data))

                if "result" in chunk_data and "status" in chunk_data:
                    if text_index is None:
                        partial.content.append(TextContent(text=""))
                        text_index = len(partial.content) - 1
                        yield StreamTextStartEvent(
                            content_index=text_index,
                            partial=partial,
                        )

                    delta_text = str(chunk_data.get("result") or "")
                    if delta_text:
                        text_block = partial.content[text_index]
                        if isinstance(text_block, TextContent):
                            text_block.text += delta_text
                        yield StreamTextDeltaEvent(
                            content_index=text_index,
                            delta=delta_text,
                            partial=partial,
                        )

                    status = str(chunk_data.get("status") or "").lower()
                    if status == "running" and chunk_context:
                        logger.debug("Qwen stream running: %s", chunk_context)
                    if status == "completed":
                        completed = True
                        final_text = ""
                        if text_index is not None:
                            text_block = partial.content[text_index]
                            if isinstance(text_block, TextContent):
                                final_text = text_block.text
                        if chunk_context:
                            logger.info("Qwen stream completed: %s", chunk_context)
                        yield StreamTextEndEvent(
                            content_index=text_index or 0,
                            content=final_text,
                            partial=partial,
                        )
                        partial.stop_reason = "stop"
                        yield StreamDoneEvent(reason=partial.stop_reason, message=partial)
                        return

                    continue

                output = chunk_data.get("output") or {}
                choice = (output.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                finish_reason = choice.get("finish_reason") or output.get("finish_reason") or "stop"

                reasoning_text = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or message.get("reasoning_text")
                )
                if reasoning_text:
                    if thinking_index is None:
                        partial.content.append(ThinkingContent(thinking=""))
                        thinking_index = len(partial.content) - 1
                        yield StreamThinkingStartEvent(
                            content_index=thinking_index,
                            partial=partial,
                        )
                    thinking_block = partial.content[thinking_index]
                    if isinstance(thinking_block, ThinkingContent):
                        thinking_block.thinking += reasoning_text
                    yield StreamThinkingDeltaEvent(
                        content_index=thinking_index,
                        delta=reasoning_text,
                        partial=partial,
                    )
                    yield StreamThinkingEndEvent(
                        content_index=thinking_index,
                        content=thinking_block.thinking if isinstance(thinking_block, ThinkingContent) else reasoning_text,
                        partial=partial,
                    )

                content_text = message.get("content") or ""
                if content_text:
                    if text_index is None:
                        partial.content.append(TextContent(text=""))
                        text_index = len(partial.content) - 1
                        yield StreamTextStartEvent(content_index=text_index, partial=partial)
                    text_block = partial.content[text_index]
                    if isinstance(text_block, TextContent):
                        text_block.text += content_text
                    yield StreamTextDeltaEvent(
                        content_index=text_index,
                        delta=content_text,
                        partial=partial,
                    )

                tool_calls = message.get("tool_calls") or []
                for tool_call in tool_calls:
                    function_data = tool_call.get("function") or {}
                    raw_arguments = function_data.get("arguments") or "{}"
                    try:
                        parsed_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        parsed_arguments = {}
                    tool_call_obj = ToolCall(
                        id=tool_call.get("id", ""),
                        name=function_data.get("name", ""),
                        arguments=parsed_arguments,
                    )
                    partial.content.append(tool_call_obj)
                    content_index = len(partial.content) - 1
                    yield StreamToolCallStartEvent(
                        content_index=content_index,
                        partial=partial,
                    )
                    yield StreamToolCallDeltaEvent(
                        content_index=content_index,
                        delta=raw_arguments,
                        partial=partial,
                    )
                    yield StreamToolCallEndEvent(
                        content_index=content_index,
                        tool_call=tool_call_obj,
                        partial=partial,
                    )

                usage = chunk_data.get("usage") or {}
                partial.usage = {
                    "input": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    "output": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                    "cache_read": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                    "cache_write": usage.get("prompt_tokens_details", {}).get("cache_write_tokens", 0),
                    "total_tokens": usage.get(
                        "total_tokens",
                        usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    ),
                    "cost": {
                        "input": usage.get("cost_details", {}).get("upstream_inference_prompt_cost", 0),
                        "output": usage.get("cost_details", {}).get("upstream_inference_completions_cost", 0),
                        "total": usage.get("cost", 0),
                    },
                }

                if text_index is not None:
                    text_block = partial.content[text_index]
                    final_text = text_block.text if isinstance(text_block, TextContent) else ""
                    yield StreamTextEndEvent(
                        content_index=text_index,
                        content=final_text,
                        partial=partial,
                    )

                if finish_reason == "tool_calls":
                    partial.stop_reason = "toolUse"
                elif finish_reason == "length":
                    partial.stop_reason = "length"
                else:
                    partial.stop_reason = "stop"

                yield StreamDoneEvent(reason=partial.stop_reason, message=partial)
                return

            if not completed:
                if text_index is not None:
                    text_block = partial.content[text_index]
                    final_text = text_block.text if isinstance(text_block, TextContent) else ""
                    yield StreamTextEndEvent(
                        content_index=text_index,
                        content=final_text,
                        partial=partial,
                    )
                partial.stop_reason = "stop"
                yield StreamDoneEvent(reason=partial.stop_reason, message=partial)

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            error_str = str(exc)
            partial.stop_reason = "error"
            partial.error_message = error_str
            yield StreamErrorEvent(reason="error", error=partial)
            logger.warning("Qwen HTTP error: %s", error_str)
            if status_code == 401:
                raise LLMAuthenticationError(provider=model.provider) from exc
            if status_code == 429:
                raise LLMRateLimitError(provider=model.provider) from exc
            raise LLMConnectionError(provider=model.provider) from exc
        except requests.RequestException as exc:
            partial.stop_reason = "error"
            partial.error_message = str(exc)
            yield StreamErrorEvent(reason="error", error=partial)
            logger.warning("Qwen request error: %s", exc)
            raise LLMConnectionError(provider=model.provider) from exc
        except Exception as exc:
            partial.stop_reason = "error"
            partial.error_message = str(exc)
            yield StreamErrorEvent(reason="error", error=partial)
            logger.warning("Qwen stream error: %s", exc)

    async def _stream_request(
        self,
        base_url: str,
        api_key: str,
        payload: Dict[str, Any],
        timeout: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        sentinel = object()

        def worker() -> None:
            try:
                with requests.post(
                    base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        loop.call_soon_threadsafe(queue.put_nowait, parsed)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        worker_task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await worker_task

    def _send_request(
        self,
        base_url: str,
        api_key: str,
        payload: Dict[str, Any],
        timeout: int,
    ) -> requests.Response:
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response


# =============================================================================
# Provider 注册表
# =============================================================================

_PROVIDERS: Dict[str, ProviderAdapter] = {}


def register_provider(name: str, provider: ProviderAdapter) -> None:
    """注册一个 LLM Provider"""
    _PROVIDERS[name] = provider


def get_provider(name: str) -> ProviderAdapter:
    """获取 Provider 实例"""
    if name not in _PROVIDERS:
        # 尝试自动注册
        if name == "openai":
            _PROVIDERS[name] = OpenAIProvider()
        elif name in ("QwenLLMprovider", "QwenLLMProvider"):
            provider = QwenLLMProvider()
            _PROVIDERS["QwenLLMprovider"] = provider
            _PROVIDERS["QwenLLMProvider"] = provider
        else:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available: {list(_PROVIDERS.keys())}. "
                f"Use register_provider() to register custom providers."
            )
    return _PROVIDERS[name]


# =============================================================================
# stream_simple (核心流式调用)
# =============================================================================


async def stream_simple(
    model: Model,
    context: Dict[str, Any],
    **options,
) -> StreamResponse:
    """
    统一的 LLM 流式调用函数。

    对应 TypeScript 版本的 streamSimple()。
    根据 model.provider 选择对应的 Provider 进行调用。

    Args:
        model: LLM 模型
        context: 上下文字典 (system_prompt, messages, tools)
        **options: 额外选项 (api_key, reasoning, signal 等)

    Returns:
        StreamResponse 包装器
    """
    provider = get_provider(model.provider)

    system_prompt = context.get("system_prompt", "")
    messages = context.get("messages", [])
    tools = context.get("tools", None)

    async def _generate():
        async for event in provider.stream(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            **options,
        ):
            yield event

    return StreamResponse(_generate())
