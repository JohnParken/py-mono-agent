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
import importlib.resources
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
import pi_logger as _pi_logger

from .prompts.qwen_tools import build_qwen_tool_prompt
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

getattr(_pi_logger, "configure_logging", lambda *args, **kwargs: None)()
logger = logging.getLogger(__name__)


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _truncate_log_text(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def _summarize_tool_call_for_log(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, ToolCall):
        arguments = tool_call.arguments
        tool_name = tool_call.name
        tool_id = tool_call.id
    elif isinstance(tool_call, dict):
        function_data = tool_call.get("function") or {}
        arguments = function_data.get("arguments")
        tool_name = function_data.get("name") or tool_call.get("name", "")
        tool_id = tool_call.get("id", "")
    else:
        arguments = getattr(tool_call, "arguments", None)
        tool_name = getattr(tool_call, "name", "")
        tool_id = getattr(tool_call, "id", "")

    argument_keys: List[str] = []
    if isinstance(arguments, dict):
        argument_keys = sorted(str(key) for key in arguments.keys())

    return {
        "id": tool_id,
        "name": tool_name,
        "argument_keys": argument_keys,
        "arguments_preview": _truncate_log_text(_safe_json_dumps(arguments)),
    }


def _summarize_tool_calls_for_log(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    return [_summarize_tool_call_for_log(tool_call) for tool_call in tool_calls]


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
        logger.debug(
            "[TOOL-VALIDATE] start tool=%s payload=%s",
            tool.name,
            _summarize_tool_call_for_log(tool_call),
        )
        validated = tool.parameters(**tool_call.arguments)
        logger.debug(
            "[TOOL-VALIDATE] success tool=%s validated_type=%s",
            tool.name,
            type(validated).__name__,
        )
        return validated
    except Exception as e:
        logger.warning(
            "[TOOL-VALIDATE] failed tool=%s payload=%s error=%s",
            tool.name,
            _summarize_tool_call_for_log(tool_call),
            e,
        )
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
            logger.debug(
                "[OPENAI-TOOLS] request tool_count=%s tools=%s",
                len(api_tools),
                [tool["function"]["name"] for tool in api_tools],
            )

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
                    logger.debug(
                        "[OPENAI-TOOLS] delta tool_calls=%s",
                        _summarize_tool_calls_for_log(
                            [
                                {
                                    "id": tc.id or "",
                                    "function": {
                                        "name": tc.function.name if tc.function else "",
                                        "arguments": tc.function.arguments if tc.function else "",
                                    },
                                }
                                for tc in delta.tool_calls
                            ]
                        ),
                    )
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
                                logger.debug(
                                    "[OPENAI-TOOLS] completed tool_call=%s",
                                    _summarize_tool_call_for_log(tc_content),
                                )
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
                        logger.info(
                            "[OPENAI-TOOLS] finish_reason=tool_calls tool_calls=%s",
                            _summarize_tool_calls_for_log(
                                [c for c in partial.content if isinstance(c, ToolCall)]
                            ),
                        )
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

    使用 requests 库请求 Qwen 自定义接口。
    construct_request() 基于 JSON 模板字符串做深拷贝，再按 token /
    appInfo / variable / data.messages 等字段进行定制构造。
    """

    DEFAULT_BASE_URL = (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    )
    NATIVE_TOOL_MODEL_PREFIXES = ("qwen3.6-35b-a3b",)
    REQUEST_TEMPLATE_RESOURCE = "qwen_request_template.json"
    _request_template_json: Optional[str] = None

    @classmethod
    def get_request_template_json(cls) -> str:
        if cls._request_template_json is None:
            cls._request_template_json = (
                importlib.resources.files("pi_ai")
                .joinpath(cls.REQUEST_TEMPLATE_RESOURCE)
                .read_text(encoding="utf-8")
            )
        return cls._request_template_json

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

        基于固定 JSON 模板字符串做深拷贝，仅动态填充 data.messages。
        """
        template_json = kwargs.pop(
            "request_template_json",
            self.get_request_template_json(),
        )
        payload = self._clone_request_template(template_json)
        resolved_api_key = kwargs.pop("api_key", None) or model.api_key or os.environ.get(
            "DASHSCOPE_API_KEY", ""
        )
        app_api_key = kwargs.pop("app_api_key", None) or os.environ.get(
            "QWEN_APP_API_KEY", ""
        )
        app_info_override = kwargs.pop("app_info", None) or {}
        payload_override = kwargs.pop("payload_override", None) or {}
        requested_tool_calling_mode = kwargs.pop("tool_calling_mode", "text")
        resolved_tool_calling_mode = self._resolve_tool_calling_mode(
            model,
            tools,
            {"tool_calling_mode": requested_tool_calling_mode},
        )

        if resolved_api_key:
            payload["token"] = resolved_api_key
        if app_api_key:
            payload["apikey"] = app_api_key
        payload["modelId"] = model.id

        if app_info_override:
            payload["appInfo"] = self._deep_merge(
                payload.get("appInfo", {}),
                app_info_override,
            )

        api_tools = self._build_tools(tools) if tools else []
        prompt_variable_value = (
            ""
            if resolved_tool_calling_mode == "native"
            else self._build_sys_prompt_content(
                system_prompt=system_prompt,
                messages=messages,
                api_tools=api_tools,
                tool_calling_mode=resolved_tool_calling_mode,
            )
        )
        payload["variable"] = self._build_variables(
            prompt_variable_value,
            payload.get("variable"),
        )

        data = payload.setdefault("data", {})
        if resolved_tool_calling_mode == "native":
            data["messages"] = self._build_native_messages(messages)
        else:
            data["messages"] = self._build_messages(messages, system_prompt)
        if resolved_tool_calling_mode == "native" and api_tools:
            data["tools"] = self._deep_copy_json_value(api_tools)
        else:
            data.pop("tools", None)

        if payload_override:
            payload = self._deep_merge(payload, payload_override)

        logger.debug(
            "[QWEN-TOOLS] construct_request model=%s requested_mode=%s resolved_mode=%s tool_count=%s tools=%s variable_names=%s",
            model.id,
            requested_tool_calling_mode,
            resolved_tool_calling_mode,
            len(api_tools),
            [tool["function"]["name"] for tool in api_tools],
            [item.get("name") for item in payload.get("variable", []) if isinstance(item, dict)],
        )

        return payload

    def _clone_request_template(self, template_json: str) -> Dict[str, Any]:
        if not isinstance(template_json, str):
            raise TypeError("request_template_json must be a JSON string.")
        return json.loads(template_json)

    def _mask_secret(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        if len(text) <= 8:
            return "*" * len(text)
        return f"{text[:4]}...{text[-4:]}"

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = payload.get("data") or {}
        messages = data.get("messages") or []
        last_role = messages[-1].get("role") if messages else None
        last_content = messages[-1].get("content") if messages else []
        last_preview = ""
        if isinstance(last_content, list) and last_content:
            first_block = last_content[0] or {}
            last_preview = str(
                first_block.get("text")
                or first_block.get("value")
                or ""
            )[:80]
        return {
            "modelId": payload.get("modelId"),
            "type": payload.get("type"),
            "stream": data.get("stream"),
            "message_count": len(messages),
            "last_role": last_role,
            "last_preview": last_preview,
            "tool_count": self._count_tools_from_variables(payload.get("variable")),
            "token": self._mask_secret(payload.get("token")),
            "apikey": self._mask_secret(payload.get("apikey")),
        }

    def _sanitize_payload_for_logging(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = self._deep_copy_json_value(payload)
        if "token" in sanitized:
            sanitized["token"] = self._mask_secret(sanitized.get("token"))
        if "apikey" in sanitized:
            sanitized["apikey"] = self._mask_secret(sanitized.get("apikey"))
        return sanitized

    def _format_json_for_logging(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)

    def _truncate_for_logging(self, value: Any, limit: int = 4000) -> str:
        text = value if isinstance(value, str) else str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...[truncated {len(text) - limit} chars]"

    def _summarize_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        result_preview = str(chunk_data.get("result") or "")[:80]
        output = chunk_data.get("output") or {}
        choices = output.get("choices") or []
        finish_reason = None
        if choices:
            finish_reason = (choices[0] or {}).get("finish_reason")
        return {
            "status": chunk_data.get("status"),
            "resCode": chunk_data.get("resCode"),
            "questionId": chunk_data.get("questionId"),
            "sessionId": chunk_data.get("sessionId"),
            "result_preview": result_preview,
            "finish_reason": finish_reason,
        }

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

        for msg in messages:
            if msg.role == "user":
                api_messages.append(
                    {
                        "role": "user",
                        "content": self._build_message_content(msg.content),
                    }
                )

            elif msg.role == "assistant":
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                content_parts = self._build_message_content(msg.content)
                if content_parts:
                    assistant_msg["content"] = content_parts

                tool_calls_list = []
                for content in msg.content:
                    if getattr(content, "type", None) != "toolCall":
                        continue
                    tool_calls_list.append(
                        {
                            "id": getattr(content, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(content, "name", ""),
                                "arguments": json.dumps(
                                    getattr(content, "arguments", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )
                if tool_calls_list:
                    assistant_msg["tool_calls"] = tool_calls_list
                api_messages.append(assistant_msg)

            elif msg.role == "toolResult":
                api_messages.append(
                    {
                        "role": "tool",
                        "content": self._build_message_content(msg.content),
                        "tool_call_id": msg.tool_call_id,
                    }
                )

        return api_messages

    def _build_native_messages(
        self,
        messages: List[Any],
    ) -> List[Dict[str, Any]]:
        api_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "user":
                text_parts = [
                    c.text
                    for c in msg.content
                    if getattr(c, "type", None) == "text"
                ]
                api_messages.append({"role": "user", "content": "".join(text_parts)})

            elif msg.role == "assistant":
                content_text = ""
                tool_calls_list = []
                for c in msg.content:
                    if getattr(c, "type", None) == "text":
                        content_text += c.text
                    elif getattr(c, "type", None) == "toolCall":
                        tool_calls_list.append(
                            {
                                "id": getattr(c, "id", ""),
                                "type": "function",
                                "function": {
                                    "name": getattr(c, "name", ""),
                                    "arguments": json.dumps(
                                        getattr(c, "arguments", {}),
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        )

                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if content_text:
                    assistant_msg["content"] = content_text
                if tool_calls_list:
                    assistant_msg["tool_calls"] = tool_calls_list
                api_messages.append(assistant_msg)

            elif msg.role == "toolResult":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": "\n".join(
                            c.text for c in msg.content if getattr(c, "type", None) == "text"
                        ),
                    }
                )

        return api_messages

    def _count_tools_from_variables(self, variables: Any) -> int:
        if not isinstance(variables, list):
            return 0
        for item in variables:
            if item.get("name") != "tools":
                continue
            value = item.get("value")
            if not isinstance(value, str):
                continue
            return value.count('"type": "function"')
        return 0

    def _build_message_content(self, content_blocks: List[Any]) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        for content in content_blocks:
            if content.type == "text":
                parts.append({"type": "text", "text": content.text})
            elif content.type == "thinking":
                parts.append({"type": "text", "text": content.thinking})
            elif content.type == "toolCall":
                continue
            elif content.type == "image":
                raise ValueError(
                    "QwenLLMProvider 当前使用文本生成接口，不支持 image content。"
                )
        return parts

    def _extract_message_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("value") or ""
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        if isinstance(content, dict):
            return str(content.get("text") or content.get("value") or "")
        return str(content)

    def _merge_stream_fragment(self, existing: str, incoming: Any) -> str:
        incoming_text = ""
        if incoming is not None:
            incoming_text = str(incoming)
        if not incoming_text:
            return existing
        if not existing:
            return incoming_text
        existing_stripped = existing.strip()
        incoming_stripped = incoming_text.strip()
        if (
            existing_stripped.startswith(("{", "["))
            and existing_stripped.endswith(("}", "]"))
            and incoming_stripped.startswith(("{", "["))
            and incoming_stripped.endswith(("}", "]"))
        ):
            return incoming_text
        if incoming_text.startswith(existing):
            return incoming_text
        if existing.startswith(incoming_text) or incoming_text in existing:
            return existing
        return f"{existing}{incoming_text}"

    def _compute_stream_text_delta(self, existing_full: str, incoming: Any) -> Tuple[str, str]:
        incoming_text = ""
        if incoming is not None:
            incoming_text = str(incoming)
        if not incoming_text:
            return "", existing_full
        if not existing_full:
            return incoming_text, incoming_text
        if incoming_text.startswith(existing_full):
            return incoming_text[len(existing_full) :], incoming_text
        if existing_full.startswith(incoming_text):
            return "", incoming_text
        return incoming_text, f"{existing_full}{incoming_text}"

    def _parse_tool_call_arguments(self, raw_arguments: Any) -> Tuple[Dict[str, Any], str]:
        if raw_arguments is None:
            return {}, "{}"
        if isinstance(raw_arguments, dict):
            return raw_arguments, json.dumps(raw_arguments, ensure_ascii=False)
        if isinstance(raw_arguments, str):
            raw_arguments = self._normalize_tool_call_text(raw_arguments)
            try:
                parsed = json.loads(raw_arguments)
                if isinstance(parsed, dict):
                    return parsed, raw_arguments
            except json.JSONDecodeError:
                return {}, raw_arguments
            return {}, raw_arguments
        try:
            text = json.dumps(raw_arguments, ensure_ascii=False)
        except TypeError:
            text = str(raw_arguments)
        if isinstance(raw_arguments, dict):
            return raw_arguments, text
        return {}, text

    def _normalize_tool_call_text(self, text: str) -> str:
        return (
            text.replace("\ufeff", "")
            .replace("\u200b", "")
            .replace("\u200c", "")
            .replace("\u200d", "")
            .replace("\u2060", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

    def _extract_json_candidates(self, text: str) -> List[Any]:
        candidates: List[Any] = []
        stripped = self._normalize_tool_call_text(text)
        if not stripped:
            return candidates

        decoder = json.JSONDecoder()
        for start, char in enumerate(stripped):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                continue
            candidates.append(value)

        return candidates

    def _normalize_text_tool_call(
        self,
        raw_tool_call: Any,
        available_tool_names: set[str],
    ) -> Optional[ToolCall]:
        if not isinstance(raw_tool_call, dict):
            return None

        function_data = raw_tool_call.get("function") or {}
        name = (
            function_data.get("name")
            or raw_tool_call.get("name")
            or raw_tool_call.get("tool_name")
        )
        if not name or name not in available_tool_names:
            return None

        raw_arguments = (
            function_data.get("arguments")
            if "arguments" in function_data
            else raw_tool_call.get("arguments", raw_tool_call.get("args", {}))
        )
        parsed_arguments, _ = self._parse_tool_call_arguments(raw_arguments)
        return ToolCall(
            id=str(raw_tool_call.get("id") or f"call_{uuid.uuid4().hex}"),
            name=str(name),
            arguments=parsed_arguments,
        )

    def _extract_text_tool_calls(
        self,
        text: str,
        tools: Optional[List[ToolDef]],
    ) -> List[ToolCall]:
        available_tool_names = {tool.name for tool in (tools or [])}
        if not available_tool_names:
            return []

        extracted: List[ToolCall] = []
        seen_tool_calls: set[Tuple[str, str, str]] = set()
        for candidate in self._extract_json_candidates(text):
            raw_tool_calls: List[Any] = []
            if isinstance(candidate, dict):
                if isinstance(candidate.get("tool_calls"), list):
                    raw_tool_calls.extend(candidate["tool_calls"])
                elif isinstance(candidate.get("tool_call"), dict):
                    raw_tool_calls.append(candidate["tool_call"])
                else:
                    raw_tool_calls.append(candidate)
            elif isinstance(candidate, list):
                raw_tool_calls.extend(candidate)

            for raw_tool_call in raw_tool_calls:
                tool_call = self._normalize_text_tool_call(
                    raw_tool_call,
                    available_tool_names,
                )
                if tool_call:
                    dedupe_key = (
                        tool_call.name,
                        _safe_json_dumps(tool_call.arguments),
                    )
                    if dedupe_key in seen_tool_calls:
                        continue
                    seen_tool_calls.add(dedupe_key)
                    extracted.append(tool_call)

        if extracted:
            logger.info(
                "[QWEN-TOOLS] parsed text tool_calls=%s",
                _summarize_tool_calls_for_log(extracted),
            )
        return extracted

    async def _emit_qwen_tool_call(
        self,
        partial: AssistantMessage,
        tool_call_obj: ToolCall,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        partial.content.append(tool_call_obj)
        content_index = len(partial.content) - 1
        yield StreamToolCallStartEvent(
            content_index=content_index,
            partial=partial,
        )
        yield StreamToolCallDeltaEvent(
            content_index=content_index,
            delta=json.dumps(tool_call_obj.arguments, ensure_ascii=False),
            partial=partial,
        )
        yield StreamToolCallEndEvent(
            content_index=content_index,
            tool_call=tool_call_obj,
            partial=partial,
        )

    def _build_variables(
        self,
        tools_prompt_content: str,
        variables_override: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        variables = [{"name": "tools", "value": tools_prompt_content or ""}]
        if not variables_override:
            return variables

        copied = self._deep_copy_json_value(variables_override)
        has_system_var = False
        for item in copied:
            if item.get("name") == "tools":
                item["value"] = tools_prompt_content or ""
                has_system_var = True
        if not has_system_var:
            copied.append({"name": "tools", "value": tools_prompt_content or ""})
        return copied

    def _build_sys_prompt_content(
        self,
        system_prompt: str,
        messages: List[Any],
        api_tools: List[Dict[str, Any]],
        tool_calling_mode: str = "text",
    ) -> str:
        _ = messages
        include_fewshots = tool_calling_mode != "native"
        return build_qwen_tool_prompt(
            system_prompt,
            api_tools,
            mode=tool_calling_mode,
            include_fewshots=include_fewshots,
        )

    def _model_supports_native_tool_calling(self, model: Model) -> bool:
        model_id = (model.id or "").lower()
        return any(
            model_id.startswith(prefix.lower())
            for prefix in self.NATIVE_TOOL_MODEL_PREFIXES
        )

    def _resolve_tool_calling_mode(
        self,
        model: Model,
        tools: Optional[List[ToolDef]],
        kwargs: Dict[str, Any],
    ) -> str:
        requested_mode = str(kwargs.get("tool_calling_mode", "auto") or "auto").lower()
        if requested_mode not in {"auto", "native", "text"}:
            requested_mode = "auto"
        if not tools:
            return "text"
        if requested_mode == "native":
            return "native"
        if requested_mode == "text":
            return "text"
        return "native" if self._model_supports_native_tool_calling(model) else "text"

    async def _collect_attempt_events(
        self,
        model: Model,
        messages: List[Any],
        system_prompt: str,
        tools: Optional[List[ToolDef]],
        api_key: Optional[str],
        tool_calling_mode: str,
        **kwargs,
    ) -> Tuple[List[AssistantMessageEvent], Optional[Exception]]:
        events: List[AssistantMessageEvent] = []
        try:
            async for event in self._stream_once(
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                api_key=api_key,
                tool_calling_mode=tool_calling_mode,
                **kwargs,
            ):
                events.append(event)
        except Exception as exc:
            return events, exc
        return events, None

    def _native_attempt_requires_text_fallback(
        self,
        events: List[AssistantMessageEvent],
    ) -> bool:
        if any(getattr(event, "type", None) == "error" for event in events):
            return True
        done_event = next(
            (event for event in reversed(events) if getattr(event, "type", None) == "done"),
            None,
        )
        if done_event is None:
            return False
        message = done_event.message
        tool_calls = [content for content in message.content if isinstance(content, ToolCall)]
        return done_event.reason == "toolUse" and len(tool_calls) == 0

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
        resolved_mode = self._resolve_tool_calling_mode(model, tools, kwargs)
        request_kwargs = dict(kwargs)
        if resolved_mode == "native":
            native_events, native_error = await self._collect_attempt_events(
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                api_key=api_key,
                tool_calling_mode="native",
                **request_kwargs,
            )
            if native_error is None and not self._native_attempt_requires_text_fallback(native_events):
                for event in native_events:
                    yield event
                return

            logger.warning(
                "[QWEN-TOOLS] native tool calling fallback to text mode model=%s reason=%s",
                model.id,
                str(native_error) if native_error else "missing_structured_tool_calls",
            )

        text_events, text_error = await self._collect_attempt_events(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            api_key=api_key,
            tool_calling_mode="text",
            **request_kwargs,
        )
        for event in text_events:
            yield event
        if text_error is not None:
            raise text_error

    async def _stream_once(
        self,
        model: Model,
        messages: List[Any],
        system_prompt: str,
        tools: Optional[List[ToolDef]] = None,
        api_key: Optional[str] = None,
        tool_calling_mode: str = "text",
        **kwargs,
    ) -> AsyncGenerator[AssistantMessageEvent, None]:
        key = api_key or model.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base_url = model.base_url or self.DEFAULT_BASE_URL
        payload = self.construct_request(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            api_key=key,
            tool_calling_mode=tool_calling_mode,
            **kwargs,
        )

        logger.debug(
            "[QWEN-STREAM] start provider=%s model=%s base_url=%s has_api_key=%s tool_mode=%s payload=%s",
            model.provider,
            model.id,
            base_url,
            bool(key),
            tool_calling_mode,
            self._summarize_payload(payload),
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
            streamed_text_full = ""
            streamed_thinking_full = ""
            pending_tool_calls: Dict[str, Dict[str, str]] = {}
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
                logger.debug(
                    "[QWEN-PARSED] chunk=%s",
                    self._summarize_chunk(chunk_data),
                )

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
                        logger.debug(
                            "[QWEN-PARSED] text-delta status=%s len=%s preview=%r",
                            chunk_data.get("status"),
                            len(delta_text),
                            delta_text[:80],
                        )
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
                        text_tool_calls = self._extract_text_tool_calls(final_text, tools)
                        logger.debug(
                            "[QWEN-STREAM] completed-via-status total_text_len=%s usage=%s",
                            len(final_text),
                            partial.usage,
                        )
                        if chunk_context:
                            logger.info("[QWEN-STREAM] completed: %s", chunk_context)
                        yield StreamTextEndEvent(
                            content_index=text_index or 0,
                            content=final_text,
                            partial=partial,
                        )
                        for tool_call_obj in text_tool_calls:
                            async for tool_event in self._emit_qwen_tool_call(
                                partial,
                                tool_call_obj,
                            ):
                                yield tool_event
                        partial.stop_reason = "toolUse" if text_tool_calls else "stop"
                        if text_tool_calls:
                            logger.info(
                                "[QWEN-TOOLS] completed-via-status tool_calls=%s",
                                _summarize_tool_calls_for_log(text_tool_calls),
                            )
                        yield StreamDoneEvent(reason=partial.stop_reason, message=partial)
                        return

                    continue

                output = chunk_data.get("output") or {}
                choice = (output.get("choices") or [{}])[0]
                message = choice.get("message") or choice.get("delta") or {}
                finish_reason = choice.get("finish_reason") or output.get("finish_reason")

                reasoning_text = (
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or message.get("reasoning_text")
                )
                if reasoning_text:
                    thinking_delta, streamed_thinking_full = self._compute_stream_text_delta(
                        streamed_thinking_full,
                        reasoning_text,
                    )
                    if thinking_index is None:
                        partial.content.append(ThinkingContent(thinking=""))
                        thinking_index = len(partial.content) - 1
                        yield StreamThinkingStartEvent(
                            content_index=thinking_index,
                            partial=partial,
                        )
                    if thinking_delta:
                        thinking_block = partial.content[thinking_index]
                        if isinstance(thinking_block, ThinkingContent):
                            thinking_block.thinking += thinking_delta
                        yield StreamThinkingDeltaEvent(
                            content_index=thinking_index,
                            delta=thinking_delta,
                            partial=partial,
                        )
                    thinking_block = partial.content[thinking_index]
                    yield StreamThinkingEndEvent(
                        content_index=thinking_index,
                        content=thinking_block.thinking if isinstance(thinking_block, ThinkingContent) else streamed_thinking_full,
                        partial=partial,
                    )

                content_text = self._extract_message_text(message.get("content"))
                if content_text:
                    text_delta, streamed_text_full = self._compute_stream_text_delta(
                        streamed_text_full,
                        content_text,
                    )
                    if text_index is None:
                        partial.content.append(TextContent(text=""))
                        text_index = len(partial.content) - 1
                        yield StreamTextStartEvent(content_index=text_index, partial=partial)
                    if text_delta:
                        text_block = partial.content[text_index]
                        if isinstance(text_block, TextContent):
                            text_block.text += text_delta
                        yield StreamTextDeltaEvent(
                            content_index=text_index,
                            delta=text_delta,
                            partial=partial,
                        )

                tool_calls = message.get("tool_calls") or []
                for idx, tool_call in enumerate(tool_calls):
                    logger.debug(
                        "[QWEN-TOOLS] chunk tool_call=%s finish_reason=%s",
                        _summarize_tool_call_for_log(tool_call),
                        finish_reason,
                    )
                    if not isinstance(tool_call, dict):
                        continue
                    function_data = tool_call.get("function") or {}
                    call_index = tool_call.get("index")
                    call_id_raw = tool_call.get("id")
                    fallback_key = str(call_index) if call_index is not None else ""
                    if not fallback_key:
                        fallback_key = str(call_id_raw or idx)
                    pending = pending_tool_calls.setdefault(
                        fallback_key,
                        {"id": "", "name": "", "arguments": ""},
                    )

                    pending["id"] = self._merge_stream_fragment(pending["id"], call_id_raw)
                    pending["name"] = self._merge_stream_fragment(
                        pending["name"],
                        function_data.get("name") or tool_call.get("name"),
                    )
                    raw_arguments = function_data.get("arguments")
                    if isinstance(raw_arguments, dict):
                        pending["arguments"] = json.dumps(raw_arguments, ensure_ascii=False)
                    else:
                        pending["arguments"] = self._merge_stream_fragment(
                            pending["arguments"],
                            raw_arguments,
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
                logger.debug(
                    "[QWEN-PARSED] usage finish_reason=%s usage=%s",
                    finish_reason,
                    partial.usage,
                )

                if not finish_reason:
                    continue

                existing_tool_calls = [
                    c for c in partial.content if isinstance(c, ToolCall)
                ]
                text_tool_calls: List[ToolCall] = []
                if not existing_tool_calls and text_index is not None:
                    text_block = partial.content[text_index]
                    final_text_for_tools = (
                        text_block.text if isinstance(text_block, TextContent) else ""
                    )
                    text_tool_calls = self._extract_text_tool_calls(
                        final_text_for_tools,
                        tools,
                    )

                if text_index is not None:
                    text_block = partial.content[text_index]
                    final_text = text_block.text if isinstance(text_block, TextContent) else ""
                    yield StreamTextEndEvent(
                        content_index=text_index,
                        content=final_text,
                        partial=partial,
                    )

                for tool_call_obj in text_tool_calls:
                    async for tool_event in self._emit_qwen_tool_call(
                        partial,
                        tool_call_obj,
                    ):
                        yield tool_event

                streamed_tool_calls: List[ToolCall] = []
                existing_tool_call_keys = {
                    (
                        c.id,
                        c.name,
                        _safe_json_dumps(c.arguments),
                    )
                    for c in partial.content
                    if isinstance(c, ToolCall)
                }
                for pending in pending_tool_calls.values():
                    tool_name = pending.get("name", "")
                    if not tool_name:
                        logger.warning(
                            "[QWEN-TOOLS] skipped blank tool_call aggregated=%s",
                            pending,
                        )
                        continue
                    parsed_arguments, raw_arguments_text = self._parse_tool_call_arguments(
                        pending.get("arguments", "")
                    )
                    tool_call_obj = ToolCall(
                        id=pending.get("id", "") or f"call_{uuid.uuid4().hex}",
                        name=tool_name,
                        arguments=parsed_arguments,
                    )
                    dedupe_key = (
                        tool_call_obj.id,
                        tool_call_obj.name,
                        _safe_json_dumps(tool_call_obj.arguments),
                    )
                    if dedupe_key in existing_tool_call_keys:
                        continue
                    existing_tool_call_keys.add(dedupe_key)
                    streamed_tool_calls.append(tool_call_obj)
                    partial.content.append(tool_call_obj)
                    content_index = len(partial.content) - 1
                    yield StreamToolCallStartEvent(
                        content_index=content_index,
                        partial=partial,
                    )
                    yield StreamToolCallDeltaEvent(
                        content_index=content_index,
                        delta=raw_arguments_text,
                        partial=partial,
                    )
                    yield StreamToolCallEndEvent(
                        content_index=content_index,
                        tool_call=tool_call_obj,
                        partial=partial,
                    )

                if finish_reason == "tool_calls" or text_tool_calls or streamed_tool_calls:
                    logger.info(
                        "[QWEN-TOOLS] stop=toolUse finish_reason=%s tool_calls=%s",
                        finish_reason,
                        _summarize_tool_calls_for_log(
                            [c for c in partial.content if isinstance(c, ToolCall)]
                        ),
                    )
                    partial.stop_reason = "toolUse"
                elif finish_reason == "length":
                    partial.stop_reason = "length"
                else:
                    partial.stop_reason = "stop"

                logger.debug(
                    "[QWEN-STREAM] done stop_reason=%s content_count=%s",
                    partial.stop_reason,
                    len(partial.content),
                )
                yield StreamDoneEvent(reason=partial.stop_reason, message=partial)
                return

            if not completed:
                text_tool_calls: List[ToolCall] = []
                if text_index is not None:
                    text_block = partial.content[text_index]
                    final_text = text_block.text if isinstance(text_block, TextContent) else ""
                    text_tool_calls = self._extract_text_tool_calls(final_text, tools)
                    yield StreamTextEndEvent(
                        content_index=text_index,
                        content=final_text,
                        partial=partial,
                    )
                for tool_call_obj in text_tool_calls:
                    async for tool_event in self._emit_qwen_tool_call(
                        partial,
                        tool_call_obj,
                    ):
                        yield tool_event
                partial.stop_reason = "toolUse" if text_tool_calls else "stop"
                logger.debug(
                    "[QWEN-STREAM] ended-without-explicit-completed content_count=%s",
                    len(partial.content),
                )
                yield StreamDoneEvent(reason=partial.stop_reason, message=partial)

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            error_str = str(exc)
            partial.stop_reason = "error"
            partial.error_message = error_str
            yield StreamErrorEvent(reason="error", error=partial)
            logger.warning("[QWEN-ERROR] HTTP error: %s", error_str)
            if status_code == 401:
                raise LLMAuthenticationError(provider=model.provider) from exc
            if status_code == 429:
                raise LLMRateLimitError(provider=model.provider) from exc
            raise LLMConnectionError(provider=model.provider) from exc
        except requests.RequestException as exc:
            partial.stop_reason = "error"
            partial.error_message = str(exc)
            yield StreamErrorEvent(reason="error", error=partial)
            logger.warning("[QWEN-ERROR] request error: %s", exc)
            raise LLMConnectionError(provider=model.provider) from exc
        except Exception as exc:
            partial.stop_reason = "error"
            partial.error_message = str(exc)
            yield StreamErrorEvent(reason="error", error=partial)
            logger.warning("[QWEN-ERROR] stream error: %s", exc)
            raise

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
                request_headers = {
                    "Authorization": f"Bearer {self._mask_secret(api_key)}",
                    "Content-Type": "application/json",
                }
                logger.info(
                    "[QWEN-REQUEST] base_url=%s timeout=%s payload=%s",
                    base_url,
                    timeout,
                    self._summarize_payload(payload),
                )
                logger.info(
                    "[QWEN-REQUEST] headers:\n%s",
                    self._format_json_for_logging(request_headers),
                )
                logger.info(
                    "[QWEN-REQUEST] payload(full, masked):\n%s",
                    self._format_json_for_logging(
                        self._sanitize_payload_for_logging(payload)
                    ),
                )
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
                    logger.debug(
                        "[QWEN-RESPONSE] status=%s content_type=%s",
                        response.status_code,
                        response.headers.get("Content-Type"),
                    )
                    logger.debug(
                        "[QWEN-RESPONSE] headers:\n%s",
                        self._format_json_for_logging(dict(response.headers)),
                    )
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
                            logger.debug("[QWEN-RAW] received DONE marker")
                            continue
                        logger.debug(
                            "[QWEN-RAW] response line(full):\n%s",
                            self._truncate_for_logging(line),
                        )
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug(
                                "[QWEN-RAW] line is not valid JSON, skipped:\n%s",
                                self._truncate_for_logging(line),
                            )
                            continue
                        logger.debug(
                            "[QWEN-PARSED] response chunk(full):\n%s",
                            self._format_json_for_logging(parsed),
                        )
                        loop.call_soon_threadsafe(queue.put_nowait, parsed)
            except Exception as exc:
                logger.debug("[QWEN-ERROR] worker raised: %r", exc)
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

    logger.debug(
        "[LLM-STREAM] provider=%s model=%s messages=%s tool_count=%s tool_names=%s",
        model.provider,
        model.id,
        len(messages),
        len(tools or []),
        [tool.name for tool in (tools or [])],
    )

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
