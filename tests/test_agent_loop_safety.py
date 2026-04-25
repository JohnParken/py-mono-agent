import unittest

from pydantic import BaseModel

from pi_agent_core.agent_loop import agent_loop
from pi_agent_core.types import AgentContext, AgentLoopConfig, AgentTool, AgentToolResult
from pi_ai.llm import (
    Model,
    StreamDoneEvent,
    StreamResponse,
    StreamStartEvent,
    StreamToolCallEndEvent,
    StreamToolCallStartEvent,
)
from pi_ai.types import AssistantMessage, TextContent, ToolCall, UserMessage


class EchoArgs(BaseModel):
    text: str


class LooseEchoArgs(BaseModel):
    text: str = "default"


async def echo_tool(tool_call_id, args, cancel_event, update_callback):
    return AgentToolResult(content=[TextContent(text=args.text)])


def make_echo_tool():
    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo text.",
        parameters=EchoArgs,
        execute=echo_tool,
    )


def make_loose_echo_tool(execute_fn):
    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo text.",
        parameters=LooseEchoArgs,
        execute=execute_fn,
    )


async def repeated_tool_stream(model, context, **options):
    partial = AssistantMessage(
        content=[],
        provider=model.provider,
        model=model.id,
        stop_reason="toolUse",
    )

    async def generate():
        yield StreamStartEvent(partial=partial)
        tool_call = ToolCall(id="call_1", name="echo", arguments={"text": "again"})
        partial.content.append(tool_call)
        yield StreamToolCallStartEvent(content_index=0, partial=partial)
        yield StreamToolCallEndEvent(
            content_index=0,
            tool_call=tool_call,
            partial=partial,
        )
        yield StreamDoneEvent(reason="toolUse", message=partial)

    return StreamResponse(generate())


async def parse_error_tool_stream(model, context, **options):
    has_tool_result = any(getattr(m, "role", None) == "toolResult" for m in context["messages"])

    partial = AssistantMessage(
        content=[],
        provider=model.provider,
        model=model.id,
        stop_reason="toolUse" if not has_tool_result else "stop",
    )

    async def generate():
        yield StreamStartEvent(partial=partial)
        if not has_tool_result:
            tool_call = ToolCall(
                id="call_parse_error",
                name="echo",
                arguments={},
                parse_error="Failed to parse tool arguments for source=text:echo.",
            )
            partial.content.append(tool_call)
            yield StreamDoneEvent(reason="toolUse", message=partial)
            return

        partial.content.append(TextContent(text="done"))
        yield StreamDoneEvent(reason="stop", message=partial)

    return StreamResponse(generate())


class AgentLoopSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_tool_call_stops_loop(self):
        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[make_echo_tool()],
        )
        prompt = UserMessage(content=[TextContent(text="repeat")])
        config = AgentLoopConfig(
            model=Model(provider="test", id="test"),
            max_tool_iterations=8,
        )

        stream = agent_loop(
            prompts=[prompt],
            context=context,
            config=config,
            stream_fn=repeated_tool_stream,
        )

        messages = []
        async for event in stream:
            if event.type == "message_end" and event.message is not None:
                messages.append(event.message)

        final_texts = [
            content.text
            for message in messages
            for content in getattr(message, "content", [])
            if getattr(content, "type", None) == "text"
        ]
        self.assertTrue(
            any("repeated the same tool call" in text for text in final_texts)
        )

    async def test_strict_tool_arguments_returns_structured_error_and_skips_execution(self):
        called = {"value": False}

        async def loose_echo_tool(tool_call_id, args, cancel_event, update_callback):
            called["value"] = True
            return AgentToolResult(content=[TextContent(text=args.text)])

        context = AgentContext(
            system_prompt="",
            messages=[],
            tools=[make_loose_echo_tool(loose_echo_tool)],
        )
        prompt = UserMessage(content=[TextContent(text="run strict parse check")])
        config = AgentLoopConfig(
            model=Model(provider="test", id="test"),
            strict_tool_arguments=True,
        )

        stream = agent_loop(
            prompts=[prompt],
            context=context,
            config=config,
            stream_fn=parse_error_tool_stream,
        )

        tool_results = []
        async for event in stream:
            if event.type == "message_end" and getattr(event.message, "role", None) == "toolResult":
                tool_results.append(event.message)

        self.assertFalse(called["value"])
        self.assertEqual(len(tool_results), 1)
        self.assertTrue(tool_results[0].is_error)
        self.assertEqual(tool_results[0].details.get("error_type"), "tool_arguments_parse_error")


if __name__ == "__main__":
    unittest.main()
