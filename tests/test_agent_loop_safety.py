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


if __name__ == "__main__":
    unittest.main()
