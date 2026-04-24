import unittest

from pydantic import BaseModel

from pi_ai.llm import QwenLLMProvider, Model
from pi_ai.types import AssistantMessage, TextContent, ToolCall


class EchoArgs(BaseModel):
    text: str


class DummyTool:
    name = "echo"
    description = "Echo text."
    parameters = EchoArgs


class QwenProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = QwenLLMProvider()
        self.model = Model(provider="QwenLLMprovider", id="lightapplication")

    def test_default_request_template_loads_from_resource(self):
        template = self.provider.get_request_template_json()

        self.assertIn('"prompt": "(tools)"', template)
        self.assertIn('"name": "tools"', template)

    def test_construct_request_populates_keys_model_and_app_info(self):
        payload = self.provider.construct_request(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            app_api_key="app-token",
            app_info={"agent_id": "custom-agent", "temperature": 0.1},
        )

        self.assertEqual(payload["token"], "dash-token")
        self.assertEqual(payload["apikey"], "app-token")
        self.assertEqual(payload["modelId"], "lightapplication")
        self.assertEqual(payload["appInfo"]["agent_id"], "custom-agent")
        self.assertEqual(payload["appInfo"]["temperature"], 0.1)
        self.assertIn("Tools description:", payload["variable"][0]["value"])
        self.assertIn("<tool_call>", payload["variable"][0]["value"])
        self.assertNotIn("tools", payload["data"])

    def test_build_messages_keeps_assistant_tool_calls(self):
        assistant_message = AssistantMessage(
            content=[
                TextContent(text="I will inspect the file."),
                ToolCall(id="call_1", name="echo", arguments={"text": "hi"}),
            ]
        )

        messages = self.provider._build_messages([assistant_message], system_prompt="")

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["name"], "echo")
        self.assertEqual(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            '{"text": "hi"}',
        )

    async def test_stream_waits_for_finish_reason_and_accepts_dict_arguments(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": "Inspecting repository...",
                            }
                        }
                    ]
                }
            },
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "echo",
                                            "arguments": {"text": "hello"},
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ]

        async def fake_stream_request(**kwargs):
            for chunk in chunks:
                yield chunk

        self.provider._stream_request = fake_stream_request

        events = []
        async for event in self.provider.stream(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
        ):
            events.append(event)

        done_event = events[-1]
        self.assertEqual(done_event.type, "done")
        self.assertEqual(done_event.reason, "toolUse")

        tool_calls = [
            content for content in done_event.message.content if isinstance(content, ToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})

    async def test_stream_skips_blank_native_tool_calls(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call_blank",
                                        "function": {
                                            "name": "",
                                            "arguments": {"text": "hello"},
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            },
        ]

        async def fake_stream_request(**kwargs):
            for chunk in chunks:
                yield chunk

        self.provider._stream_request = fake_stream_request

        events = []
        async for event in self.provider.stream(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
        ):
            events.append(event)

        done_event = events[-1]
        self.assertEqual(done_event.type, "done")
        self.assertEqual(done_event.reason, "toolUse")
        tool_calls = [
            content for content in done_event.message.content if isinstance(content, ToolCall)
        ]
        self.assertEqual(tool_calls, [])

    async def test_stream_extracts_text_json_tool_calls(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"tool_calls":[{"id":"call_1","function":'
                                    '{"name":"echo","arguments":{"text":"hello"}}}]}'
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            },
        ]

        async def fake_stream_request(**kwargs):
            for chunk in chunks:
                yield chunk

        self.provider._stream_request = fake_stream_request

        events = []
        async for event in self.provider.stream(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
        ):
            events.append(event)

        done_event = events[-1]
        self.assertEqual(done_event.type, "done")
        self.assertEqual(done_event.reason, "toolUse")

        tool_calls = [
            content for content in done_event.message.content if isinstance(content, ToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})

    async def test_stream_extracts_windows_wrapped_tool_calls(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "\ufeff<tool_call>\r\n"
                                    "```json\r\n"
                                    '{"name":"echo","arguments":{"text":"hello"}}\r\n'
                                    "```\r\n"
                                    "</tool_call>"
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            },
        ]

        async def fake_stream_request(**kwargs):
            for chunk in chunks:
                yield chunk

        self.provider._stream_request = fake_stream_request

        events = []
        async for event in self.provider.stream(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
        ):
            events.append(event)

        done_event = events[-1]
        self.assertEqual(done_event.reason, "toolUse")

        tool_calls = [
            content for content in done_event.message.content if isinstance(content, ToolCall)
        ]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
