import unittest
import tempfile

from pydantic import BaseModel

from pi_ai.llm import QwenLLMProvider, Model
from pi_ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage
from pi_code_agent.tools import create_coding_tools


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
        self.assertIn("# Tools", payload["variable"][0]["value"])
        self.assertIn("<tools>", payload["variable"][0]["value"])
        self.assertIn("<tool_call>", payload["variable"][0]["value"])
        self.assertNotIn("tools", payload["data"])

    def test_construct_request_native_mode_uses_api_tools(self):
        native_model = Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct")

        payload = self.provider.construct_request(
            model=native_model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="native",
        )

        self.assertEqual(payload["data"]["tools"][0]["function"]["name"], "echo")
        native_prompt = payload["variable"][0]["value"]
        self.assertEqual(native_prompt, "")
        self.assertNotIn("<tool_call>", native_prompt)
        self.assertNotIn("<tools>", native_prompt)

    def test_construct_request_auto_mode_resolves_to_native_for_supported_models(self):
        native_model = Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct")

        payload = self.provider.construct_request(
            model=native_model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="auto",
        )

        self.assertEqual(payload["data"]["tools"][0]["function"]["name"], "echo")
        auto_prompt = payload["variable"][0]["value"]
        self.assertEqual(auto_prompt, "")
        self.assertNotIn("<tool_call>", auto_prompt)
        self.assertNotIn("<tools>", auto_prompt)

    def test_construct_request_native_mode_is_case_insensitive(self):
        native_model = Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct")

        payload = self.provider.construct_request(
            model=native_model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="NATIVE",
        )

        self.assertEqual(payload["data"]["tools"][0]["function"]["name"], "echo")
        native_prompt = payload["variable"][0]["value"]
        self.assertEqual(native_prompt, "")
        self.assertNotIn("<tool_call>", native_prompt)
        self.assertNotIn("<tools>", native_prompt)

    def test_construct_request_text_mode_does_not_inject_tool_interactions(self):
        assistant_message = AssistantMessage(
            content=[
                TextContent(text="I will use tool."),
                ToolCall(id="call_1", name="echo", arguments={"text": "hi"}),
            ]
        )

        payload = self.provider.construct_request(
            model=self.model,
            messages=[assistant_message],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="text",
        )

        tools_prompt = payload["variable"][0]["value"]
        self.assertIn("# Shared Tool Use Rules", tools_prompt)
        self.assertIn("# Text Fallback Tool Protocol", tools_prompt)
        self.assertIn("# Tools", tools_prompt)
        self.assertIn("<tools>", tools_prompt)
        self.assertIn("<tool_call>", tools_prompt)
        self.assertIn("# Examples", tools_prompt)
        self.assertNotIn("tools", payload["data"])

    def test_construct_request_text_mode_injects_static_memory_variable(self):
        payload = self.provider.construct_request(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="text",
            static_memory="Project Facts: keep naming style snake_case.",
        )

        variable_map = {item["name"]: item["value"] for item in payload["variable"]}
        self.assertIn("static_memory", variable_map)
        self.assertEqual(
            variable_map["static_memory"],
            "Project Facts: keep naming style snake_case.",
        )
        self.assertIn("(static_memory)", payload["appInfo"]["prompt"])
        self.assertIn("(tools)", payload["appInfo"]["prompt"])

    def test_construct_request_static_memory_prompt_placeholder_not_duplicated(self):
        payload = self.provider.construct_request(
            model=self.model,
            messages=[],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="text",
            static_memory="Stable context",
            app_info={"prompt": "(static_memory)\n(tools)"},
        )
        self.assertEqual(payload["appInfo"]["prompt"].count("(static_memory)"), 1)

    def test_construct_request_native_messages_match_openai_compatible_shape(self):
        assistant_message = AssistantMessage(
            content=[
                TextContent(text="I will inspect the file."),
                ToolCall(id="call_1", name="echo", arguments={"text": "hi"}),
            ]
        )
        tool_result = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="echo",
            content=[TextContent(text="ok")],
        )
        payload = self.provider.construct_request(
            model=Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct"),
            messages=[
                assistant_message,
                tool_result,
            ],
            system_prompt="You are a tool-using assistant.",
            tools=[DummyTool()],
            api_key="dash-token",
            tool_calling_mode="native",
        )

        native_messages = payload["data"]["messages"]
        self.assertEqual(native_messages[0]["role"], "assistant")
        self.assertIsInstance(native_messages[0].get("content"), str)
        self.assertIn("tool_calls", native_messages[0])
        self.assertEqual(native_messages[1]["role"], "tool")
        self.assertEqual(native_messages[1]["content"], "ok")
        self.assertEqual(native_messages[1]["tool_call_id"], "call_1")

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

    def test_parse_tool_call_arguments_extracts_nested_arguments_object(self):
        parsed_arguments, raw_text = self.provider._parse_tool_call_arguments(
            '{"name":"echo","arguments":{"text":"hello"}}',
            source="test",
        )

        self.assertEqual(parsed_arguments, {"text": "hello"})
        self.assertEqual(raw_text, '{"text": "hello"}')

    def test_extract_text_tool_calls_recovers_python_literal_argument_string(self):
        text = (
            '<tool_call>{"name":"echo","arguments":"{\'text\': \'hello\'}"}'
            "</tool_call>"
        )

        tool_calls = self.provider._extract_text_tool_calls(text, [DummyTool()])

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})

    def test_extract_text_tool_calls_recovers_embedded_json_in_argument_string(self):
        text = (
            '<tool_call>{"name":"echo","arguments":"call tool with payload '
            '{\\"text\\":\\"hello\\"} right now"}'
            "</tool_call>"
        )

        tool_calls = self.provider._extract_text_tool_calls(text, [DummyTool()])

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})

    def test_extract_text_tool_calls_marks_parse_error_for_bad_arguments(self):
        text = (
            '<tool_call>{"name":"echo","arguments":"{text: hello}"}'
            "</tool_call>"
        )

        tool_calls = self.provider._extract_text_tool_calls(text, [DummyTool()])

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].arguments, {})
        self.assertIsNotNone(tool_calls[0].parse_error)

    def test_construct_request_exposes_edit_file_edits_schema_in_text_and_native_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edit_tool = next(
                tool for tool in create_coding_tools(tmpdir) if tool.name == "edit_file"
            )
            native_model = Model(
                provider="QwenLLMprovider",
                id="qwen3.6-35b-a3b-instruct",
            )

            text_payload = self.provider.construct_request(
                model=self.model,
                messages=[],
                system_prompt="You are a tool-using assistant.",
                tools=[edit_tool],
                api_key="dash-token",
                tool_calling_mode="text",
            )
            native_payload = self.provider.construct_request(
                model=native_model,
                messages=[],
                system_prompt="You are a tool-using assistant.",
                tools=[edit_tool],
                api_key="dash-token",
                tool_calling_mode="native",
            )

            text_prompt = text_payload["variable"][0]["value"]
            self.assertIn('"name": "edit_file"', text_prompt)
            self.assertIn('"edits"', text_prompt)
            self.assertIn("single edit_file", text_prompt)

            native_schema = native_payload["data"]["tools"][0]["function"]["parameters"]
            self.assertIn("edits", native_schema["properties"])
            edit_block_schema = native_schema["$defs"]["EditBlock"]
            self.assertEqual(
                edit_block_schema["properties"]["old_text"]["type"],
                "string",
            )
            self.assertEqual(
                edit_block_schema["properties"]["new_text"]["type"],
                "string",
            )

    def test_extract_text_tool_calls_recovers_edit_file_edits_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edit_tool = next(
                tool for tool in create_coding_tools(tmpdir) if tool.name == "edit_file"
            )
            text = (
                '<tool_call>{"name":"edit_file","arguments":{"path":"app.py","edits":['
                '{"old_text":"foo","new_text":"bar"},'
                '{"old_text":"hello","new_text":"world"}]}}</tool_call>'
            )

            tool_calls = self.provider._extract_text_tool_calls(text, [edit_tool])

            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].name, "edit_file")
            self.assertEqual(tool_calls[0].arguments["path"], "app.py")
            self.assertEqual(
                tool_calls[0].arguments["edits"],
                [
                    {"old_text": "foo", "new_text": "bar"},
                    {"old_text": "hello", "new_text": "world"},
                ],
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

    async def test_stream_native_merges_incremental_tool_call_fragments(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": None,
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            {
                "output": {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": None,
                                            "arguments": '{"text":"hello"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
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
            model=Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct"),
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
        self.assertEqual(tool_calls[0].id, "call_1")
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})

    async def test_stream_handles_cumulative_text_chunks_without_duplicate_content(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": "Hel",
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
                                "content": "Hello",
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
                                "content": "Hello!",
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
        self.assertEqual(done_event.reason, "stop")
        text_blocks = [
            content for content in done_event.message.content if isinstance(content, TextContent)
        ]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0].text, "Hello!")

    async def test_stream_handles_cumulative_tool_calls_without_duplicates(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"text":"hel"}',
                                        },
                                    }
                                ]
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
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"text":"hello"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
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
        self.assertEqual(tool_calls[0].id, "call_1")
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

    async def test_stream_native_request_failure_falls_back_to_text_mode(self):
        native_model = Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct")

        async def fake_stream_request(**kwargs):
            payload = kwargs["payload"]
            if payload["data"].get("tools"):
                raise RuntimeError("native tools unsupported")
            yield {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '<tool_call>{"name":"echo","arguments":{"text":"fallback"}}'
                                    "</tool_call>"
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            }

        self.provider._stream_request = fake_stream_request

        events = []
        async for event in self.provider.stream(
            model=native_model,
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
        self.assertEqual(tool_calls[0].arguments, {"text": "fallback"})

    async def test_stream_native_blank_tool_calls_falls_back_to_text_mode(self):
        native_model = Model(provider="QwenLLMprovider", id="qwen3.6-35b-a3b-instruct")

        async def fake_stream_request(**kwargs):
            payload = kwargs["payload"]
            if payload["data"].get("tools"):
                yield {
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "id": "call_blank",
                                            "function": {
                                                "name": "",
                                                "arguments": {"text": "ignored"},
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                }
                return
            yield {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '<tool_call>{"name":"echo","arguments":{"text":"recovered"}}'
                                    "</tool_call>"
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            }

        self.provider._stream_request = fake_stream_request

        events = []
        async for event in self.provider.stream(
            model=native_model,
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
        self.assertEqual(tool_calls[0].arguments, {"text": "recovered"})

    async def test_stream_text_mode_re_raises_generic_errors_after_error_event(self):
        async def fake_stream_request(**kwargs):
            raise RuntimeError("boom")
            yield

        self.provider._stream_request = fake_stream_request

        events = []
        with self.assertRaisesRegex(RuntimeError, "boom"):
            async for event in self.provider.stream(
                model=self.model,
                messages=[],
                system_prompt="You are a tool-using assistant.",
                tools=None,
                api_key="dash-token",
            ):
                events.append(event)

        self.assertEqual([event.type for event in events], ["start", "error"])

    async def test_stream_accepts_explicit_text_tool_calling_mode_without_conflict(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            }
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
            tool_calling_mode="text",
        ):
            events.append(event)

        self.assertEqual(events[-1].type, "done")
        self.assertEqual(events[-1].reason, "stop")

    def test_extract_text_tool_calls_parses_multiple_tool_call_blocks(self):
        text = (
            '<tool_call>{"name":"echo","arguments":{"text":"hello"}}</tool_call>\n'
            '<tool_call>{"name":"echo","arguments":{"text":"world"}}</tool_call>'
        )

        tool_calls = self.provider._extract_text_tool_calls(text, [DummyTool()])

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})
        self.assertEqual(tool_calls[1].name, "echo")
        self.assertEqual(tool_calls[1].arguments, {"text": "world"})

    def test_extract_text_tool_calls_parses_markdown_wrapped_tool_calls(self):
        text = (
            '<tool_call>\n```json\n'
            '{"name":"echo","arguments":{"text":"markdown"}}\n'
            '```\n</tool_call>'
        )

        tool_calls = self.provider._extract_text_tool_calls(text, [DummyTool()])

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "markdown"})

    def test_extract_text_tool_calls_parses_multiple_markdown_wrapped_tool_calls(self):
        text = (
            '<tool_call>\n```json\n'
            '{"name":"echo","arguments":{"text":"first"}}\n'
            '```\n</tool_call>\n'
            '<tool_call>\n```json\n'
            '{"name":"echo","arguments":{"text":"second"}}\n'
            '```\n</tool_call>'
        )

        tool_calls = self.provider._extract_text_tool_calls(text, [DummyTool()])

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].arguments, {"text": "first"})
        self.assertEqual(tool_calls[1].arguments, {"text": "second"})

    async def test_stream_extracts_multiple_text_tool_calls(self):
        chunks = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '<tool_call>{"name":"echo","arguments":{"text":"hello"}}</tool_call>\n'
                                    '<tool_call>{"name":"echo","arguments":{"text":"world"}}</tool_call>'
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
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0].name, "echo")
        self.assertEqual(tool_calls[0].arguments, {"text": "hello"})
        self.assertEqual(tool_calls[1].name, "echo")
        self.assertEqual(tool_calls[1].arguments, {"text": "world"})


if __name__ == "__main__":
    unittest.main()
