import unittest

from pydantic import BaseModel

from pi_ai.prompts.qwen_tools import build_qwen_tool_prompt


class EchoArgs(BaseModel):
    text: str


class DummyTool:
    name = "echo"
    description = "Echo text."
    parameters = EchoArgs


class QwenPromptBuilderTests(unittest.TestCase):
    def test_text_prompt_contains_protocol_tools_and_examples(self):
        api_tools = [
            {
                "type": "function",
                "function": {
                    "name": DummyTool.name,
                    "description": DummyTool.description,
                    "parameters": DummyTool.parameters.model_json_schema(),
                },
            }
        ]

        prompt = build_qwen_tool_prompt(
            "You are a tool-using assistant.",
            api_tools,
            mode="text",
            include_fewshots=True,
        )

        self.assertIn("You are a tool-using assistant.", prompt)
        self.assertIn("# Shared Tool Use Rules", prompt)
        self.assertIn("# Text Fallback Tool Protocol", prompt)
        self.assertIn("<tools>", prompt)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("# Examples", prompt)
        self.assertIn('"name": "echo"', prompt)

    def test_unsupported_mode_is_aliased_to_text_prompt(self):
        prompt = build_qwen_tool_prompt(
            "You are a tool-using assistant.",
            [],
            mode="native",
            include_fewshots=False,
        )

        self.assertIn("You are a tool-using assistant.", prompt)
        self.assertIn("# Shared Tool Use Rules", prompt)
        self.assertIn("# Text Fallback Tool Protocol", prompt)
        self.assertIn("<tool_call>", prompt)


if __name__ == "__main__":
    unittest.main()
