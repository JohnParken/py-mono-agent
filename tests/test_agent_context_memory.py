import unittest

from pi_agent_core import Agent, AgentOptions
from pi_ai.types import AssistantMessage, TextContent, ToolResultMessage, UserMessage


class AgentContextMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_transform_context_keeps_short_history_under_budget(self):
        agent = Agent(
            AgentOptions(
                enable_context_memory=True,
                context_max_tokens=4096,
            )
        )
        messages = [
            UserMessage(content=[TextContent(text="hello")]),
            AssistantMessage(content=[TextContent(text="hi")]),
        ]

        transformed = await agent._transform_context(messages, None)

        self.assertEqual(len(transformed), 2)
        self.assertEqual(getattr(transformed[0], "role", ""), "user")
        self.assertEqual(getattr(transformed[1], "role", ""), "assistant")

    async def test_default_transform_context_builds_summary_and_trims_history(self):
        agent = Agent(
            AgentOptions(
                enable_context_memory=True,
                context_recent_messages=4,
                context_tool_results_to_keep=2,
                context_max_tokens=120,
                context_summary_max_chars=1200,
            )
        )
        messages = []
        for idx in range(12):
            messages.append(
                UserMessage(
                    content=[TextContent(text=f"user request {idx} with detailed constraints and acceptance criteria")]
                )
            )
            messages.append(
                AssistantMessage(
                    content=[TextContent(text=f"assistant answer {idx} with implementation notes and rationale")]
                )
            )
            if idx % 3 == 0:
                messages.append(
                    ToolResultMessage(
                        tool_call_id=f"call_{idx}",
                        tool_name="echo",
                        content=[TextContent(text=f"tool result payload {idx}")],
                    )
                )

        transformed = await agent._transform_context(messages, None)

        self.assertLess(len(transformed), len(messages))
        self.assertEqual(getattr(transformed[0], "role", ""), "user")
        summary_text = transformed[0].content[0].text
        self.assertIn("[MEMORY SUMMARY]", summary_text)
        self.assertTrue(
            "Latest Tool Findings" in summary_text
            or "Historical Tool Findings" in summary_text
        )
        # The newest user turn should remain in recent window.
        self.assertTrue(
            any(
                "user request 11" in content.text
                for message in transformed
                for content in getattr(message, "content", [])
                if getattr(content, "type", None) == "text"
            )
        )


if __name__ == "__main__":
    unittest.main()
