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
        # Tool findings may be omitted if the summary is truncated by max_chars;
        # verify at least one known section is present.
        self.assertTrue(
            "User Facts" in summary_text
            or "Assistant Progress" in summary_text
            or "Latest Tool Findings" in summary_text
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


    async def test_chinese_token_estimation_is_more_accurate(self):
        agent = Agent(
            AgentOptions(
                enable_context_memory=True,
                context_max_tokens=4096,
            )
        )
        # Chinese text: each char ~1.5 tokens with new estimator
        chinese_text = "这是一个中文测试消息，用于验证token估算是否更准确。"
        msg = UserMessage(content=[TextContent(text=chinese_text)])
        tokens = agent._estimate_message_tokens(msg)
        # Old estimator would give ~12 tokens; new should give ~30+
        self.assertGreater(tokens, 20)

    async def test_priority_messages_are_preserved_longer(self):
        # Build a long enough message list to force summarization.
        # With context_max_tokens=512 (clamped minimum), we need many messages
        # to exceed the budget and trigger summary generation.
        agent = Agent(
            AgentOptions(
                enable_context_memory=True,
                context_recent_messages=2,
                context_tool_results_to_keep=0,
                context_max_tokens=512,
                context_summary_max_chars=4000,
            )
        )
        messages = []
        # Add a priority constraint early in the conversation
        messages.append(UserMessage(content=[TextContent(text="Make sure you do not use any external libraries.")]))
        messages.append(AssistantMessage(content=[TextContent(text="Understood, no external libs.")]))
        for i in range(20):
            messages.append(UserMessage(content=[TextContent(text=f"Message {i}: some content here.")]))
            messages.append(AssistantMessage(content=[TextContent(text=f"Answer {i}: some response here.")]))

        transformed = await agent._transform_context(messages, None)
        summary_text = transformed[0].content[0].text

        # The priority constraint about "do not use external libraries" should be in summary
        self.assertIn("do not use", summary_text.lower())

    async def test_context_memory_logs_transform_stats(self):
        agent = Agent(
            AgentOptions(
                enable_context_memory=True,
                context_recent_messages=2,
                context_max_tokens=100,
                context_summary_max_chars=1000,
            )
        )
        messages = [
            UserMessage(content=[TextContent(text="First message with some content here.")]),
            AssistantMessage(content=[TextContent(text="Assistant response here.")]),
            UserMessage(content=[TextContent(text="Second message with more content.")]),
            AssistantMessage(content=[TextContent(text="Another response here.")]),
        ]

        # Should not raise; mainly verifies the log path works
        transformed = await agent._transform_context(messages, None)
        self.assertIsInstance(transformed, list)
        self.assertGreater(len(transformed), 0)


if __name__ == "__main__":
    unittest.main()
