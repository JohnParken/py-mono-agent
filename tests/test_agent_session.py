import tempfile
import unittest
from pathlib import Path

from pi_ai import Model
from pi_ai.llm import (
    StreamDoneEvent,
    StreamResponse,
    StreamStartEvent,
    StreamTextDeltaEvent,
    StreamTextEndEvent,
    StreamTextStartEvent,
)
from pi_ai.types import AssistantMessage, TextContent
from pi_code_agent.session import AgentSession, SessionCreateOptions


async def _fake_stream_fn(model, context, **options):
    _ = model
    _ = options

    async def _events():
        final = AssistantMessage(
            content=[TextContent(text="session-ok")],
            provider="fake",
            model="fake-model",
            stop_reason="stop",
        )
        partial = AssistantMessage(
            content=[],
            provider="fake",
            model="fake-model",
            stop_reason="stop",
        )
        yield StreamStartEvent(partial=partial)
        partial.content.append(TextContent(text=""))
        yield StreamTextStartEvent(content_index=0, partial=partial)
        partial.content[0].text += "session-ok"
        yield StreamTextDeltaEvent(content_index=0, delta="session-ok", partial=partial)
        yield StreamTextEndEvent(content_index=0, content="session-ok", partial=partial)
        yield StreamDoneEvent(reason="stop", message=final)

    return StreamResponse(_events())


class AgentSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_prompt_persists_and_restores_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            session = AgentSession.create(
                SessionCreateOptions(
                    workspace=workspace,
                    model_instance=Model(provider="fake", id="fake-model"),
                    stream_fn=_fake_stream_fn,
                    custom_system_prompt="You are a test agent.",
                )
            )

            await session.prompt("hello session")

            stats = session.get_stats()
            self.assertEqual(stats.user_messages, 1)
            self.assertEqual(stats.assistant_messages, 1)
            self.assertTrue(Path(stats.session_file).exists())

            restored = AgentSession.create(
                SessionCreateOptions(
                    workspace=workspace,
                    session_file=Path(stats.session_file),
                    model_instance=Model(provider="fake", id="fake-model"),
                    stream_fn=_fake_stream_fn,
                    custom_system_prompt="You are a test agent.",
                )
            )

            restored_messages = restored.agent.state.messages
            self.assertEqual(len(restored_messages), 2)
            self.assertEqual(getattr(restored_messages[0], "role", None), "user")
            self.assertEqual(getattr(restored_messages[1], "role", None), "assistant")
            self.assertEqual(restored_messages[1].content[0].text, "session-ok")
