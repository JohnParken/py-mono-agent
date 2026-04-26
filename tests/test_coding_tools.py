import tempfile
import unittest
from pathlib import Path

from pi_code_agent.tools import create_coding_tools


class CodingToolsTests(unittest.IsolatedAsyncioTestCase):
    def _get_tool(self, tools, name):
        return next(tool for tool in tools if tool.name == name)

    async def test_write_file_accepts_content_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = create_coding_tools(tmpdir)
            write_file = self._get_tool(tools, "write_file")
            args = write_file.parameters(
                path="nested/example.txt",
                content_lines=["alpha", "beta", "gamma"],
            )

            result = await write_file.execute("call_1", args, None, None)

            self.assertEqual(
                Path(tmpdir, "nested/example.txt").read_text(encoding="utf-8"),
                "alpha\nbeta\ngamma",
            )
            self.assertIn("Wrote nested/example.txt", result.content[0].text)

    async def test_write_file_rejects_content_and_content_lines_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = create_coding_tools(tmpdir)
            write_file = self._get_tool(tools, "write_file")
            args = write_file.parameters(
                path="example.txt",
                content="alpha",
                content_lines=["beta"],
            )

            with self.assertRaisesRegex(ValueError, "either content or content_lines"):
                await write_file.execute("call_1", args, None, None)

    async def test_edit_file_supports_multiple_edits_and_preserves_bom_and_crlf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "sample.txt")
            target.write_bytes("\ufeffalpha\r\nbeta\r\ngamma\r\n".encode("utf-8"))

            tools = create_coding_tools(tmpdir)
            edit_file = self._get_tool(tools, "edit_file")
            args = edit_file.parameters(
                path="sample.txt",
                edits=[
                    {"old_text": "alpha", "new_text": "ALPHA"},
                    {"old_text": "gamma", "new_text": "GAMMA"},
                ],
            )

            result = await edit_file.execute("call_1", args, None, None)

            self.assertEqual(
                target.read_bytes(),
                "\ufeffALPHA\r\nbeta\r\nGAMMA\r\n".encode("utf-8"),
            )
            self.assertIn("Successfully replaced 2 block(s)", result.content[0].text)
            self.assertIn("--- a/sample.txt", result.details["diff"])
            self.assertEqual(result.details["first_changed_line"], 1)

    async def test_edit_file_rejects_ambiguous_legacy_edit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir, "sample.txt")
            target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")

            tools = create_coding_tools(tmpdir)
            edit_file = self._get_tool(tools, "edit_file")
            args = edit_file.parameters(
                path="sample.txt",
                old_text="alpha",
                new_text="ALPHA",
            )

            with self.assertRaisesRegex(ValueError, "matched 2 times"):
                await edit_file.execute("call_1", args, None, None)

    async def test_run_command_streams_updates_and_reports_truncation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = create_coding_tools(tmpdir)
            run_command = self._get_tool(tools, "run_command")
            args = run_command.parameters(
                command="yes x | head -n 300",
                timeout=5,
            )

            updates = []

            def on_update(result):
                updates.append(result)

            result = await run_command.execute("call_1", args, None, on_update)

            self.assertGreaterEqual(len(updates), 1)
            self.assertIsNotNone(result.details)
            self.assertIsNotNone(result.details["truncation"])
            self.assertTrue(result.details["full_output_path"])
            self.assertTrue(Path(result.details["full_output_path"]).exists())
            self.assertIn("Showing lines", result.content[0].text)

    async def test_run_command_raises_on_non_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tools = create_coding_tools(tmpdir)
            run_command = self._get_tool(tools, "run_command")
            args = run_command.parameters(
                command="printf 'boom'; exit 7",
                timeout=5,
            )

            with self.assertRaisesRegex(RuntimeError, "exit 7|code 7"):
                await run_command.execute("call_1", args, None, None)


if __name__ == "__main__":
    unittest.main()
