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


if __name__ == "__main__":
    unittest.main()
