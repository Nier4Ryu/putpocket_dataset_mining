from __future__ import annotations

import unittest

from putpocket_dataset_mining.cline_tools import parse_cline_tool_calls


class ClineToolParserTests(unittest.TestCase):
    def test_parse_original_cline_write_tool(self) -> None:
        text = """
I will update the file.
<write_to_file>
<path>solution.py</path>
<content>def add(a, b):
    return a + b
</content>
</write_to_file>
"""
        calls = parse_cline_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "write_to_file")
        self.assertEqual(calls[0].params["path"], "solution.py")
        self.assertIn("return a + b", calls[0].params["content"])

    def test_parse_rejects_json_actions(self) -> None:
        with self.assertRaises(Exception) as context:
            parse_cline_tool_calls('{"action":"write_file","path":"solution.py"}')
        self.assertEqual(context.exception.__class__.__name__, "ToolParseError")


if __name__ == "__main__":
    unittest.main()
