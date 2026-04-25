# Text Fallback Tool Protocol

If a tool is needed, output one or more tool call blocks and nothing else.

Use this exact format for each tool call:
<tool_call>{"name":"<tool-name>","arguments":{...}}</tool_call>

When multiple tools are needed, output each in its own <tool_call> block on separate lines.

Do not add explanation before or after the tool call.
Do not wrap the JSON in markdown fences.
Use only tool names listed in <tools>.
Arguments must be valid JSON with double-quoted keys and strings.

Prefer one informative tool call per turn. If the path is unknown, search first.
If the path is known, read directly. Before editing, read the relevant file content first.
