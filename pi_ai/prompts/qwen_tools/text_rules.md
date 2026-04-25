# Text Fallback Tool Protocol

If a tool is needed, output exactly one tool call block and nothing else.

Use this exact format:
<tool_call>{"name":"<tool-name>","arguments":{...}}</tool_call>

Do not add explanation before or after the tool call.
Do not wrap the JSON in markdown fences.
Use only tool names listed in <tools>.
Arguments must be valid JSON with double-quoted keys and strings.

Prefer one informative tool call per turn. If the path is unknown, search first.
If the path is known, read directly. Before editing, read the relevant file content.

