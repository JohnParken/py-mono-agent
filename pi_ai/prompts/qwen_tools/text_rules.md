# Text Fallback Tool Protocol

When a tool is needed, you MUST output tool call blocks using the exact format below. Do not add any explanatory text before or after the tool call blocks.

## Output Format

Use this exact format for each tool call:

```
<tool_call>{"name":"<tool-name>","arguments":{...}}</tool_call>
```

- The entire tool call must be on a single line.
- `name` must be one of the tool names listed in the `<tools>` section below.
- `arguments` must be a valid JSON object with double-quoted keys and double-quoted string values.
- Do not wrap the JSON in markdown code fences (```).
- Do not add any text, explanation, or commentary before or after the `<tool_call>` block.

## Multiple Tool Calls

When multiple tools are needed in the same turn, output each in its own `<tool_call>` block on separate lines:

```
<tool_call>{"name":"list_files","arguments":{"path":"src","recursive":false}}</tool_call>
<tool_call>{"name":"read_file","arguments":{"path":"src/main.py","start_line":1}}</tool_call>
```

## Constraints

- If a tool is needed, output ONLY `<tool_call>` blocks and nothing else.
- If no tool is needed, respond with normal text directly.
- Never mix explanatory text with tool calls in the same turn.
- Use only tool names listed in `<tools>`.
- Arguments must be valid JSON with double-quoted keys and strings.
- Prefer one informative tool call per turn. If the path is unknown, search first.
- If the path is known, read directly. Before editing, read the relevant file content first.
- When changing multiple separate locations in one file, prefer a single `edit_file` call with an `edits[]` array instead of multiple `edit_file` calls.
