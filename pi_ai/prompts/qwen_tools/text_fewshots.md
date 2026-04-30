# Examples

## Example 1: Reading a known file

User: Read pi_code_agent/tools.py.

Assistant:
<tool_call>{"name":"read_file","arguments":{"path":"pi_code_agent/tools.py","start_line":1}}</tool_call>

---

## Example 2: Searching for code

User: Find where Qwen disables unsupported tool calling modes.

Assistant:
<tool_call>{"name":"search_code","arguments":{"pattern":"tool calling is disabled","path":".","literal":true}}</tool_call>

---

## Example 3: Listing files before reading

User: List the files in pi_agent_core and then read agent_loop.py.

Assistant:
<tool_call>{"name":"list_files","arguments":{"path":"pi_agent_core","recursive":false}}</tool_call>
<tool_call>{"name":"read_file","arguments":{"path":"pi_agent_core/agent_loop.py","start_line":1}}</tool_call>

---

## Example 4: Writing a new file

User: Create docs/todo.txt with three lines: alpha, beta, gamma.

Assistant:
<tool_call>{"name":"write_file","arguments":{"path":"docs/todo.txt","content_lines":["alpha","beta","gamma"]}}</tool_call>

---

## Example 5: Editing a file with multiple replacements

User: Update app.py so "foo" becomes "bar" and "hello" becomes "world".

Assistant:
<tool_call>{"name":"edit_file","arguments":{"path":"app.py","edits":[{"old_text":"foo","new_text":"bar"},{"old_text":"hello","new_text":"world"}]}}</tool_call>

---

## Example 6: Searching before editing

User: Fix the bug in the auth module.

Assistant:
<tool_call>{"name":"search_code","arguments":{"pattern":"def authenticate","path":".","literal":false}}</tool_call>

User: [Tool result shows auth.py contains the authenticate function]

Assistant:
<tool_call>{"name":"read_file","arguments":{"path":"auth.py","start_line":1}}</tool_call>

User: [Tool result shows file content]

Assistant:
<tool_call>{"name":"edit_file","arguments":{"path":"auth.py","edits":[{"old_text":"if not token:","new_text":"if not token or token == '':"}]}}</tool_call>

---

## Example 7: Running a command after changes

User: Add a new test case and run the tests.

Assistant:
<tool_call>{"name":"read_file","arguments":{"path":"tests/test_example.py","start_line":1}}</tool_call>

User: [Tool result shows file content]

Assistant:
<tool_call>{"name":"edit_file","arguments":{"path":"tests/test_example.py","edits":[{"old_text":"def test_addition():\n    assert add(2, 3) == 5","new_text":"def test_addition():\n    assert add(2, 3) == 5\n\ndef test_subtraction():\n    assert sub(5, 3) == 2"}]}}</tool_call>

User: [Tool result shows edit succeeded]

Assistant:
<tool_call>{"name":"run_command","arguments":{"command":"python -m pytest tests/test_example.py -v","timeout":60}}</tool_call>

---

## Example 8: Normal response (no tool needed)

User: What is the capital of France?

Assistant:
The capital of France is Paris.
