# Examples

User: Read pi_code_agent/tools.py.
Assistant:
<tool_call>{"name":"read_file","arguments":{"path":"pi_code_agent/tools.py","start_line":1}}</tool_call>

User: Find where Qwen falls back from native tool calling to text mode.
Assistant:
<tool_call>{"name":"search_code","arguments":{"pattern":"fallback to text mode","path":".","literal":true}}</tool_call>

User: Create docs/todo.txt with three lines: alpha, beta, gamma.
Assistant:
<tool_call>{"name":"write_file","arguments":{"path":"docs/todo.txt","content_lines":["alpha","beta","gamma"]}}</tool_call>

User: List the files in pi_agent_core and then read agent_loop.py.
Assistant:
<tool_call>{"name":"list_files","arguments":{"path":"pi_agent_core","recursive":false}}</tool_call>
<tool_call>{"name":"read_file","arguments":{"path":"pi_agent_core/agent_loop.py","start_line":1}}</tool_call>
