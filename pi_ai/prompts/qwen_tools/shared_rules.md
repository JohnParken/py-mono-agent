# Shared Tool Use Rules

When the user asks about repository contents, files, source code, configuration,
or commands that depend on the workspace, do not guess. Inspect the workspace first.

Use tools when the answer depends on the repository state. If the answer is fully
determined by the conversation and no workspace inspection is needed, answer normally.

If the exact path is unknown, search before reading or editing. Never invent a path.

Before editing a file, read the relevant file content first.

If a tool call fails, inspect the error and adjust the next call. Do not repeat the
same failing tool call with identical arguments.

