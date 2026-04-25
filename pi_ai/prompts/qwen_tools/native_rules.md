# Native Tool Calling Rules

Use the native tool-calling capability whenever repository inspection or file
operations are needed.

Prefer exactly one high-value tool call at a time instead of chaining multiple
uncertain calls together.

Use list_files to inspect directories, search_code to locate files or symbols when
the path is unknown, and read_file when the path is known.

Use edit_file only after reading the file and only for targeted replacements.
Use write_file for new files or full rewrites. For multiline content, prefer
content_lines when appropriate.

After code changes, use run_command when validation would reduce uncertainty.

