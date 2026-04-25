---
name: qwen-provider-api
description: Use when generating, modifying, or reviewing Qwen provider code for either this Python repo or TypeScript code in pi-mono. Covers the custom JSON wrapper API, text-protocol tool calling, non-OpenAI request and response shapes, and includes self-contained TypeScript scaffolding for pi-mono-style provider generation.
---

# Qwen Provider API

Use this skill when implementing a Qwen provider that talks to a custom wrapper API instead
of a standard OpenAI-compatible endpoint.

This skill supports two targets:

- this Python repo, where the provider lives in `pi_ai/llm.py`
- the TypeScript `pi-mono` repo, where a custom provider would typically live under
  `packages/ai/src/providers/`

## When To Use This Skill

Use this skill when the task involves any of these:

- generating a new Qwen provider in TypeScript for `pi-mono`
- porting this repo's `QwenLLMProvider` logic into TypeScript
- adapting a custom proprietary wrapper API that is not OpenAI-compatible
- preserving text-protocol tool-calling behavior
- parsing nonstandard Qwen streaming chunks
- injecting prompt templates for text fallback without hardcoding them inline

## Read These Files First

### In this repo

- `pi_ai/llm.py`
- `pi_ai/qwen_request_template.json`
- `pi_ai/prompts/qwen_tools/shared_rules.md`
- `pi_ai/prompts/qwen_tools/text_rules.md`
- `tests/test_qwen_provider.py`
- `tests/test_qwen_prompt_builder.py`

### In pi-mono

Read these in the upstream repo before integrating generated TS code:

- `packages/ai/src/types.ts`
- `packages/ai/src/providers/openai-completions.ts`
- `packages/coding-agent/src/core/system-prompt.ts`

Use `openai-completions.ts` mainly as a reference for:

- `StreamFunction` shape
- `AssistantMessageEventStream` usage
- output event conventions (`text_start`, `text_delta`, `toolcall_start`, etc.)
- usage / stop reason handling

Do not copy its request shape for Qwen wrapper APIs.

## Important Difference From OpenAI

This Qwen integration is not a normal OpenAI-compatible wrapper.

- Do not assume request bodies follow `{ model, messages, tools, stream }`.
- Do not assume tools are always sent in a `tools` field.
- Do not assume responses only contain `choices[].delta`.
- Treat this wrapper as a text-protocol tool-calling integration.

If the API requires:

- `token`
- `apikey`
- `appInfo`
- `variable`
- `data.messages`

then build exactly that shape. Do not normalize it into OpenAI format just because the
target repo already has OpenAI providers.

## Core Concepts

### Request shape

This provider constructs a wrapper JSON payload. Typical top-level fields:

- `token`: resolved provider token
- `apikey`: optional app API key
- `modelId`: model id
- `appInfo`: wrapper-side settings
- `variable`: prompt variables, especially the `tools` prompt variable
- `data.messages`: message history in provider-specific shape

### Tool-calling mode

This skill assumes `tool_calling_mode="text"` only.

Rules:

- omit `data.tools`
- place tool definitions and protocol instructions into the `variable.tools` prompt
- require the model to emit `<tool_call>...</tool_call>` blocks in plain text
- keep the implementation and documentation scoped to text mode only

### Message mapping

- user messages can be structured content arrays
- assistant messages may include prior tool calls
- tool results map to tool-role messages

Keep one clear text-mode serializer in generated code.

### Response parsing

Support both of these response families:

1. status/result wrapper chunks
   - fields like `status`, `result`, `resCode`, `resMessage`
   - common statuses may include `running`, `success`, and `completed`
   - `result` should be treated as a streamed text chunk or delta, not a guaranteed final full response
   - do not assume `status="success"` means the stream is finished; it may still be only one incremental fragment
2. OpenAI-like output chunks
   - fields like `output.choices[0].message`, `finish_reason`, `tool_calls`

Do not simplify parsing to a single chunk shape.

## Request And Response Examples

Use concrete wrapper-shaped examples when generating code or reviewing payload handling.

### Sample request payload

This is the shape the provider should send in text mode:

```json
{
  "token": "dashscope-token",
  "apikey": "wrapper-app-key",
  "type": "txt",
  "modelId": "lightapplication",
  "appInfo": {
    "agent_id": "e76d09a-fed2-4ac1-9317-bf419f624c21",
    "sensitive_judge": false,
    "safe_model_judge": false,
    "max_new_tokens": 81920,
    "temperature": 0.3,
    "name": "Chat-Medium",
    "prompt": "(static_memory)\n(tools)"
  },
  "variable": [
    {
      "name": "tools",
      "value": "# Shared Tool Use Rules\n# Text Fallback Tool Protocol\n<tools>\n[{\"type\":\"function\",\"function\":{\"name\":\"echo\",\"description\":\"Echo text.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}}}]\n</tools>\nReturn tool calls inside <tool_call>...</tool_call>."
    },
    {
      "name": "static_memory",
      "value": "Project Facts:\n- Repository root is /workspace\n- Coding style: snake_case"
    }
  ],
  "data": {
    "messages": [
      {
        "role": "system",
        "content": "You are a tool-using assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Read config and summarize it."
          }
        ]
      }
    ],
    "stream": true
  }
}
```

Key expectations:

- `variable.tools` carries the tool schema and text protocol instructions
- `data.messages` stays in wrapper-specific message format
- `data.tools` is absent
- The request template's default `appInfo.prompt` is `"(tools)"`; when `static_memory` is provided, the provider dynamically prepends `"(static_memory)\n"` to the prompt without duplicating the placeholder

### Sample streaming response: status/result wrapper

Some deployments stream plain wrapper chunks like this. These chunks are incremental:

```json
{
  "status": "running",
  "result": "I will inspect the config first.\n",
  "resCode": "PLA0000",
  "resMessage": "OK",
  "questionId": "q-123",
  "sessionId": "s-456"
}
```

```json
{
  "status": "success",
  "result": "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hello\"}}</tool_call>",
  "resCode": "PLA0000",
  "resMessage": "OK",
  "questionId": "q-123",
  "sessionId": "s-456"
}
```

```json
{
  "status": "completed",
  "result": "",
  "resCode": "PLA0000",
  "resMessage": "OK",
  "questionId": "q-123",
  "sessionId": "s-456"
}
```

Parsing expectations:

- append each `result` fragment in arrival order
- keep reading while `status` is a non-terminal success state such as `running` or `success`
- do not finalize merely because a chunk says `success`
- finalize only when the wrapper emits an explicit terminal state such as `completed`
- after concatenation, extract any `<tool_call>...</tool_call>` block from the full text

### Sample streaming response: OpenAI-like output

Other deployments may stream OpenAI-like envelopes:

```json
{
  "output": {
    "choices": [
      {
        "message": {
          "role": "assistant",
          "content": "I will inspect the config first.\n<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hello\"}}</tool_call>"
        },
        "finish_reason": "stop"
      }
    ]
  }
}
```

The parser should read `output.choices[0].message.content`, preserve any visible text,
and still extract the embedded `<tool_call>` JSON safely.

## Prompt Rules

Keep prompt assembly outside the provider transport logic.

Separate:

- shared rules
- text-only protocol rules
- optional few-shot examples

In this Python repo, prompt assembly is already split into:

- `pi_ai/prompts/qwen_tools/shared_rules.md`
- `pi_ai/prompts/qwen_tools/text_rules.md`
- `pi_ai/prompts/qwen_tools/builder.py`

When generating TS code for `pi-mono`, preserve the same separation. Do not inline large
prompt templates into the provider file unless the user explicitly asks for a single-file version.

## TypeScript Guidance For pi-mono

In `pi-mono`, a good first implementation shape is:

- file: `packages/ai/src/providers/qwen-wrapper.ts`
- exported stream function:
  - `streamQwenWrapper`
  - `streamSimpleQwenWrapper`
- custom API name:
  - `"qwen-wrapper"` or another custom string API identifier

Prefer a dedicated provider file instead of trying to force this API into
`openai-completions.ts`.

### Minimal implementation plan in pi-mono

1. Define a custom options type extending `StreamOptions`
2. Build a wrapper payload from a JSON template or inline object
3. Add `buildMessages()` for text-mode history
4. Add `buildTools()` and prompt assembly helpers
5. Implement one streaming function that:
   - creates an `AssistantMessageEventStream`
   - initializes an `AssistantMessage` partial
   - streams and parses chunks
   - emits text/tool/thinking events
   - finalizes `done` or `error`
6. Add tests for:
   - request construction
   - text-mode prompt shape
   - text tool extraction

## TypeScript Provider Template

When generating TS code for `pi-mono`, read:

- `references/pi-mono-qwen-provider-template.ts`
- `references/pi-mono-integration-notes.md`

The template is intentionally self-contained so code generation does not need to infer the
whole implementation from this repo.

## Python Mapping Notes

If you are porting from this repo, preserve these semantic equivalents:

- `construct_request()` -> TS payload builder
- `_build_messages()` -> TS text-mode serializer
- `_extract_text_tool_calls()` -> TS text tool parser
- `stream()` -> TS `StreamFunction`
- `_stream_request()` -> TS fetch/SSE/JSON wrapper reader

## Tool Call Normalization Rules

Text-mode tool extraction should remain defensive.

Preserve these behaviors:

- normalize BOM and Windows newlines
- parse JSON candidates from free text
- accept:
  - `{"tool_calls":[...]}`
  - `{"tool_call": {...}}`
  - direct `{"name":"...","arguments":{...}}`
- accept arguments as either dict/object or JSON string
- reject tool names not in the available tool set
- deduplicate equivalent tool calls

## Safe Change Checklist

Before finishing a Qwen-related change, verify:

- request payload still matches the custom wrapper contract
- text mode still omits `data.tools`
- text prompt still contains `<tools>` and `<tool_call>` instructions
- wrapped JSON and Windows-formatted tool-call text still parse

## Tests To Run In This Repo

- `uv run python -m unittest tests.test_qwen_prompt_builder`
- `uv run python -m unittest tests.test_qwen_provider`
- `uv run python -m unittest tests.test_agent_loop_safety`

## What To Avoid

- do not rewrite the wrapper API into OpenAI chat format
- do not add extra tool-calling branches beyond text mode
- do not hide the text tool protocol in transport code without tests
- do not bury prompt templates inline in the transport layer
- do not assume pi-mono provider code can reuse OpenAI request construction unchanged
