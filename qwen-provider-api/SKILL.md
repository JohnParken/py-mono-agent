---
name: qwen-provider-api
description: Use when generating, modifying, or reviewing Qwen provider code for either this Python repo or TypeScript code in pi-mono. Covers the custom JSON wrapper API, native vs text tool-calling modes, non-OpenAI request and response shapes, and includes self-contained TypeScript scaffolding for pi-mono-style provider generation.
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
- preserving native/text/auto tool-calling behavior
- parsing nonstandard Qwen streaming chunks
- injecting prompt templates for text fallback without hardcoding them inline

## Read These Files First

### In this repo

- `pi_ai/llm.py`
- `pi_ai/qwen_request_template.json`
- `pi_ai/prompts/qwen_tools/shared_rules.md`
- `pi_ai/prompts/qwen_tools/native_rules.md`
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
- Do not assume native and text modes share the same prompt contract.

If the API requires:

- `token`
- `apikey`
- `appInfo`
- `variable`
- `data.messages`
- `data.tools`

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
- `data.tools`: native tool schema list, only in native mode

### Tool-calling modes

`tool_calling_mode` should be treated as a runtime option:

- `auto`
- `native`
- `text`

Rules:

- `native`: send actual tool schemas in `data.tools`
- `text`: omit `data.tools` and make the model emit `<tool_call>...</tool_call>`
- `auto`: try native first, then fall back to text if native fails or returns tool-use
  without usable structured tool calls

### Message mapping

Keep separate serializers for native vs text mode.

For text mode:

- user messages can be structured content arrays
- assistant messages may include prior tool calls
- tool results map to tool-role messages

For native mode:

- user content is usually flattened to text
- assistant text is flattened to string
- assistant tool calls go into `tool_calls`
- tool results become plain text tool messages

Do not casually merge these serializers.

### Response parsing

Support both of these response families:

1. status/result wrapper chunks
   - fields like `status`, `result`, `resCode`, `resMessage`
2. OpenAI-like output chunks
   - fields like `output.choices[0].message`, `finish_reason`, `tool_calls`

Do not simplify parsing to a single chunk shape.

## Prompt Rules

Keep prompt assembly outside the provider transport logic.

Separate:

- shared rules
- native-only rules
- text-only protocol rules
- optional few-shot examples

In this Python repo, prompt assembly is already split into:

- `pi_ai/prompts/qwen_tools/shared_rules.md`
- `pi_ai/prompts/qwen_tools/native_rules.md`
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
3. Add separate `buildMessages()` and `buildNativeMessages()`
4. Add `buildTools()` and prompt assembly helpers
5. Implement `resolveToolCallingMode()`
6. Implement one streaming function that:
   - creates an `AssistantMessageEventStream`
   - initializes an `AssistantMessage` partial
   - streams and parses chunks
   - emits text/tool/thinking events
   - finalizes `done` or `error`
7. Add fallback from native to text
8. Add tests for:
   - request construction
   - native/text prompt shape
   - native fallback to text
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
- `_build_native_messages()` -> TS native-mode serializer
- `_extract_text_tool_calls()` -> TS text tool parser
- `_resolve_tool_calling_mode()` -> TS runtime mode resolver
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
- native mode still sends `data.tools`
- text mode still omits `data.tools`
- native prompt does not contain text protocol instructions
- text prompt still contains `<tools>` and `<tool_call>` instructions
- native-to-text fallback still works
- blank native tool calls still trigger fallback recovery
- wrapped JSON and Windows-formatted tool-call text still parse

## Tests To Run In This Repo

- `uv run python -m unittest tests.test_qwen_prompt_builder`
- `uv run python -m unittest tests.test_qwen_provider`
- `uv run python -m unittest tests.test_agent_loop_safety`

## What To Avoid

- do not rewrite the wrapper API into OpenAI chat format
- do not collapse native/text prompt logic into one undifferentiated template
- do not remove fallback behavior without updating tests and product assumptions
- do not bury prompt templates inline in the transport layer
- do not assume pi-mono provider code can reuse OpenAI request construction unchanged

