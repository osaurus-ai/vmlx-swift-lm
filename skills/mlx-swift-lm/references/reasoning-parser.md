# Reasoning Parser (`<think>...</think>` streaming)

## Overview

Several modern open-weight families emit chain-of-thought inside `<think>...</think>` tags that the tokenizer deliberately marks as `special: false`, so every consumer sees them as literal text. `ReasoningParser` is the streaming-safe splitter that peels the reasoning region out of a token-by-token stream.

**Files:**
- `Libraries/MLXLMCommon/ReasoningParser.swift` — the parser value type
- `Libraries/MLXLMCommon/JangLoader.swift` — `ParserResolution.reasoning(capabilities:modelType:)` façade
- `Libraries/MLXLMCommon/Evaluate.swift` — wired into `TextToolTokenLoopHandler`
- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift` — wired into `BatchEngine.generate(...)`

## Quick Reference

| Type / method | Purpose |
|---|---|
| `public struct ReasoningParser: Sendable` | Streaming splitter. Value type — pass by copy and write back after `.feed` to keep state. |
| `init(startTag: String = "<think>", endTag: String = "</think>")` | Defaults work for Qwen 3.5 / 3.6 / DeepSeek-R1 / GLM 4.x / Nemotron-H. |
| `mutating func feed(_ chunk: String) -> [ReasoningSegment]` | Per-chunk streaming. Returns zero or more segments. |
| `mutating func flush() -> [ReasoningSegment]` | End-of-stream drain. Anything still buffered becomes `.content` (or `.reasoning` if we're mid-`<think>`). |
| `enum ReasoningSegment: Sendable, Equatable { case content(String), case reasoning(String) }` | `.content` → visible answer; `.reasoning` → think-pane affordance. |
| `static fromCapabilityName(_: String?) -> ReasoningParser?` | JANG-stamp → parser. `"qwen3"`/`"qwen3_5"`/`"qwen3_6"`/`"deepseek_r1"`/`"glm4"`/`"nemotron"`/`"minimax"`/`"think_xml"` → `ReasoningParser()`. `"none"`/`"mistral"`/`"gemma4"` → `nil`. |
| `static split(_ text: String, startTag: =, endTag: =) -> (reasoning: String, content: String)` | One-shot non-streaming convenience. |

## Streaming contract

Token streams arrive fragmented — `<think>` may split across chunks as `<thi` + `nk>`. The parser **holds back** the minimum tail (up to `max(startTag.count, endTag.count) - 1`) while streaming, so a tag split across chunks isn't mistakenly emitted as content. On `flush()`, the holdback is drained.

```swift
var parser = ReasoningParser()  // <think>/</think> default
for chunk in stream {
    for segment in parser.feed(chunk) {
        switch segment {
        case .content(let s):   appendVisibleAnswer(s)
        case .reasoning(let s): appendThinkPane(s)
        }
    }
}
for segment in parser.flush() { /* same dispatch */ }
```

## JANG capability auto-pickup

JANG / JANGTQ bundles ship a `jang_config.json` whose `capabilities.reasoning_parser` stamp tells downstream consumers which parser to use. `ParserResolution.reasoning(capabilities:modelType:)` picks the right parser with this priority:

1. Stamp present → honour exactly (including `"none"` meaning *no reasoning*).
2. No stamp, but `model_type` looks like `mistral*` / `gemma*` → `nil`.
3. Otherwise → default `ReasoningParser()`.

Both `LLMModelFactory` and `VLMModelFactory` stamp `ModelConfiguration.reasoningParserName` at load time so the rest of the pipeline can resolve the parser without re-reading disk. `BatchEngine.generate(...)` and `Evaluate.generate(...)` both call `ReasoningParser.fromCapabilityName(context.configuration.reasoningParserName)` during setup and pipeline each decoded chunk through it **before** the tool-call processor.

## How `<think>` ties into `ToolCallProcessor`

The streaming pipeline is:

```
detokenized text chunk
    → ReasoningParser.feed(_:)        // strips <think>…</think>
    → ToolCallProcessor.processChunk  // extracts .toolCall, returns user-visible text
    → emit .chunk(String) / .toolCall(ToolCall)
```

`Generation` has no `.reasoning(String)` case — the library silently drops reasoning to stay byte-compatible with upstream ml-explore/mlx-swift-lm. Consumers that want a think-pane should either:

- Hold their own `ReasoningParser` alongside `generateTokens(...)` raw token stream, or
- Call `ReasoningParser.split(_:)` on accumulated assistant text after the stream finishes (non-streaming reveal).

## osaurus integration pattern

`osaurus/Packages/OsaurusCore/Utils/StreamingDeltaProcessor.swift` holds a `vmlxReasoningParser: ReasoningParser?` field, copies into a local `var parser = vmlxReasoningParser`, calls `parser.feed(text)` then writes back (`vmlxReasoningParser = parser`). This is the canonical value-type streaming pattern — each `feed` mutates internal buffer state, so you must reassign after every call. On stream end osaurus calls `parser.flush()` once.

`JANGReasoningResolver.resolve(modelKey:directory:)` caches a `Resolution { reasoningParser: ReasoningParser?; toolCallFormat: ToolCallFormat?; reasoningSource; toolCallSource }` per model so the `jang_config.json` read happens once per load.

See `Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md` §5 for the exact call-site inventory.

## Verifying the parser engages

Two cheap checks:

1. Log `ModelConfiguration.reasoningParserName` after `loadModelContainer` — should be `"think_xml"` (or JANG stamp value) for Qwen / DeepSeek / GLM / Nemotron; `"none"` for Gemma-4 / Mistral; `nil` only if neither factory matched (unknown model type, no JANG stamp).
2. Feed a string containing `<think>test</think>hello` through `ReasoningParser().feed(...)` then `.flush()`. Must produce `[.reasoning("test"), .content("hello")]`.

The library's own regression tests are in `Tests/MLXLMTests/ReasoningParserTests.swift` and `Tests/MLXLMTests/ToolCallEdgeCasesTests.swift` (streaming pipeline, character-by-character, JANG-stamp resolution, pipelined-with-tool-call).
