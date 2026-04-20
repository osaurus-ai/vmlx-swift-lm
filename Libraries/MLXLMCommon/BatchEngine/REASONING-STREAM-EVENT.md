# Reasoning stream event — `.reasoning(String)` on `Generation`

**Status:** Landed on branch `fix/osaurus-integration-issues` (2026-04-20).
**Closes:** tpae's (2026-04-20) "thinking parsers should be handled at
library level (same way we do tool calls, we should display it streaming)."

## The contract

`Libraries/MLXLMCommon/Evaluate.swift` now ships a fourth `Generation` case:

```swift
public enum Generation: Sendable {
    case chunk(String)            // pure user-visible text (reasoning peeled, tool-calls stripped)
    case reasoning(String)        // streaming chain-of-thought delta
    case toolCall(ToolCall)       // authoritative tool call
    case info(GenerateCompletionInfo)
}
```

Every generation path that carries tokens through a reasoning parser now
emits `.reasoning(String)` deltas in real time instead of silently
dropping the `<think>…</think>` bytes:

| Emitter | File |
|---|---|
| `Evaluate.generate(…)` | `Libraries/MLXLMCommon/Evaluate.swift` — `TextToolTokenLoopHandler.dispatch` + `onGenerationEnd` flush |
| `BatchEngine.generate(…)` | `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift` — `pump` + `flush` closures inside `generate(input:parameters:)` |
| `SpecDecStream.streamDflashLinear(…)` / `streamDDTree(…)` | `Libraries/MLXLMCommon/SpecDec/SpecDecStream.swift` — `pushBatch` + `flush` statics |

All three emitters share the same pipeline order:

```
detokenized chunk
    → ReasoningParser.feed(_:)                    emits .reasoning(String); forwards content
    → ToolCallProcessor.processChunk(_:)          extracts .toolCall + pure text
    → emit .chunk / .reasoning / .toolCall
```

Flush happens on end-of-stream: `ReasoningParser.flush()` drains any tail
(a trailing `<think>…` with no closing tag becomes `.reasoning(String)`,
matching `ReasoningParser.split(_:)` semantics), and
`ToolCallProcessor.processEOS()` releases any complete tool call queued
in its inline-JSON buffer.

## Streaming granularity

`.reasoning(String)` deltas are NOT end-of-thought boundaries — they are
per-parser-segment. A typical `<think>…</think>` block produces many
`.reasoning` events as the detokenizer delivers deltas. Callers that want
the full reasoning transcript should concatenate all consecutive
`.reasoning` events into a rolling buffer.

There is no `.reasoningEnd` / `.reasoningStart` event. The boundary is
implicit: when you receive a `.chunk` after one or more `.reasoning`
deltas, the think block is closed. If the model emits multiple think
blocks interleaved with assistant text (Qwen 3.6 interleaved-thinking
families), the library emits them in stream order: reasoning-delta(s),
then chunk-delta(s), then reasoning-delta(s), etc.

## When `.reasoning` fires

Only when the model's `ModelConfiguration.reasoningParserName` resolves
to a non-nil `ReasoningParser` via
`ReasoningParser.fromCapabilityName(_:)`:

- **Qwen 3 / 3.5 / 3.6 / Nemotron** → `think_xml` parser active → `.reasoning` fires.
- **MiniMax M2 / M2.5** → `minimax` parser (interleaved) → `.reasoning` fires.
- **GLM 4.x / DeepSeek-R1** → `glm4` / `deepseek_r1` parser → `.reasoning` fires.
- **Kimi K2** → `think_xml` parser → `.reasoning` fires.
- **Gemma 3 / Gemma 4 / Mistral / LFM2** → `"none"` → no parser → **no `.reasoning` ever fires**.
- **JANG / JANGTQ** — honours the `capabilities.reasoning_parser` stamp; most
  Qwen / MiniMax / GLM JANGs set this and emit `.reasoning` accordingly.

No new flag is added. If osaurus (or any caller) wants to disable
reasoning emission for a given request at runtime, override
`ModelConfiguration.reasoningParserName = "none"` before load, or skip
the parser by consuming `Evaluate.generateTokens(…)` (the raw-token path
which has never run the parser).

## Backward compatibility

Adding a new `Generation` enum case IS a source-breaking change at
exhaustive `switch` sites. Every switch in this repository has been
updated to handle `.reasoning` (typically `case .reasoning: break` or a
compound `case .reasoning, .toolCall` arm for callers that don't render
CoT). External consumers that pattern-match exhaustively must add a
`.reasoning` handler or a `default`/`@unknown default` arm.

`Generation` is NOT marked `@frozen` (and never was), so the language
permits adding a case. We accept the source-break at callers as a
deliberate cost of correctness — the previous "silently drop `<think>`"
behaviour actively hid information from osaurus's UI layer and was the
reason it had to parse reasoning again at the app level.

The `Generation.info` / `.chunk` / `.toolCall` computed properties keep
their exact pre-change semantics (return nil for non-matching cases),
and a new `.reasoning: String?` property follows the same pattern.

## Test coverage

`Tests/MLXLMTests/ReasoningParserTests.swift` — new suite
`Generation.reasoning event`:

| Test | Covers |
|---|---|
| `testReasoningAndContentSeparated` | `<think>r</think>answer` → `reasoning == "r"`, `content == "answer"`. |
| `testReasoningStreamsAcrossChunks` | Char-by-char delivery across a start-tag boundary and end-tag boundary still produces coherent reasoning/content streams. |
| `testUnclosedReasoningFlushesOnEOS` | A `<think>…` with no closing tag surfaces as `.reasoning` on stream-end (parity with `ReasoningParser.split` semantics). |
| `testNoReasoningEmitsNoReasoningEvents` | Plain-text input produces zero `.reasoning` events. |
| `testReasoningAndToolCallCoexist` | `<think>…</think>` followed by an inline JSON tool call: three distinct streams — `.reasoning`, `.chunk` text, `.toolCall`. |
| `testGenerationEnumSurface` | Smoke test for the new `.reasoning` case + `.reasoning: String?` computed property. |

The suite uses a private `driveGenerationPipeline(chunks:)` helper that
mirrors the closure inside `BatchEngine.generate` byte-for-byte — so the
tests pin the contract shared by all three emitters without loading a
real model.

Real-model coverage rides on the existing Qwen 3.6 / MiniMax M2 / Kimi
K2 / Nemotron scenarios in `scripts/verify-engine.sh` — once the
harness is run against any think-emitting family, `.reasoning` deltas
now show up in the event stream.

## Migration for osaurus

Replace the app-layer `StreamingDeltaProcessor` fan-out logic with a
switch on the new case:

```swift
for await event in engine.generate(input: input, parameters: params) {
    switch event {
    case .chunk(let text):       deltaAccumulator.appendAssistantText(text)
    case .reasoning(let delta):  deltaAccumulator.appendReasoning(delta)
    case .toolCall(let call):    toolInvocations.append(call)
    case .info(let info):        finish(stopReason: info.stopReason)
    }
}
```

Since the library is now the single source of truth for both reasoning
and tool-call parsing, osaurus's `StreamingDeltaProcessor.swift` can
drop its `ReasoningParser?` instance entirely on the BatchEngine /
Evaluate path. `OSAURUS-API-SURFACE.md` line 104 and
`OSAURUS-INTEGRATION.md` §4 have been updated with the new switch
template.

## Not changed

- `ReasoningParser` public API — same `feed(_:) -> [ReasoningSegment]`,
  `flush() -> [ReasoningSegment]`, `split(_:)`, `fromCapabilityName(_:)`.
- `TokenGeneration` / `BatchGeneration` — raw-token streams still emit
  only `.token(Int)` / `.info`. If you want streaming reasoning you
  must go through `Generation` (detokenized) streams.
- Reasoning-parser autoloading — `LLMModelFactory` + `VLMModelFactory`
  still stamp `ModelConfiguration.reasoningParserName` with the same
  JANG-stamp → model-type heuristic priority described in §4 of
  `OSAURUS-INTEGRATION.md`.
