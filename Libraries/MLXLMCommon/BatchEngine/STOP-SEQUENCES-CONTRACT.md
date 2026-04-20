# Stop-sequence contract — `GenerateParameters.extraStopStrings`

**Status:** Landed on branch `fix/osaurus-integration-issues` (2026-04-20).
**Closes:** tpae's (2026-04-20) "StreamAccumulator does substring
stop-sequence matching against decoded text. BatchEngine.generate honors
token-level extraEOSTokens but not arbitrary text-level stops. What
should happen to text-level stop sequences?"

## The contract

`GenerateParameters` gains a new field:

```swift
public struct GenerateParameters: Sendable {
    // …
    /// Additional text-level stop sequences. When any of these strings
    /// appears in the user-visible assistant output, the library halts
    /// generation, truncates the match and everything after it, and
    /// emits `.info(stopReason: .stop)`.
    public var extraStopStrings: [String] = []
}
```

Two distinct stop mechanisms now exist at the library level, with
different scopes:

| Mechanism | Scope | Where it lives | When it fires |
|---|---|---|---|
| `ModelConfiguration.extraEOSTokens: Set<String>` | Token-level | Converted to token IDs at load time, added to `stopTokenIDs`. | On a TOKEN-ID match during generation — before detokenization. |
| `GenerateParameters.extraStopStrings: [String]` | Text-level | Matched against `.chunk(String)` output after the reasoning + tool-call pipeline. | On a SUBSTRING match in user-visible assistant text. |

They are orthogonal and can be combined. Token-level is fastest
(no text work) and preserves perfect boundaries. Text-level is the
OpenAI-API `stop: ["<string>"]` behaviour — it exists because many stop
sequences cannot be expressed as a single token (e.g., `"\n\nUser:"`).

## Match scope — `.chunk` only

`.reasoning(String)` deltas and `.toolCall(ToolCall)` events are NOT
candidates for a stop-string match. A `<think>` block containing "STOP"
will not halt a request whose `extraStopStrings` includes "STOP" — the
match only runs against the user-visible answer stream. This mirrors
the semantics an OpenAI-compatible server expects: stop sequences
gate the assistant ANSWER, not the chain-of-thought and not the tool
envelope.

Rationale: a reasoning trace naturally contains many strings that would
look stop-like ("end", "STOP", "</s>", etc.) without the model actually
wanting to terminate. Gating only on `.chunk` gives the parser room to
think in natural language without accidentally triggering terminations.

## Halt semantics

When a stop string matches:

1. The pipeline emits the pre-match `.chunk(String)` prefix — this is
   the final visible text for this turn.
2. Generation is halted upstream:
   - **`Evaluate.generate`** — the token-loop handler returns false
     from `onToken`, the loop breaks, the outer generator sets
     `stopReason = .stop` instead of `.cancelled` (see the
     `TokenLoopHandler.stopSequenceHit` protocol hook).
   - **`BatchEngine.generate`** — the consumer Task calls
     `engine.cancel(requestId)` on the actor, which flips the slot's
     `isFinished` flag. The final `.info` event's `stopReason` is
     transformed from the actor's default `.cancelled` to `.stop`
     inside the consumer Task before forwarding.
3. The final event is `.info(GenerateCompletionInfo)` with
   `stopReason == .stop`. Token counters reflect the actual tokens
   generated (including the ones inside the matched stop string); only
   the TEXT is truncated.
4. No further `.chunk`, `.reasoning`, or `.toolCall` events fire after
   the match.

Compute past the match is bounded: on BatchEngine the cancel goes
through on the next scheduling tick (one to a few tokens of overrun); on
Evaluate the loop breaks on the first post-match token. The stop string
itself is never emitted to the consumer.

## Matcher implementation

`Libraries/MLXLMCommon/StopStringMatcher.swift` is a pure-Swift rolling
tail buffer. On each `feed(_:)`:

- Append the incoming piece to the buffer.
- Check every configured stop string via `String.range(of:)`. The
  earliest match wins across stop strings.
- If matched: return `.stopped(emit: pre-match prefix)`, truncate the
  buffer.
- If no match: compute the safe-prefix length as
  `max(0, buffer.count - (maxStopLen - 1))`. Emit that prefix; keep the
  trailing `maxStopLen - 1` bytes in the buffer as potential prefix of
  a future match.

End-of-stream: `flush()` returns whatever is held (safe because no more
tokens are coming — no match is now possible).

`isEnabled` is false when the stop-strings list is empty or contains
only empty/duplicate entries. Disabled matchers are a pure pass-through
with no allocations per feed.

## Pipeline order (all three emitters)

```
detokenized chunk
    → ReasoningParser.feed(_:)        emits .reasoning(String) events; forwards .content
    → ToolCallProcessor.processChunk  emits .toolCall events; returns pure text
    → StopStringMatcher.feed          returns safe-prefix or stops; gates .chunk emission
    → emit .chunk / .reasoning / .toolCall / halt
```

`Libraries/MLXLMCommon/Evaluate.swift` — `TextToolTokenLoopHandler.dispatch` +
`onGenerationEnd` + new `emitChunkThroughStopMatcher` helper.
`Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift` — `pump` + `flush` +
`emitChunkThroughStop` closures inside `generate(input:parameters:)`.

## Not yet wired

**`SpecDecStream.streamDflashLinear` / `streamDDTree`** currently do
NOT honor `extraStopStrings`. The speculative-decoding runtimes run
their own multi-round accept/verify loops and halting mid-round would
require threading the matcher through `SpecDecRuntimeLinear` /
`SpecDecRuntimeDDTree`. Callers that need stop strings on the SpecDec
path should either:

- Route through `BatchEngine.generate(…)` (which dispatches to SpecDec
  via `draftStrategy` but wraps the stream with the matcher) — but
  currently the SpecDec dispatch at line 221 short-circuits the
  BatchEngine wrapper, so this works only after the SpecDec wrapper
  is updated to apply the matcher to its own stream.
- Fall back to non-speculative generation when `extraStopStrings` is
  non-empty.

This is tracked separately — the current commit ships matcher support
on the two primary paths osaurus uses (`Evaluate.generate` and
`BatchEngine.generate` non-SpecDec) and leaves SpecDec as a known
follow-up documented in this file.

## Test coverage

`Tests/MLXLMTests/StopStringMatcherTests.swift` — new suite
`StopStringMatcher`:

| Test | Covers |
|---|---|
| `testEmptyIsPassThrough` | No stop strings → `isEnabled == false`, `feed` always streams input unchanged. |
| `testFiltersEmptyStrings` | Init filters `""` entries. |
| `testDeduplicates` | Init de-duplicates. |
| `testSingleFeedMiddleMatch` | Match in the middle of one feed → emit pre-match, stop. |
| `testSingleFeedStartMatch` | Match at offset 0 → empty emit, stop. |
| `testSingleFeedEndMatch` | Match at the end → emit prefix, stop. |
| `testSplitAcrossChunks` | Stop string split across two feeds: first feed holds partial match; second completes + stops. |
| `testCharByCharStreaming` | Character-by-character delivery still halts correctly. |
| `testFlushDrainsTail` | `flush()` releases the held tail when no match arrives. |
| `testEarliestMatchAcrossMultiple` | When multiple stop strings would match, the EARLIEST match position wins (regardless of init order). |
| `testDifferentLengthHold` | Hold size tracks the LONGEST stop string's length. |
| `testNaturalTerminationFlushesHeld` | Per-feed partial match that never completes is released by `flush()`. |
| `testRoundTripByFlush` | Non-matching input round-trips byte-for-byte through `feed + flush`. |
| `testGenerateParametersField` | `GenerateParameters.extraStopStrings` plumbs through `init`. |

Integration with the three generation paths is exercised by the same
real-model scenarios in `scripts/verify-engine.sh` that cover reasoning
+ tool-call streaming — the matcher is a post-processing stage on
existing `.chunk` output.

## Osaurus migration

Before: `StreamAccumulator` buffers all `.chunk` output, runs substring
matching against `GenerationParameters.stopSequences`, halts its own
downstream consumption. Tokens generated past the stop are wasted
(generation continues on the vmlx side).

After: osaurus builds `GenerateParameters` with
`extraStopStrings: stopSequences` at the vmlx boundary, and can drop
the accumulator-side substring matching for this case. The library
halts on match; `StreamAccumulator` just forwards `.chunk` events as
they arrive. No post-processing needed.

`OSAURUS-API-SURFACE.md` updated: `GenerateParameters` now lists
`extraStopStrings: [String]` alongside the existing temperature/topP/
etc. fields, and the §Generation consumer example switches on
`.chunk` / `.reasoning` / `.toolCall` / `.info` with
`stopReason == .stop` as the valid text-level stop signal.
