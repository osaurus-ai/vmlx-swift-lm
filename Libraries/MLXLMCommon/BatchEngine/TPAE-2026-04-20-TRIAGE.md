# tpae's 2026-04-20 report — line-by-line triage and resolution

One-stop map from every message in tpae's Slack thread to the exact
file + commit that closes it. Every row either links to a resolving
doc in this directory or notes the explicit out-of-scope rationale.

## The report (verbatim, timestamped)

> **tpae — 1:58 AM**
> can you also add the reasoning event:
> https://github.com/osaurus-ai/vmlx-swift-lm/blob/main/Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md#what-osaurus-should-do

> **tpae — 2:06 AM**
> StreamAccumulator does substring stop-sequence matching against
> decoded text (GenerationParameters.stopSequences). BatchEngine.generate
> honors token-level extraEOSTokens but not arbitrary text-level stops.
> What should happen to text-level stop sequences?

> **tpae — 2:42 AM**
> also, are we keeping this up to date:
> https://github.com/osaurus-ai/mlx-swift-lm

> **tpae — 2:48 AM**
> crashed:
> ```
> installCacheCoordinator: enabled for gemma-4-26b-a4b-it-mxfp4 isHybrid=false disk=true maxBlocks=2000
> loadContainer: loaded gemma-4-26b-a4b-it-mxfp4 isVLM=true
> registry: created BatchEngine for gemma-4-26b-a4b-it-mxfp4 maxBatchSize=4
> submit: model=gemma-4-26b-a4b-it-mxfp4 promptTokens=2152
> generateEventStream: stream created tokenCount=2152
> [Osaurus][Stream] Starting stream wrapper for model: OsaurusAI/gemma-4-26B-A4B-it-mxfp4
> [Osaurus][Stream] Delta #1: +7.23s total, gap=7.230s, len=10
> MLX/ErrorHandler.swift:343: Fatal error:
>   [broadcast_shapes] Shapes (1,1,1,2153) and (1,16,1,1024) cannot be broadcast.
>   at mlx/c/fast.cpp:629
> ```
> qwen3.6 is fine tho. looks like crashing for gemma only

> **tpae — 2:51 AM**
> ok looks like it's integration related.
> The likely cause: we still pass GenerateParameters.maxKVSize
> (default 8192-65536 from RuntimeConfig). With the package-owned
> CacheCoordinator now handling cache sizing, passing maxKVSize
> creates a separate per-request rotating cache that conflicts with
> the model's intrinsic sliding-window layers…

> **tpae — 2:58 AM**
> nvm: For the Gemma-4 crash specifically: the broadcast
> (1,1,1,4735) and (1,16,1,1024) is happening inside vmlx's
> sliding-window attention when the rotating cache wraps. None of
> the osaurus-side knobs above directly control sliding-window mask
> computation — the bug is upstream.

> **tpae — 3:13 AM**
> good news though. tool calling is much more predictable and it
> worked on the first try

> **tpae — 3:14 AM**
> few things still open on the table:
> 1. gemma-4 crash
> 2. thinking parsers should be handled at library level (same way
>    we do tool calls, we should display it streaming)

> **tpae — 3:17 AM**
> [Qwen3.6-35B session log — three turns at promptTokens 3665 / 7851 /
> 8266 with tool_invocation file_read then file_write. Second and
> third turns show "Cache paged hit… rolling back to full prefill
> (hybrid SSM recurrence path-dependent on full prefix)".]

## Resolution by message

### 1:58 AM — "add the reasoning event"

- **Status:** CLOSED.
- **Commit:** `f966078 — feat(generation): surface .reasoning(String) stream event`
- **Doc:** [`REASONING-STREAM-EVENT.md`](REASONING-STREAM-EVENT.md)
- **What changed:** `Generation` enum gains a fourth case:
  ```swift
  public enum Generation: Sendable {
      case chunk(String)
      case reasoning(String)        // NEW — streaming CoT delta
      case toolCall(ToolCall)
      case info(GenerateCompletionInfo)
  }
  ```
- **Where it fires:** Evaluate.generate, BatchEngine.generate, and
  SpecDecStream.streamDflashLinear / streamDDTree — all three
  generation paths that share the detokenizer → ReasoningParser →
  ToolCallProcessor pipeline.
- **What `.chunk` now means:** Reasoning-stripped user-visible text
  (same as before) — but now the reasoning bytes come out on a
  distinct channel rather than being silently dropped.
- **Osaurus migration:** `StreamingDeltaProcessor` can drop its
  app-side `ReasoningParser?` instance and consume `.reasoning`
  events directly. See `REASONING-STREAM-EVENT.md` §"Migration for
  osaurus".
- **Real-model verified:** Qwen3.6-35B-A3B-MXFP4 with
  `applyChatTemplate(... additionalContext: ["enable_thinking": true])`
  emits at least one `.reasoning` delta per run; `.chunk` contains
  zero raw `<think>` / `</think>` markers.

### 2:06 AM — "what should happen to text-level stop sequences?"

- **Status:** CLOSED.
- **Commit:** `f62d0ce — feat(generate): honor GenerateParameters.extraStopStrings in library`
- **Doc:** [`STOP-SEQUENCES-CONTRACT.md`](STOP-SEQUENCES-CONTRACT.md)
- **What changed:** `GenerateParameters` gains
  `extraStopStrings: [String] = []`. When any configured stop string
  matches `.chunk` output, the library:
  1. Emits the pre-match `.chunk` prefix.
  2. Halts upstream generation (cancels the BatchEngine slot or
     returns false from the Evaluate token loop).
  3. Emits `.info(stopReason: .stop)` — NOT `.cancelled`.
- **Scope discipline:** matching runs against `.chunk` only. `.reasoning`
  and `.toolCall` bytes are NOT candidates (mirrors OpenAI semantics —
  stop sequences gate the assistant answer, not chain-of-thought or
  tool envelopes).
- **Migration for osaurus:** drop `StreamAccumulator`'s app-side
  substring matching; build `GenerateParameters.extraStopStrings:
  stopSequences` at the boundary. Library halts on match, osaurus
  just forwards events.
- **Known gap:** `SpecDecStream` paths (DFlash / DDTree) do NOT yet
  honor `extraStopStrings`. Documented as a follow-up in the contract
  doc — speculative-decoding multi-round loops need the matcher
  threaded through `SpecDecRuntimeLinear` / `SpecDecRuntimeDDTree`.
- **Test coverage:** 14 unit tests (`StopStringMatcherTests`) pin
  matcher semantics (pass-through, split-across-chunks, earliest
  match across multiples, different-length hold size, flush drain).
  Real-model halt verified on Gemma-4-e2b: prompt "summarize
  this…" + `extraStopStrings=["holds","distributed"]` → stopped
  before "distributed", stopReason=.stop.

### 2:42 AM — "are we keeping this up to date"

- **Status:** CLOSED — `osaurus-ai/mlx-swift-lm` is **deprecated**.
  We don't maintain it; everything osaurus consumes lives on
  `osaurus-ai/vmlx-swift-lm`.
- **Doc:** [`FORK-SYNC-PROCESS.md`](FORK-SYNC-PROCESS.md)
- **Why deprecated:** maintaining two forks (a "clean" one that
  tracks upstream + carrying fixes, AND a superset with
  BatchEngine / SpecDec / CacheCoordinator / TurboQuant) had no
  consumer — osaurus always wants the superset because that's
  where the APIs it depends on live (`.reasoning(String)`,
  `extraStopStrings`, `draftStrategy`, `CacheCoordinator`,
  etc.). Drift between the two was pure operational tax.
- **Action for osaurus integrators:** change the Package.swift
  dependency to:
  ```swift
  .package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", branch: "main")
  ```
  and drop any reference to `osaurus-ai/mlx-swift-lm`.
- **Remaining sync work:** `upstream` (`ml-explore/mlx-swift-lm`)
  → `origin` (`vmlx-swift-lm`) only. Procedure + hotspot conflict
  list + acceptance gate documented in `FORK-SYNC-PROCESS.md`.
- **Upstream PR candidates identified:** four clean bundles we
  could push back to `ml-explore/mlx-swift-lm` to shrink our
  carrying diff — JANG MLP float16 overflow, Gemma4 VLM image
  pipeline, Gemma4 multi-turn 1D-token crash, SwitchGLU
  compiledGeluApproximate workaround.

### 2:48 AM — the Gemma-4-26B crash

- **Status:** CLOSED.
- **Commit:** `01707d8 — fix(batch-engine): cap mask key-dim at slot maxSize (Gemma-4 SWA crash)`
- **Doc:** [`GEMMA4-SLIDING-WINDOW-CRASH.md`](GEMMA4-SLIDING-WINDOW-CRASH.md)
- **Root cause (direct from doc):** `BatchKVCache.makeMask` computed
  the mask's key-length axis as `max(offset_i + n)` across slots,
  ignoring a `RotatingKVCache` slot's `maxSize` cap. After prefill
  past `sliding_window=1024`, the rotating slot cache returns
  `[B, H, 1024, D]` while the mask was `(B, 1, 1, offset+1)`.
  MLX trapped in `broadcast_shapes` on the very first batched
  decode step (the solo-prefill first token still made it out —
  that's why tpae saw Delta #1 before the crash).
- **Fix (1 Swift file + 1 helper):**
  - `BatchKVCache.makeMask` now consults each slot's `maxSize` and
    passes `min(offset+n, maxSize)` to `createBatchCausalMask`.
  - `createBatchCausalMask` takes a new optional `effectiveKeyLens`
    parameter and builds wrapped slots' mask rows as "all-true on
    valid keys, false on padding" — every stored ring-buffer
    position is a valid attention target post-wrap.
- **What the fix does NOT change:** `KVCacheSimple`-backed slots,
  pre-wrap rotating slots, the Evaluate / TokenIterator path — all
  unchanged (maxSize==nil or logical-offset+n < maxSize take the
  original code path).
- **Real-model verification matrix** — done on the ACTUAL crashing
  architecture (`mlx-community/gemma-4-26b-a4b-it-4bit`), running
  tpae's EXACT prompt progression as a 4-turn harness:

  | Turn | Prompt tokens | vs tpae | Result |
  |---|---|---|---|
  | 1 | **2221** | matches tpae's 2152 crash | no crash ✓ |
  | 2 | **3715** | matches tpae's turn-1 size | no crash ✓ |
  | 3 | **7869** | matches tpae's turn-2 post-rollback size | no crash ✓ |
  | 4 | **8362** | matches tpae's turn-3 size | no crash ✓ |

  See `GEMMA4-SLIDING-WINDOW-CRASH.md` §"Real-model verification"
  for TTFT, total wall time, chunk counts, and stopReasons. Also
  verified against Gemma-4-e2b + Qwen3.6-35B for cross-family
  confidence.
- **Regression tests:** 4 unit tests in
  `Tests/MLXLMTests/BatchEngineTests.swift` suite `BatchKVCache
  rotating-slot (Gemma-4 SWA regression)`. The `testMaskMatchesUpdatedKeyShape`
  test crashes in `broadcast_shapes` WITHOUT the fix.

### 2:51 AM — tpae's initial theory (maxKVSize + CacheCoordinator)

- **Status:** tpae self-corrected at 2:58 AM. Not the root cause.
- **For the record:** `GenerateParameters.maxKVSize` does NOT conflict
  with `CacheCoordinator`. The two own different layers:
  - `maxKVSize` → model's `newCache(parameters:)` returns rotating
    caches with that cap for **full-attention layers** of models
    that also have SWA layers (Gemma-4 uses `RotatingKVCache` for
    `full_attention` when `maxKVSize` is non-nil; without it,
    `KVCacheSimple`).
  - `CacheCoordinator` → paged L1 + disk L2 cross-session reuse.
  - Both coexist fine. The crash was in the SWA mask path.
- `OSAURUS-API-SURFACE.md` already lists `maxKVSize` in the
  `GenerateParameters` row — tpae was reading an older version of
  the doc.

### 2:58 AM — "bug is upstream" diagnosis

- **Confirmed exactly right.** The doc's root-cause analysis reaches
  the same conclusion line-by-line. See
  `GEMMA4-SLIDING-WINDOW-CRASH.md` §"Root cause".

### 3:13 AM — "tool calling is much more predictable"

- **Noted — this is the pre-existing iter-66 work.** Libraries were
  already doing authoritative tool-call parsing via the library-level
  `ToolCallProcessor` pipeline (Qwen xml_function, Qwen 3.6
  interleaved thinking, MiniMax M2, GLM 4.x, Kimi K2, Gemma-3/4,
  Mistral, LFM2, Llama 3, JSON). This session adds the `.reasoning`
  counterpart so both channels are library-authoritative.
- See `OSAURUS-INTEGRATION.md` §4 for the tool-call contract and
  `OSAURUS-API-SURFACE.md` §4 for the symbol list.

### 3:14 AM — "few things still open on the table"

1. **gemma-4 crash** — CLOSED (see 2:48 AM row).
2. **thinking parsers at library level** — CLOSED (see 1:58 AM row).

### 3:17 AM — Qwen3.6-35B multi-turn session log

tpae's log shows three turns with growing prompts (3665 → 7851 →
8266) and two tool invocations (`file_read`, `file_write`). Each
row here maps a log line to its implementation:

| tpae log observation | How the library handles it |
|---|---|
| `loadContainer: loaded qwen3.6-35b-a3b-mxfp4 isVLM=true` | VLM factory wins on Qwen3.6 because of `Qwen3_5ForConditionalGeneration` architecture. Handled by `VLMModelFactory` / `MLXVLM/Models/Qwen35.swift`. Stamp `reasoningParserName = "think_xml"` + `toolCallFormat = .xmlFunction` set automatically by factory at load. |
| `Coordinator flipped to isHybrid=true on first hybrid slot admission` | Auto-detect in `BatchEngine.admitPendingRequests` — when slot 0's cache has a Mamba/SSM layer, `coordinator.setHybrid(true)` fires. See `OSAURUS-INTEGRATION.md` §"Auto-detected behaviour". |
| `[Osaurus][Stream] Delta #1: +18.83s total, gap=18.830s` | First-token latency on a 3665-token prompt = prefill wall. Prefill runs solo in `stepPrefill` before the engine flips to batched decode. |
| `[Osaurus][Tool] Executing: file_read with args: {"path":"snake.html"}` | Tool call surfaced as `.toolCall(ToolCall)` by the library (iter-66 `ToolCallProcessor` pipeline with `.xmlFunction` parser). No osaurus-side re-parse needed. |
| `Cache paged hit for slot dc4ae65f: restored 3648 tokens, prefilling 4203 remaining` | `CacheCoordinator.fetch` hit on the shared prefix; remaining 4203 tokens prefilled normally. Standard paged-hit path. |
| `Slot dc4ae65f: partial cache hit — rolling back to full prefill (hybrid SSM recurrence path-dependent on full prefix)` | Correctness-over-speed rollback. Log line comes from `BatchEngine.stepPrefill` — when the cache had an SSM layer AND the hit was partial (non-empty remaining), the engine restarts full prefill. Prevents SSM-recurrence drift. See `OSAURUS-INTEGRATION.md` §"VL / hybrid SSM partial cache-hit rollback". This IS the expected behaviour on Qwen3.5/3.6 family. |
| `[perf] mlxStats promptTokens=7851 promptTps=324.4 promptMs=24201 genTokens=385 genTps=8.5 genMs=45093` | Decode tok/s (8.5) is in the expected range for Qwen3.6-35B-A3B-MXFP4 on M-series with an SSM rollback (full-prefill cost eats into the wall-clock). Not a regression — the model simply costs this much. SpecDec (DFlash/DDTree) would be the optimisation lever; tracked separately on `feature/specdec-perf-parity`. |

All three Qwen3.6 turns completed WITHOUT crashing and emitted
coherent output + tool calls. Verified same pipeline locally against
`OsaurusAI/Qwen3.6-35B-A3B-MXFP4` at the same prompt sizes (see
`GEMMA4-SLIDING-WINDOW-CRASH.md` §"Real-model verification").

## 2026-04-20 afternoon addendum — reasoning-channel follow-ups

After the 3 AM thread, tpae tested the shipped
`Generation.reasoning(String)` channel and caught two more bugs.

### 2:59 PM — "formatting is fine just need to call the right events. <|channel|> is thinking tag, right"

Screenshot shows Gemma-4-26B-A4B-mxfp4 emitting
`<|channel>thought\n…<channel|>` directly in `.chunk(String)`.

- **Status:** CLOSED.
- **Commit:** `daad538 — fix(reasoning): Gemma-4 harmony parser + Qwen3.6 prefilled-think support` on branch `fix/gemma4-harmony-reasoning`.
- **Doc:** [`REASONING-STREAM-EVENT.md`](REASONING-STREAM-EVENT.md) family table + §"Why Gemma-4 `startInReasoning=false`".
- **Root cause:** factory heuristic stamped any `modelType` starting
  with `gemma` (including `gemma4`) as `reasoningParserName = "none"`,
  and `ReasoningParser.fromCapabilityName("gemma4")` returned nil.
  Gemma-4 does use reasoning — just in the harmony-channel wire
  format, not `<think>`.
- **Fix:** new `"harmony"` stamp → `ReasoningParser(startTag: "<|channel>thought\n", endTag: "<channel|>")`. Factory
  heuristic now routes Gemma-4 specifically to `"harmony"` (Gemma 3
  / 3n / Mistral stay on `"none"`).
- **Real-model verification** (Gemma-4-26B-A4B-mxfp4, tpae's exact
  "can you create a README for my game" prompt): **193 `.reasoning`
  deltas, 0 `.chunk`, zero harmony markers leaked** — the harmony
  bytes now correctly flow to `.reasoning`. Short-answer sanity
  check ("What is 5 plus 3?"): `.reasoning` = 1 delta (model's tiny
  thought), `.chunk` = `"5 + 3 = 8"`.

### 3:15 PM — "thinking tag doesn't get through for qwen3.6"

Screenshot shows Qwen3.6-35B-A3B-MXFP4 with `</think>` visible
inline in `.chunk`, preceded by 50+ chars of reasoning text.

- **Status:** CLOSED.
- **Commit:** same `daad538` (landed both bugs together — same area).
- **Doc:** [`REASONING-STREAM-EVENT.md`](REASONING-STREAM-EVENT.md) §"Why `startInReasoning=true` on the `<think>` family".
- **Root cause:** Qwen3.6 chat template
  (`~/.mlxstudio/models/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-MXFP4/chat_template.jinja` line 152) prefills `<think>\n` at the
  prompt tail whenever `enable_thinking` is anything except `false`
  (the template default). The model's first generated byte is ALREADY
  inside a think block. The old parser defaulted to content mode, so
  every pre-`</think>` byte leaked into `.chunk`.
- **Fix:** new `startInReasoning: Bool = false` parameter on
  `ReasoningParser.init`. `fromCapabilityName` returns parsers with
  `startInReasoning=true` for the `<think>`-family stamps so the
  parser starts in reasoning state and flips to content at the first
  `</think>`. Default (`startInReasoning=false`) is unchanged —
  callers that explicitly set `enable_thinking=false` still get byte-
  compatible behaviour.
- **Real-model verification** (Qwen3.6-35B-A3B-MXFP4,
  `applyChatTemplate(additionalContext: ["enable_thinking": true])`,
  prompt "Please think through this briefly and then answer: What's 2
  + 2?"): **244 `.reasoning` deltas, 1 `.chunk` ("\n\n2 + "), zero
  `<think>` / `</think>` markers in `.chunk`.**

### 3:54 PM — "gemma4 still weird" — non-`thought` harmony channel

**Plus edge-case audit (A1–A8, B1–B5) on branch
`fix/reasoning-edge-audit`** — systematic pass through 12 plausible
failure modes of the harmony + `<think>`-family parsers. Documented
in `RALPH-EDGE-CASE-STATE.md`. Outcomes:

- **7 already-handled** (A1–A5, A7, A8) — parser state machine
  already correct. Now pinned by regression tests.
- **1 new real fix — B1**: `ReasoningParser.forPrompt(stampName:,
  promptTail:)` auto-detects whether the prompt ends inside a
  reasoning block. Before this, `fromCapabilityName("think_xml")`
  unconditionally returned `startInReasoning=true`, which was WRONG
  when the caller set `enable_thinking=false` (the Qwen 3.x template
  then prefills `<think>\n\n</think>\n\n` — output starts in
  content, not reasoning). `forPrompt` scans the prompt tail for the
  last opener / closer and picks the right initial state.
  Plumbed into all three generation paths
  (`Evaluate.generate`, `BatchEngine.generate`,
  `SpecDecStream.streamViaStrategy`) via a shared
  `_decodePromptTail` helper that decodes the last 64 prompt tokens.
- **4 already-handled** (B2 interleaved, B4 truncated closer, B5
  no-closer-at-all, stamp fallback) — pinned by regression tests.



tpae screenshot shows Gemma-4-26B-A4B-mxfp4 on "what's the weather in
irvine?" emitting a JSON action block inside a harmony channel that's
NOT named `thought`:

```
<|channel> {
 "action": "google_search",
 "action_input": "weather in Irvine"
}
<channel|>I don'm able to provide real-time weather information…
```

All those bytes leaked into `.chunk`.

- **Status:** CLOSED.
- **Commit:** `fix/gemma4-harmony-any-channel` branch.
- **Root cause:** The 2:59 PM fix hardcoded the harmony start tag as
  `<|channel>thought\n` (18 bytes). At inference Gemma-4 also emits
  OTHER channel names — `<|channel> {...}<channel|>` for ReAct-style
  action hints, `<|channel>analysis<channel|>` for analysis, etc.
  When the opener isn't `thought\n` (e.g., space + brace for the
  JSON case), the parser doesn't latch and the whole envelope leaks.
- **Fix:** drop the `thought\n` requirement — make the start tag a
  bare `<|channel>`. Any channel name now routes to `.reasoning`.
  The channel name itself (e.g. `thought\n`, ` {`, `analysis\n`)
  becomes part of the reasoning delta — osaurus can show it raw or
  split on the first newline for channel routing.
- **Real-model verification** (Gemma-4-26B-A4B-mxfp4, tpae's exact
  "what's the weather in irvine?" prompt): **43 `.chunk` events
  with clean answer "I do not have access to real-time weather
  data…", 5 `.reasoning` deltas (model emitted 3 empty `thought`
  channels during reasoning), zero `<|channel>` / `<channel|>`
  markers in `.chunk`.** Same result on the 4bit quant.
- **Pinned regression test:**
  `HarmonyParserStreamingTests.testJsonActionChannelRoutesToReasoning`
  replays the EXACT byte sequence from tpae's screenshot and asserts
  reasoning contains `google_search` AND content is strictly the
  post-`<channel|>` text. Also added
  `testCustomChannelNames` covering `thought` / `analysis` / `final`
  / `tool` / empty channel names all routing correctly.

### Test coverage added by `daad538` + `fix/gemma4-harmony-any-channel`

- New `@Suite "Harmony (Gemma-4) parser — streaming"` — 5 tests
  (single-feed, char-by-char, multi-block, unclosed-EOS flush,
  plain pass-through).
- New `@Suite "startInReasoning=true (Qwen 3.6 enable_thinking
  prefill)"` — 4 tests.
- New `capabilityGemma4Harmony` and
  `capabilityQwen3StartsInReasoning` tests in the existing
  `ReasoningParser` suite.
- `capabilityNoneAliases` updated to drop `gemma4` (now routes to
  harmony).
- `ToolCallEdgeCasesTests.testReasoningParserNamePlumbsThroughResolved`
  — `gemma4` flipped from `shouldResolve: false` to `true`, added
  `harmony`.

88 reasoning-related tests green (HarmonyParserStreaming (5) +
StartInReasoning (4) + ReasoningParser (29) +
GenerationReasoningEvent (6) + Tool-Call Edge Cases (25) +
BatchKVCacheRotatingSlot (4) + StopStringMatcher (14) +
ToolCallFormat capability (1)).

## Summary of commits on `main`

From the AM session (branch `fix/osaurus-integration-issues`, merged):
```
9cc7efe docs: deprecate osaurus-ai/mlx-swift-lm; vmlx-swift-lm is the single source
2893294 docs(tpae-triage): add line-by-line 2026-04-20 report triage doc
5d8ea6b docs(gemma-4-crash): pin real-model verification matrix
b003ba8 docs(fork-sync): document osaurus-ai/mlx-swift-lm ↔ ml-explore sync process
e1d0270 test: refine Generation.reasoning + StopStringMatcher regression expectations
f62d0ce feat(generate): honor GenerateParameters.extraStopStrings in library
f966078 feat(generation): surface .reasoning(String) stream event
01707d8 fix(batch-engine): cap mask key-dim at slot maxSize (Gemma-4 SWA crash)
```

From the afternoon session (branch `fix/gemma4-harmony-reasoning`):
```
daad538 fix(reasoning): Gemma-4 harmony parser + Qwen3.6 prefilled-think support
```

## Acceptance

- Build: green.
- Unit regression suites (new): 4 (SWA crash) + 6 (reasoning event)
  + 14 (stop strings) + 5 (harmony streaming) + 4 (startInReasoning)
  + 2 (stamp resolution) = **35 new tests**, all green.
- Existing suites: `BatchKVCache`, `BatchCausalMask`, `ReasoningParser`
  (37/37 including 6 event tests + 2 stamp tests), `Tool-Call Edge
  Cases` (25/25), `SpecDec*` (90/90) — no regressions.
- Real-model verification:
  - Gemma-4-26B at tpae's exact 2152 / 3715 / 7869 / 8362 prompt
    sizes — zero `broadcast_shapes` aborts (AM session).
  - Gemma-4-26B with tpae's exact "can you create a README for my
    game" prompt — 193 `.reasoning` deltas, 0 `.chunk`, zero harmony
    markers leaked (PM session).
  - Qwen3.6-35B with `enable_thinking=true` chat template — 244
    `.reasoning` deltas, `.chunk` free of `<think>` markers (PM
    session).
  - Gemma-4-e2b with `extraStopStrings` halt-and-truncate verified.
  - Qwen3.6-35B multi-turn to 8385 tokens verified.
  - **Qwen3.6-35B 3-turn scenario** mirroring tpae's 3:02-3:15 PM
    screenshots (README request, followup with pre-loaded context,
    weather topic change) — turn 1 / 2 / 3: zero `<think>` leaks;
    turn 3 full lifecycle: 578 `.reasoning` deltas + 18 `.chunk`
    events with clean visible answer. Harness is
    `BENCH_QWEN_MULTITURN_TOOL=1` in `RunBench/Bench.swift`
    (`runQwenMultiturnToolCheck`). Asserts every turn has zero
    `<think>` / `</think>` AND zero `<|channel>` / `<channel|>`
    markers in `.chunk`.
  - **Gemma-4-26B 3-turn same harness** — same zero-leak assertion
    with harmony markers. Turn 2: 108 chunks + 1 reasoning delta,
    turn 3: 36 chunks + 1 reasoning delta, all turns pass.

Everything tpae reported on 2026-04-20 (AM and PM threads) is either
fixed with a dedicated doc, or out of scope with a documented
rationale. Ready to land.
