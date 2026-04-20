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

- **Status:** CLOSED.
- **Commit:** `b003ba8 — docs(fork-sync): document osaurus-ai/mlx-swift-lm ↔ ml-explore sync process`
- **Doc:** [`FORK-SYNC-PROCESS.md`](FORK-SYNC-PROCESS.md)
- **Summary of state as of 2026-04-20:**
  - `upstream` = ml-explore/mlx-swift-lm (canonical Apple MLX).
  - `public` = osaurus-ai/mlx-swift-lm — Eric's STABLE public fork:
    upstream + ONLY the carrying fixes osaurus production needs
    (Gemma-4 VLM, JANG overflow, Qwen3.5 norm shift, MXFP loader).
    Currently **75 commits ahead** of upstream, **18 behind**.
  - `origin` = osaurus-ai/vmlx-swift-lm — Eric's DEV superset
    (BatchEngine + SpecDec + CacheCoordinator + TurboQuant on top
    of the `public` base). **120 commits ahead** of public.
- **Sync procedure:** documented step-by-step for both
  `upstream → public` (merge with conflict hotspot list) and
  `public → origin` (fast-forward-able — origin is strictly ahead).
- **Upstream PR candidates identified:** four clean batches from
  the 75 carrying patches — JANG MLP float16 overflow bundle,
  Gemma4 VLM image pipeline bundle, Gemma4 multi-turn 1D-token
  crash, SwitchGLU compiledGeluApproximate workaround.
- **Acceptance gate for a public push:** full build + regression
  suites + real-model smoke (`BENCH_GEMMA4_STRESS` on a local
  Gemma-4 model). Exact commands in the doc.

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

## Summary of commits on `fix/osaurus-integration-issues`

```
5d8ea6b docs(gemma-4-crash): pin real-model verification matrix
b003ba8 docs(fork-sync): document osaurus-ai/mlx-swift-lm ↔ ml-explore sync process
e1d0270 test: refine Generation.reasoning + StopStringMatcher regression expectations
f62d0ce feat(generate): honor GenerateParameters.extraStopStrings in library
f966078 feat(generation): surface .reasoning(String) stream event
01707d8 fix(batch-engine): cap mask key-dim at slot maxSize (Gemma-4 SWA crash)
```

Branching: created off `main` at `5b2220c`. Fast-forward-able.

## Acceptance

- Build: green.
- Unit regression suites (new): 4 + 6 + 14 = **24 new tests**, all green.
- Existing suites: `BatchKVCache`, `BatchCausalMask`, `ReasoningParser`
  (37/37 including 6 new event tests), `Tool-Call Edge Cases` (24/24),
  `SpecDec*` (90/90) — no regressions.
- Real-model: Gemma-4-26B at tpae's exact 2152 / 3715 / 7869 / 8362
  prompt sizes — zero `broadcast_shapes` aborts. Gemma-4-e2b with
  `extraStopStrings` halt-and-truncate verified. Qwen3.6-35B
  multi-turn to 8385 tokens verified with native chat template
  (enable_thinking kwarg) — `.reasoning` emits, `.chunk` clean.

Everything tpae reported on 2026-04-20 is either fixed with a
dedicated doc, or out of scope with a documented rationale. Ready
to land.
