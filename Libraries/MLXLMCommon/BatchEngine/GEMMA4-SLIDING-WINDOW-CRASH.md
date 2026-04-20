# Gemma-4 sliding-window crash under BatchEngine

**Status:** Fixed on branch `fix/osaurus-integration-issues` (2026-04-20).
**Scope:** Any sliding-window model served through `BatchEngine` — Gemma-3,
Gemma-4 (text + VLM, incl. E2B / E4B / 26B-a4b / 31B), Mistral-3, Mistral-4,
MiMoV2Flash, BaichuanM1, Qwen3.5-VL inherited sliding layers.

## Symptom

Reported by tpae (2026-04-20, osaurus #TBD):

```
installCacheCoordinator: enabled for gemma-4-26b-a4b-it-mxfp4 isHybrid=false disk=true maxBlocks=2000
loadContainer: loaded gemma-4-26b-a4b-it-mxfp4 isVLM=true
registry: created BatchEngine for gemma-4-26b-a4b-it-mxfp4 maxBatchSize=4
submit: model=gemma-4-26b-a4b-it-mxfp4 promptTokens=2152
generateEventStream: stream created tokenCount=2152
[Osaurus][Stream] Starting stream wrapper for model: OsaurusAI/gemma-4-26B-A4B-it-mxfp4
[Osaurus][Stream] Delta #1: +7.23s total, gap=7.230s, len=10
MLX/ErrorHandler.swift:343: Fatal error:
  [broadcast_shapes] Shapes (1,1,1,2153) and (1,16,1,1024) cannot be broadcast.
  at mlx/c/fast.cpp:629
```

Qwen3.6-35B (no sliding layers) ran fine in the same session. tpae bisected
into vmlx and noted "the broadcast … is happening inside vmlx's sliding-window
attention when the rotating cache wraps." That diagnosis is correct.

The crash always lands on the FIRST batched decode step after prefill —
Delta #1 makes it out (it comes from the solo `stepPrefill` path), then the
second-token forward through `stepBatchDecode` crashes. Affects any prompt
larger than the model's `sliding_window` (Gemma-4 = 1024).

## Root cause

`BatchKVCache` is the per-layer wrapper that the batch engine builds around
each slot's cache every decode step. `BatchKVCache.makeMask` used to compute
the mask's key-length axis as `max(offset_i + n)` across slots, ignoring the
slot cache's `maxSize`:

```swift
// BatchKVCache.swift — BEFORE
public override func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let offsets = slotCaches.map(\.offset)
    return .array(createBatchCausalMask(queryLen: n, offsets: offsets, windowSize: windowSize))
}

// BatchMask.swift — BEFORE
let maxTotal = offsets.map { $0 + n }.max()!
// … mask shape [B, 1, n, maxTotal]
```

For the failing run:

| Piece | Value |
|---|---|
| `slot.offset` (after prefilling 2152 tokens) | 2152 |
| `n` (decode query length) | 1 |
| `offsets.map { $0 + n }.max()` (old mask key dim) | **2153** |
| `slotCache.maxSize` (Gemma-4 sliding layer) | **1024** |
| Shape actually returned by `RotatingKVCache.update` | `[1, kvHeads, 1024, D]` |
| Scores shape at SDPA | `(1, 16, 1, 1024)` |
| Mask shape the old code built | `(1, 1, 1, 2153)` |

`RotatingKVCache.update` for `n=1` once the ring has filled (`offset >=
maxCacheSize`) runs through `updateInPlace`, trims to `maxCacheSize`, writes
the new token at the rotated index, and returns the FULL ring buffer
`[B, H, maxCacheSize, D]`. That is the `(1, 16, 1, 1024)` tail-dim 1024 seen
in the trap. The mask's 2153 never matches. MLX's `broadcast_shapes` aborts.

Everything downstream was built on the assumption that `cache.update`
returns `[B, H, offset_total, D]` (the `KVCacheSimple` contract). That
assumption is wrong for `RotatingKVCache`.

### Why the non-batch path did not crash

On the `Evaluate` / `TokenIterator` path the same model's sliding layer is
a raw `RotatingKVCache` (no `BatchKVCache` wrapping). `RotatingKVCache.makeMask`
checks `maxCacheSize > windowSize` before building a post-wrap mask. For
Gemma-4 `sliding_window == maxCacheSize == 1024`, so that branch returns
`.none` — no mask, symbolic causal, no shape conflict. Wrong for batch
because `BatchKVCache.makeMask` overrides the slot cache's `makeMask` and
never consults the slot's `maxSize`.

### Why the initial first-delta token made it out

Prefill runs solo per slot in `BatchEngine.stepPrefill` — no BatchKVCache,
same raw-cache path as `Evaluate`. First decode token is sampled from the
LAST logit of the prefill forward. That is how Delta #1 emits before the
crash. The second forward goes through `stepBatchDecode`, which wraps each
slot in `BatchKVCache` → mask mismatch → crash.

## Fix

**`Libraries/MLXLMCommon/BatchEngine/BatchMask.swift`** — `createBatchCausalMask`
takes an optional `effectiveKeyLens: [Int]?`. When nil it defaults to
`offset + n` (unchanged behaviour for unbounded caches). When a slot has
wrapped (`effectiveKeyLen_i < offset_i + n`), that slot's row of the mask
is built as "all-true across valid positions, false on padding" — the ring
is full, every stored key is a valid attention target, and keys are in ring
order rather than logical position order so a logical-position causal test
does not apply.

**`Libraries/MLXLMCommon/BatchEngine/BatchKVCache.swift`** — `makeMask`
consults each slot's `maxSize` and passes the capped effective key lengths
to `createBatchCausalMask`:

```swift
let effectiveKeyLens: [Int] = slotCaches.map { slot in
    let logical = slot.offset + n
    if let maxSize = slot.maxSize, logical > maxSize {
        return maxSize
    }
    return logical
}
```

For the failing run the new mask is `(1, 1, 1, 1024)` — exactly what the
attention scores expect. `broadcast_shapes` succeeds, SDPA runs.

### What the fix does NOT change

- Slots backed by `KVCacheSimple` (maxSize nil) behave identically — the
  default `offset + n` key length is passed through untouched.
- Pre-wrap rotating slots (offset < maxSize) also hit the `offset + n`
  path — the ring is still in linear-write mode, positions are logical,
  standard causal applies.
- The `Evaluate` / `TokenIterator` path is untouched. `RotatingKVCache.makeMask`
  still short-circuits to `.none` for n=1 post-wrap because `maxCacheSize ==
  windowSize` (`Gemma4Text.swift:735`, `Gemma4.swift:766`).

### Window-size semantics post-wrap

The old code applied `rinds >= linds - (windowSize - 1)` as a window
constraint even for wrapped caches. This was never meaningful on a wrapped
ring because:

1. The ring stores exactly `maxCacheSize` tokens. If `windowSize == maxCacheSize`
   (the Gemma-3/4 SWA case), every stored key is trivially in-window.
2. If `windowSize < maxCacheSize` (hypothetical), the window test would
   reference LOGICAL positions, but the stored keys are in RING order after
   wrap — the test gives the wrong answer anyway.

The fix drops the logical-position window test for wrapped slots. The
correct semantics — "every stored key is a valid attention target post-wrap"
— is what `RotatingKVCache`'s temporal-order guarantee already delivers
(the slot's `update` returns the buffer; the buffer is the window).

## Reproducer

`Tests/MLXLMTests/BatchEngineTests.swift` → suite `BatchKVCache rotating-slot
(Gemma-4 SWA regression)`. Four tests:

| Test | What it exercises |
|---|---|
| `testMaskMatchesUpdatedKeyShape` | RotatingKVCache wrapped, BatchKVCache.makeMask last axis == slot.update's last axis. Without fix: 41 vs 16, broadcast crash. |
| `testPreWrapMaskUnchanged` | Pre-wrap rotating slot still gets `offset + n` key length. Fix must not regress this. |
| `testMixedBatchWrappedAndUnbounded` | Belt-and-braces cross-topology case. Rotating + KVCacheSimple in the same batch produce a well-formed joint mask. |
| `testCreateBatchCausalMaskWithEffectiveKeyLens` | Low-level helper contract — explicit `effectiveKeyLens` argument caps `maxTotal` correctly. |

Run with:

```bash
swift test --filter 'BatchKVCacheRotatingSlot'
```

All four pass with the fix; `testMaskMatchesUpdatedKeyShape` crashes in MLX
`broadcast_shapes` without it.

## Real-model verification

The user's reported crash does not reproduce after the fix. Verified
end-to-end on the actual crashing architecture
`mlx-community/gemma-4-26b-a4b-it-4bit` (same sliding_window=1024 as the
crashing `OsaurusAI/gemma-4-26B-A4B-it-mxfp4`), running the exact
prompt-token progression tpae reported — 2152 → 3665 → 7851 → 8266 — as
a 4-turn conversation through `BatchEngine.generate(...)`.

Apple M4 Max 128GB, `swift-build -c release`, 2026-04-20 (commits
01707d8 / f966078 / f62d0ce on `fix/osaurus-integration-issues`):

### Gemma-4-26B 4-turn regression (`BENCH_OSAURUS_MULTITURN=1 BENCH_OSAURUS_SIZE=big`)

| Turn | Prompt tokens | TTFT | Total | Chunks | stopReason |
|---|---|---|---|---|---|
| 1 | **2221** (hit tpae's 2152 crash point) | 2016 ms | 2.63 s | 40 | length |
| 2 | **3715** (tpae's turn 1 prompt size) | 3150 ms | 3.78 s | 40 | length |
| 3 | **7869** (tpae's turn 2 post-partial-hit-rollback size) | 8406 ms | 9.08 s | 40 | length |
| 4 | **8362** (tpae's turn 3 size) | 10098 ms | 10.79 s | 40 | length |

All four turns completed. No `broadcast_shapes` abort at any point. All
prompts exceed `sliding_window = 1024`, so each one exercises the ring
wrap + post-wrap mask path that used to crash.

### Gemma-4-e2b (`BENCH_GEMMA4_STRESS=1` + `extraStopStrings`)

Smaller E2B variant to verify the fix across Gemma-4 sub-families:

| Scenario | Turn 1 prompt | Turn 2 prompt | `.chunk` halted at stop? |
|---|---|---|---|
| No stop strings | 1371 tokens, 23 chunks, stop=stop | 1385 tokens, 40 chunks, stop=length | — |
| `extraStopStrings=["holds","distributed"]` | halted before "distributed" | halted before "distributed" | ✓ (stopReason=.stop) |

### Qwen3.6-35B-A3B-MXFP4 4-turn regression (same harness)

Same multi-turn harness against a reasoning + tool-call capable model
to prove the other two library-level features tpae asked about also
work at scale:

| Turn | Prompt | Chunks | Reasoning deltas | Tool calls | stopReason |
|---|---|---|---|---|---|
| 1 | 2194 | 59 | 0 | 0 | length |
| 2 | 3753 | 59 | 0 | 0 | length |
| 3 | 7860 | 59 | 0 | 0 | length |
| 4 | 8385 | 60 | 0 | 0 | length |

(Reasoning delta count is 0 here because the multi-turn harness uses a
raw-string prompt instead of the model's native chat template; the
`BENCH_OSAURUS_REASONING=1` scenario below exercises reasoning events
with proper chat templating and confirms `.reasoning` fires.)

Qwen3.6 is not a sliding-window model (pure attention + SSM), so it
wasn't crashing on tpae's end either. Included here to confirm the
fix doesn't regress non-SWA families and the multi-turn prefix-cache
interaction still works at 8k+ tokens.

### Reasoning channel (`BENCH_OSAURUS_REASONING=1`)

Qwen3.6-35B-A3B-MXFP4 with proper chat template
(`tokenizer.applyChatTemplate(messages:, additionalContext:
["enable_thinking": true])`):

- Reasoning stamp detected: `think_xml`
- Emitted at least one `.reasoning(String)` delta per run
- `.chunk` text contains zero `<think>` or `</think>` markers — the
  parser correctly routes think content to `.reasoning`

### Reproduction commands

```bash
# 1. Gemma-4 26B — the actual crashing architecture, exact tpae
#    prompt progression.
pkill -f xctest; pkill -f RunBench; pkill -f ollama; pkill -f lms
VMLX_CHAT_TEMPLATE_OVERRIDE=$PWD/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja \
  BENCH_OSAURUS_MULTITURN=1 BENCH_OSAURUS_SIZE=big \
  BENCH_MODEL=~/.mlxstudio/models/MLXModels/mlx-community/gemma-4-26b-a4b-it-4bit \
  BENCH_MAX_TOKENS=40 \
  ./.build/release/RunBench

# 2. Gemma-4 e2b with stop strings — confirms extraStopStrings halts
#    upstream and truncates .chunk pre-match.
pkill -f RunBench
VMLX_CHAT_TEMPLATE_OVERRIDE=$PWD/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja \
  BENCH_GEMMA4_STRESS=1 BENCH_STOP_STRINGS='holds,distributed' \
  BENCH_MODEL=~/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit \
  BENCH_MAX_TOKENS=80 \
  ./.build/release/RunBench

# 3. Qwen3.6-35B reasoning channel with enable_thinking template kwarg.
pkill -f RunBench
BENCH_OSAURUS_REASONING=1 \
  BENCH_MODEL=~/.mlxstudio/models/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-MXFP4 \
  BENCH_MAX_TOKENS=200 \
  ./.build/release/RunBench
```

All three scenarios are implemented in `RunBench/Bench.swift` (functions
`runOsaurusMultiturnIntegration`, `runGemma4SlidingStress`,
`runOsaurusReasoningChannel`). RunBench is gitignored — these scenarios
are kept alive via this doc.

## Upstream story

`BatchKVCache` is vmlx-only (no upstream `ml-explore/mlx-swift-lm`
equivalent). No upstream PR to file. If / when ml-explore adds a batch
engine with a similar wrapper, this bug class will need to be avoided
there too — the post-wrap `all-true-on-valid-positions` mask is the
general answer.

## Related

- `docs/SLIDING-WINDOW.md` — canonical sliding-window architecture doc.
  This fix complements SLIDING-1 (disk persistence of rotating caches)
  by closing the last mask-shape correctness gap on the batch path.
- `Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md` §"Production
  bugs fixed" — add this crash class to the table.
- `Libraries/MLXLMCommon/BatchEngine/CompilableRotatingKVCache.swift`
  already ships the "post-wrap mask = all-true on valid positions"
  pattern for the compiled path. The uncompiled `BatchKVCache` path had
  the same requirement but no implementation.
