# DSV4-Flash Production + JangPress Hardening + Land-All-To-Main

**Date:** 2026-05-04
**Author:** Eric (with assistant)
**Status:** APPROVED — implementation in progress (revision 2)

## Revision history

- **r2 (2026-05-04):** User directive — eliminate the `DSV4_LONG_CTX={0,1}`
  toggle and the legacy "sliding-window-only fallback" path. DSV4-Flash is a
  hybrid SWA + CSA + HSA architecture by definition; the no-long-context path
  was actively producing degraded output and was a regression risk on every
  run. **All DSV4 forwards now use the full hybrid attention path
  unconditionally.** Paged cache and prefix cache are upgraded to
  first-class understanding of the hybrid `DeepseekV4Cache` shape
  (rotating window + compressor pool + indexer pool + per-layer
  `compress_ratio`). No regressions tolerated — verification step is
  mandatory after every commit.
- **r1 (2026-05-04):** Initial spec.
**Branch policy:** every change lands directly on `main` (per user directive)

## Goal

Three things, in priority order:

1. **DSV4-Flash JANGTQ** runs at production quality on osaurus through
   vmlx-swift-lm — the same correctness contracts that the codex_dsv4_fixkit
   and `jang_tools.dsv4` Python paths already enforce.
2. **JangPress** is shipped as a stable production cache (post Iter-25/26
   refactor: `JangPressController` + `JangPressMachCache` removed, replaced
   by `JangPressPrestacker` + `LoadConfiguration` + `MmapSafetensorsLoader`).
3. **Everything else uncommitted** (Phase-5 distributed-TP scaffolding under
   `Tools/`, `MLXDistributedTP` Linear/Collectives edits, model touch-ups for
   factory wiring, JangPress test suite refresh) lands on `main`.

Plus: **osaurus integration `.md` documentation** describing what the host app
must observe / call / configure for each subsystem.

## Non-goals

- Re-quantizing DSV4 bundles (this is `jang_tools` work, not Swift runtime).
- Implementing actual JACCL ring `all_sum` runtime (upstream MLX C++ work).
- Any new feature for non-DSV4 models (Llama, Qwen, GLM, etc.) — only
  the wiring touch-ups that already exist in the diff.

## Decomposition

| Stream | Subject | LOC est | Risk |
|---|---|---:|---|
| **A** | Stabilize + commit uncommitted Swift to `main` | (no new code) | low |
| **B** | DSV4 SWA/CSA/HSA correctness fixes | ~600 | medium |
| **C** | DSV4 cache rework for hetero-attention multi-turn | ~300 | medium |
| **D** | DSV4 reasoning-parser stamp verification | ~30 | trivial |
| **E** | JangPress production hardening + tests | ~400 | low |
| **F** | osaurus integration `.md` docs (3 files) | ~700 lines text | low |

Streams B/C are the technical core; A/D/E/F are landing + docs work.

---

## Stream A — Land uncommitted to main

### Inventory of uncommitted changes (from `git status` snapshot)

**Modified (44 files):**

- `Libraries/MLXDistributedTP/{Collectives.swift,LinearLayers.swift}` — Phase-5
  TP collectives + `AllToShardedLinear` / `ShardedToAllLinear` polish.
- `Libraries/MLXLLM/Models/*` (24 files) — model factory adjustments related
  to LoadConfiguration plumbing + minor `JangPressCanonicalExpertAdvisor`
  observation hooks (already used in `DeepseekV4MoE.callAsFunction`).
- `Libraries/MLXLMCommon/Cache/JangPressActivation.swift`,
  `Libraries/MLXLMCommon/Cache/JangPressMmapTier.swift`,
  `Libraries/MLXLMCommon/Cache/JangPressShard.swift`,
  `Libraries/MLXLMCommon/Cache/JangPressLoadOptions.swift` —
  Iter 24-26 jangpress refactor.
- `Libraries/MLXLMCommon/JangLoader.swift`, `Load.swift`,
  `ModelContainer.swift`, `ModelFactory.swift` — LoadConfiguration plumbing.
- `Libraries/MLXVLM/Models/Mistral4VLM.swift` — minor.
- `Package.swift` — adds new targets/products.
- Tests: `JangPressActivationTests.swift`, `JangPressMmapTierTests.swift`,
  `KimiK25RoutingTests.swift`.

**Added (9 new files):**

- `Libraries/MLXDistributedTP/{ShardingPlan.swift, ShardingPlans+Llama.swift}`
- `Libraries/MLXLMCommon/Cache/{JangPressCanonicalExpertAdvisor.swift, JangPressPrestacker.swift, LoadConfiguration.swift}`
- `Libraries/MLXLMCommon/MmapSafetensorsLoader.swift`
- `Tests/MLXDistributedTPTests/{LlamaTPBitIdentityTests.swift, ShardingPlanTests.swift}`
- `Tests/MLXLMTests/{LoadConfigurationTests.swift, MmapSafetensorsLoaderTests.swift}`
- `Tools/{tp-launch-2host.sh, tp-launch.sh, TPRankWorker/main.swift}` — Phase-5
  TP rank-worker launcher (compiles, hangs at upstream `all_sum` per memory;
  scaffold validates plumbing).

**Deleted (5 files):**

- `JangPressController.swift`, `JangPressMachCache.swift` and their three test
  files — Iter 24/25 refactor cleanup. Replacement is `JangPressPrestacker`.

### Commit grouping (target ~6 commits, all on `main`)

1. `feat(jangpress): Iter 26 — Prestacker + LoadConfiguration + MmapSafetensorsLoader`
   — bundles the JangPress refactor + new loader + new options.
2. `refactor(jangpress): drop Controller + MachCache (replaced by Prestacker)`
   — file deletions + corresponding test deletions, no behavior change.
3. `feat(distributed-tp): ShardingPlan + Llama plan + collectives polish + TPRankWorker`
   — bundles the TP scaffolding (compiles + tests pass; runtime hang is
   upstream and tracked).
4. `feat(dsv4): production SWA/CSA/HSA correctness pass` — Stream B (P0 fixes).
5. `feat(dsv4): hetero-attention cache + multi-turn pool persistence` — Stream C.
6. `docs(osaurus): integration guides for DSV4, JangPress, multi-turn cache` — Stream F.

(D goes in 4; E goes in 1 + 2.)

### Verification

- `swift build` clean.
- `swift test --filter <subset>` for each stream's targeted tests.
- Real-bundle smoke decode against `~/models/DeepSeekV4-Flash-JANGTQ2` and
  `~/.mlxstudio/models/JANGQ-AI/DeepSeekV4-Flash-JANGTQ` after Stream B/C.

---

## Stream B — DSV4 SWA/CSA/HSA correctness

### Architecture clarification (per user's "pure SWA CSA HSA combo")

- **SWA** = Sliding-Window Attention — the `RotatingKVCache(maxSize=128)`
  local path used on every layer.
- **CSA** = Compressed-Summary Attention — `Compressor` projects raw KV
  windows into pooled summaries that augment local KV. Active on layers
  with `compress_ratio ∈ {4, 128}` (~41 of 43 layers in DSV4-Flash).
- **HSA** = Hash-Selected Attention — `Indexer` picks `index_topk=512` rows
  out of the compressor's pool per query position. Only on `compress_ratio=4`
  layers.

DSV4-Flash's per-layer `compress_ratios` array determines which combination
fires per layer: layers `0` and `n-1` get `0` (pure SWA, full RoPE/no YaRN);
middle layers alternate `4` (SWA+CSA+HSA, with YaRN on
`compress_rope_theta=160000`) and `128` (SWA+CSA only, with YaRN).

### P0 correctness gaps in current Swift (vs Python ref)

Identified by reading
`~/jang/jang-tools/jang_tools/dsv4/mlx_model.py` 1–1458 and
diffing against
`Libraries/MLXLLM/Models/DeepseekV4.swift` + `DeepseekV4Compressor.swift`
+ `DeepseekV4JANGTQ.swift` + `Libraries/MLXLMCommon/JANGTQKernels.swift`.

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `DeepseekV4.swift:455-470` (affine MoE `SwitchGLU` activation) | activation only clamps gate; up escapes the `±swiglu_limit=10` clip → silu(gate)·up overflows for native FP4 source | wrap activation so SwitchGLU sees the **clamped-up branch baked in**: pass `up * clip-mask` via a custom activation closure that takes both gate and up. Easier path: switch to applying `_dsv4_swiglu` via an MLX-side post-multiply that re-applies the clip after SwitchGLU returns. **Concrete:** add `DeepseekV4Math.dsv4SwitchGLUActivation(limit:)` that returns a closure compatible with MLX `SwitchGLU(activation:)` that internally captures both legs. If `SwitchGLU` cannot accept a 2-arg activation in mlx-swift, replace with a manual `gather_qmm`-equivalent path mirroring Python `_dsv4_swiglu(gate, up, swiglu_limit)`. |
| 2 | `JANGTQKernels.swift:215` (`jangtq_fused_gate_up_swiglu` Metal kernel) | computes `(gv / (1 + exp(-gv))) * uv` — no clamp; same bug the codex Python kit fixes | rewrite kernel source to `gv_limited = min(gv, 10.0f); uv_limited = clamp(uv, -10.0f, 10.0f); out = (gv_limited / (1 + exp(-gv_limited))) * uv_limited;` Add a `swiglu_limit` template/macro arg so the kernel works for both DSV4 (limit=10) and other models (limit=None ↔ 0 sentinel = skip clamp). Update `fusedGateUpSwiGLU` Swift wrapper to pass the limit per call. |
| 3 | `DeepseekV4.swift:411-414` (hash-routing weights) | uses synthetic uniform `routedScalingFactor / topK`; Python uses `mx.take_along_axis(scores, inds, axis=-1)` — actual gate scores per selected expert, then renormalized | match Python: compute `scores = sqrtSoftplus(logits)`, then `weights = take_along(scores, inds, axis=-1)`, normalize if `norm_topk_prob`, scale by `routed_scaling_factor`. |
| 4 | `DeepseekV4.swift:288-297` (compressor-mask pad with all-ones) | for prefill multi-query, allows query `q` to see pooled rows summarizing tokens with positions > q (causal violation); for HSA-gather path it concatenates `(B, 1, L*k, D)` with no mask, so query `i` mixes into query `j`'s selected pool | port Python `_dsv4_window_visibility` and `_dsv4_compressed_visibility` into Swift (`DeepseekV4Math` extension). For prefill (L>1): keep pool flat at `(B, 1, P, D)`, build `local_mask ∥ comp_mask` bool, where `comp_mask = (k_idx+1)*ratio ≤ q+1` AND (when topk available) `k_idx ∈ topK[q]`. For decode (L=1): keep current gather path but use `(B, 1, k, D)` not `(B, 1, L*k, D)`. |
| 5 | `DeepseekV4.swift:875` (`lm_head` matmul) | runs in input dtype (bf16 / fp16) — 4096-dim contraction drifts logits enough to flip arithmetic answers per Python ref | cast both `h` and `lm_head.weight` (or dequantized weight, if `lmHead is QuantizedLinear`) to `fp32` before matmul; cast result back to host dtype. Mirror Python `Model.__call__` lines 1287-1306. |
| 6 | `DeepseekV4.swift:213-216` (per-head Q RMSNorm) | manual `rsqrt(mean(q^2) + eps)` is 3 ops per call across 64 heads × 43 layers; bf16 precision OK on M4 but margin is tight | swap to `MLXFast.rmsNorm(q, weight: cachedOnes, eps:)` with a per-(head_dim, dtype) cached ones tensor (mirror Python `_get_q_norm_ones`). |
| 7 | `DeepseekV4MoE` activation interface | the closure-only `SwitchGLU` API blocks fix #1 | add a `TwoArgActivation` overload to the Swift `SwitchGLU` (or fall back to manual matmul path for affine routed experts when `swiglu_limit > 0`). Reuse for the JANGTQ path so kernel-level fix #2 isn't the only line of defense. |

### Files touched (Stream B)

- `Libraries/MLXLLM/Models/DeepseekV4.swift`
- `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift`
- `Libraries/MLXLLM/Models/DeepseekV4JANGTQ.swift`
- `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` (add visibility-mask helpers + `_get_q_norm_ones` cache)
- `Libraries/MLXLMCommon/JANGTQKernels.swift` (kernel patch + Swift wrapper signature)
- `Libraries/MLXLMCommon/TurboQuantSwitchLinear.swift` (thread `swiglu_limit` through to kernel call)

### Tests (Stream B)

- New `Tests/MLXLMTests/DeepseekV4SwiGLULimitTests.swift` — verify dense + routed
  paths produce values bit-identical (within fp32 tolerance) to a fp32
  reference computation of `silu(min(gate,10))·clip(up,±10)`.
- New `Tests/MLXLMTests/DeepseekV4VisibilityMaskTests.swift` — verify causal +
  window + pool masks for both prefill (L>1) and decode (L=1) shapes.
- New `Tests/MLXLMTests/DeepseekV4HashRoutingTests.swift` — verify hash gate
  produces non-uniform weights matching `take_along(scores, inds)`.
- Existing `KimiK25RoutingTests.swift` keeps passing (regression guard).

---

## Stream C — Cache rework for hetero attention + multi-turn

### Problems

Per Python `DeepseekV4Cache.trim()` (mlx_model.py:453-550):

- Multi-turn chat (`/v1/chat/completions` + prefix-cache reuse) trims the
  KV by `n` tokens between turns. The pool buffers in `compressor_state` /
  `indexer_state` were built from raw KV positions including the just-trimmed
  output tokens — restoring them on the next turn re-introduces those
  contaminated rows.
- Symptom on Python before the fix: "polite-assistant attractor loops" on
  long multi-turn chats. Bench mode (no prefix-cache reuse) was unaffected.

Current Swift `DeepseekV4Cache.trim(_:)` (`DeepseekV4Compressor.swift:81`)
delegates only to `local.trim(n)` — same bug.

### Fix (r2 — pure long-context, no fallback)

1. **Remove `DSV4_LONG_CTX` env knob entirely.** By default, both
   `DeepseekV4Model.newCache(parameters:)` and
   `DeepseekV4JANGTQModel.newCache(parameters:)` return:
   - `RotatingKVCache(maxSize: slidingWindow, keep: 0)` for layers with
     `compress_ratio == 0` (pure SWA — layers 0 and n-1).
   - `DeepseekV4Cache(slidingWindow: ..., compressRatio: cr)` for every
     other layer (CSA on `cr=128`, CSA+HSA on `cr=4`).
   - `DSV4_KV_MODE` env override is **kept** (`sliding | full | tq`) only
     for diagnostics. `full` and `tq` deliberately replace the hybrid pool
     with `KVCacheSimple`; `tq` can then let `BatchEngine.maybeCompress`
     swap to `TurboQuantKVCache` once offset crosses the min-token threshold.
2. `DeepseekV4Cache.init` REQUIRES `compressRatio: Int` — no optional.
   Old `DeepseekV4Cache(slidingWindow:)` removed; one callsite migrated.
3. `DeepseekV4Cache.trim(_:)` runs proportional pool-row truncation
   matching `llama.cpp dsv4_clear_rows`:
   - delegate to `local.trim(n)`
   - clear `compBufferKV`/`compBufferGate`/`idxBufferKV`/`idxBufferGate`
     unconditionally (incomplete-window buffers are start_pos-keyed and
     invalidated by ANY trim)
   - drop trailing `max(1, n / compressRatio)` pool rows from
     `compPooled` and `idxPooled` (zero rows when `n == 0`)
4. `DeepseekV4Cache.copy()` deep-copies the pool state arrays (currently
   constructs a fresh cache without copying pool — silent multi-session
   state leak).
5. **Prefix cache integration (TQDiskSerializer)** — the rotating window
   state plus the compressor pool, indexer pool, and per-branch
   incomplete-window buffers ALL serialize and round-trip. Each pool
   tensor stamps its `compressRatio` and `pooledOffset` so a restored
   cache can resume its proportional-row arithmetic exactly. New
   `LayerKind.deepseekV4 = 7` discriminator added; canonical doc updated.
6. **Paged cache (PagedCacheManager) — first-class hybrid support.**
   `PagedCacheManager` learns about `DeepseekV4Cache` as a layer kind:
   - allocates an SWA block ring identical to `RotatingKVCache`-backed
     layers (size = `sliding_window`)
   - allocates a separate "pool block ring" per branch (compressor +
     indexer) sized in pool-rows (each pool row is one
     `compress_ratio`-sized chunk of raw KV)
   - block hash includes per-layer `compress_ratio` so blocks are not
     accidentally shared between `cr=4` and `cr=128` layers.
   For BatchEngine, `PagedCacheManager.allocate(model:)` dispatches per
   layer's actual cache class; the existing scheduler still sees a
   uniform `[KVCache]` array.
7. **No fallback path remains.** Every code path that previously branched
   on `longCtxEnabled` is removed. Any callsite that constructed a
   `DeepseekV4Cache(slidingWindow:)` without `compressRatio` is migrated.

### Files touched (Stream C r2)

- `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift` — `DeepseekV4Cache`:
  required `compressRatio`, proportional `trim`, deep `copy`, full
  `state`/`metaState` round-trip including pools.
- `Libraries/MLXLLM/Models/DeepseekV4.swift` — `newCache` rewritten to
  always allocate the hybrid cache; `DSV4_LONG_CTX` removed.
- `Libraries/MLXLLM/Models/DeepseekV4JANGTQ.swift` — same.
- `Libraries/MLXLMCommon/Cache/PagedCacheManager.swift` — first-class
  hybrid support: SWA block ring + per-branch pool block ring keyed on
  `compress_ratio`.
- `Libraries/MLXLMCommon/Cache/CacheCoordinator.swift` — admission path
  honors `DeepseekV4Cache`'s `pooledNbytes` separately from rotating
  `nbytes` so the coordinator KV-budget includes pool rows.
- `Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift` — `LayerKind.deepseekV4 = 7`
  discriminator + serialize/restore of pool tensors and per-branch
  buffer state. Canonical doc `docs/SLIDING-WINDOW.md` updated to
  describe the new layer kind.
- `Libraries/MLXLMCommon/Cache/CacheBlock.swift` — pool-block variant
  + hash key extension `compress_ratio:Int`.
- `Libraries/MLXLMCommon/Cache/BlockHashMap.swift` — pool-block lookup.

### Tests (Stream C r2)

- New `Tests/MLXLMTests/DeepseekV4CacheTrimTests.swift` — verify pool-row
  truncation matches `n // compress_ratio + 1` boundary semantics.
- New `Tests/MLXLMTests/DeepseekV4MultiTurnTests.swift` — simulate two-turn
  chat with prefix-cache reuse on a short synthetic input; verify generation
  is deterministic and pool buffers are cleared between turns.
- New `Tests/MLXLMTests/DeepseekV4CacheRoundTripTests.swift` — disk round-trip
  with `TQDiskSerializer` of a populated `DeepseekV4Cache` (pool +
  buffers + rotating window) yields a byte-identical second cache.
- New `Tests/MLXLMTests/PagedCacheHybridTests.swift` — exercise
  `PagedCacheManager` against a synthetic two-layer model with one
  `cr=0` SWA layer and one `cr=4` hybrid layer; verify block reuse +
  hash isolation.

---

## Stream D — Reasoning parser stamp

### Status

Already covered. `Libraries/MLXLMCommon/ReasoningParser.swift:487` includes
`"deepseek"` in `thinkXmlPrefixes`, which catches `deepseek_v3`, `deepseek_v4`,
`deepseek_r1`. `reasoningStampFromModelType("deepseek_v4")` → `"think_xml"`.

### Verification action

Add a one-line case to `Tests/MLXLMTests/ReasoningParserTests.swift`
(or wherever it lives — verify) that explicitly asserts
`reasoningStampFromModelType("deepseek_v4") == "think_xml"`. No code change
beyond the test.

---

## Stream E — JangPress production hardening

### Status

Iter 24-26 refactor partly committed:

- 332e20a, 7050f7d, a3366a5, a59ccc3 — distributed
- d2bc526 — defer all tier work until first inference (iter 24)
- 9ea2d4c — cap allocator cache + detect mixed shard sets + split stacked tiles (iter 25)
- 1bc6e6b — bundleStats/startCold for stacked-tile split

Uncommitted:

- `JangPressController.swift` + `JangPressMachCache.swift` deleted (replaced
  by `JangPressPrestacker.swift`)
- `JangPressActivation.swift` reduced from 133 lines diff (mostly deletions
  + simplifications)
- `JangPressMmapTier.swift` reduced ~200 lines
- New `JangPressCanonicalExpertAdvisor.swift`, `JangPressPrestacker.swift`,
  `LoadConfiguration.swift`, `MmapSafetensorsLoader.swift`
- Tests: 3 deleted (Controller, MachCache, two perf-bench files); 1 modified
  (Activation), 1 modified (MmapTier)

### Production-quality bar

1. **Tests**: replace deleted Controller/MachCache tests with equivalent
   coverage on `JangPressPrestacker`:
   - cold-start ordering (defer-tier-work behavior from iter 24)
   - mixed shard sets (iter 25)
   - stacked-tile split (iter 25)
   - allocator-cache cap (iter 25)
2. **Existing failing tests**: re-run `swift test --filter JangPress` and
   ensure the modified `JangPressActivationTests` + `JangPressMmapTierTests`
   pass on M5 Max.
3. **Documentation**: docstring sweep of `JangPressPrestacker` + the new
   `LoadConfiguration` contract (which is the JangPress entry point from
   the loader) — what callers must opt into and what defaults look like.

### Files touched (Stream E)

- `Tests/MLXLMTests/JangPressPrestackerTests.swift` (new — replaces deleted
  Controller/MachCache tests)
- `Tests/MLXLMTests/LoadConfigurationTests.swift` (already exists, untracked)
- `Tests/MLXLMTests/MmapSafetensorsLoaderTests.swift` (already exists, untracked)
- doc-comment touch-ups inside the new files (no behavior change)

### Verification

- `swift test --filter "JangPress|LoadConfiguration|MmapSafetensors"` green.
- Real-bundle JangPress decode against a JANGTQ2 model from `~/models`.

---

## Stream F — osaurus integration `.md` documentation

Three new docs under `docs/`:

### `docs/OSAURUS-DSV4-INTEGRATION.md`

Audience: osaurus host-app engineers integrating DSV4-Flash JANGTQ.

Sections:

1. Bundle requirements (`config.json`, `jang_config.json`, chat-template
   present, JANGTQ2 / JANGTQ4 layout, MTP weights kept).
2. Minimum SDK call sequence — `LoadConfiguration` → `ModelFactory.load` →
   chat session.
3. Cache mode env vars: `DSV4_KV_MODE={sliding,full,tq}`; unset/`sliding`
   is the production SWA+CSA+HSA path, while `full` and `tq` are
   diagnostics that deliberately drop the hybrid pool.
4. Reasoning parser (`<think>` envelope is auto-stripped before tool-call
   processor runs; how to surface it in the host UI as a separate stream).
5. SWA/CSA/HSA architecture summary (one paragraph + table) so on-call
   engineers know what to expect from the per-layer attention.
6. Multi-turn chat caveats — pool buffers are intentionally ephemeral; no
   action needed but document the rationale so nobody re-enables a "save
   pool to disk" path that breaks correctness.
7. Known-good bundle paths and smoke-test commands.

### `docs/OSAURUS-JANGPRESS.md`

Audience: osaurus host-app engineers using JangPress as the production
weights cache.

Sections:

1. What JangPress is (memory-mapped tile cache with prestack defer-to-first-
   inference).
2. `LoadConfiguration` knobs the host can set: enable/disable JangPress,
   eviction caps, prestack budget, expert-advisor opt-in, allocator cap
   override.
3. Cold-start cost (first inference triggers tier work) vs warm decode.
4. Disk + memory pressure: what to clean up, what to keep across launches.
5. Diagnostic logging hooks the host can subscribe to.
6. What changed between Iter 23 (Controller / MachCache) and Iter 26 (Pre-
   stacker) so anyone migrating off the old API knows the rename map.

### `docs/OSAURUS-CACHE-CONTRACT.md`

Audience: anyone integrating prefix cache, paged cache, or the BatchEngine
multi-turn flow.

Sections:

1. Per-layer cache types: `KVCacheSimple`, `RotatingKVCache`,
   `DeepseekV4Cache`, `TurboQuantKVCache`, SSM caches — what each
   guarantees and when to use it.
2. Hetero-attention cache rules — DSV4 mixes `RotatingKVCache` and
   `DeepseekV4Cache` per layer; the BatchEngine + scheduler must call
   `make_cache()` per session and not share caches across requests.
3. Multi-turn / prefix-cache reuse contract — what gets serialized
   (rotating window state) vs what's recomputed (compressor + indexer
   pools). Why pool state is intentionally ephemeral.
4. Paged cache compatibility: SWA-only models OK; DSV4 with `compress_ratio>0`
   layers explicitly NOT supported (and why).
5. Trim semantics — what `cache.trim(n)` does on each cache type, especially
   the pool-row proportional truncation on DSV4.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `SwitchGLU` API may not accept 2-arg activation in mlx-swift, blocking fix #1 | Fallback path: replace `SwitchGLU` with explicit `gather_qmm` + post-multiply when `swiglu_limit > 0`. Keep behind `DeepseekV4MoE.useFusedSwiGLU` runtime flag during validation. |
| Patching `JANGTQKernels.swift` Metal source could regress non-DSV4 JANGTQ models | Pass `swiglu_limit=0` (sentinel "no clamp") for non-DSV4 callers; existing callers default to 0 → zero behavior change. New unit test covers both branches. |
| Visibility-mask helpers introduce subtle off-by-one | New unit test compares Swift mask shape + values against handwritten reference for several `(L, offset, window, ratio)` tuples. |
| Phase-5 TP scaffolding may not compile after rebase | `swift build` runs in the verification step before Stream A's commits. |
| Pool-truncation regression breaks single-turn decode | Existing single-turn tests stay green; new multi-turn test isolates the regression surface. |

## Build sequence

1. **Stream A (commit 1, 2, 3)** — land jangpress + distributed plumbing as
   pure refactor with no behavior change in DSV4. Verify `swift build` +
   `swift test` per group.
2. **Stream B + D (commit 4)** — DSV4 correctness pass. Verify `swift test
   --filter "DeepseekV4|JANGTQKernel"` + JANGTQ2 real-bundle decode smoke.
3. **Stream C (commit 5)** — cache rework + multi-turn. Verify the new
   multi-turn test + a 5-turn synthetic chat smoke.
4. **Stream F (commit 6)** — osaurus integration docs.
5. **Push to `main`** in one final `git push`.

## Approval gate

Before implementation begins, user reviews this spec and either:

- ✅ Approves as-is.
- ✏️ Asks for revisions (different commit grouping, different doc audience,
  different fix priority, etc.).

After approval the implementation proceeds task-by-task, with the spec as
the authoritative reference for what "done" looks like.
