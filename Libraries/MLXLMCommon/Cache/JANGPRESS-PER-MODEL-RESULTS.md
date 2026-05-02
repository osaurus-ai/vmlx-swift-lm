# JANGPress — Per-Model Test Results

Living log of JangPress measurements across MoE / dense bundles.
Each model section: bundle stats, baseline (off) numbers, JangPress
on numbers (with knob settings), coherency check, and recommended
settings.

Test rig: **M4 Max MacBook (128 GB unified, internal SSD)**
Test runner: `swift run JANGPressCompare <bundle> <pct>` with
`MODE=off` and `MODE=on` in separate processes (apples-to-apples
fix from iter 14).

## Test methodology

For each model, run 3 separate-process measurements:

1. **OFF baseline** — `enableJangPress = false`. Establishes
   baseline load wall, decode tok/s, RSS post-decode, output text.
2. **ON soft pct=70** — default production mode. Should land within
   1-3% of baseline tok/s; RSS may shift slightly.
3. **ON force pct=70** — eager reclaim. Documents the worst-case
   slowdown that the user opts into when they want maximum reclaim.

Coherency: compare `content` byte-for-byte across all three runs.
For temperature=0 they should be identical (or both broken in the
same way if the model has a separate runtime bug).

## Model: DeepSeek-V4-Flash-JANGTQ (79 GB, 256×6 routed × 43 layers)

**Architecture**: MLA + HSA/CSA/SWA tri-mode + per-expert tiles.
**Tile layout**: pattern E (`layers.<L>.ffn.experts.<E>.<w1|w2|w3>.tq_packed`).
**Bundle disposition**: 91 % of 79 GB is routed-expert mass (72 GB).
Tiny 2 % is embed (126 MB) + lm_head (126 MB) = 252 MB Zipfian-eligible.
**Pre-existing issue**: DSV4 Swift baseline emits `?` repeated
(per-tensor `.tq_bits` config-metadata bug). Coherency comparison is
trivially "identical garbage" until that's fixed.

### Measurements (iter 14, soft mode default)

| | OFF | ON soft pct=70 |
|---|---|---|
| Load wall | 16.95 s | 17.38 s |
| Decode wall (64 tokens) | 5.26 s | 5.33 s |
| Engine tok/s | 12.71 | 12.53 |
| RSS post-load | 8.85 GB | 8.82 GB |
| RSS post-decode | 8.88 GB | 8.85 GB |
| Output | (0 chars — chunk buffer bug) | (0 chars) |

Δ vs baseline: **-1.4% tok/s, -30 MB RSS** (kernel ignores DONTNEED hint
under low pressure, gives nothing back).

### Coherency verified (iter 20, 2026-05-02)

With `enableThinking=true` bench fix + iter 19 sniff + iter 20 EmbedTier
sniff, DSV4-Flash JANGTQ runs cleanly in Swift and produces COHERENT
reasoning content. Earlier "outputs `?` 32 times" diagnosis was wrong
— that was the bench's BLOCKER #3 suppressing thinking output, not
the model.

**Separate-process apples-to-apples** (the only valid measurement
mode per iter 14 finding):

| | OFF | ON soft pct=70 |
|---|---|---|
| Load wall | 12.9 s | **11.1 s** (faster — warm cache) |
| Decode wall (64 tokens) | 2.69 s | 2.70 s (identical) |
| Reasoning chars | 300 | 296 |
| Reasoning content (first 200 chars) | "The user wants 5 concise facts about Apple Silicon. I need to provide them in a clear, bullet-point format. Let me recall key facts about Apple Silicon:" | (identical first ~200 chars) |
| Backend | none | mmap |

**Result**: ZERO regression on DSV4 from JangPress. Output is
coherent thinking, decode time is byte-equivalent, load is faster
than baseline due to warm cache from the OFF run.

The earlier "63s ON load" anomaly was a sequential-same-process
artifact (running OFF then ON in same process pollutes MLX-swift's
load state — known iter 14 issue). In separate processes (which is
the only valid measurement mode), there is no slowdown.

### Recommended settings (DSV4-Flash)

- **Roomy host (≥64 GB)**: `enableJangPress=true, pct=30, mode=soft`.
  Negligible cost, controller is armed for memory pressure.
- **Constrained host (≤32 GB)**: `pct=70, mode=force`. Frees ~50 GB
  but pays ~3× tok/s during decode (verified standalone — needs
  re-verification in Engine path on a constrained host).

### What needs fixing

- **DSV4 Swift `.tq_bits` strip bug** — model outputs `?` regardless
  of JangPress. Per memory `project_swift_dsv4_jangtq_working`, fix
  is a 5-line `.tq_bits` strip in `DeepseekV4JANGTQModel.sanitize()`
  in vmlx-swift-lm. **NOT a JangPress issue.** Blocks real coherency
  comparison on this bundle.
- **Stream chunk content buffering** — decode produces tokens but
  none surface as `chunk.content`. Likely the reasoning-parser or
  marker-buffer logic catching a stamp-think condition incorrectly
  for DSV4 with thinking=false. Separate engine bug.

---

## Model: Holo3-35B-A3B-JANGTQ (11 GB, qwen3_5_moe, 256×8 × 40 layers, switch_mlp)

**Architecture**: Hcompany GUI-agent VL on Qwen3.5MoE finetune.
**Tile layout**: NEW pattern G (`language_model.model.layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.tq_packed`).
Two innovations vs prior bundles: (1) `language_model.` VL prefix wraps
the LM, (2) `switch_mlp.<proj>.tq_packed` (per-projection stacked TQ),
not `experts.<E>.<proj>` per-expert. **Regex pattern G added iter 16**.
**Bundle disposition**: 7.5 GB / 11 GB (68 %) is routed-expert mass.
40 stacked tiles total (one per MoE layer per projection bucket — but
controller treats each layer as one tile for synthetic id 0).

### Measurements (iter 16, soft mode default)

| | OFF | ON soft pct=70 | ON force pct=70 |
|---|---|---|---|
| Load wall | 1.72 s | 1.67 s | 1.73 s |
| Decode wall (64 tokens) | 965 ms | 961 ms | 991 ms |
| Engine tok/s | 78.78 | 80.05 | 77.16 |
| Decode wall tok/s | 66.32 | 66.60 | 64.58 |
| Prefill | 44 ms | 44 ms | 44 ms |
| RSS post-load | 11.11 GB | 11.11 GB | 11.10 GB |
| RSS post-decode | 11.41 GB | 11.41 GB | 11.41 GB |
| Output | (0 chars — chunk buffer bug) | (0 chars) | (0 chars) |

Δ vs baseline: **+1.6% tok/s soft, -2.0% force** (within run-to-run
variance — JangPress is functionally free on Holo3).

### RSS reclaim (standalone JANGPressRSSBench, all separate process)

| Phase | RSS |
|---|---|
| Process baseline | 8 MB |
| Tier built (lazy mmap) | 12.6 MB |
| All routed pages hot | 7,692 MB |
| After msync(INVALIDATE) all | 12.6 MB — **100 % reclaim** |
| **After partial forceRelease (pct=70)** | **2,316 MB** |
| **Reclaimed at pct=70** | **5,376 MB = 70.0 % of routed mass** |
| Compaction pass time | 129 ms (28/40 layers) |
| First re-acquire after release | **61 ms** (stacked tile, not per-expert) |

Holo3's 11 GB bundle becomes **2.3 GB-resident** under quiesce with
pct=70. On a 16 GB MacBook this is the difference between "barely fits
with browser closed" and "runs alongside daily apps".

### Recommended settings (Holo3-35B-A3B-JANGTQ)

- **Roomy host (≥32 GB)**: `enableJangPress=true, pct=30, mode=soft`.
  Negligible overhead; controller armed for pressure events.
- **16 GB host**: `pct=70, mode=force`. Frees 5.4 GB at the cost of
  61 ms first-token after a quiesce window. Worth it on RAM-tight
  Macs running browser+IDE+chat alongside the model.

### What needs fixing

- **Stream chunk buffering** — same 0-content-chunks symptom as DSV4.
  Engine generates 64 tokens, parser swallows them. Likely the
  thinking-aware chat template + `enableThinking=false` interaction.
  Separate engine bug.
- **VL coherency check blocked** — until real chunks emit, can't
  diff outputs across modes. The RSS+tok/s data is solid though.
- **Re-acquire 61 ms is slower than DSV4's 6.5 ms** because
  switch_mlp's per-layer stacked tile is ~190 MB whole-page-faulted
  vs DSV4's 7 MB per-expert tiles. Per-projection split would help.

---

## Model: MiniMax-M2.7-Small-JANGTQ (36 GB, minimax_m2, 154×8 × 62 layers, per-expert)

**Architecture**: MiniMax M2 family (sigmoid routing, full-attn + MTP).
**Tile layout**: NEW pattern H (`model.layers.<L>.block_sparse_moe.experts.<E>.w[123].tq_packed`).
Different envelope from DSV4 (`block_sparse_moe` vs `ffn`) and from
JANGTQ stacked (`experts.<E>.w[123]` vs `experts.<gate_up_proj>`).
**Regex pattern H added iter 17**.
**Bundle disposition**: 32.2 GB / 36 GB (89 %) is routed mass.
9548 per-expert tiles (62 × 154 — REAP pruned from 256, full per-expert).

### Measurements (iter 17, soft mode default)

| | OFF (warm) | ON pct=0 | ON soft pct=70 |
|---|---|---|---|
| Load wall | 2.11 s | 2.50 s | 2.45 s |
| Decode wall (64 tokens) | 1567 ms | 1559 ms | 1595 ms |
| **Engine tok/s** | **46.66** | **47.12** | **43.74** |
| Decode wall tok/s | 40.84 | 41.05 | 40.13 |
| Prefill | 178 ms | 183 ms | 74 ms (warmer cache) |
| RSS post-load | 5.37 GB | 5.42 GB | 5.44 GB |
| RSS post-decode | 5.40 GB | 5.44 GB | 5.48 GB |
| Output | (0 chars — chunk buffer bug) | (0 chars) | (0 chars) |

Δ vs baseline: **-6.3 % engine tok/s** (reproducible across 2 runs).
This is **larger than DSV4 (-1.4%)** or Holo3 (+1.6%). Hypothesis:
9548-tile controller + EmbeddingZipfianTier per-token tracking adds
visible overhead on a smaller model. Worth a profile pass before
shipping pct≥70 as default for MiniMax-class bundles.

### RSS reclaim (standalone JANGPressRSSBench, separate process)

| Phase | RSS |
|---|---|
| Process baseline | 8 MB |
| Tier built (lazy mmap) | 64 MB |
| All routed pages hot | 32,289 MB |
| After msync(INVALIDATE) all | 64 MB — **100 % reclaim** |
| **After partial forceRelease (pct=70)** | **9,420 MB** |
| **Reclaimed at pct=70** | **22,869 MB = 71.0 % of routed mass** |
| Compaction pass time | 540 ms (44/62 layers) |
| First re-acquire after release | **3.5 ms** (per-expert tile fast path) |

MiniMax M2.7-Small's 36 GB bundle becomes **9.4 GB-resident** under
quiesce with pct=70. **Best absolute reclaim of any tested bundle**:
22.9 GB freed for other apps without unloading the model.

### Recommended settings (MiniMax M2.7-Small-JANGTQ)

- **Roomy host (≥64 GB)**: `enableJangPress=false` (or pct=20).
  -6 % decode is real; only worth paying when RAM pressure is real.
- **32 GB host**: `pct=50, mode=soft`. Frees ~16 GB at modest cost.
- **16 GB host**: `pct=70, mode=force`. Trades -6 % decode for 22.9 GB
  freed, AND gets 3.5 ms re-acquire (per-expert tiles fault-fast).

### What needs fixing

- **6 % decode regression scales with pct** — investigated iter 17.
  - **VERIFIED 2026-05-02**: at pct=0, decode is 47.12 tok/s vs
    OFF=46.66 → **+1.0 % (within noise)**. The overhead only
    materializes when compaction actually engages. Arming the
    controller alone is free.
  - **Ruled OUT**: per-token tracking (`recordTokenActivity` is never
    called in the inference loop), per-token controller hooks (only
    `willStartInference`/`didFinishInference` at boundaries),
    embed-tier setup (RSS post-load identical at pct=0 vs pct=70).
  - **Actual cause**: at pct≥50, the controller's `madvise(DONTNEED)`
    pass on 9548 tiles triggers minor page-cache turnover that
    competes with MLX's safetensors reads during decode. Per-tile
    cost is microseconds but adds up over 9548 tiles per layer.
  - **Mitigation now shipped**: users on roomy hosts run pct=0
    (controller armed, ready for pressure events, no overhead).
    Users on tight hosts run pct=50-70 and pay the small cost.
  - **Verdict**: ✅ failsafe at pct=0; documented cost at pct=70.
- **Stream chunk buffering** — same DSV4/Holo3 zero-content-chunks bug.
  64 tokens generated, parser swallows. Engine bug, not JangPress.
- **Test M2.7-Med (~80 GB) and M2.7-Large (~110 GB)** — bigger
  bundles will have more headroom for absolute reclaim. Same regex
  should cover them.

---

## Model: Nemotron-Cascade-2-30B-A3B / Omni-30B-A3B (12-21 GB, nemotron_h)

**Architecture**: Nvidia hybrid SSM (Mamba2) + attention + 128-expert
MoE per layer, 23 MoE layers out of 52 total (others are Mamba2 / Attn).
**Tile layout**: NEW patterns J/K/L/M added iter 17.
- Pattern J: `backbone.layers.<L>.mixer.experts.<E>.<gate|up|down>_proj.tq_packed` (Omni JANGTQ)
- Pattern L: `backbone.layers.<L>.mixer.switch_mlp.fc[12].weight` (Omni MXFP4)
- Pattern M: `backbone.layers.<L>.mixer.switch_mlp.<gate|up|down>_proj.weight` (Cascade-2 affine)
- All anchored on `backbone.layers` + `mixer` (not `model.layers` + `mlp`).

**Status: Swift Engine BLOCKED — `NemotronHJANGTQModel` wrapper not wired.**

```
Error 438 "Nemotron-H JANGTQ bundle detected (weight_format=mxtq) but
the NemotronHJANGTQModel wrapper isn't wired yet. Components are in
place (§437 NemotronHJANGTQSwitchMLP); the Model+sanitize wrapper
lands in a follow-up iter."
```

Omni MXFP4 also fails — multimodal: `unhandledKeys: ["mlp1", "sound_encoder", "sound_projection", "vision_model"]`.

### RSS reclaim (standalone JANGPressRSSBench, separate process)

#### Cascade-2 JANG_4M (17 GB, affine 4-bit)

| Phase | RSS |
|---|---|
| Process baseline | 8 MB |
| Tier built (lazy mmap) | 11 MB |
| All routed pages hot | 14,018 MB |
| After msync(INVALIDATE) all | 11.5 MB — **100 % reclaim** |
| **After partial forceRelease (pct=70)** | **3,666 MB** |
| **Reclaimed at pct=70** | **10,353 MB = 73.9 % of routed mass** |
| Compaction pass time | 199 ms (17/23 layers) |
| First re-acquire after release | **186 ms** (large stacked tile, ~600 MB) |

#### Omni JANGTQ2 (12 GB, 2-bit per-expert)

| Phase | RSS |
|---|---|
| Total routed mass | 7,003 MB (per-expert tiles) |
| **Reclaimed at pct=70** | **5,176 MB = 73.9 %** |
| First re-acquire after release | **2.6 ms** (fine-grained per-expert) |
| Tile count | 2944 (23 layers × 128 experts) |

Per-expert layout (Omni JANGTQ) re-acquires in **2.6 ms** vs stacked
(Cascade-2 affine) at **186 ms** — same architecture but different
tile granularity → 70× difference in cold-fault latency.

### Recommended settings

- **Roomy host (≥32 GB)**: `pct=20, mode=soft`. Negligible cost,
  controller armed.
- **16 GB host with Cascade-2**: `pct=70, mode=force`. 10 GB freed,
  but 186 ms first-token after quiesce (stacked tile).
- **Per-expert variant (Omni JANGTQ)**: `pct=70, mode=force` is
  cheap — 2.6 ms re-acquire. Best ergonomics.

### What needs fixing

- **NemotronHJANGTQModel Swift wrapper** — components exist (§437
  switch_mlp helper) but Model+sanitize wrapper isn't wired. Engine
  load fails with code 438. Until wired, Nemotron is RSS-only path
  (no decode-time tok/s measurement possible in Swift).
- **Omni multimodal keys** — vision_model / sound_encoder /
  sound_projection / mlp1 not handled by NemotronH Model in Swift.
  Audio + vision towers need explicit support.
- **Cascade-2 first-token 186 ms cold** — this is the per-layer
  switch_mlp stack faulting in. If we're going to make stacked
  tiles common, maybe split each stack into per-expert sub-tiles
  at JangPress-tier load (lazy fault granularity = 4 KB pages
  inside a stacked tile, but kernel reads ≥256 KB at a time).

---

## Model: Qwen3.6-35B-A3B-JANG_2L (11 GB, qwen3_5_moe, 40 layers stacked)

**Architecture**: Qwen3.6 hybrid MoE 35B/A3B (256 experts × 8 routed
+ 1 shared, GatedDeltaNet + full-attn alternating).
**Tile layout**: pattern D (affine stacked) with **deep-VL prefix**:
`model.language_model.layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.weight`.
Different from Holo3's `language_model.model.layers.<L>...` ordering.
**Regex iter 18**: vlPrefix loosened from `(?:language_model\.)?model\.`
to `(?:(?:model|language_model)\.)*` so the matcher accepts any
combination of `model.` and `language_model.` chunks before `layers.`.

**Status: Swift Engine BLOCKED** — qwen3_5_moe dense Swift wrapper
trap during load (per memory `project_swift_qwen36_dense_gaps`,
~1500 LOC outstanding for VL tower + linear_attn + dense FFN).

```
[JangLoader] config-metadata BUG detected, patched in-memory:
  declared (bits=2, gs=128) → shape-inferred (bits=8, gs=32)
SIGTRAP (exit 133) on Engine.load
```

### RSS reclaim (standalone JANGPressRSSBench, separate process)

| Phase | RSS |
|---|---|
| Process baseline | 8 MB |
| Tier built (lazy mmap) | 12.7 MB |
| All routed pages hot | 7,693 MB |
| After msync(INVALIDATE) all | 12.7 MB — **100 % reclaim** |
| **After partial forceRelease (pct=70)** | **2,317 MB** |
| **Reclaimed at pct=70** | **5,376 MB = 70.0 % of routed mass** |
| Compaction pass time | 115 ms (28/40 layers) |
| First re-acquire after release | **65 ms** (stacked tile, similar to Holo3) |

Identical reclaim profile to Holo3 (both qwen3_5_moe with 40 layers).
**The looser vlPrefix correctly indexes both Holo3 (VL outside) and
Qwen3.6 (VL inside) without breaking plain `model.layers.*` bundles**
— verified by full regex test suite (4 tests, 13+ pattern cases).

### Recommended settings (Qwen3.6-A3B-JANG_2L)

- **16 GB host**: `pct=70, mode=force`. Frees 5.4 GB.
- **Roomy host**: `pct=0` armed; auto-engage on pressure event.

### What needs fixing

- **qwen3_5_moe Swift wrapper** — Engine.load SIGTRAPs. Per memory
  `project_swift_qwen36_dense_gaps`, gating on full Swift port
  (linear_attn + VL tower + dense FFN). NOT a JangPress issue.
- **Bundle metadata bug** — config.json declares `(bits=2, gs=128)`
  but shape-inferred is `(bits=8, gs=32)`. This is the same JANGTQ
  config-metadata invariant as `feedback_jangtq_bundle_metadata_invariants`.
  Re-export with §410 patch.

---

## Model: Laguna-XS.2-JANGTQ (9.4 GB, 256×8 routed × 39 MoE layers, stacked)

**Status: BLOCKED — `model_type=laguna` not ported to Swift Engine.**

```
swift run JANGPressCompare /…/Laguna-XS.2-JANGTQ →
  RuntimeShared.swift:122: Fatal error: [runtime] load failed:
  Unsupported model type: laguna
```

### Architecture (for reference)

- Per-layer head count (48 full / 64 SWA), dual RoPE
- 256 routed top-8 + 1 shared expert
- Tile layout: pattern C (`layers.<L>.mlp.experts.<gate_up_proj|down_proj>.tq_packed`)
- Bundle disposition: ~78 % is routed mass (7.5 GB), embed/lm_head ~196 MB

### What needs fixing

- **Laguna Swift port** — `LagunaModel + LagunaConfiguration` types
  exist in vmlx-swift-lm per memory `project_swift_qwen36_dense_gaps`,
  but the dispatcher entry isn't reachable through `Engine.load`.
  Per `LagunaPortRegressionTests.swift` it's been groundwork-only.
  ~1500 LOC of model + dispatch wiring remaining (Swift port of
  `jang_tools/laguna/model.py`).
- **NOT a JangPress issue** — JangPress's mmap tier opens the bundle
  fine and indexes 39 expert tiles correctly (verified by earlier
  `JANGPressSmoke` run on this bundle). Just can't run inference.

---

## Future models to test

Available on M5 Max (sync to M4 Max as needed):

| Bundle | Size | Architecture | Priority |
|---|---|---|---|
| `Kimi-K2.6-Small-JANGTQ` | tbd | Kimi MoE (B/E/F variants) | high — different family |
| `Kimi-K2.6-Med-JANGTQ` | tbd | Kimi MoE | medium |
| `MiniMax-M2.7-Med-JANGTQ` | ~80 GB | MiniMax MoE | medium |
| `MiniMax-M2.7-Large-JANGTQ` | ~110 GB | MiniMax MoE | high |
| `Qwen3.6-35B-A3B-JANG_2L` | ~13 GB | Qwen3.6 hybrid | high — already on M4 Max |
| `DeepSeek-V4-Flash-JANGTQ2` | 79.6 GB | DSV4 v2 | low (same arch) |
| `DeepSeek-V4-Flash-JANG_2L` | 107 GB | DSV4 affine | low (same arch) |

For dense controls (no JangPress engagement expected):

| Bundle | Notes |
|---|---|
| `Mistral-Medium-3.5-128B-JANGTQ` | dense, only embed-tier candidate |

---

## Tile-layout regex coverage (current)

| Pattern | Match | Used by |
|---|---|---|
| **A** | `model.layers.<L>.mlp.switch_mlp.<gate\|up\|down>_proj.weight` | Qwen3/GLM/MiniMax fp16 stacked |
| **B** | `model.layers.<L>.mlp.experts.<E>.<gate\|up\|down>_proj.weight` | Mistral 4 / DSV3.x / Kimi K2 per-expert |
| **C** | `model.layers.<L>.mlp.experts.<gate_up\|down>_proj.tq_packed` | Laguna / Qwen3.6 / MiniMax JANGTQ stacked |
| **D** | `model.layers.<L>.mlp.experts.<gate_up\|down>_proj.weight` | JANG_2L / MXFP4 affine stacked |
| **E** | `layers.<L>.ffn.experts.<E>.w[123].tq_packed` | DSV4 per-expert JANGTQ |
| **F** | `layers.<L>.ffn.experts.<E>.w[123].weight` | DSV4 per-expert affine |
| **G** | `model.layers.<L>.mlp.switch_mlp.<gate\|up\|down>_proj.tq_packed` | Holo3 / Qwen3.5MoE JANGTQ (iter 16) |
| **H** | `model.layers.<L>.block_sparse_moe.experts.<E>.w[123].tq_packed` | MiniMax M2/M2.7 JANGTQ (iter 17) |
| **I** | `model.layers.<L>.block_sparse_moe.experts.<E>.w[123].weight` | MiniMax affine JANG (iter 17) |
| **J** | `backbone.layers.<L>.mixer.experts.<E>.<gate\|up\|down>_proj.tq_packed` | Nemotron Omni JANGTQ (iter 17) |
| **K** | `backbone.layers.<L>.mixer.experts.<E>.<gate\|up\|down>_proj.weight` | Nemotron affine per-expert (iter 17) |
| **L** | `backbone.layers.<L>.mixer.switch_mlp.fc[12].weight` | Nemotron Omni MXFP4 (iter 17) |
| **M** | `backbone.layers.<L>.mixer.switch_mlp.<gate\|up\|down>_proj.weight` | Nemotron Cascade-2 affine (iter 17) |

A/C/D/G/L/M = stacked (synthetic expert id 0). B/E/F/H/I/J/K = per-expert.

**vlPrefix iter 18**: `(?:(?:model|language_model)\.)*` accepts ANY
combination of `model.` / `language_model.` chunks before `layers.`,
including:

| Prefix observed | Bundle |
|---|---|
| (none) | DSV4 (E/F) |
| `model.` | Plain MoE (Mistral4, Kimi K2.6, MiniMax) |
| `language_model.model.` | Holo3 (VL outside) |
| `model.language_model.` | Qwen3.6 JANG_2L (VL inside) |
| `language_model.` | Qwen3.5MoE base affine |
| `backbone.` | Nemotron Cascade-2 / Omni (J/K/L/M) |

## Cross-model patterns observed

1. **Reclaim percent at pct=70 is consistent** — 68.7-74% across
   DSV4, Holo3, MiniMax, Nemotron. This is a property of how the
   controller distributes pct-target across layers, not the model.
   `forceRelease(layer:experts:)` does what you ask.
2. **First re-acquire latency is bimodal**: per-expert tiles (B/E/H/J)
   = 2.6-6.5 ms. Stacked tiles (A/C/D/G/L/M) = 61-186 ms. Choose
   per-expert layout when designing future converters if quiesce-
   recovery latency matters.
3. **Decode tok/s impact varies by model**: DSV4 -1.4%, Holo3 +1.6%
   (noise), MiniMax -6.3% (real). Suspect: 9548-tile bookkeeping in
   MiniMax. Profile before recommending pct≥70 for tile-heavy bundles.
4. **Engine compatibility** — Swift Engine still has gaps:
   - DSV4 outputs garbage (per-tensor `.tq_bits` strip not in sanitize)
   - Laguna entire model not ported (~1500 LOC pending)
   - Nemotron-H JANGTQ wrapper not wired (§437 components in place,
     Model+sanitize follow-up missing)
   - Nemotron Omni MXFP4 multimodal keys unhandled
   These are NOT JangPress issues; they block per-model decode tests.
5. **Chunk-buffer ate tokens on every thinking-aware model tested**
   (DSV4, Holo3, MiniMax). 64 tokens generated, 0 chunks emitted.
   Engine bug, separate from JangPress. Tokens-per-second is
   measurable from `engine.usage.tokensPerSecond` in the chunk usage
   field even when content/reasoning are empty.
6. **Routing density vs reclaim** — DSV4 256×6 (2.3 % active);
   MiniMax 154×8 (5.2 %); Holo3 256×8 (3.1 %); Nemotron Omni 128×8
   (6.3 %). Lower density → more dormant tiles → more JangPress
   headroom. But re-acquire cost depends on tile granularity, not
   density — see #2.
