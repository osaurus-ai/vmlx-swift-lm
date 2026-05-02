# JANGPress — Deep Trace of Issues, Fixes, and Open Questions

Companion to `JANGPRESS-PER-MODEL-RESULTS.md`. This doc traces the
**actual** root causes of each anomaly observed during multi-model
testing iter 14-18, separating JangPress bugs from test-rig bugs
from upstream Engine/bundle bugs.

## Glossary

- **JangPress (jangpress)** — the cold-weight tier feature itself
  (controller + mmap tier + mach tier + embed tier).
- **Engine** — vMLXEngine.Engine, the high-level inference orchestrator.
- **Bundle** — a `~/.mlxstudio/models/...` directory with config.json
  + safetensors + jang_config.json.
- **§NNN** — section in the Swift loader with that line number,
  references match in-code comments.

## Issue 1 — "0 content chunks" across every thinking-aware bundle

### Symptom

Initial JANGPressCompare bench output:

```
[decode] chunks=0 content_chars=0 reasoning_chars=0
[decode] tokens: prompt=23 completion=64 approx=64
[decode] engine tok/s=86.01
[content first 200]: (empty)
```

64 tokens generated per `chunk.usage.completionTokens`, but `chunk.content`
and `chunk.reasoning` both empty.

### Initial misdiagnosis

I initially assumed an Engine streaming bug — that the parser was
swallowing the model's output. Spent ~2 iters confirming via the
`JANGPressDebug` example that ALL StreamChunk fields were empty.

### True root cause (iter 18)

This is a **TEST RIG BUG**, not an Engine bug.

The bench was passing `enableThinking: false, includeReasoning: true`.
Combined with every tested model having `thinkInTemplate: true` in
`ModelCapabilities.swift`:

| Bundle | `model_type` | `reasoningParser` | `thinkInTemplate` |
|---|---|---|---|
| DSV4-Flash JANGTQ | `deepseek_v4` | `deepseek_r1` | true |
| Holo3-A3B JANGTQ | `qwen3_5_moe` | `qwen3` | true |
| MiniMax M2.7 Small JANGTQ | `minimax_m2` | `qwen3` | true |
| Nemotron Cascade-2 / Omni | `nemotron_h` | `deepseek_r1` | true |
| Qwen3.6-A3B JANG_2L | `qwen3_5_moe` | `qwen3` | true |
| Laguna-XS.2 JANGTQ | `laguna` | `qwen3` | true |

The matrix triggers BLOCKER #3 in `Stream.swift`:

```swift
// §1588 — modelStampsThink && !effectiveThinking && !budgetExhausted
if modelStampsThink && !effectiveThinking
    && !thinkingBudgetExhausted
{
    // Drop on the floor — suppression sink.
}
```

Logic flow:
1. Chat template stamps `<think>` (template owns this, regardless of
   `enable_thinking` flag). Parser is `resetState(thinkInPrompt: true)`.
2. `inThinkBlock = true` from the start.
3. Every token routes through `parser.extractReasoningStreaming` →
   `splitReasoning` accumulates.
4. Because user requested `enableThinking=false`, the stream code at
   §1588 **drops `splitReasoning` on the floor** (BLOCKER #3
   "suppression sink").
5. `splitContent` only gets bytes AFTER parser sees `</think>`.
6. Model still inside think block at `maxTokens=64` → no `</think>` →
   `splitContent` stays empty → 0 chunks.

This was an **intentional design** (BLOCKER #3 per the comment block
at §1572-1583), not a regression: when the user says "no thinking",
we suppress the structural `<think>` content so it doesn't leak as
visible output. The cost is that the user sees nothing until the
model actually closes its think block.

### Fix

Changed bench to pass `enableThinking: true`. Now reasoning emits
into `chunk.reasoning`, and we can verify coherency via that channel
(still deterministic at temperature=0).

**Verified by re-running on Holo3** (iter 18):
- OFF: `reasoning_chars=291`, content "I need to list 5 concise facts about Apple Silicon..."
- ON soft pct=70: `reasoning_chars=331`, content "I need to list 5 concise facts about Apple Silicon..."
- Within-process control (OFF→OFF): byte-identical output (deterministic at T=0).

### Lesson

Test rigs that exercise thinking-aware models MUST either:
- Pass `enableThinking=true` AND consume `chunk.reasoning`, OR
- Use a maxTokens >> the model's typical think length (1024+), OR
- Use a non-thinking-aware model (Mistral 3.5, Qwen2, etc.)

Without one of those, BLOCKER #3 swallows everything until `</think>`
arrives — which may take the full reasoning budget on math/logic
prompts. Documented in updated bench comments.

---

## Issue 2 — VL prefix variations across model families

### Symptom

JANGPressRSSBench on Holo3 reported `0 experts indexed, 0 MB routed`.
Same on Qwen3.6-A3B JANG_2L. Other bundles (DSV4, MiniMax) indexed
fine.

### Root cause

Original regex hardcoded `^model\.layers\.` as the prefix anchor.
But VL bundles wrap the language tower under varying namespace
chunks:

| Layout | Bundle |
|---|---|
| `layers.X` | DSV4 (E/F) |
| `model.layers.X` | Plain MoE |
| `language_model.model.layers.X` | Holo3 (VL outside) |
| `model.language_model.layers.X` | Qwen3.6 JANG_2L (VL inside) |
| `language_model.layers.X` | some Qwen3.5MoE base bundles |
| `backbone.layers.X` | Nemotron Cascade-2 / Omni |

### Fix (iter 18)

Loosened `vlPrefix` from `(?:language_model\.)?` to
`(?:(?:model|language_model)\.)*`. Accepts any combination of `model.`
and `language_model.` chunks before `layers.`. One backtrack step
per call (cheap).

Backbone-prefixed Nemotron patterns are separate (`backbone.layers.X`
doesn't start with `model.` or `language_model.`).

### Verified

All 13 named patterns (A-M) now match across 5 prefix variants.
Test in `JangPressMmapTierTests.swift` covers each variant.

---

## Issue 3 — Tile naming explosion (13 patterns)

### Story

Every new MoE family ships its own tensor-naming convention. Iter
6 had 4 patterns (A-D); iter 12 added DSV4 patterns (E/F); iter 16
added pattern G (Holo3 switch_mlp+tq_packed); iter 17 added H/I/J/K/L/M
(MiniMax block_sparse_moe + Nemotron backbone+mixer).

### Pattern catalog

| Pattern | Anchor | Stacked? | Bundle |
|---|---|---|---|
| **A** | `mlp.switch_mlp.<g\|u\|d>_proj.weight` | yes | Qwen/GLM fp16 stacked |
| **B** | `mlp.experts.<E>.<g\|u\|d>_proj.weight` | no | Mistral 4 / DSV3.x / Kimi |
| **C** | `mlp.experts.<gate_up\|down>_proj.tq_packed` | yes | Laguna / Qwen3.6 / MiniMax JANGTQ |
| **D** | `mlp.experts.<gate_up\|down>_proj.weight` | yes | JANG_2L / MXFP4 affine |
| **E** | `ffn.experts.<E>.w[123].tq_packed` | no | DSV4 JANGTQ |
| **F** | `ffn.experts.<E>.w[123].weight` | no | DSV4 affine |
| **G** | `mlp.switch_mlp.<g\|u\|d>_proj.tq_packed` | yes | Holo3 / Qwen3.5MoE JANGTQ |
| **H** | `block_sparse_moe.experts.<E>.w[123].tq_packed` | no | MiniMax M2.x JANGTQ |
| **I** | `block_sparse_moe.experts.<E>.w[123].weight` | no | MiniMax affine |
| **J** | `backbone.…mixer.experts.<E>.<g\|u\|d>_proj.tq_packed` | no | Nemotron Omni JANGTQ |
| **K** | `backbone.…mixer.experts.<E>.<g\|u\|d>_proj.weight` | no | Nemotron affine per-expert |
| **L** | `backbone.…mixer.switch_mlp.fc[12].weight` | yes | Nemotron Omni MXFP4 |
| **M** | `backbone.…mixer.switch_mlp.<g\|u\|d>_proj.weight` | yes | Nemotron Cascade-2 |

### Thought: future-proofing

Adding a regex per family is sustainable so far (13 patterns over
6 family ports). When this hits ~20+ patterns, it's worth replacing
with a structural parser that recognizes:
- prefix (zero or more `model.` / `language_model.` / `backbone.`)
- layer container (`layers`)
- layer index (digits)
- moe envelope (`mlp` / `ffn` / `mixer` / `block_sparse_moe`)
- expert addressing (`experts.<E>.<proj>` or `switch_mlp.<proj>`)
- proj names (gate/up/down/gate_up/w1/w2/w3/fc1/fc2)
- suffix (`.weight` / `.tq_packed`)

For now: explicit patterns + good test coverage + add a new pattern
when a new family ships. Total cost: ~10 lines per new pattern,
~30 seconds to write the test case.

### Lesson

The naming chaos isn't JangPress's problem — it's a converter-level
inconsistency that affects every consumer of JANGTQ bundles. A future
JANG converter pass could standardize naming (e.g., always emit
`<vl_prefix>layers.<L>.<envelope>.experts.<E>.<proj>.tq_packed`)
which would simplify EVERY downstream tool, not just JangPress.

---

## Issue 4 — MiniMax-M2.7-Small-JANGTQ outputs garbage AND is non-deterministic

### Symptom

Two separate runs in the SAME process at T=0:
- Run 1: ` RE外国语ap_tRue_tRue_n_pa__t_`
- Run 2: `author_pa_l_l_`

Different garbage across runs in same process. T=0 should be
deterministic.

### Initial hypothesis: JANGTQ config-metadata bug

Per project memory `feedback_jangtq_bundle_metadata_invariants`:
> Every JANGTQ bundle MUST set `mxtq_bits` OR `routed_expert_bits`
> in config.json. Without it, §418 fallback picks `top.bits=8` (the
> §410-patched affine setting) → wrong codebook → garbage.

Inspected MiniMax bundle config:
```json
"mxtq_bits": null,
"weight_format": null,
"routed_expert_bits": 2,
"quantization": {"bits": 8, ...}
```

`mxtq_bits` IS missing. But `routed_expert_bits=2` IS set, which
the memory says should be sufficient.

### Attempted fix: patch config in place

```python
c['mxtq_bits'] = 2
c['weight_format'] = 'mxtq'
```

**Did not fix the garbage.** Same non-deterministic output after patch.

### Open: deeper investigation needed

The §418 fallback chain in Swift is more complex than the memory
captures. Possibilities:
1. Swift loader doesn't actually consult `routed_expert_bits` — only
   `mxtq_bits` or top-level `bits`.
2. Some other loader path (sanitize, factory dispatch) overrides
   the bit count.
3. The bundle has additional issues beyond metadata (e.g., wrong
   quant grouping, mismatched scales).
4. Non-determinism suggests memory is being read past the end of
   a tile or zero-init'd buffer is getting picked up.

### Conclusion: NOT a JangPress issue

JangPress was OFF for both runs (`MODE=control`). The garbage
appears whether or not the cold-weight tier is engaged. This is
a Swift-side MiniMax-M2 inference bug. Tracked separately; documented
here so we don't conflate it with JangPress when running benches.

---

## Issue 5b — Within-process OFF→ON output drift at T=0 (NEW iter 18)

### Symptom

Holo3 same-process pair shows non-identical outputs:
- `MODE=control` (OFF→OFF): byte-identical (292 chars match)
- `MODE=both` (OFF→ON pct=70): differ by ~3 chars at the end (292 vs 295)

OFF: "...focus on key, distinct points. I recall that Apple Silicon is a cu"
ON: "...focus on key points. I recall that Apple Silicon is a custom chip,"

### Root cause hypothesis

JangPress IS causing the drift, but at the FP-order level:

1. **OFF run**: MLX-swift `mlx_load_safetensors_lazy` opens mmap; pages
   fault on first touch in MLX's natural read order.
2. **ON run**: Engine.setupCacheCoordinator constructs JangPressMmapTier
   which opens its OWN mmap of the same shards. The controller calls
   `pinHotExperts` → `madvise(WILLNEED)` on the top 30% (`1.0 -
   keepHotFraction`) of tiles. Kernel **prefetches** those pages into
   the page cache.
3. By the time MLX reads weights for inference, some pages are already
   resident (from JangPress prefetch) — others fault cold. The page-
   read order differs subtly from the OFF run.
4. Different read order → different memory layout in MLX's internal
   buffers → tiny FP differences in intermediate accumulations →
   greedy-argmax flips on a few late-position tokens.

### Magnitude

- 3 chars out of 292 = **1.0% character drift**
- Structural meaning preserved (same prompt addressed, same content
  domain, same approach)
- All within the model's own next-token-tie sensitivity envelope.

### Why this is acceptable for shipping

1. T=0 + greedy is a degenerate test condition. Production users
   sample at T≥0.6 (DSV4) or T≥0.3 (Holo3) where this drift is
   invisible relative to sampling noise.
2. The drift is bounded (sub-1%) and structural, not garbage.
3. JangPress only affects the page-fault order; it does not change
   weight bytes. MLX's FP-order sensitivity is the actual source of
   non-determinism — JangPress just exposes it.
4. We can disable the prefetch (`jangPressEnablePrefetch = false`) if
   anyone needs byte-identical reproducibility at T=0.

### Documented caveat

JangPress is failsafe for **correctness** (no garbage, no infinite
loops, no NaN propagation), but introduces **sub-1% drift at greedy
T=0** due to FP-order non-determinism in MLX-swift's load path.
For byte-exact reproducibility, run with `pct=0` (controller armed
but no prefetch) or set `jangPressEnablePrefetch=false`.

### Verified iter 18: prefetch is NOT the source

Tested with `JANGPRESS_PREFETCH=0`:

| Mode | OFF chars | ON chars | First divergence |
|---|---|---|---|
| prefetch=true | 295 | 292 | char ~270 (very late) |
| prefetch=false | 273 | 276 | char ~80 (early) |

Prefetch=false makes drift WORSE, not better. So WILLNEED isn't the
source — it was actually attenuating the drift by warming the page
cache to match the OFF run's warmth.

### True root cause (revised)

The act of constructing JangPressMmapTier opens 12 ADDITIONAL mmap
views of the safetensors shards, then parses their headers. These
extra mmaps:
1. Compete with MLX's mmap views for kernel page-cache slots.
2. Trigger speculative kernel readahead independent of madvise.
3. Add fd entries that change how the kernel schedules I/O for
   the file group.

The drift comes from MLX-swift seeing **slightly different page
residency** for its weight reads due to JangPress's parallel
mappings — even when we do nothing else with them.

### Why prefetch=true was BETTER

`pinHotExperts` calls `madvise(WILLNEED)` on the top 30% of tiles
across all layers. This pre-warms the kernel page cache UNIFORMLY,
so MLX's later reads see consistent residency. Without it, MLX's
reads hit a partially-warm cache that depends on JangPress's mmap
activation pattern — more variable, more drift.

### Implication for shipping

Recommended configuration for byte-exact reproducibility at T=0:
- `enableJangPress=false` (mmap not opened at all) — best
- `enableJangPress=true, pct=0, prefetch=true` — still slight drift
- `enableJangPress=true, pct=70, prefetch=true` — bounded drift (1%)
- `enableJangPress=true, pct=70, prefetch=false` — worst drift (3%)

For production (T>0), all configurations are equivalent — the drift
is masked by sampling noise. The `prefetch=true` default is the
right ship setting.

### iter 19 FIX: shard-sniff to reduce mmap surface area

**Hypothesis**: if the drift comes from JangPress's parallel mmap
views competing with MLX's mmap views, then reducing the number of
JangPress mmap views to ONLY the shards that contain routed experts
should reduce the competition proportionally.

**Implementation**:
- New static method `JangPressShard.sniffTensorNames(at:)` — opens fd,
  reads first 8 bytes (header size) + next H bytes (header JSON),
  parses tensor names. NO mmap. Closes fd.
- `JangPressMmapTier.init` calls this for each shard, runs
  `parseRoutedExpertName` over the names. Only mmap's shards that
  contain at least one routed-expert tile.

**Verified iter 19 on Holo3** (13 shards total, 10 with experts):

```
[JangPressMmapTier] sniffed 13 shards, mmap'd 10, skipped 3
```

OFF→ON byte-identical reasoning output (276 chars):
```
✅ content output IDENTICAL (0 chars)
✅ reasoning output IDENTICAL (276 chars)
```

**This eliminates the iter 18 drift on Holo3.** JangPress no longer
opens mmap views of attention/embed/lm_head shards, so MLX's reads
from those shards see the original page-cache state (not perturbed
by an additional read-only mapping).

**Applicability across bundles**:

| Bundle | Total shards | After sniff | Skipped | Drift improvement |
|---|---|---|---|---|
| Holo3-A3B JANGTQ | 13 | 10 | 3 | ✅ byte-identical |
| MiniMax-M2.7-Small | 40 | 39 | 1 | minimal — most shards have experts |
| (DSV4-Flash) | ~86 | ~86 | ~0 | minimal expected — every shard has experts |

For bundles where ALL shards contain routed experts (DSV4 case), the
sniff doesn't reduce mmap surface but also doesn't HURT — same
behavior as before. For bundles like Holo3 where attention/embed
live in separate shards, sniff yields full coherency.

**Cost**: one open + one read of (8 + headerSize) bytes per shard at
init time (~1 MB read on Holo3, < 50 MB on the largest bundles).
Compared to the dropped mmap regions (multi-GB each), this is a
net win at every layer:
- Less RSS from redundant virtual mappings
- Less page-cache competition with MLX
- Same physical-memory-share semantics (page cache is shared)

**Test coverage**:
- New unit test `sniffTensorNames returns names without mmap'ing data`
- Existing tests still pass (5/5 JangPressShard, 4/4 JangPressMmapTier)
- Live bench validation on Holo3 (drift eliminated) and MiniMax
  (sniff works, garbage unchanged due to upstream bundle bug).

### iter 19 long-generation finding: drift is MLX-intrinsic past ~150 tokens

After fixing the early-token drift via sniff, ran the bench at
`MAX_TOKENS=256` to verify durability. Result:

| Mode | OFF chars | ON chars | Drift |
|---|---|---|---|
| MODE=both (OFF/ON) | 1298 | 1329 | 31 chars (2.4%) |
| MODE=control (OFF/OFF in same process) | **1331** | **1281** | **50 chars (3.7%)** |

**MODE=control diverges MORE than MODE=both.** This means the long-
generation drift is NOT introduced by JangPress — it's MLX-swift's own
FP-order non-determinism compounding over many decode steps.

In fact, the JangPress=ON path's drift (2.4%) is LESS than the
control's intrinsic drift (3.7%). JangPress isn't adding noise; it's
operating within MLX's own noise floor.

**Implication**: at maxTokens >150 tokens, byte-exact reproducibility
is impossible at T=0 in Swift even WITHOUT JangPress. Users who need
reproducibility must either:
1. Pin a fixed seed AND temperature > 0 (then sampling is the only
   source of variance, dominated by seed).
2. Use Python (which has different FP-order, same problem in different
   form).
3. Accept structural-similarity coherency rather than byte-match.

**JangPress production verdict**: ✅ No coherency regression vs
baseline. Drift at long generation is upstream MLX behavior. The
sniff fix eliminates the SHORT-generation extra-drift JangPress was
contributing; everything beyond that is MLX's domain.

---

## Issue 5 — Cross-process word drift at T=0

### Symptom

Holo3 reasoning text differs slightly between separate processes
even at temperature=0:

| Process | Output |
|---|---|
| Cold-cache OFF (1st process) | "User specifically asked for brevity, so I should avoid lengthy explanations" |
| Warm-cache ON pct=70 (2nd process) | "User specifically requested brevity, so I must avoid lengthy explanations" |
| Within-process OFF→OFF (control) | byte-identical |

### Root cause

Within-process: deterministic. Across processes: fp drift driven by
**cold vs warm OS page cache**.

When the kernel page cache is cold, MLX-swift's safetensors mmap
loads weights via different read patterns (potentially different
chunk sizes, different prefetch). Tiny FP32 vs bf16 accumulation
order differences can flip greedy-argmax ties on a small number of
positions. With 64 tokens of generation, 1-3 token positions ending
up at a different logit max produces noticeably different word
choices.

### Conclusion: NOT a JangPress issue

The drift appears between two OFF runs across separate processes
just as it does between OFF/ON runs. JangPress is not introducing
non-determinism; it's surfacing existing FP order-sensitivity in
MLX-swift's load path.

### Implication for the bench

Coherency check should compare:
- Same-process: byte-exact (passing)
- Cross-process: structurally identical (same prompt addressed,
  same approach, same domain knowledge surfaced) — LLM output
  similarity, not byte match.

The bench's `MODE=both` runs both in same process (apples-to-apples
deterministic). `MODE=off` + `MODE=on` separate processes give
real load timings but with this drift baked in.

---

## Issue 6 — MiniMax 6% decode slowdown at pct≥50

### Symptom

| | Engine tok/s |
|---|---|
| OFF baseline | 46.7 |
| ON pct=0 | 47.1 (within noise) |
| ON soft pct=70 | 43.7 (-6.3%) |

Reproducible across multiple runs.

### Root cause (iter 17 trace)

Ruled OUT:
- Per-token tracking (`recordTokenActivity` is unwired in Engine)
- Per-token controller hooks (only fire at start/end of inference)
- Embed-tier setup (RSS post-load identical at pct=0 vs pct=70)
- Mach VM purgeable scaffolding (we use `.mmap`, not `.mach`)

Identified:
- At pct≥50 the controller's `compressColdTiles` does
  `madvise(DONTNEED)` on 9548 tile byte ranges (62 layers ×
  154 experts). Per-tile cost is microseconds but adds up.
- Page-cache turnover competes with MLX safetensors reads during
  the next decode, causing modest serialization.

### Mitigation

`pct=0` is verified failsafe: controller arms (ready for pressure
events) but never compacts. Engine tok/s at pct=0 = 47.1 vs OFF
46.7 = +0.9% (noise). Users on roomy hosts can run pct=0 with no
cost.

### Open: deeper investigation could reduce or eliminate the cost

1. Profile `JangPressMmapTier.forceRelease` on 9548-tile path —
   maybe batch the madvise calls into fewer larger ranges.
2. Test with `mode=force` (msync) to see if the cost scales
   identically (would localize cost to the syscall side vs
   page-cache side).
3. Test on Mistral 4 / Kimi K2.6 (similar per-expert tile counts)
   to see if the cost is tile-count-dependent vs MiniMax-specific.

For shipping iter 18: documented + workaround verified.

---

## Issue 7 — Stacked tiles re-acquire 60-186 ms vs per-expert 2-7 ms

### Observation

| Layout | First re-acquire after release |
|---|---|
| DSV4 per-expert (E/F) | 6.5 ms |
| Nemotron Omni per-expert (J/K) | 2.6 ms |
| MiniMax per-expert (H/I) | 3.5 ms |
| Holo3 stacked switch_mlp (G) | 61 ms |
| Qwen3.6 stacked (D) | 65 ms |
| Cascade-2 stacked (M) | 186 ms |
| Nemotron Omni MXFP4 stacked (L) | 186 ms |

Stacked layouts pay 10-70× more for the first cold fault after
quiesce.

### Why

A stacked tile is `[num_experts, intermediate, hidden]`. The mmap
range covers ALL experts. When `WILLNEED` is hinted on this range,
the kernel reads ahead the WHOLE 67-200 MB block, not just the few
experts we'll actually use.

Per-expert tiles cover ~262 KB - 7 MB each, so the first fault
loads only what's needed.

### Idea (deferred): split stacked tiles into per-expert sub-ranges

For stacked layouts (A/C/D/G/L/M), parse the safetensors header to
get the first-axis size = num_experts, then split the byte range
into N sub-ranges of equal size. The Tier could expose
`acquire(layer, experts: [Int])` that hits only those sub-ranges.

Pros:
- 10-70× faster cold fault
- Per-expert routing decisions can drive per-expert release/acquire

Cons:
- Tier needs shape info (currently only stores byte range)
- Storage is shared at page level; sub-tile granularity may not
  reduce total fault cost much (page reads are ≥ 16 KB on Apple
  Silicon and ≥ 256 KB with prefetch)
- Adds complexity

### Decision

Defer. The 60-186 ms one-time cost only applies post-quiesce
(>30s idle). For a roomy host on pct=0, this never fires. For a
constrained host using pct=70, the user opts into this trade-off
(more reclaim ↔ slower wake). Documented in
`JANGPRESS-PER-MODEL-RESULTS.md` per-model recommendations.

---

## Issue 8 — Engine port gaps blocking decode tests

These are NOT JangPress issues but block our coherency-test
matrix.

### DSV4 outputs `?` repeated

Per memory `project_swift_dsv4_jangtq_working`: 5-line `.tq_bits`
strip needed in `DeepseekV4JANGTQModel.sanitize()` in
`vmlx-swift-lm`. Inference produces `?` 32 times instead of real
text. JangPress on/off don't change this — both produce identical
garbage.

### Laguna entire model not ported

`Engine.load` errors with "Unsupported model type: laguna". Per
memory `project_swift_qwen36_dense_gaps`: `LagunaModel +
LagunaConfiguration` types exist in vmlx-swift-lm but the dispatcher
entry isn't reachable. ~1500 LOC of Swift port remaining.

### Nemotron-H JANGTQ wrapper missing

```
Error 438: Nemotron-H JANGTQ bundle detected (weight_format=mxtq) but
the NemotronHJANGTQModel wrapper isn't wired yet. Components are in
place (§437 NemotronHJANGTQSwitchMLP); the Model+sanitize wrapper
lands in a follow-up iter.
```

### Nemotron Omni multimodal keys unhandled

```
unhandledKeys: ["mlp1", "sound_encoder", "sound_projection", "vision_model"]
```

Audio + vision towers in the multimodal Omni bundle aren't in the
Swift NemotronH model class.

### Qwen3.6 qwen3_5_moe Swift wrapper SIGTRAP

`Engine.load` hits SIGTRAP on Qwen3.6-35B-A3B-JANG_2L. Per memory
`project_swift_qwen36_dense_gaps`: VL tower + linear_attn
(GatedDeltaNet) + dense FFN block ~1500 LOC outstanding.

### Conclusion

For the four bundles where Engine load works (DSV4, Holo3, MiniMax,
and the rest is RSS-only):
- DSV4: garbage output (Engine bug)
- Holo3: ✅ works, JangPress verified
- MiniMax: garbage output (Engine bug + bundle metadata)
- Nemotron / Qwen3.6 / Laguna: BLOCKED at Engine.load

Only **Holo3** gives a clean OFF/ON coherency comparison. RSS
reclaim works on all six bundles regardless of Engine status
(JangPress's mmap path is decoupled from inference).

---

## Summary of session fixes shipped

| # | Issue | Fix | Iter |
|---|---|---|---|
| 1 | Bench had `enableThinking=false` (BLOCKER #3 swallowed output) | Switched to `enableThinking=true` + documented why | 18 |
| 2 | VL-prefix variations broke regex on Holo3, Qwen3.6 | Loosened `vlPrefix` to `(?:(?:model\|language_model)\.)*` | 18 |
| 3 | Holo3 `switch_mlp.tq_packed` not matched | Added pattern G | 16 |
| 4 | MiniMax `block_sparse_moe.experts.<E>.w[123]` not matched | Added patterns H + I | 17 |
| 5 | Nemotron `backbone.layers.<L>.mixer.*` not matched | Added patterns J/K/L/M | 17 |
| 6 | MiniMax 6% slowdown was assumed pct-independent | Verified pct=0 is free; documented pct-driven cost | 17 |
| 7 | Force-mode wasn't testable in compare bench | Added `MODE=on-force` env var | 17 |
| 8 | `--jangpress-force-mode` flag undocumented | Updated `JANGPRESS-USAGE.md` with flag table + tradeoff | 18 |

## Open items (logged, not blocking ship)

| # | Issue | Owner | Priority |
|---|---|---|---|
| A | DSV4 `?` output (`.tq_bits` strip) | vmlx-swift-lm | high (affects flagship bundle) |
| B | MiniMax garbage despite metadata patch | unknown — needs §418 trace | medium |
| C | Laguna full Swift port | vmlx-swift-lm | medium |
| D | Nemotron-H JANGTQ wrapper | vmlx-swift-lm | medium |
| E | Qwen3.6 qwen3_5_moe Swift wrapper | vmlx-swift-lm | medium |
| F | Stacked-tile sub-tile granularity | JangPress | low (deferred) |
| G | `recordRoute` wiring (Engine → controller) | JangPress | low (improvement) |
| H | Constrained-host pressure verification (32 GB or balloon) | testing | high (validates the entire thesis) |

## Closing thoughts

JangPress as a **mechanism** is correct and verified across 6 model
families:
- All 13 tile-naming patterns matched by regex
- 100% RSS reclaim with `msync(MS_INVALIDATE)` on every bundle
- 68-74% partial reclaim at pct=70 (knob does what it says)
- Failsafe at pct=0 (engine tok/s within noise)
- pct=70 cost varies 0-6% by tile count (MiniMax worst at 6.3%)

Most "issues" surfaced during testing are **upstream Engine port
gaps** or **bundle metadata bugs**, not JangPress bugs. The decoupling
is structural: JangPress only touches mmap'd byte ranges via syscall;
it never touches MLX tensors or quantization metadata.

The per-model test matrix is the deliverable — not "tested on every
bundle ever made" but "tested across 6 distinct model families
covering all 13 tile-naming variants and 5 prefix wrappings". Any
new bundle ships through the same regex without code change as long
as it adheres to the existing patterns.
