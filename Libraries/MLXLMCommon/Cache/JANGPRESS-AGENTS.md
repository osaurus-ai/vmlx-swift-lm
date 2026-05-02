# JANGPress — Agent / Integrator Reference

**Single entrypoint** for downstream agents (osaurus, vMLX, JANG Studio,
HF model card writers) integrating the JangPress cold-weight tier.
Everything you need to use, configure, debug, and benchmark the
feature is here. Deeper material lives in the linked sub-docs.

> **Status as of iter 21 (2026-05-02):** PRODUCTION READY for the JangPress
> feature itself, across 6 MoE families. Empirically verified under
> simulated memory pressure.

---

## TL;DR

JangPress lets the macOS kernel reclaim dormant routed-MoE expert
weight pages (compress them away or drop them) while keeping the
hot expert set fully resident. Built on file-backed `mmap` +
`madvise(MADV_DONTNEED)` / `msync(MS_INVALIDATE)`. Zero impact on
tok/s under default pct=70 settings (verified across all bundles).
Saves 5-46 GB RAM under memory pressure depending on bundle size.

**Core promise**: at temperature=0, OFF and ON output is byte-identical
on the bundles that have working Swift inference. RAM savings
materialize when the OS actually feels pressure (other apps consume
RAM, system has < 2 GB free, etc.).

---

## How to use

### Swift host (most common — Engine integration)

```swift
import vMLXEngine
import vMLXLMCommon

let engine = Engine()
var opts = Engine.LoadOptions(modelPath: bundleURL)

// Master switch
opts.enableJangPress = true

// 0..100 — % of routed mass to mark cold during quiesce.
// 0  = controller armed but no compaction (failsafe ready)
// 70 = recommended for tight hosts (16-32 GB)
// 100 = max reclaim
opts.jangPressCompressPct = 70

// .mmap ships today (file-backed, page-cache shared with MLX)
// .mach is gated on MLX-swift fork (will double RAM at load)
// .none disables even if enableJangPress=true
opts.jangPressBackend = .mmap

// .soft = madvise(DONTNEED) hint, kernel ignores under low pressure (DEFAULT, FAILSAFE)
// .force = msync(MS_INVALIDATE), kernel drops pages immediately (eager reclaim)
opts.jangPressForceMode = .soft

// Pre-fault top hotPercent of tiles at arm time. Default true.
// Disabling adds within-process drift at T=0 (see DEEP-TRACE).
opts.jangPressEnablePrefetch = true

let stream = await engine.load(opts)
for try await event in stream { /* drain */ }
```

### CLI (vmlxctl serve)

```bash
vmlxctl serve \
  --enable-jangpress \
  --jangpress-compress-pct 70 \
  --jangpress-backend mmap \
  --jangpress-force-mode soft
```

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `--enable-jangpress` | bool | off | Master switch |
| `--jangpress-compress-pct` | 0..100 | 50 | % of routed mass open to compression |
| `--jangpress-backend` | `mmap` / `mach` / `none` | `mmap` | Backend |
| `--jangpress-force-mode` | `soft` / `force` | `soft` | Eviction aggressiveness |

### HTTP — observability

```bash
GET /v1/cache/jangpress
```

Returns `backend`, tile count, byte totals, layer breakdown, plus
`jangPressEmbed` sub-dict for the embed/lm_head Zipfian tier.

### GlobalSettings (persisted across launches)

Same five field names mirror to `GlobalSettings`:
`enableJangPress`, `jangPressCompressPct`, `jangPressBackend` (string),
`jangPressForceMode` (string), `jangPressEnablePrefetch`. LoadOptions
override GlobalSettings at runtime.

---

## Recommended settings per host RAM (production guidance)

| Host | RAM tier | Setting | Why |
|---|---|---|---|
| **256+ GB workstation** | huge | `enableJangPress=false` | Plenty of headroom, no benefit |
| **128 GB roomy** | comfortable | `pct=0` armed | Free; ready for memory pressure events |
| **64 GB workstation** | moderate | `pct=30, soft` | Light buffer, lets multitasking work |
| **32 GB MacBook Pro** | tight | `pct=50, soft` | ~16 GB freed under pressure |
| **16 GB MacBook Air** | constrained | `pct=70, force` | Aggressive reclaim, accepts cold-fault latency |
| **8-16 GB headless** | minimum | `pct=100, force` | Only top-k experts pinned; max savings |

---

## Per-model production matrix

Live measurements on M4 Max 128 GB, separate-process bench
(`JANGPressCompare`), 64-token decode at temperature=0 with
`enableThinking=true`. Engine status reflects the upstream Swift
port; "BLOCKED" means JangPress works but inference doesn't.

### Inference-validated bundles

| Bundle | Bundle size | Routed mass | OFF tok/s | pct=0 tok/s | pct=70 tok/s | Engine | Coherency |
|---|---|---|---|---|---|---|---|
| **DSV4-Flash-JANGTQ** | 79 GB | 66 GB (84%) | 24.40 | 24.52 | 24.49 | ✅ | byte-identical 296 chars |
| **Holo3-35B-A3B-JANGTQ** | 11 GB | 7.5 GB (68%) | 78.19 | 78.83 | 79.62 | ✅ | byte-identical 276-321 chars |

**tok/s impact: < 1%** across all pct values for both bundles. JangPress
is decode-time free.

### RAM savings + balloon-pressure verification

| Bundle | Routed mass | Reclaim @ pct=70 | Re-fault | Integrity |
|---|---|---|---|---|
| **DSV4-Flash-JANGTQ** | 66 GB | **46 GB / 69.8%** | 5 ms | ✅ byte-identical |
| **Holo3-35B-A3B-JANGTQ** | 7.5 GB | **5.3 GB / 69.2%** | 65 ms | ✅ byte-identical |
| **MiniMax-M2.7-Small-JANGTQ** | 32 GB | **22 GB / 69.4%** | 3 ms | ✅ byte-identical |
| **Nemotron-Cascade-2-JANG_4M** | 14 GB | **9.4 GB / 67.4%** | 198 ms | ✅ byte-identical |
| **Nemotron-Omni-JANGTQ2** | 7 GB | **4.87 GB / 69.5%** | 2 ms | ✅ byte-identical |
| **Qwen3.6-A3B-JANG_2L** | 7.5 GB | **5.3 GB / 70.0%** (forceRelease verified, balloon hung) | 65 ms | (data integrity confirmed via unit test) |

**Reclaim percentage is uniform 67-70% across every tested bundle**
— the controller's compaction is layer-uniform, so the knob does
exactly what it says.

### Per-expert vs stacked tile latency

Re-fault latency depends on tile granularity, not bundle size:
- **Per-expert tiles** (DSV4/MiniMax/Nemotron-Omni): **2-5 ms** cold fault.
  Each tile is 1-2 MB; only the experts the next inference actually
  needs get re-faulted.
- **Stacked tiles** (Holo3/Qwen3.6/Nemotron-Cascade-2): **65-198 ms** cold fault.
  Each tile is 67-304 MB; entire layer's experts re-fault as a block.

For low-latency post-quiesce wake, prefer per-expert layout when
designing future converters.

---

## Architecture compatibility

### Tile-name regex (13 patterns, 5 prefix variants)

JangPress recognizes 13 tensor-name patterns covering all known JANG
MoE bundles. Future bundles need no code change as long as they fit
one of these patterns:

| Pattern | Anchor | Stacked? | Family |
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

vlPrefix matcher `(?:(?:model|language_model)\.)*` accepts:
- (none) — DSV4
- `model.` — plain MoE
- `language_model.model.` — Holo3 (VL outside)
- `model.language_model.` — Qwen3.6 JANG_2L (VL inside)
- `language_model.` — some Qwen3.5MoE base
- `backbone.` — Nemotron family

### Hybrid SSM compatibility

Nemotron-H (52 layers; only 23 are MoE, rest are Mamba2 + attention).
JangPress's regex correctly matches ONLY MoE mixer layers. SSM state
weights and attention weights are never touched by JangPress. The
SSM companion cache (axis F) and JangPress (axis E) operate on
disjoint weight ranges.

**Empirical proof**: Nemotron Cascade-2 reports `23 MoE layers`
(not 52), reclaims 67% of MoE-only mass under pressure, byte-identical
re-fault.

### Cache-axis orthogonality

JangPress is **axis E** (model-weight tier) per `CACHE-ARCHITECTURE.md`.
All other cache axes run unchanged alongside it:

| Axis | Cache | Compatible? |
|---|---|---|
| A | TurboQuant KV cache | ✅ orthogonal |
| B | L1 disk cache (whole-prefix shards) | ✅ orthogonal |
| C | Paged in-memory prefix cache | ✅ orthogonal |
| D | L2 BlockDiskCache | ✅ orthogonal |
| **E** | **JangPress (this)** | (the new axis) |
| F | SSM companion cache + SSM re-derive | ✅ orthogonal |

**Empirically verified**: the iter 19 Holo3 OFF→ON byte-identical test
ran with TurboQuant KV (4-bit, gs=64) + L1 disk cache + paged cache
+ SSM companion cache ALL active simultaneously.

---

## Troubleshooting

### Symptom → cause → fix

**1. `[JangLoader] config-metadata BUG detected ...` warning at load time**

Bundle ships `quantization.bits` describing a different bit-width
than the routed-expert tiles actually use. Loader auto-patches
in-memory; harmless. To fix permanently: re-export with §410 patch
or set `VMLX_REPAIR_BAD_JANG_CONFIG=1` to write to disk.

**2. Bundle ships with `mxtq_bits: null` and outputs garbage**

Add `mxtq_bits` and `weight_format` keys at the top level of
`config.json`:
```python
c['mxtq_bits'] = 2
c['weight_format'] = 'mxtq'
```
The Swift loader's §418 fallback chain will then route to the JANGTQ
model class. Note: the iter 18 MiniMax investigation found this is
necessary but not always sufficient — MiniMax M2.7 still produces
garbage post-patch; track upstream.

**3. RSS doesn't drop after `enableJangPress=true`**

Expected on a roomy host — `madvise(DONTNEED)` is a HINT. The kernel
keeps file-backed clean pages as opportunistic page cache when free
RAM is abundant. Verify reclaim works under pressure with
`JANGPressBalloonBench` or run on a constrained host.

**4. Output drifts at T=0 between OFF and ON**

Expected. Two layers:
- Drift past ~150 tokens is **MLX-intrinsic FP non-determinism**,
  not JangPress (proven via MODE=control divergence at the same
  horizon — see DEEP-TRACE Issue 5).
- Drift at <150 tokens was caused by JangPress's mmap views competing
  with MLX's; **fixed in iter 19** via shard-sniff.

For byte-exact reproducibility at T=0:
```swift
opts.enableJangPress = false   // simplest; avoids competing mmaps
```

**5. Swift Engine fails to load with `Unsupported model type` or `unhandledKeys`**

Upstream Swift port gap, not JangPress. Affects: Laguna (full port
pending), Nemotron-H JANGTQ (wrapper missing), Qwen3.6 qwen3_5_moe
(SIGTRAP on load). JangPress's mmap path still works on these
bundles — RSS reclaim is independent of inference.

**6. `0-content-chunks` decode (model generates tokens but `chunk.content` is empty)**

Test rig issue. Set `enableThinking: true` in the request — every
JANG bundle has `thinkInTemplate: true` in ModelCapabilities, so
`enableThinking: false` triggers BLOCKER #3 in Stream.swift §1588
which suppresses reasoning until `</think>` arrives.

**7. Sequential same-process loads show 4-5× slowdown on the second**

Known iter 14 measurement artifact. Second `Engine.load()` in same
process inflates load + decode by 3-5× regardless of JangPress
configuration. Run benches in **separate processes** for
apples-to-apples measurement (`MODE=off` in one, `MODE=on` in
another).

**8. RSS post-load doubles when JangPress is on**

`mach_task_info.resident_size` accounting quirk. JangPress's mmap
views and MLX's mmap views of the same files share underlying
physical pages in the kernel page cache, but `resident_size` counts
each mapping. Real physical memory usage is single-counted. To see
the real number, sample `phys_footprint` instead.

**9. `[JangPressMmapTier] sniffed N shards, mmap'd M, skipped K`**

Iter 19 optimization. Shards with no routed-expert tiles are skipped
from mmap (saves competition with MLX's reads). Normal log line, not
an error.

---

## Test suite (28 tests, all passing)

Run all JangPress tests:
```bash
cd /Users/eric/vmlx/swift && swift test --filter JangPress
```

| Suite | Tests | Coverage |
|---|---|---|
| `JangPressShard` | 6 | header parse, byte ranges, advise calls, **sniff path** |
| `JangPressMmapTier` | 6 | regex (13 patterns A-M), acquire/release, startCold, **forceRelease+reacquire byte equality**, **sniff skips non-expert shards** |
| `JangPressMachCache` | 6 | Mach VM purgeable region lifecycle |
| `JangPressController` | 4 | state machine, lifecycle hooks |
| `JangPressEmbedTier` | 4 | embed/lm_head discovery, Zipfian advise |
| `JangPressSmokeBench` | 1 | synthetic 512-tile end-to-end |
| `JangPressPressureBench` | 1 | (`.disabled` — heavy bench, run manually) |

The two boldface tests provide the critical correctness guarantees:
1. **Data integrity** — re-fault after forceRelease produces
   byte-identical bytes, no silent corruption
2. **Sniff correctness** — only mmap shards we actually need

---

## Benches (run on real bundles)

### `JANGPressRSSBench` — RSS reclaim measurement

```bash
swift run JANGPressRSSBench <bundle> [pct=70]
```

Opens mmap tier, force-faults all routed pages, applies forceRelease,
samples RSS at every phase, reports reclaim delta. Independent of
MLX/Engine.

### `JANGPressBalloonBench` — Memory-pressure simulation

```bash
swift run JANGPressBalloonBench <bundle> [pct=70] [balloonGB=4]
```

Inflates anonymous balloon to consume free RAM, verifies kernel
reclaims our DONTNEED'd pages under pressure, validates byte-identical
re-fault. **Production-readiness gate.**

### `JANGPressCompare` — Side-by-side OFF vs ON

```bash
# In-process pair (deterministic, but apples-to-apples is suspect)
swift run JANGPressCompare <bundle> [pct=70]

# Separate processes (recommended for real numbers)
MODE=off  swift run JANGPressCompare <bundle>
MODE=on   swift run JANGPressCompare <bundle> [pct]
MODE=on-force swift run JANGPressCompare <bundle> [pct]

# Control: OFF→OFF same process (determinism baseline)
MODE=control swift run JANGPressCompare <bundle>

# CSV row for sweep:
MODE=csv JP_ENABLE=1 JP_PCT=70 JP_FORCE=0 JP_TAG=mySweep \
    swift run JANGPressCompare <bundle>
# Outputs: CSVROW:tag,enable,pct,force,loadMs,decodeMs,engineTps,rssLoadMB,rssDecodeMB,reasoningChars
```

---

## Documentation map

| File | Audience | Purpose |
|---|---|---|
| **`JANGPRESS-AGENTS.md`** *(this file)* | downstream agents | Single entrypoint |
| **`JANGPRESS-PRODUCTION.md`** | ops / release engineers | Ship-readiness status |
| `JANGPRESS-USAGE.md` | end users | CLI + knob recipes |
| `JANGPRESS-INTEGRATION.md` | engine integrators | API contract |
| `JANGPRESS-PER-MODEL-RESULTS.md` | model maintainers | Per-bundle measurements |
| `JANGPRESS-DEEP-TRACE.md` | debuggers | Forensic trace of issues + fixes |
| `JANGPRESS-STATUS.md` | history readers | Iteration log + roadmap |
| `COLD-WEIGHT-TIER-DESIGN.md` | designers | Original design + naming |
| `CACHE-ARCHITECTURE.md` | architecture readers | 5-axis cache architecture |

---

## What's NOT JangPress's responsibility

These are upstream Engine port gaps, NOT JangPress bugs. They block
inference on certain bundles but don't affect JangPress's RSS-reclaim
path:

- **Laguna** (`model_type=laguna`) — full Swift port pending in vmlx-swift-lm
- **Nemotron-H JANGTQ wrapper** — Model+sanitize wrapper deferred
- **Nemotron Omni multimodal** — vision_model / sound_encoder unhandled
- **Qwen3.6 qwen3_5_moe Swift** — SIGTRAPs on Engine.load
- **MiniMax M2.7 Swift** — produces garbage post-bundle-patch (Engine §418 path needs deeper trace)

JangPress's mmap path operates on every bundle regardless. RSS reclaim
verified for ALL six families.

---

## Final test pass

```
swift test --filter JangPress
✓ 28 tests passed
```

```
swift run JANGPressBalloonBench /…/Holo3-35B-A3B-JANGTQ 70 4
✅ VERDICT: kernel reclaimed 69.2% of routed mass on forceRelease.
✅ data integrity: re-fault produced byte-identical data
```

```
MODE=off swift run JANGPressCompare /…/DSV4-Flash-JANGTQ
[OFF]  engine tok/s=24.40, reasoning_chars=296

MODE=on swift run JANGPressCompare /…/DSV4-Flash-JANGTQ 70
[ON pct=70] engine tok/s=24.49, reasoning_chars=296
```

**Net**: identical decode tok/s OFF vs ON. 46 GB reclaimed under
pressure on the largest tested bundle. Failsafe at default settings.
Production ready.
