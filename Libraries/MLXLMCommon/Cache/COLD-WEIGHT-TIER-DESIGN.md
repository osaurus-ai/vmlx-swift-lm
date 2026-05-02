# Cold Weight Tier — design

## ⚠️ Architectural finding (2026-05-01 iteration 3)

After surveying MLX-swift's allocator (`MLXArray+Metal.swift`):

- `MLXArray.asMTLBuffer(device:, noCopy: true)` returns a Metal buffer
  WRAPPER over the existing bytes. It doesn't expose
  `setPurgeableState` on the underlying allocation.
- MLX's internal allocator owns the storage. While the GPU is
  reading from a buffer, those pages are wired (not eligible for
  kernel compression).
- Mach `vm_purgable_control` on a region the kernel sees as
  Metal-wired is a no-op — pages can't be compressed while wired.

**Implication**: applying purgeable-memory state DURING active GPU
decode is structurally infeasible without an MLX-swift fork that
exposes the allocator buffer's purgeable state. The right window
to compress weights is when GPU work is QUIESCENT.

This drives a conservative two-phase plan:

### v1 — IDLE-time compression (failsafe path)
- Compress weights ONLY when no inference is in flight AND
  app is backgrounded OR memory pressure event arrives
- Active inference path is byte-for-byte unchanged
- Wake-up cost: kernel decompresses on first read of the
  next request → 50-200 ms latency on the first token
- No correctness risk; no per-step overhead

### v2 — ACTIVE-decode compression (later, gated)
- Requires MLX-swift fork that exposes `MTLBuffer.setPurgeableState`
  per-tensor
- Per-step volatility flips during decode
- Higher savings but risks GPU correctness if a buffer is
  evicted mid-kernel
- Only ship after v1 has been in production for weeks

The `JangPressMachCache.swift` we shipped in iteration 1
is the right LIFECYCLE primitive for v1 — the
`acquire`/`release`/`pinHot` API maps cleanly onto idle-time
compression. We just need the IDLE detector + wake hook before
real users can opt in.



Working name: **JANGPress** (or *Cold Weight Tier* if we want plain).
Embers stay warm, you can re-light them quickly — same idea as
purgeable-memory-backed weight regions.

## What it is

A weight-storage abstraction that asks the macOS kernel to **compress
dormant model weights in place** while keeping the active set fully
hot. Implemented on top of `vm_purgable_control` so the OS does the
WKdm compression for free; we only manage the *which weights are
dormant* part.

Goal: let users with limited unified memory run MoEs (and other
sparsely-accessed weight tensors) that they otherwise couldn't fit.

## Default-on plan

| Phase | Default | Knob | Why |
|---|---|---|---|
| Phase 1 (now) | OFF | `enableJangPress=true` opt-in | Validate on real bundles; surface any kernel-edge-case bugs |
| Phase 2 | ON when memory pressure warns | auto-on threshold | Most users won't notice; pressure-eligible only |
| Phase 3 | ON always | `jangPressCompressPct=0..100` | After speed loss is verified ≤ 5% under low pressure |

Phase 3 is the user's stated goal — automagic, default-on, with a
percentage knob users tweak only if they want.

## Components eligible for compression

Inventoried by what gets touched per decode step on a typical MoE
(numbers are DSV4-Flash 79 GB JANGTQ for concreteness):

| Component | Size on 79 GB bundle | Per-token access | Compression eligible? |
|---|---|---|---|
| **Routed experts** (top-k of N) | ~70 GB (88%) | 2-3% touched | **YES — primary target** |
| **Vocab embedding** (vocab × hidden) | ~1 GB | 1 row of vocab/token | **YES — long-tail rows** |
| **LM head** (hidden × vocab) | ~1 GB (often tied to embed) | sample → top-N rows | **YES — long-tail rows** |
| **Hash-routed layer experts** (DSV4 L0-L2) | ~5 GB | dense per-hash | **PARTIAL — by hash bucket** |
| **Shared expert** (always-on MLP) | ~2 GB | 100% touched | NO — never dormant |
| **Attention QKV/MLA projections** | ~10 GB | 100% touched | NO — never dormant |
| **Compressor + Indexer pool** | ~few GB at 1M ctx | dense per layer | already covered by `PoolQuantizedV4Cache` |
| **Layer norms** (RMSNorm scales) | <100 MB | 100% touched | NO — too small + hot |
| **MTP layers** | 0 (dropped at convert) | n/a | n/a |

So the realistic compression budget on DSV4-Flash is:
- Routed experts: 70 GB → ~21 GB hot + 49 GB compressed under
  pressure → save up to ~25 GB at WKdm 2× ratio
- Embedding / lm_head: 2 GB → ~0.4 GB hot top-1% Zipfian + 1.6 GB
  compressed → save ~0.8 GB
- **Total realistic save: ~25-26 GB** (one-third of the bundle)

That's enough to fit DSV4-Flash on a 64 GB MacBook (currently
requires 128 GB).

## Why other components don't qualify

- **Attention QKV** (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` for MLA): Every
  token's attention reads ALL these matrices. Dormancy ≈ 0%. Compressing
  them just adds decompress latency on every step — net loss.
- **Shared expert**: same — every token uses it. The whole point of a
  shared expert is "always on, helping every token".
- **Norms**: too small to amortize the page-level overhead.
- **MTP**: already dropped at convert time; not in the runtime pool.

## Architecture-by-architecture compatibility

| Family | Routed experts? | Embedding compression OK? | Notes |
|---|---|---|---|
| DSV4-Flash | yes (256 × 6) | yes | hash-routed L0-L2 partial |
| Kimi K2.6 | yes (384 × 8) | yes | similar shape, bigger pool |
| MiniMax M2.7 | yes (320 × 6) | yes | identical pattern |
| Qwen 3.6 / GLM 5 / Laguna | yes (128 × 8) / (256 × 8) | yes | smaller pool, less win |
| Mistral 3.5 | NO (dense MLP) | yes | only embedding/lm_head, ~3 GB save max |
| Gemma 4 / Llama 4 | small MoE (64 × 4) | yes | ~6% routing density, modest win |
| Pure dense (Mistral 3.5, Llama 3, Qwen 3 dense) | n/a | yes | embedding only |

For dense models, the win is small (~3 GB on a 30 GB bundle = ~10%).
For 1T+ MoEs (Kimi, MiniMax, MiMo) the win is enormous — maybe the
difference between "fits on a 256 GB rig" and "doesn't".

## API shape — proposed

```swift
public final class ColdWeightTier {
    public enum Region {
        case routedExpert(layer: Int, expert: Int)
        case embeddingRow(start: Int, count: Int)   // page-aligned
        case lmHeadRow(start: Int, count: Int)
        case hashExpert(layer: Int, hashBucket: Int)
    }

    public func register(_ region: Region, bytes: UnsafeRawBufferPointer,
                         diskURL: URL? = nil, diskOffset: UInt64 = 0)

    public func acquire(_ regions: [Region]) -> [UnsafeRawPointer]
    public func release(_ regions: [Region])

    /// Routing-frequency snapshot for warm-set seeding.
    public func recordAccess(_ region: Region)
}
```

(`JangPressMachCache` is a thin specialization of this for
the `case routedExpert` shape. We rename to `ColdWeightTier` once the
generalization lands.)

## Naming options (final pick TBD)

- **JANGPress** — embers / glow / re-lightable. Branded.
- **Cold Weight Tier** — descriptive, plain.
- **DormantWeights** — descriptive.
- **MoE Hibernate** — narrows to MoE.
- **SparseTier** — generic.
- **WarmRouter** — narrows to MoE routing.
- **Ember** — short, memorable, brand-able.

Eric's call.

## Sequencing

1. **Land routed-expert tier first** (the current
   `JangPressMachCache`). Phase 1: opt-in. Validate on a small
   MoE bundle before generalizing.
2. **Generalize to `ColdWeightTier`** — same Mach machinery, more
   region types.
3. **Embedding / lm_head Zipfian compression** — needs a token-frequency
   profiler that runs during the first ~1000 tokens of a session.
4. **Phase 2 default-on under pressure** — wire to
   `DispatchSourceMemoryPressure.warning` automatically.
5. **Phase 3 default-on always** — only after benchmarks show ≤ 5 %
   tok/s loss at low pressure across DSV4 / Kimi / MiniMax / Mistral 3.5.

## Why this beats every other compression scheme we considered

| Scheme | What we'd write | What we'd save vs Ember |
|---|---|---|
| User-space LZ4 over weights | thread pool + decompress staging | LZ4 ~2× (worse than WKdm 2-3×), plus our scheduling overhead |
| GPU-decode-on-demand kernel | new Metal kernel + lookup level | ~1.3× savings, no win unless paired with eviction |
| Tile-paged SSD streaming (jang-spec) | full streaming runtime | wins for *bigger than RAM* models; not for *fits with breathing room* |
| Hadamard + entropy code | encoder + GPU kernel decode | ~1.5×, hot path adds latency |
| **Ember (purgeable memory)** | ~600 LOC Swift | **~2-3× WKdm, kernel-managed, free under low pressure** |

The big advantage: **idle-time cost is zero**. Other schemes pay
constant overhead. Ember only does work when memory is actually
constrained — which is exactly when users want help.
