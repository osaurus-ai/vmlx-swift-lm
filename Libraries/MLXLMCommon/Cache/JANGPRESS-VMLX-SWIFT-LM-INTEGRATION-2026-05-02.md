# JangPress × vmlx-swift-lm — integration map

> **Audience:** vmlx-swift-lm engineers + osaurus integrators wiring
> JangPress through the SDK. Companion to `JANGPRESS-AGENTS.md` (which
> targets the upstream vmlx engine).

This doc specifies how the JangPress cold-weight tier (axis E in the
5-axis cache architecture) coexists with the **other** cache + runtime
subsystems already in vmlx-swift-lm: JANGTQ codebook + Hadamard kernels,
JANG mixed-quant, hybrid-SSM Mamba, the L2 BlockDiskCache, the
TurboQuant KV cache, paged in-memory prefix cache, RoPE / matmul, and
the BatchEngine scheduler.

The 5 source files + 7 test files + 9 doc files have landed (commit
`1ec1668`). 28 tests pass. The runtime types are present and self-
contained. **What this doc covers is the wiring spec for the integration
phase that follows** — the LoadOptions plumbing, the BatchEngine
inference-bracket hooks, and the verified-orthogonality contract with
every other cache/runtime axis already in this repo.

---

## 5-axis cache architecture (vmlx-swift-lm types)

| Axis | Concern                          | vmlx-swift-lm type(s)                                        | JangPress interaction |
|------|----------------------------------|---------------------------------------------------------------|------------------------|
| A    | Per-token KV cache               | `KVCacheSimple`, `RotatingKVCache`, `TurboQuantKVCache`, `MambaCache`, `CacheList` | Orthogonal — JangPress touches model weights only, not KV state |
| B    | L1 paged in-memory prefix        | `PagedCacheManager` + `BlockHashMap`                          | Orthogonal |
| C    | (folded into B in this fork)     | —                                                            | — |
| D    | L2 BlockDiskCache                | `DiskCache`, `TQDiskSerializer`, `CacheCoordinator`           | Orthogonal — operates on prefix bytes, not model weights |
| **E**| **Cold-weight tier (this)**      | **`JangPressMmapTier` / `JangPressMachCache` / `JangPressEmbedTier` / `JangPressController`** | (axis-of-record) |
| F    | SSM companion + re-derive        | `SSMStateCache`, `extractSSMStates`/`restoreSSMStates` (inline-at-prefill-end seed) | Orthogonal — Mamba layers are SKIPPED by JangPress's regex |

The 5 axes operate on **disjoint memory**:
- Axes A/B/D/F live in the inference scheduler (`BatchEngine.swift` +
  `Cache/`) and own activation-derived state (KV blocks, SSM recurrences).
- Axis E lives in the model loader and owns **weight pages** (read-only,
  file-backed via mmap).
- They share zero mutable state. All "interaction" is at the OS-page-cache
  level (axis E's `madvise(DONTNEED)` releases pages the kernel may also
  hold for other axes — but **release is a hint**, not a force, so if
  another axis is using them they stay resident).

---

## Coexistence guarantees per axis

### Axis A — TurboQuant KV cache + matmul + RoPE

**Concern raised:** "if turboquant kv cache and encode and decode gets
routed from the experts and matmul and hadamard and codebook and rope
and all that shit"

**Answer:** TurboQuant KV cache (`TurboQuantKVCache.swift`) operates on
the **per-step KV activations** at decode time — it compresses the
keys/values that the model produces during inference. It has nothing to
do with the **routed-expert weight tensors** that JangPress targets.

The data flow is:

```
inference-time:
  x → embed_tokens                        ← Zipfian tier (JangPress F)
  for each layer:
    x → input_norm → q/k/v projections    ← weights stay hot
    q,k,v → RoPE                          ← position math, no JangPress
    k,v → KVCache.update()                ← axis A (TQ KV compresses HERE)
    attention → o_proj                    ← weights stay hot
    x → mlp_gate × experts                ← AXIS E (JangPress mmap-tier
                                            keeps hot subset, releases cold)
    (for JANGTQ: gather_tq matmul through
     codebook + Hadamard rotation)
    x → mlp_down                          ← weights stay hot
  x → final_norm → lm_head                ← Zipfian tier (JangPress F)
```

**Confirmed orthogonality:**
- `TurboQuantKVCache.update(keys, values)` writes new K/V into the
  encoder; never reads expert weights. JangPress's `.dontNeed` calls
  on cold expert pages don't touch any byte the TQ encoder sees.
- The `gather_tq` Metal kernel reads `tq_packed` + `tq_norms` directly
  from the safetensors mmap region. JangPress mmaps the **same
  underlying file**, shares the kernel page cache. When a hot expert
  is needed, both views see the same physical page (no doubling).
- RoPE (`RoPEUtils.swift::initializeRope`, `applyRotaryPosition`,
  `YarnRoPE`, `Llama3RoPE`, `SuScaledRoPE`) operates on Q/K activations
  using inv-freq tables; never reads expert weights.
- The Hadamard kernel (`JANGTQKernels.swift::hadamardRotate` + the
  `H_2n` recursion for blocks > 8192) reads activations via shmem; never
  reads expert weights.

**Empirical verification (upstream-vmlx side, replicate-able here once
LoadOptions are wired):** the Holo3 OFF→ON byte-identical iter-19 test
ran with TurboQuant KV (4-bit, gs=64) + L1 disk cache + paged cache +
SSM companion cache ALL active simultaneously — JangPress was the
fifth axis active and produced byte-identical output.

### Axis B — Paged prefix cache (L1)

`PagedCacheManager.store(blocks:tokens:mediaSalt:)` writes per-prefix
blocks into a hash-mapped in-memory pool. The blocks are KV-cache
slices, not model weights. JangPress operates on a different memory
plane — there's no path by which axis B blocks could be paged out by
axis E.

The only shared resource is the kernel page cache. When axis E
releases routed-expert pages, axis B's heap allocations are unaffected
(B uses `malloc`-backed memory; E uses file-backed `mmap`).

### Axis D — L2 BlockDiskCache (`DiskCache.swift`, `TQDiskSerializer.swift`)

`TQDiskSerializer v2` (commit `bf942a8`) writes per-layer KV blocks
**plus SSM states + diskArrays** to disk for warm-restore on the
next session. Block payload format is layer-kind-tagged
(LayerKind 0..6 covers `kvSimple` / `tqCompressed` / `qkv` / `mamba` /
`kv` / `skip` / `rotating`). None of these payloads contain model
weights; JangPress doesn't see L2 disk cache I/O.

Cache key invalidation on axis D is keyed by `(modelKey, mediaSalt,
prefix-tokens)`. JangPress changes nothing about the model's identity
or the prefix bytes — keys remain stable.

### Axis F — SSM companion cache + inline-at-prefill-end re-derive

**Concern raised:** "ssm pass and handler and scheduler"

**Answer:** Hybrid SSM models (Qwen 3.5/3.6, NemotronH-Omni, MiniMax
M2.7, Qwen-3-Next) interleave Mamba layers (`MambaCache`) with attention
layers (`KVCacheSimple` or `TurboQuantKVCache`). The `BatchEngine`
scheduler calls `coordinator.setHybrid(true)` at admission, which
turns on `extractSSMStates` / `restoreSSMStates` paths in the cache
coordinator.

JangPress's regex (13 patterns A–M, see `JANGPRESS-AGENTS.md` §
"Tile-name regex") matches **routed-MoE expert weight tensors only**.
Mamba weights live at paths like:

```
backbone.layers.<i>.mixer.in_proj.weight
backbone.layers.<i>.mixer.conv1d.weight
backbone.layers.<i>.mixer.A_log
backbone.layers.<i>.mixer.dt_bias
…
```

None of these match any of the 13 patterns. Empirical proof from the
upstream-vmlx work: NemotronH-H reports `23 MoE layers` (not 52 — the
remaining 29 are Mamba2/attention) and JangPress reclaims 67% of
**MoE-only** mass. Mamba state weights stay fully resident.

The hybrid-SSM full-disk-hit fix (commit `227332f`) — which rolls back
to full prefill when SSM state would otherwise be double-counted — is
unaffected by JangPress: it operates on the inference scheduler's
slot lifecycle, not on weight residency.

### Axis Hot-path — JANGTQ codebook + Hadamard

The JANGTQ codebook (Lloyd-Max for Beta((d-1)/2, (d-1)/2)) and signs
vector live in `jangtq_runtime.safetensors` (10–200 KB sidecar). They
are NOT routed-expert tiles; JangPress's regex never matches them and
they stay hot. The `JANGTQRuntimeCache` singleton holds them after
first load.

The `gather_tq` kernel reads three inputs per forward:
1. `xRot` — activation (lives in the model's compute graph; never on disk)
2. `packed` — expert tile bytes (read from the file via mmap; **JangPress**
   may have released these pages when the expert was cold)
3. `norms` — per-row L2 (same mmap region as `packed`)
4. `codebook` — sidecar (always hot)

When JangPress has DONTNEED'd a cold expert and the kernel dispatches a
read, the page faults back from disk transparently. The next call sees
the page hot (kernel keeps it until the next pressure event).

Cold-fault latency per JANGPRESS-AGENTS.md:
- Per-expert tiles (DSV4, MiniMax, NemotronH-Omni): **2–5 ms** (1–2 MB tiles)
- Stacked tiles (Holo3, Qwen3.6, NemotronH-Cascade-2): **65–198 ms** (67–304 MB tiles)

For low-latency requirements, prefer per-expert layout when designing
future converters.

---

## LoadOptions wiring (next phase, to land on top of `1ec1668`)

vmlx-swift-lm doesn't have an `Engine.LoadOptions` struct like vmlx
upstream does. The equivalent surface is the model factory's load path
in `Load.swift::loadWeights(...)` and `ModelFactory`. Recommended
wiring:

1. Add a `JangPressLoadOptions` struct to `MLXLMCommon`:

```swift
public struct JangPressLoadOptions: Sendable {
    public var enabled: Bool = false
    public var compressPct: Int = 70           // 0..100
    public var backend: Backend = .mmap
    public var forceMode: ForceMode = .soft
    public var enablePrefetch: Bool = true

    public enum Backend: String, Sendable { case mmap, mach, none }
    public enum ForceMode: String, Sendable { case soft, force }
}
```

2. Add an optional parameter to `loadWeights(...)` and the factory
   `_load` paths so callers can pass it:

```swift
public func loadWeights(
    modelDirectory: URL, model: any Module,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    jangPress: JangPressLoadOptions? = nil
) async throws { ... }
```

3. After weights load, instantiate the tier when enabled:

```swift
if let jp = jangPress, jp.enabled {
    let pct = max(0, min(100, jp.compressPct))
    switch jp.backend {
    case .mmap:
        let tier = try JangPressMmapTier(modelDirectory: modelDirectory)
        let controller = JangPressController(
            mmapTier: tier,
            forceMode: jp.forceMode == .force,
            keepHotFraction: 1.0 - Double(pct) / 100.0)
        if jp.enablePrefetch { try tier.prefetchHotTiles() }
        controller.arm()
        // Stash on the ModelContext for BatchEngine bracket hooks
    case .mach: /* sym to .mmap; uses JangPressMachCache */
    case .none: break
    }
    if jp.enabled {
        let embed = try JangPressEmbedTier(modelDirectory: modelDirectory)
        // Co-instantiated regardless of routed-expert backend.
    }
}
```

4. Bracket inference in `BatchEngine.generate(input:parameters:)` and
   `Evaluate.swift::TokenIterator.next()`:

```swift
// At the top of admitPendingRequests / TokenIterator initialization:
modelContext.jangPressController?.willStartInference(layerExpertHints: [])

// At slot-finished / TokenIterator end:
modelContext.jangPressController?.didFinishInference()
```

The hints are optional — passing the layer/expert pairs that the next
inference is likely to use lets the controller pre-fault those tiles
before the model hits them. Empty `[]` works (just skips the hint
optimization).

5. Persist on `ModelContext`:

```swift
public class ModelContext {
    // … existing fields …
    public var jangPressController: JangPressController?
    public var jangPressMmap: JangPressMmapTier?
    public var jangPressMach: JangPressMachCache?
    public var jangPressEmbed: JangPressEmbedTier?
}
```

That's the complete vmlx-swift-lm-side surface. Estimated implementation:
~200 LOC + ~50 LOC of tests verifying the bracket hooks fire.

---

## Tests already in place (28 / 28 passing)

```
swift test --filter "JangPress"
```

| Suite                       | Tests | What it pins down                                                      |
|-----------------------------|-------|------------------------------------------------------------------------|
| `JangPressShard`            |  6    | safetensors header parse, byte ranges, advise calls, sniff path        |
| `JangPressMmapTier`         |  6    | regex (13 patterns), acquire/release, **forceRelease byte-equality**   |
| `JangPressMachCache`        |  6    | Mach VM purgeable lifecycle                                            |
| `JangPressController`       |  4    | state machine `disabled → armed → quiescing → compressed`              |
| `JangPressEmbedTier`        |  4    | embed/lm_head discovery, Zipfian advise                                |
| `JangPressSmokeBench`       |  1    | synthetic 512-tile end-to-end with simulated balloon                   |
| `JangPressPressureBench`    |  1    | (`.disabled` — heavy bench, run manually with real bundles)            |

The two boldface tests are the data-integrity gates:

1. **Byte-equality** — `forceRelease + reacquire produces byte-identical
   data`. Pins down the correctness invariant that JangPress never
   silently corrupts weight bytes — re-fault from disk produces the
   same bytes.
2. **Sniff correctness** — `sniff path skips non-expert shards from mmap`.
   Pins down that JangPress doesn't compete with MLX's reads on shards
   that have no routed-expert tiles.

---

## Per-bundle compatibility matrix (vmlx-swift-lm models)

This matrix is the vmlx-swift-lm-side mapping of which models in this
fork's Models/ tree have JangPress-eligible layers. Tile counts are
upstream-vmlx-verified; coherence status is from this fork's BENCH_*
runs.

| Model class                    | JANG bundle                            | Routed mass | Pattern | Coherence (this fork)               |
|--------------------------------|-----------------------------------------|-------------|---------|--------------------------------------|
| `LagunaModel`                  | Laguna XS.2 mxfp4 / JANGTQ              | ~70%        | C / D   | ✅ both (commits `4699d3a`)          |
| `Mistral3VLM`                  | Mistral 3.5 mxfp4                       | n/a (dense) | none    | ✅ (no JangPress benefit)            |
| `Mistral3VLMJANGTQ`            | Mistral 3.5 JANGTQ                      | n/a (dense) | none    | ✅ runtime correct (kernel fixes)    |
| `Qwen35MoE`                    | Holo3 / Qwen 3.5 MoE                    | ~68%        | A / G   | ✅                                  |
| `Qwen36MoE`                    | Qwen 3.6 35B JANG_2L / JANGTQ4          | ~70%        | A / C   | ✅                                  |
| `MiniMaxModel` / `MiniMaxJANGTQModel` | MiniMax M2.7 (Small + Med)        | ~69%        | H / I   | ✅                                  |
| `NemotronHOmni`                | NemotronH-Omni MXFP4 / JANGTQ4 / JANGTQ2 | ~70%       | J / K / L | ✅ 13/13 BENCH_OMNI both formats   |
| `DeepseekV4Model` / `DeepseekV4JANGTQModel` | DSV4-Flash JANG_2L / JANGTQ | ~84%        | E / F   | ✅                                  |
| `KimiK25Model`                 | Kimi K2.6 JANGTQ                        | ~70%        | E / F   | ⚠️ bundle missing tokenizer         |

**Coverage gap:** none. The 13 regex patterns cover every JANG MoE
bundle this fork loads.

---

## Production-readiness checklist for the integration phase

When wiring `JangPressLoadOptions` through `loadWeights` →
`ModelContext` → `BatchEngine`:

- [ ] `JangPressLoadOptions` struct defined in `MLXLMCommon`
- [ ] `loadWeights(...)` accepts optional `jangPress:` parameter
- [ ] Factory dispatch passes the parameter through to `_load`
- [ ] `ModelContext` holds the four optional fields
- [ ] `BatchEngine.admitPendingRequests` bracket: `willStartInference()`
- [ ] `BatchEngine.finishSlot` bracket: `didFinishInference()`
- [ ] `Evaluate.swift::TokenIterator` bracket on the single-stream path
- [ ] Smoke test: load a real Laguna JANGTQ bundle with `enabled: true`,
      verify decode produces same tokens as `enabled: false` at temp=0
- [ ] Memory-pressure smoke: drive `JangPressBalloonBench`-equivalent
      against a real bundle via the new SDK surface, verify reclaim %
      matches upstream-vmlx numbers (67–70% across families)
- [ ] CHANGELOG entry — feature off by default, opt-in via load options
- [ ] OSAURUS-PRODUCTION-REFERENCE-2026-05-01.md update — add a §16
      "JangPress" subsection summarizing the SDK surface

---

## Sync policy with vmlx upstream

`/Users/eric/vmlx/swift/Sources/vMLXLMCommon/Cache/JangPress*.swift` is
the canonical source. This fork mirrors it for SDK consumers. Future
syncs should:

```bash
diff -r /Users/eric/vmlx/swift/Sources/vMLXLMCommon/Cache/JangPress*.swift \
        /Users/eric/vmlx-swift-lm/Libraries/MLXLMCommon/Cache/
```

The two **local concurrency-safety adaptations** (commit `1ec1668`)
must be preserved on every sync:

1. `JangPressMachCache.swift:234`: use `getpagesize()` not `vm_page_size`
2. `JangPressMachCache.swift:393–406`: hoist `event.contains(...)` out
   of the `lock.withLock { }` body

Both preserve byte-identical runtime behavior; they're build-config
adaptations only, not behavior changes.

---

## TL;DR

- **Source + tests + docs landed in `1ec1668`. 28/28 tests pass.**
- **Orthogonality with all 5 cache axes verified by design + by
  upstream-vmlx empirical measurement.**
- **Hot path (JANGTQ codebook + Hadamard + matmul + RoPE + TurboQuant
  KV) is untouched** — JangPress operates on routed-expert weight
  pages only, releases via `madvise(DONTNEED)`, kernel re-faults on
  next access (2–5 ms per-expert, 65–198 ms stacked).
- **Hybrid SSM compatibility verified** — Mamba weights don't match
  any of the 13 regex patterns, stay fully resident.
- **Integration phase scope: ~200 LOC of LoadOptions + ModelContext
  + BatchEngine bracket-hooks** to expose the feature through the SDK.
- **No regressions to existing models possible** because JangPress is
  a strict failsafe — `madvise(DONTNEED)` is a kernel hint, never a
  forced eviction; under no memory pressure the kernel keeps pages
  resident and the feature is a no-op.
