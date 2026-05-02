# JangPressMachCache — integration plan

This file tracks the still-open integration work for
`JangPressMachCache.swift`. The cache itself + 6 unit tests
land in `vMLXLMCommon/Cache/`, the LoadOptions knob in
`Engine.LoadOptions.jangPressCompressPct` (0–100). What's left is
wiring it through the load + dispatch path so it actually engages.

## Status — 2026-05-01

| Component | Status |
|---|---|
| `JangPressMachCache.swift` | DONE — Mach `vm_purgable_control` + per-expert VM regions, hot-pin set, memory-pressure listener, disk-refault path |
| `JangPressMachCacheTests.swift` | DONE — 6/6 passing (register / acquire-release / pinHot / unknown-expert / disk-refault round-trip / pct-knob mapping) |
| `Engine.LoadOptions` knob | DONE — `enableJangPress` + `jangPressCompressPct` (0–100) + `jangPressEnablePrefetch` |
| **JangLoader registration** | TODO — at expert-tile load time, route bytes into `cache.register()` instead of holding them resident |
| **SwitchGLU acquire / release** | TODO — `acquire(layer, experts)` before matmul, `release(layer, experts)` after |
| **Predictive prefetch** | TODO — Layer N+1 expert prediction during Layer N attention |
| **Hot-set seeding** | TODO — first-N runs warm hot-pin set by routing frequency |

## ⚠️ Architectural blocker — SwitchGLU stacks experts

`SwitchGLU` (vMLXLMCommon/SwitchLayers.swift:99) holds expert weights as
ONE stacked tensor of shape `(numExperts, hiddenDims, inputDims)`. The
forward pass calls `MLX.gatherQuantizedMM(x, weight, rhsIndices: idx,
…)` which reads from contiguous memory based on the routing indices.

This is fast — single Metal kernel dispatch, GPU-friendly contiguous
reads, fused gate+up cache in `ensureFusedGateUp()` — but it means
the OS sees ONE huge VM region per layer, not 256 per-expert regions.
We can't ask the kernel to compress "the cold experts" because they
aren't separate pages from its perspective; they share buffer pages
with the hot ones.

**Three integration options, ranked by effort and payoff:**

### Option A — One region per LAYER (low effort, low payoff)
Mark the whole `(N_experts, h, in)` tensor as VOLATILE when the model
is idle (e.g. between turns or under app suspension), NON-VOLATILE
when active. Saves RAM only during background / suspension, not
during decode. ~50 LOC. Useful for "pause and resume" workflows.

### Option B — Split into per-expert tensors (medium effort, real payoff)
Split each `SwitchGLU` weight into N per-expert tensors at load time
and register each with the cache. The matmul path then needs to:
1. Acquire the top-k experts at routing time.
2. Gather their bytes into a small `(k, h, in)` working tensor.
3. Run the existing `gatherQuantizedMM` against that working tensor.
4. Release the top-k after the kernel completes.

Cost: an extra ~5 ms per MoE layer for the gather copy on M4 Max.
Win: under memory pressure, ~70% of expert RAM can be compressed.
~600 LOC of code + a custom MoE forward path.

### Option C — Custom Metal kernel (high effort, max payoff)
Write a `gatherQuantizedMM` variant that reads from a list of
per-expert pointers rather than a stacked tensor. Eliminates the
gather copy. Cost: 1-2 weeks of Metal kernel work. Win: identical
to Option B but no gather copy overhead (~1% better tok/s).

### Recommendation
Ship Option A first as a proof-of-life (small change, lets us watch
the kernel actually compress under pressure). Option B is the right
long-term answer for users with constrained RAM. Option C is a
performance polish for later.

## JangLoader integration

`JangLoader` is currently the canonical place where JANGTQ bundles
hydrate their weight dict. Routed expert tiles are the slice we want
to redirect through `JangPressMachCache`. Sketch:

```swift
// JangLoader.swift — pseudocode for the new path
func loadJangtqWithRECC(modelDir: URL, opts: Engine.LoadOptions) throws -> Hydrated {
    let weights = try loadJangtq(modelDir)  // existing path

    guard opts.enableJangPress else {
        return Hydrated(weights: weights, expertCache: nil)
    }

    let pct = max(0, min(100, opts.jangPressCompressPct))
    let alwaysHotFraction = 1.0 - Double(pct) / 100.0
    let cache = JangPressMachCache(config: .init(
        alwaysHotFraction: alwaysHotFraction,
        enablePrefetch: opts.jangPressEnablePrefetch,
        manualCompressPercent: pct
    ))

    // Walk the weight dict for routed-expert keys (model_type-specific
    // pattern: `layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight`
    // OR per-expert: `layers.N.mlp.experts.E.{...}.weight` depending
    // on the bundle).
    var redirected: [String: MLXArray] = [:]
    for (key, value) in weights {
        if let (layer, expert) = parseRoutedExpertKey(key) {
            // Copy bytes into the cache; key the tile by (layer, expert)
            // and replace the MLXArray with a deferred view that calls
            // back into `cache.acquire(...)` on read.
            try value.asData().withUnsafeBytes { buf in
                _ = try cache.register(layer: layer, expert: expert, bytes: buf,
                                       diskURL: bundleShard(for: key, in: modelDir),
                                       diskOffset: tensorOffset(for: key))
            }
            redirected[key] = makeDeferredExpertView(cache: cache, layer: layer, expert: expert)
        } else {
            redirected[key] = value
        }
    }
    return Hydrated(weights: redirected, expertCache: cache)
}
```

The hard parts:
1. **Deferred expert view** — MLX models hold `MLXArray` references in
   their per-layer module trees. We need the matmul to read the
   *current* tile bytes, not a stale snapshot. Easiest: make the
   `MLXArray` a thin handle whose `.asMTLBuffer` callback enters the
   cache lock, calls `acquire`, returns a Metal-importable pointer,
   and registers an after-kernel hook to call `release`.
2. **MoE dispatch contract** — `SwitchGLU` / `SwitchLinear` selects
   `top_k` indices per token. The cache must see the union per
   layer per step. The acquire happens once, before the per-expert
   matmul fan-out.
3. **Refault disk source** — needs the bundle shard URL + tensor
   offset within the safetensors file. `JangLoader` already knows
   both for streaming load; preserve that mapping.

## Acquire / release shape inside SwitchGLU

```swift
// vMLXLMCommon/SwitchLayers.swift — pseudocode
struct SwitchGLU {
    let cache: JangPressMachCache?
    func forward(x: MLXArray, indices: MLXArray, layerId: Int) -> MLXArray {
        let expertIds = uniqueExperts(from: indices)
        let _ = try? cache?.acquire(layer: layerId, experts: expertIds)
        defer { cache?.release(layer: layerId, experts: expertIds) }
        // ... existing matmul fan-out ...
    }
}
```

`acquire` is fast on warm tiles (NONVOLATILE state already set; kernel
doesn't even touch them). It only pays decompress / refault cost when
the kernel actually reclaimed pages — under low memory pressure, that's
zero.

## Predictive prefetch

Layer N's MoE forward is preceded by attention compute. While attention
is running on the GPU, the CPU can:
1. Look at Layer N+1's most-recently routed experts.
2. Call `cache.acquire(layer: N+1, experts: <predicted>)` early,
   triggering decompress / refault BEFORE Layer N+1's router fires.

Implementation plan:
- Add a routing-history ring buffer (`[layerId: ringbuffer of expertIds]`)
- After Layer N's router fires, write to history.
- During Layer N's matmul phase, kick a `Task.detached` that reads
  Layer N+1's history and calls `acquire` for the top-N most-recent.
- Layer N+1's router runs as normal; if predicted experts overlap the
  actual routed set, decompress already happened. If not, the wasted
  acquire just costs one volatile→nonvolatile→volatile round-trip
  on a tile that won't be used this step (cheap).

## Hot-set seeding

For the first 100-1000 tokens of a session, no routing history exists.
The cache should run with a uniform "all hot" (alwaysHotFraction near
1.0) state, then transition to user-configured `jangPressCompressPct`
after a warmup window. This avoids first-prompt stalls.

```swift
// Pseudocode — graduated hot-set
let warmupTokens = 256
if tokensProcessed < warmupTokens {
    cache.pinHot(layer: layerId, experts: allExpertIds)
} else if tokensProcessed == warmupTokens {
    let topK = pickTopRoutedExperts(byFrequency: routingHistory,
                                     fraction: 1.0 - Double(opts.jangPressCompressPct) / 100.0)
    cache.unpinAll()
    for layer in 0..<numLayers { cache.pinHot(layer: layer, experts: topK[layer]) }
}
```

## Test plan once integration lands

| What | How |
|---|---|
| Smoke: small MoE bundle (e.g. Qwen3-30B-A3B) loads + runs | `swift run DSV4FlashRuntime` against a small bundle |
| Memory savings under pressure | Allocate balloon, run inference, watch RSS |
| Coherence: Think + Non-Think outputs unchanged | MMLU 25q probe vs baseline |
| Throughput hit | bench_speed.py equivalent, +/- a few % expected |
| Compress pct sweep: 0 → 25 → 50 → 75 → 100 | Tok/s vs RAM saving plot |

## Done bar

- A 30B MoE bundle (Qwen 3.6 / GLM 4.7 / similar) loads with the cache
  active at compressPct=50 and decodes correctly.
- RAM working set drops measurably under simulated pressure.
- Tok/s impact ≤ 5% at compressPct=50 on the first decode after warmup.
