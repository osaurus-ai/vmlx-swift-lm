# JANGPress — Usage Guide

How to turn on the cold-weight tier from your code, your CLI, or the
Settings UI; how to interpret the stats; and how to pick the backend.

## Toggling

### From `vmlxctl serve`

```bash
vmlxctl serve \
  --enable-jangpress \
  --jangpress-compress-pct 50 \
  --jangpress-backend mmap \
  --jangpress-force-mode soft
```

| Flag | Values | Default | Meaning |
|---|---|---|---|
| `--enable-jangpress` | bool | off | Master switch |
| `--jangpress-compress-pct` | 0..100 | 50 | % of routed mass open to compression |
| `--jangpress-backend` | `mmap`, `mach`, `none` | `mmap` | Compression backend |
| `--jangpress-force-mode` | `soft`, `force` | `soft` | Eviction aggressiveness (see below) |

### From a custom Swift host

```swift
import vMLXEngine

let engine = Engine()

var opts = Engine.LoadOptions(modelPath: bundleURL)
opts.enableJangPress = true
opts.jangPressCompressPct = 50         // 0-100, % open to compression
opts.jangPressBackend = .mmap          // .mmap | .mach | .none
opts.jangPressForceMode = .soft        // .soft (default) | .force

let stream = await engine.load(opts)
for try await event in stream { /* drain */ }
```

### From the SwiftUI app

(future iter — Settings UI panel not yet built)

The corresponding `GlobalSettings` fields exist already:
`enableJangPress`, `jangPressCompressPct`,
`jangPressBackend`, `jangPressForceMode`. A future iteration will
surface these in the SwiftUI Settings sheet under "Memory".

## Backend selection

Two backends ship; both opt-in via `enableJangPress=true`,
both configurable via the same `jangPressCompressPct` knob.

### `.mmap` (default — file-backed, recommended)

- Reads bundle safetensors via `mmap(PROT_READ)`.
- Zero RAM overhead — pages share with the kernel page cache.
- `madvise(MADV_DONTNEED)` on dormant routed-expert byte ranges
  lets the kernel reclaim those pages on demand.
- Survival under pressure: kernel evicts our mmap'd pages first
  (since they're cleanly file-backed). Re-fault from disk on access.
- **Ships today.** Doesn't conflict with MLX's own storage.

```swift
opts.jangPressBackend = .mmap
```

### `.mach` (Mach VM purgeable, gated on MLX-swift fork)

- Allocates fresh `VM_FLAGS_PURGABLE` regions and copies weights in.
- Kernel does WKdm compression on dormant pages (faster decompress
  than disk re-fault).
- **Doubles RAM at load** until MLX-swift is forked to read from
  our regions instead of holding its own copy.
- Right answer once that fork lands; until then, `.mmap` is
  strictly better.

```swift
opts.jangPressBackend = .mach
```

### `.none`

Disables the feature even if `enableJangPress=true`. Useful
for A/B testing.

```swift
opts.jangPressBackend = .none
```

## Force mode (`.soft` vs `.force`)

The pressure-driven release path has two flavors:

### `.soft` (default — recommended)

- `release()` calls `madvise(MADV_DONTNEED)` only.
- Kernel treats the call as a HINT — under low memory pressure,
  pages may stay resident and the RSS drop is small (or zero).
- **Failsafe.** Worst case = no reclaim, no slowdown.
- Picks up real reclaim under genuine memory pressure (the
  `DispatchSourceMemoryPressure` warn/critical events).

### `.force` — eager reclaim

- `release()` additionally calls `msync(MS_INVALIDATE | MS_ASYNC)`.
- On Darwin, this drops file-backed clean pages **immediately**
  regardless of pressure level.
- Pays a real cost: re-acquiring an evicted expert causes a disk
  re-fault. First-token after compaction takes 6.5 ms (DSV4 per-expert)
  to 80 ms (Laguna stacked-per-layer).
- Full-decode cost (force pct=70) measured at ~3× decode slowdown
  on DSV4. Use only when RAM is the binding constraint.

```swift
opts.jangPressForceMode = .force
```

**When to pick which**:

| Scenario | Mode |
|---|---|
| RAM-roomy host (≥64 GB free) | `.soft` |
| Tight host (≤32 GB), MoE bundle ≥40 GB | `.force` (with low compressPct) |
| Production user-facing | `.soft` (always) |
| Internal benchmarking / RSS verification | `.force` |
| Headless batch jobs OK with first-token cold lag | `.force` |

## Reading stats

The `Engine.cacheStats()` method returns a dict with a `jangPress`
sub-dict reporting backend status + counters:

```swift
let stats = try await engine.cacheStats()
if let ember = stats["jangPress"] as? [String: Any] {
    print("backend: \(ember["backend"] ?? "?")")
    print("expertCount: \(ember["expertCount"] ?? 0)")
    print("totalRoutedBytes: \(ember["totalRoutedBytes"] ?? 0)")
}
```

### `.mmap` backend stats

| key | type | meaning |
|---|---|---|
| `backend` | String | `"mmap"` |
| `shardCount` | Int | safetensors shards opened |
| `expertCount` | Int | routed experts indexed |
| `totalRoutedBytes` | UInt64 | total bytes managed |
| `byLayer` | [(Int, Int)] | (layer_id, expert_count) |

### `.mach` backend stats

| key | type | meaning |
|---|---|---|
| `backend` | String | `"mach"` |
| `totalTiles` | Int | registered VM regions |
| `totalBytes` | Int | total bytes copied into purgeable regions |
| `hotPinned` | Int | tiles always-resident |
| `acquireCount` | UInt64 | NONVOLATILE flips |
| `releaseCount` | UInt64 | VOLATILE flips |
| `refaultCount` | UInt64 | tiles reloaded from disk after kernel discarded |
| `discardCount` | UInt64 | tiles kernel discarded entirely (vs just compressed) |
| `pressureLow`, `pressureWarn`, `pressureCrit` | UInt64 | memory pressure events seen |

## Picking `jangPressCompressPct`

The slider goes 0..100 — what value to start with?

| Setup | Suggested | Why |
|---|---|---|
| 256+ GB RAM, single MoE | 0 | No pressure expected; running the cache is wasted CPU |
| 128 GB, single MoE | 30 | Light buffer for multitasking / second model |
| 64 GB, single MoE | 50 | Moderate compression; expect modest first-token latency |
| 32 GB, big MoE | 70 | Aggressive — most experts compressed when idle |
| 16 GB, anything ≥ 8 GB model | 100 | Maximum compression; first-token latency is real |

100 means "only the actively-routed top-k experts pinned hot at any
moment" — and right now that needs the MLX storage hook to actually
save RAM. With `.mmap`, even 100 doesn't shrink MLX's copy; the kernel
just keeps our mmap pages reclaimable.

## Failure modes (and what to expect)

| Symptom | Likely cause | Fix |
|---|---|---|
| Tok/s drops 5-15 % under load | First-token decompress / refault on cold experts | Lower `compressPct`, or pick `.mmap` over `.mach` |
| First request after long idle is 200-500 ms slow | Quiesce-time compression woke up (`.mach`) | Expected; subsequent requests are fast |
| `acquire` throws `alreadyDiscarded` | `.mach` tile was dropped; no diskURL registered | Wire diskURL via JangLoader integration (TODO) |
| RAM doesn't drop with `.mmap` | MLX still holds its own copy | Expected; .mmap helps under pressure but not under low load |
| RAM doesn't drop with `.mach` | Need MLX storage replacement | Gated on MLX-swift fork |

## Compatibility matrix

| Cache axis | Compatible with JANGPress? |
|---|---|
| Prefix cache (memory) | ✓ orthogonal |
| L1 disk cache | ✓ orthogonal |
| L2 BlockDiskCache | ✓ orthogonal |
| TurboQuant KV cache | ✓ orthogonal |
| PoolQuantizedV4Cache (DSV4) | ✓ orthogonal |
| SSM companion cache | ✓ orthogonal |
| SSM re-derive | ✓ orthogonal — re-derive forward calls go through ember.willStartInference() |

JANGPress lives on **axis E** in the cache architecture (model
weights, model lifetime). All other axes (KV state, prefix cache,
SSM state) keep working unchanged.

## Logs to look for

`enableJangPress=true` produces a single info-level line at
load time, in the `cache` category:

```
JANGPress [.mmap] armed: shards=85 experts=11008 routedBytes=70114MB compressPct=50
```

or

```
JANGPress [.mach] armed: compressPct=50 keepHot=50% prefetch=true
```

If init failed (e.g. mmap couldn't open a shard) you'll see:

```
JANGPress [.mmap] init failed: <error> — falling back to disabled
```

The model load proceeds with the cache disabled — failsafe.

## What to expect on a real bundle (measured)

### Laguna-XS.2-JANGTQ (9.4 GB, 256 experts × 39 MoE layers, stacked-tile layout)

`swift run JANGPressSmoke <bundle>`:
- 11 shards opened in 8 ms
- 39 routed-expert tiles indexed (one stacked tensor per MoE layer)
- 7488 MB of routed-expert mass managed
- 196 MB of embedding + lm_head mass managed
- Per-acquire latency: 1.6 ms cold (external SSD), <1 µs warm

`swift run JANGPressRSSBench <bundle>` with `compressPct=70`:

| Phase | RSS |
|---|---|
| Process baseline | 12.5 MB |
| All routed pages resident | 7504 MB |
| **After partial forceRelease (70 %)** | **2129 MB** |
| Reclaimed | **5376 MB = 71.8 % of routed mass** |

Compaction pass takes 137 ms, first re-acquire 80 ms.

### DSV4-Flash-JANGTQ (79 GB, 256 experts × 43 layers, per-expert layout)

`swift run JANGPressSmoke <bundle>`:
- 86 shards opened in 697 ms
- **11,008 routed-expert tiles** indexed (256 × 43)
- **73,728 MB (72 GB)** of routed-expert mass — **91 %** of the bundle
- 252 MB of embed + head mass (129K vocab × 512 hidden bf16 each)
- Per-acquire: 112 µs avg (per-expert tiles smaller than Laguna's stacked)

`swift run JANGPressRSSBench <bundle>` measured numbers:

| Phase | RSS |
|---|---|
| Process baseline | 12.5 MB |
| All routed pages hot | 73,743 MB |
| After msync(INVALIDATE) all | **15 MB** — 100 % reclaim |
| **After partial forceRelease (compressPct=70)** | **23,055 MB** |
| **Reclaimed at pct=70** | **50,687 MB = 68.7 % of routed mass** |
| Compaction pass time | 1.6 s (28k tile parts) |
| First re-acquire after release | **6.5 ms** (per-expert tiles fault fast) |

**This is the headline number**: a 79 GB bundle becomes
~23 GB-resident under quiesce with `compressPct=70`. On a 64 GB
Mac that's the difference between "can't fit" and "fits with
~40 GB free for other apps".

DSV4's per-expert tile layout makes re-acquire ~12 × faster than
Laguna's stacked-per-layer layout (6.5 ms vs 80 ms first-token
latency on the next inference after compaction).

### Mistral-Medium-3.5-128B (dense, NO MoE)

Mistral 3.5 has no routed experts — only the embedding tier engages.
Estimated savings: ~3 GB on a 30 GB bundle (10 % of model). Smaller
absolute win, but still material on constrained-RAM systems.
