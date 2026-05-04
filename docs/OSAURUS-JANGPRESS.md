# Osaurus Integration: JangPress Production Cache

**Status:** production-ready (Iter 26) as of vmlx-swift-lm `main` 2026-05-04.
**Audience:** osaurus host-app engineers using JangPress as the
production weights cache.
**Companion docs:** `OSAURUS-DSV4-INTEGRATION.md`,
`OSAURUS-CACHE-CONTRACT.md`.

## What is JangPress?

JangPress is a memory-mapped tile cache for routed-MoE expert weights.
It allows large models (DSV4-Flash 74 GB, NemotronH ~30 GB,
Qwen3.5-35B-A3B ~35 GB, etc.) to load with a fraction of the resident
RAM that an eager `mx.load(...)` would require — by mmap'ing the
safetensors shards and only paging in the routed-expert tiles when an
inference forward actually touches them.

Iter 24/25/26 architecture (current production):

- **`JangPressPrestacker`** (one-time): scans the bundle, detects
  per-expert `.tq_packed` / `.tq_norms` tensors, and writes a
  pre-stacked overlay in a per-bundle cache directory. The overlay
  has `switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}` keys
  shaped `(n_experts, …)` directly, so the model loader doesn't have
  to call `MLX.stacked(...)` (which would materialize the whole
  routed expert bank into resident Metal buffers and defeat mmap).
- **`MmapSafetensorsLoader`**: direct-mmap path that bypasses the
  heavyweight safetensors framework allocations. Used by the
  prestacker AND the runtime when `LoadConfiguration.useMmapSafetensors
  = true`.
- **`JangPressMmapTier`**: per-tile mmap cache backing the routed
  experts. Splits stacked tiles when the kernel needs only one expert
  slice (iter 25 stacked-tile split fix).
- **`JangPressActivation`**: orchestrates the deferred tier setup —
  all tier work waits until the first inference forward (iter 24 defer
  fix).
- **`JangPressCanonicalExpertAdvisor`**: observes per-layer
  routed-expert selection across every MoE family and feeds eviction
  hints back into the mmap tier.
- **`LoadConfiguration`**: host-app surface for opting in / out of
  JangPress and tuning its knobs.

The previous iter 23 controller (`JangPressController`) and iter 22
Mach cache (`JangPressMachCache`) have been **removed**.

## What changed between iter 23 and iter 26

If you're migrating off the old API:

| Iter 23 (removed) | Iter 26 (current) |
|---|---|
| `JangPressController.shared.preload(...)` | host doesn't call anything; `JangPressPrestacker.prepareBundleIfNeeded(...)` runs once per bundle URL |
| `JangPressController.shared.evictionThreshold = ...` | `LoadConfiguration.jangPressOptions.coldFraction` (clamped to `[0, 0.95]`) |
| `JangPressMachCache` | `JangPressMmapTier` (deferred to first inference) |
| Manual `JangPressController.shared.activate(layerIdx)` | implicit — the canonical-expert-advisor observes routing and the mmap tier follows |
| `JangPressController.shared.flush()` | not exposed — eviction is automatic based on `coldFraction` |

The canonical observer hook lives in every MoE block now:
`JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx,
indices: indices)`. The host never calls `observe(...)` directly.

## Host-app entry point: `LoadConfiguration`

```swift
import MLXLMCommon

let loadConfig = LoadConfiguration(
    // JangPress policy. .auto = enable when the bundle has routed
    // experts; .disabled = always off; .enabled(...) = always on with
    // explicit options.
    jangPressPolicy: .auto,
    jangPressOptions: JangPressLoadOptions(
        coldFraction: 0.7,    // fraction of routed weights to keep cold
        residentCap: .fraction(0.7),
    ),
    useMmapSafetensors: true)

let context = try await ModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(directory: bundleURL),
    progressHandler: nil,
    loadConfiguration: loadConfig)
```

Defaults (`LoadConfiguration.default`):

- `jangPressPolicy = .auto` → enabled when `inspect(...)` detects a
  routed-MoE bundle.
- `coldFraction = 0.70` → keep 70% of routed-expert tiles cold (mmap
  pages, evictable).
- `residentCap = .fraction(0.70)` → cap on resident bytes is 70% of
  physical RAM.
- `useMmapSafetensors = true` → use the direct-mmap loader path.

## Env-var overrides

The host can override defaults at runtime:

| Env var | Effect | Example |
|---|---|---|
| `JANGPRESS` | `0`/`off` disables; integer `0..100` sets `coldFraction = N/100` | `JANGPRESS=70` |
| `JANGPRESS_PRESTACK` | `0` disables the prestack overlay (bundle is loaded as-is); useful for debugging the original safetensors layout | `JANGPRESS_PRESTACK=0` |
| `JANGPRESS_PRESTACK_STRICT` | `1` makes prestack failures fatal instead of falling back to the original bundle | `JANGPRESS_PRESTACK_STRICT=1` |
| `JANGPRESS_ALIGN_SAFETENSORS` | `0` disables the page-alignment overlay | rare |
| `JANGPRESS_ALIGN_JANGTQ` | `1` enables alignment overlay for JANGTQ bundles (off by default — would double cache footprint) | rare |
| `JANGPRESS_ALIGN_CACHE_DIR` | overrides the default cache directory location | `JANGPRESS_ALIGN_CACHE_DIR=/custom/path` |
| `MLX_SAFETENSORS_MMAP` | `0` disables the direct-mmap loader; `1` enables (passed by `LoadConfiguration.useMmapSafetensors`) | wired automatically |
| `MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS` | `1` makes individual tensors mmap'd via Metal (cold-start mode) | wired automatically when JangPress is enabled |
| `MLX_SAFETENSORS_MMAP_START_COLD` | `1` starts every tensor cold (paged out) | wired automatically when JangPress is enabled |
| `MLX_SAFETENSORS_MMAP_COLD_PCT` | integer `0..100` controlling the cold fraction the mmap loader applies; defaults to `coldFraction × 100` | wired automatically |

The env vars are designed for diagnostics. In production, set
`LoadConfiguration` directly.

## Cold-start vs warm decode

JangPress defers all tier work until the first inference forward
(iter 24). The host should expect:

- **Bundle preparation** (prestack overlay write, alignment scan):
  ~5-30 s on first load of a new bundle. Subsequent loads from the
  same cache directory are ~instant.
- **Cold inference** (first forward): seconds to tens of seconds
  while the mmap tier paginates routed-expert tiles for the experts
  the prompt actually selects. DSV4-Flash JANGTQ2 cold first turn:
  ~30-60 s for ~200 tokens of generated output.
- **Warm decode** (after the routing pattern is stable): bound by
  GPU compute, not by I/O. DSV4-Flash JANGTQ2 warm decode: ~5-10
  tok/s on M5 Max.

The `JangPressCanonicalExpertAdvisor` observes routing patterns over
the warm-up window; its hints drive the mmap tier's eviction
decisions.

## Bundle preparation cache

JangPress writes its prestack + alignment overlays into a per-bundle
cache directory under `~/Library/Caches/jangpress/<bundle-hash>/`
(or `JANGPRESS_ALIGN_CACHE_DIR` if set). The cache directory contains:

- `jangpress-prestacked.safetensors` — pre-stacked routed weights.
- `jangpress-prestack-manifest.json` — what got stacked, source files,
  bytes.
- `jangpress-align-manifest.json` (optional) — alignment overlay
  manifest.
- Symlinks for files that didn't need rewriting (`config.json`,
  `tokenizer.*`, etc.).

Safe to delete — JangPress will re-derive on the next load.

## Diagnostic logging

JangPress emits diagnostics on the standard log handler. Notable
messages:

- `[jangpress] using existing prestacked overlay <path>` — cache hit.
- `[jangpress] wrote N prestacked routed tensors (X.X GB) into <path>`
  — cache miss, overlay being written.
- `[jangpress] prestack failed, falling back to original bundle: <err>`
  — non-strict fallback. Set `JANGPRESS_PRESTACK_STRICT=1` to make
  this fatal.
- `[jangpress] alignment overlay flag set but unimplemented; returning
  bundle as-is` — `JANGPRESS_ALIGN_OVERLAY=1` is reserved for a future
  implementation and currently no-ops.

## Tests

`Tests/MLXLMTests/JangPressActivationTests.swift`,
`JangPressMmapTierTests.swift`,
`JangPressEmbedTierTests.swift`,
`JangPressShardTests.swift`,
`JangPressSafetensorsAlignmentTests.swift`,
`LoadConfigurationTests.swift`,
`MmapSafetensorsLoaderTests.swift`. Run with:

```bash
swift test --filter "JangPress|LoadConfiguration|MmapSafetensors"
```

## Known limitations

- Iter 26 ships without the in-flight alignment-overlay rewrite path
  (`prepareAlignedBundleIfNeeded`); the overlay flag is wired but
  no-ops. Bundles whose tensor `data_offset` values aren't page-aligned
  pay an unaligned-mmap cost on cold reads. Not a correctness issue;
  performance only.
- The mmap tier evicts based on the canonical expert advisor's
  observed routing, NOT based on token-level recency. For highly
  oscillating expert-selection patterns (rare in chat workloads) the
  eviction rate can be higher than ideal.

## P0 contracts (must not regress)

1. JangPress is opt-out from the host's perspective —
   `LoadConfiguration.default` enables it. Hosts that explicitly
   opt out must call `LoadConfiguration.off`.
2. The prestack overlay is a per-bundle cache; deleting the cache
   directory MUST NOT corrupt the original bundle.
3. JangPress failures are non-fatal by default and fall back to the
   original bundle. Set `JANGPRESS_PRESTACK_STRICT=1` to opt into
   strict mode.
4. The mmap tier defers all work until first inference (iter 24).
   Before that point, the model is loaded but no tier setup has run.
5. The canonical-expert-advisor observation hook is wired into every
   MoE family. Adding a new MoE family requires threading
   `layerIdx` and calling `observe(...)` after the gate produces
   indices.
