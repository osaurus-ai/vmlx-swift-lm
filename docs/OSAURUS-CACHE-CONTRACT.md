# Osaurus Integration: KV Cache Contract

**Status:** production-ready as of vmlx-swift-lm `main` 2026-05-04.
**Audience:** anyone integrating prefix cache, paged cache, or the
BatchEngine multi-turn flow.
**Companion docs:** `OSAURUS-DSV4-INTEGRATION.md`, `OSAURUS-JANGPRESS.md`.

## Per-layer cache types

vmlx-swift-lm allocates a `[KVCache]` array per session, one entry per
attention layer. The concrete cache class depends on the model and on
the layer's role:

| Class | Used by | Persists | Trimmable |
|---|---|---|---|
| `KVCacheSimple` | Plain attention layers (Llama, Qwen3, Mistral 4 dense) | full window | yes |
| `RotatingKVCache(maxSize:keep:)` | Sliding-window attention (Gemma 4 SWA, Mistral 4 long-ctx, MiMoV2 Flash, BaichuanM1, DSV4 layer 0/last) | last `maxSize` tokens | yes |
| `DeepseekV4Cache(slidingWindow:compressRatio:)` | DSV4-Flash layers with `cr > 0` (~41 of 43 layers) | rotating window + compressor pool + indexer pool + per-branch buffers | yes (proportional) |
| `TurboQuantKVCache(...)` | Long-reasoning runs once `BatchEngine.maybeCompress` swaps it in | compressed past tokens via QJL + signs | no (rebuild only) |
| `MambaCache(...)` | SSM hybrid models (NemotronH, Jamba, GraniteMoeHybrid, Qwen3Next) | conv state + SSM state | yes |
| `QuantizedKVCache(...)` | Models with explicit quantized KV (Mistral 4 with maxKVSize, etc.) | compressed keys + values | yes |

`CacheList` may wrap any of these for hybrid attention/SSM stacks; the
disk serializer + restore path see through the wrapper.

## Hetero-attention cache rules (DSV4 specifically)

DSV4-Flash mixes `RotatingKVCache` (cr=0 layers — pure SWA) and
`DeepseekV4Cache` (cr>0 layers — SWA + CSA [+ HSA]) per layer. The
BatchEngine + scheduler treat the per-layer array opaquely; you do
NOT need to special-case DSV4 outside of the model itself.

Rules for the host:

1. Call `model.newCache(parameters:)` once per session and KEEP that
   array for the lifetime of the conversation. Don't share caches
   across requests.
2. Don't try to construct a hybrid cache by hand; the model is the
   only authority on per-layer `compress_ratio`.
3. If you snapshot a cache (e.g. for state-saving), use
   `cache.copy()` — `DeepseekV4Cache.copy()` deep-copies pool +
   buffer state as of 2026-05-04.

## Multi-turn / prefix-cache reuse

The disk-backed prefix cache (`DiskCache` + `TQDiskSerializer`)
serializes a fresh-after-prefill cache into a safetensors blob and
restores it on a future turn whose prompt prefix matches.

What gets serialized for each cache class:

| Class | Persisted state | Tag |
|---|---|---|
| `KVCacheSimple` | `[keys, values]` | `LayerKind.kv = 1` |
| `RotatingKVCache` | `[keys, values]` + 5-tuple `(keep, maxSize, step, offset, idx)` | `LayerKind.rotating = 6` |
| `DeepseekV4Cache` | rotating 5-tuple AS ABOVE, plus `compressRatio`, `slidingWindow`, `compPooled`, `idxPooled`, `bufCompKV/Gate`, `bufIdxKV/Gate` | `LayerKind.deepseekV4 = 7` |
| `TurboQuantKVCache` | compressed-keys + compressed-values prefixes + offset | `LayerKind.tq = 2` |
| `MambaCache` | `state[0]` (conv) + `state[1]` (ssm) + offset | `LayerKind.mamba = 3` |
| `QuantizedKVCache` | 4 or 6 state arrays + group_size + bits + offset | `LayerKind.qkv = 5` |
| anything else | n/a | `LayerKind.skip = 4` (forces re-prefill) |

`DeepseekV4Cache` was previously serialized as `RotatingKVCacheWrapper`
(rotating only, pool dropped). The 2026-05-04 fix promotes it to its
own kind so multi-turn prefix-cache reuse keeps the long-context
summary across turns. **No host-side action required** — it's
transparent.

### Why pool state survives now (and why it didn't before)

The pre-2026-05-04 contract treated the compressor + indexer pool
buffers as ephemeral — the rationale was "they're recomputable from
prompt tokens on the next prefill, so we don't need to persist them."
That is technically true but two consequences hurt production:

1. Re-derivation cost on every turn was significant for long traces
   (each pool row requires `compress_ratio` raw KV positions, normed
   + RoPE'd + softmax-pooled).
2. **The bigger problem:** `DeepseekV4Cache.trim(n)` only delegated to
   `local.trim(n)`. The pool rows from output tokens that the trim was
   meant to discard kept living in `compPooled` / `idxPooled` and were
   restored on the next turn. This caused the "polite-assistant
   attractor loops" reported on `/v1/chat/completions` long traces.

The 2026-05-04 fix:

- `DeepseekV4Cache.trim(n)` does proportional pool-row truncation
  (`drop = max(1, n / compressRatio)`), matching `llama.cpp
  dsv4_clear_rows`.
- `DeepseekV4Cache.trim(n)` unconditionally clears
  `compBufferKV`/`compBufferGate`/`idxBufferKV`/`idxBufferGate` —
  these are start_pos-keyed and invalidated by ANY trim.
- `DeepseekV4Cache.copy()` deep-copies pool + buffer state.
- Disk serializer round-trips the full state.

## Paged cache compatibility

`PagedCacheManager` allocates fixed-size blocks per layer and keys
them on the prefix hash so identical prefixes share blocks across
sessions. Compatible with:

- `KVCacheSimple` (default block ring).
- `RotatingKVCache` with `maxSize > 0` (the SWA block ring uses
  `maxSize` as the ring window length).

NOT compatible with:

- `DeepseekV4Cache` (any `cr > 0` layer). Pool rows summarize variable
  spans of raw KV; they don't fit the uniform-block-per-token model.
- `TurboQuantKVCache` (compressed keys/values are not block-addressable).
- `MambaCache` (cumulative SSM state has no token boundary).
- `QuantizedKVCache` (qweight packing differs per-layer).

For DSV4-Flash specifically: the entire model is paged-incompatible
because every cr>0 layer would need its own ring shape. Use the
default `[KVCache]` array path — the disk-backed prefix cache via
`TQDiskSerializer` handles multi-turn reuse without the paged manager.

## Trim semantics

`cache.trim(n)` discards the latest `n` token positions from each layer
to support speculative-decoding rollback or scheduler revocation.

Per-class behavior:

| Class | What `trim(n)` does |
|---|---|
| `KVCacheSimple` | Drops the latest `n` token positions from `[keys, values]`. |
| `RotatingKVCache` | Decrements `idx` by `n` (with wrap-around math). |
| `DeepseekV4Cache` | Calls `local.trim(n)`, clears `bufCompKV/Gate`/`bufIdxKV/Gate` unconditionally, drops trailing `max(1, n/compressRatio)` rows from `compPooled`/`idxPooled`. |
| `TurboQuantKVCache` | Not trimmable (`isTrimmable` returns false). The scheduler avoids speculation rollback past the compressed prefix. |
| `MambaCache` | The SSM state cannot be partially un-applied; trim resets to zero. |
| `QuantizedKVCache` | Drops the latest `n` token positions from each of the 4-6 state arrays. |

## BatchEngine + Coordinator interactions

- `CacheCoordinator` (admission control) accepts `DeepseekV4Cache` via
  the `RotatingKVCacheWrapper` path. The KV-budget accounting includes
  pool rows automatically because `DeepseekV4Cache.nbytes` returns
  rotating + pool + buffer bytes.
- `BatchEngine.maybeCompress` may swap `KVCacheSimple` for
  `TurboQuantKVCache` once offset crosses the min-tokens threshold.
  This is opt-in via `GenerateParameters.kvMode = .turboQuant(...)`
  and only fires when DSV4 is loaded with `DSV4_KV_MODE = tq`.
- For DSV4 prefix-cache hits, `restoreDeepseekV4Layer` reseats the
  rotating + pool + buffer state in one shot — no incremental re-prefill
  needed.

## Common pitfalls

1. **Constructing a `DeepseekV4Cache` without `compressRatio`.**
   The init now requires it. Use `model.newCache(parameters:)` instead
   of constructing caches by hand.
2. **Sharing `DeepseekV4Cache` across requests.** The pool state IS
   conversation-specific. Never reuse a cache from a different session
   even if the prompts look similar.
3. **Treating `cr = 0` and `cr > 0` layers identically.** Layers 0 and
   `n - 1` are pure SWA; everything else is hybrid. The model handles
   this internally; the host doesn't need per-layer logic.
4. **Skipping `cache.trim(n)` after speculative-decoding rollback.**
   The DSV4 trim now does proportional pool-row truncation; skipping
   it leaves contaminated pool entries that hurt next-token quality.
5. **Force-enabling paged cache for DSV4.** It will silently produce
   wrong attention output. The model is paged-incompatible by design.
