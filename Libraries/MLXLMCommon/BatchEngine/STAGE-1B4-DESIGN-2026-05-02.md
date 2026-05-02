# Stage 1B.4 — multi-batch compile (design + scaffold)

Lifts the `maxBatchSize == 1` guard in `BatchEngine.maybePromoteToCompiledDecode`
so multi-user server deployments can also engage the compile path.

> **Status (2026-05-02):** design doc + `BucketHandle` skeleton landed.
> Production wiring is INTENTIONALLY deferred — the full implementation
> is a multi-week refactor (cache buffer + slot lifecycle + liveness
> mask wiring through attention layers + trace registry). Half-shipping
> would regress the verified Stage 1B.3 path. This file is the spec
> the next iteration follows; the skeleton ensures future work doesn't
> have to reinvent the entry points.

---

## Why it matters

Single-user osaurus chat already gets the compile speedup at
`maxBatchSize=1` (verified 9× TTFT improvement on Mistral 3.5 mxfp4
VLM). Multi-user server deployments today get the **uncompiled**
`stepBatchDecode` path because the existing compile gate at
`BatchEngine.swift:1133` fires `guard self.maxBatchSize == 1 else { return }`.
For a 4-user server, this leaves a ~5-9× per-step speedup on the table.

## Why it's hard

The shipped Stage 1B.3 path keeps each slot's cache state in its own
`CompilableKVCache(maxLength: N)` of shape `[1, H, N, D]`. The compile
trace specializes on the captured input/output array identities
(MLX captures BY ARRAY REFERENCE, not by shape). So one slot ↔ one
trace — admitting a second slot would require re-capturing.

Multi-batch compile requires a **shared per-bucket cache**: one
`[B, H, N, D]` buffer per layer, where each slot's KV state occupies
a row. Slots admit/finish during a batch's lifetime, so the bucket
needs:

- Row assignment (slot → row index)
- Liveness mask (which rows are live at this step)
- Reusing rows after slot finishes
- Possibly multiple buckets per family (e.g., bucket-of-2 vs bucket-of-4)

And the model code has to honor the liveness mask via attention masking,
which means threading the mask through every attention layer — currently
the mask is a single `MLXFast.ScaledDotProductAttentionMaskMode` value.

## Architecture

### `BucketHandle`

One handle per `(family, bucketSize, maxCacheLength)` tuple. Owns:

```
class BucketHandle {
    let key: BucketKey
    let cache: [BucketCacheLayer]              // per-layer [B, H, maxLen, D] buffer
    var slotRows: [SlotID: Int]                // slot → row mapping
    var liveMask: MLXArray                     // [B] bool, refreshed per step
    var compiledForward: ([MLXArray]) -> [MLXArray]?
}
```

A `BucketCacheLayer` is a `BaseKVCache` subclass that:
- Owns a single `[B, H, maxLen, D]` keys/values buffer
- `update(keys, values)` writes new tokens into per-row offsets atomically
- Exposes per-row `offset` as an `MLXArray[B]` (NOT an Int per row)
- `makeMask(...)` factors in `liveMask` to zero out attention to dead rows

### Slot lifecycle within a bucket

```
admit(slot):
    if no bucket exists for slot.cacheFamily:
        bucket = BucketHandle(key: ...)
    if bucket has free row:
        row = bucket.firstFreeRow()
        bucket.slotRows[slot.id] = row
        slot.cache = bucket.cacheView(row: row)  // shaped wrappers
    else:
        slot.cache = freshSlotCaches()           // uncompiled path

step:
    activeRowIndices = [bucket.slotRows[s.id] for s in active]
    bucket.liveMask = liveMaskFromIndices(activeRowIndices)
    forward(batchedTokens)                       // compiled trace

finish(slot):
    bucket.slotRows.remove(slot.id)
    // Row is now "dirty" but available — the bucket clears it lazily
    // when the next slot is admitted (via cache.trim or zeroing).
```

### Trace registry

`BatchCompile` already has `BucketKey` as a hashable identity. Stage
1B.4 adds:

```
actor BatchCompileRegistry {
    private var buckets: [BucketKey: BucketHandle] = [:]
    func handle(for: BucketKey) -> BucketHandle
    func release(slot: SlotID)
}
```

The registry is a singleton per `BatchEngine`. Bucket sizes are
configurable: `[1, 2, 4]` covers most deployments. Slots beyond the
largest bucket fall back to uncompiled.

### Liveness mask plumbing

The CompilableRotatingKVCache `makeMask` already accepts an optional
liveness mask (Stage 1A laid this groundwork via
`BatchCompile.makeLiveMask(bucketSize:, liveIndices:)`). For Stage
1B.4, every Compilable cache class must accept and apply the mask.

Plumbing path:

```
BucketHandle.step:
    liveMask = makeLiveMask(bucketSize: B, liveIndices: activeRows)
    // liveMask is part of the captured trace inputs; refreshed
    // per step via _updateInternal.
    bucket.liveMask._updateInternal(liveMask)
    forward(batchedTokens)

CompilableKVCache.makeMask:
    causalMask = ...causal pattern...
    return causalMask & liveMaskBroadcast([B, 1, 1, K])
```

Dead rows attend to nothing (mask = 0) and are attended-to by no live
row. Sampling skips dead rows post-forward.

## Scope cut

Within a single iteration, ship:

1. ✅ This design doc (here)
2. ✅ `BucketHandle` placeholder type — calls `precondition(false)` if
   instantiated, but the entry point exists so callers can be migrated
   incrementally
3. ✅ A `BatchEngine.maybePromoteToBucket(slot:)` no-op stub
4. Defer to next iteration:
   - `BucketCacheLayer` real implementation
   - Per-bucket trace registry
   - Liveness-mask plumbing through Compilable cache classes
   - Slot↔row lifecycle
   - Multi-bucket fallback ladder
   - Tests

This way the next agent picks this up with a clean architecture, not
an organic accumulation.

## Acceptance criteria for full Stage 1B.4 (when shipped)

- `maxBatchSize=4` with 4 concurrent slots produces tokens within
  fp32 precision (~5e-7) of running the same 4 prompts sequentially
  through the `maxBatchSize=1` compile path.
- TTFT for the 4-slot batch is 2-3× faster than the uncompiled
  `BatchKVCache` path on the same hardware.
- A 4-slot batch where slots finish at different times correctly
  releases rows for new admissions, with ≤5e-7 drift across the
  full conversation.
- No regression on `maxBatchSize=1` — the Stage 1B.3 path keeps
  bypassing the bucket logic.

## Open questions

- Does the model's attention layer code accept a per-row liveness
  mask without a model-class API change? (`MLXFast.scaledDotProductAttention`
  takes one mask; per-row liveness compounds with causal — needs
  expression as an `[B, 1, 1, K]` broadcastable mask.)
- Bucket size selection: static `[1, 2, 4]` or dynamic based on
  observed admit rate? Static is simpler and probably sufficient.
- Cross-family buckets (e.g., one slot is `.simple`, another is
  `.rotating`) — rejected; falls back to uncompiled.

## References

- Stage 1B.3 implementation: `BatchEngine.swift:1131` (`maybePromoteToCompiledDecode`)
- Stage 1A scaffold: `BatchCompile.swift` (`BucketKey`, `nextBucket`,
  `makeLiveMask`, `compileForward`)
- Single-slot compiled decode: `BatchEngine.swift:1044` (`stepCompiledDecode`)
- Uncompiled multi-batch path: `BatchEngine.swift:1271` (`stepBatchDecode`)
- Production reference: `OSAURUS-PRODUCTION-REFERENCE-2026-05-01.md` §13 #6
