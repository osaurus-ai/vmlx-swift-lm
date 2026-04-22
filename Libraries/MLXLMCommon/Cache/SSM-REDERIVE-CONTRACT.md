# SSM Re-derive Contract

**Landed:** 2026-04-21
**Scope:** `CacheCoordinator` + `BatchEngine` admission/stepPrefill hot path
**Motivation:** Close the "MLLM-path hybrid thinking models now get 0% SSM cache hits on the hot path until we port the LLM scheduler's async rederive to MLLM" regression. Hybrid-SSM models (Qwen3.6-MoE, Mistral4-MoE, Nemotron Cascade, every VLM that wraps one of those) were falling back to full prefill on every partial cache hit.

## The gap

Hybrid-SSM models carry two independent pieces of per-token state:

1. **Attention KV** — content-addressable, cacheable. `PagedCacheManager` and the L2 disk cache handle it cleanly.
2. **SSM companion state** — recurrence-based, path-dependent, NOT reconstructible from KV blocks alone. Without it, a partial-prefix cache hit produces garbled generation.

Before this contract existed, `BatchEngine.stepPrefill` handled the mismatch by **rolling back to full prefill** whenever a partial cache hit landed on a hybrid-SSM model:

```swift
let hasSSMLayer = slot.cache.contains { $0 is MambaCache || $0 is ArraysCache }
let unsafePartial = !remaining.isEmpty && (hasVisualContent || hasSSMLayer)
if unsafePartial {
    // throw away the restored KV, start over
    slot.cache = context.model.newCache(parameters: slot.parameters)
    inputForPrepare = slot.originalInput
}
```

That burned the entire cache benefit: **0% hit rate** on the hot path for the exact workloads (multi-turn chat with a long system prompt + growing user history) where caching would help most. The problem applied equally to the plain LLM path and to the VLM / MLLM path — any request whose cache contained a Mamba/Arrays layer hit the same rollback.

## The fix

Two new components in `Libraries/MLXLMCommon/Cache/`:

- **`SSMReDeriver.swift`** — an actor that runs chunked full prefills on matched prefixes to re-derive SSM state, with sync/async scheduling, dedup of concurrent requests for the same prefix, and an LRU-capped completed-checkpoints buffer. Modeled after osa-jang's `VMLXRuntime/Cache/SSMReDeriver.swift` but adapted to vmlx's `SSMStateCache` (`[MLXArray]` not `[SSMStateLayer]`), `KVCache` protocol (not `VMLXMambaCache`), and cross-actor wiring discipline.
- **`SSMForwardWiring.swift`** — a `@unchecked Sendable` wrapper that encapsulates the `LanguageModel`-is-not-Sendable audit in one auditable place instead of scattering `nonisolated(unsafe)` across every call site. Exposes `makeForward()` and `makeNewCache()` that build the closures `SSMReDeriver` consumes.

Integration points:

### `CacheCoordinator`

- New `public private(set) var ssmReDeriver: SSMReDeriver?`
- Lazily instantiated on `setHybrid(true)` — we need a concrete `SSMStateCache` in hand (the coordinator's own) before we can create the re-deriver.
- Toggling `setHybrid(false)` does NOT tear the re-deriver down. Flipping back and forth must not thrash actor state.

### `BatchEngine.admitPendingRequests`

- When a hybrid slot is admitted and the coordinator's `setHybrid(true)` was just flipped, wire the re-deriver's model closures from an `SSMForwardWiring` wrapper.
- Performed in a detached `Task` so the admission hot path doesn't serialize on the cross-actor hop.

### `BatchEngine.stepPrefill`

- Partial cache hit for a hybrid-SSM slot is no longer a hard rollback.
- Compute `tokenHash = SSMStateCache.makeKey(tokens, boundary: matchedTokens)` for the matched prefix.
- `await reDeriver.consumeCheckpoint(tokenHash: hash)`:
  - Checkpoint present → restore SSM state into `slot.cache`, keep the KV hit, prefill only `remaining`.
  - Checkpoint absent → launch an async `requestReDerive` (fire-and-forget detached Task) for the next turn to benefit, and fall back to full prefill this turn.
- VL content rollback is unchanged — the vision tower can't be partially replayed through a paged KV hit, and that's a separate concern from SSM recurrence.

## Sync vs async scheduling

`SSMReDeriver.syncThreshold` (default 512 tokens): prefixes shorter than this re-derive synchronously. Longer prefixes run asynchronously in a detached `Task` so the current request isn't blocked on a future turn's benefit. The re-derive itself costs roughly a full prefill over the prefix — no magic savings on the re-derive itself; the win is that the cost shifts to background work.

## Deduplication

Concurrent requests for the same `tokenHash` share a single `Task<SSMCheckpoint, Error>`. A sync caller waits on the shared task; an async caller returns `nil` after registering the hash. `deduplicatedRequests` counter exposes the behavior for diagnostics.

## Memory bounding

- `SSMReDeriver.maxCompletedCheckpoints` (default 8): LRU cap on the completed-but-not-yet-consumed buffer. Bounds per-actor memory when many concurrent sessions re-derive into the same process.
- `BatchEngine.stepPrefill` calls `consumeCheckpoint` (removes on success), which is the normal drain path.
- `SSMStateCache.store` is called from the re-deriver's sync/async ingestion path so the coordinator's own fetch path picks up the checkpoint on future turns.

## Concurrency audit

- `SSMReDeriver` is an `actor`. All mutable state is actor-isolated.
- `SSMForwardWiring` is `@unchecked Sendable`. The justification is encapsulated at the type level: MLX Metal command-buffer submissions are serialized through a global `evalLock` in the mlx core, and `LanguageModel.callAsFunction` reads parameters that are loaded once and never mutated afterwards. The `@unchecked` appears in exactly two places (`makeForward()`, `makeNewCache()`) and both are audited by the type-level doc.
- The re-deriver's background tasks use `[weak self]` on the follow-up ingestion task so an actor dying mid-re-derive doesn't keep the re-derive alive.

## Verification

- 8/8 unit tests in `Tests/MLXLMTests/SSMReDeriverTests.swift` cover: unwired nil return, sync re-derive completion, concurrent-request deduplication, consume-once semantics, invalid-boundary rejection, sync-threshold honoring, coordinator lazy instantiation, setHybrid(false) idempotence.
- 7/7 `BatchEngineIntegrationTests` pass after the `step()`/`stepPrefill()` async conversion.
- 6/6 `BatchEngineMultiTurnTests` pass — cache hit / prompt extension / concurrent multi-turn / no cross-contamination / turbo-quant / no-coordinator paths unchanged.
- 6/6 `BatchEngineTurboQuantIntegrationTests` pass.
- 85/85 reasoning / harmony / stop-string / tool-call / KV-policy tests still green.

## End-to-end validation status

The unit suite covers lifecycle + dedup without running a real model. End-to-end integration with a hybrid-SSM model — where the actual benefit shows up as measured tok/s on turn 2 of a multi-turn chat — is covered once a small hybrid test fixture (Nemotron-H-small / Mamba-370m) lands in `Tests/MLXLMTests/Resources/`. That work is tracked under the multi-turn real-model test matrix task. The current fix is provably correct in code but needs a real-model benchmark to validate the measured cache-hit rate lifts from ~0% to "close to non-hybrid baseline".

## Out-of-scope follow-ups

- **Rederive bounded by memory budget.** The adaptive chunk size (32 / 128 / 512 tokens) in `runReDerive` is a heuristic. A memory-aware scheduler that watches `Memory.activeMemory` and throttles chunk size in real time is a follow-up.
- **Checkpoint persistence to disk.** Today checkpoints live in the `SSMReDeriver` actor's completed-checkpoints array only; `SSMStateCache.store` writes the SSM state to the in-memory LRU. Persisting SSM companion state to disk across app restarts would compose well with the existing `TQDiskSerializer` v2 `.mamba` LayerKind.
- **Multi-turn hybrid regression benchmark.** Owner + hardware needed to execute; the hooks are all in place.
