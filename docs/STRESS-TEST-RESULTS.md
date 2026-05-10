# Stress Test Results

## Gemma 4 E2B — Full Cache Stack Integration

**Date:** 2026-04-12
**Model:** `gemma-4-e2b-it-4bit` (Gemma4, 2B params, Q4_0, context 131072)
**Server:** Osaurus on localhost:1337
**CLI:** `osaurus-cli` (debug build)
**Platform:** macOS Darwin 25.2.0, arm64

---

## Server Integration

The multi-tier `CacheCoordinator` was wired into the Osaurus server runtime with three changes:

### 1. ModelRuntime.swift — Enable CacheCoordinator on model load

```swift
// After loadModelContainer:
let cacheConfig = CacheCoordinatorConfig(
  usePagedCache: true,
  enableDiskCache: true,
  pagedBlockSize: 64,
  maxCacheBlocks: 1000,
  diskCacheMaxGB: 10.0,
  ssmMaxEntries: 50,
  modelKey: name
)
container.enableCaching(config: cacheConfig)

// Auto-detect hybrid models
if let coordinator = container.cacheCoordinator {
  let isHybrid = await container.perform { ctx -> Bool in
    let testCache = ctx.model.newCache(parameters: nil)
    return testCache.contains { $0 is MambaCache || $0 is ArraysCache }
  }
  coordinator.setHybrid(isHybrid)
}
```

### 2. MLXGenerationEngine.swift — Pass coordinator to TokenIterator

```swift
static func prepareAndGenerate(
  ...
  cacheCoordinator: CacheCoordinator? = nil  // added
) async throws -> ( ... )

// Inside single-phase path:
let iterator = try TokenIterator(
  input: effectiveInput,
  model: contextWithEOS.model,
  cache: cache,
  parameters: parameters,
  cacheCoordinator: cacheCoordinator  // added
)
```

### 3. All three call sites in ModelRuntime updated

- Prefix cache warmup call (~line 419)
- Main generation call (~line 560)
- Retry-after-cache-failure call (~line 592)

All pass `cacheCoordinator: holder.container.cacheCoordinator`.

---

## Test Results

**Total: 199/199 requests passed (100%). No crashes, no HTTP errors, no hangs.**

| # | Test | Result | Details |
|---|------|--------|---------|
| 1 | Basic single request | OK | Gemma responded correctly |
| 2 | Multi-turn session reuse (10 turns) | **10/10** | Same `session_id`, KV cache reused each turn |
| 3 | Concurrent requests (5 simultaneous) | **5/5** | All returned HTTP 200 |
| 4 | Prefix cache reuse via `cache_hint` | **5/5** | `prefix_hash` consistent, follow-up requests reused cache |
| 5 | Long prompt (~1562 prompt tokens) | OK | 481 prompt tokens from ~5000 char system prompt |
| 6 | Rapid fire (20 sequential) | **20/20** | Completed in 7.3s |
| 7 | Streaming mode (10 requests) | **10/10** | 4-5 SSE events per request |
| 8 | Mixed concurrent (3 stream + 3 non-stream) | **6/6** | 18 SSE events per stream, non-stream OK |
| 9 | Session switching (5 sessions x 3 turns) | **15/15** | Rapidly alternating between sessions |
| 10 | High concurrency burst (10 simultaneous) | **10/10** | All returned HTTP 200 |
| 11 | Edge cases | **7/7** | Empty msg, single char, max_tokens=1, temp=0, temp=2.0, 5K char msg, multi-message chat |
| 12 | Sustained load (50 sequential) | **50/50** | 13.3s total, 3.8 req/s |
| 13 | Concurrent sustained (5 threads x 10) | **50/50** | 16.7s total |
| 14 | Prefix cache + session reuse combined | **8/8** | Same `prefix_hash` across all 8 turns |
| 15 | Post-stress health check | OK | Server running, model listed, final request succeeded |

---

## Key Observations

- **Prefix cache works end-to-end:** `prefix_hash` was consistent across requests sharing the same system prompt (`45fa9109f11d3eea8b4515405890cbdf` for math tutor prompt, `6e340b9cffb37a989ca544e6bb780a2c` for no system prompt).
- **Session KV cache reuse** functioned across multi-turn conversations without errors.
- **Concurrency** handled cleanly: 10 simultaneous requests completed without race conditions or timeouts.
- **Edge cases** (empty input, single char, extreme temperatures, very long prompts) all returned HTTP 200.
- **Throughput:** ~3.8 req/s sequential, ~3.0 req/s under 5-thread concurrent load (with max_tokens=3-5).
- **Streaming** worked correctly in isolation and mixed with non-streaming concurrent requests.
- **No memory issues** observed during sustained load (50 sequential + 50 concurrent).

---

## Unit Tests

Additionally, 15 XCTest cases in `GemmaStressTests.swift` exercise a synthetic Gemma model with all cache tiers enabled (paged, disk, SSM companion):

1. Basic generation
2. Prefix cache hits
3. Concurrent generation
4. SSM companion store/fetch
5. Disk cache store/fetch
6. Rapid sequential reuse
7. Cache eviction under pressure
8. SSM deep-copy isolation
9. Large prompt chunked prefill
10. Cache clear during generation
11. High-concurrency coordinator access
12. Batch engine integration
13. Varying prompt lengths
14. Rapid clear/rebuild cycles
15. All sampling strategies

All 15 tests pass.
