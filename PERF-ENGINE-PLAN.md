# Performance Engine Plan — mlx-swift-lm → vmlx-class speeds

## Current: ~48 tok/s | Target: 143+ tok/s (3x)

Based on deep analysis of `/Users/eric/mlx/vllm-mlx/vmlx_engine/`.

## Phase 1: Compiled Forward Pass (est. +20-40%)

**Source:** `vmlx_engine/model_runner.py:153-176`

vmlx calls `mx.compile(model.__call__)` — compiles the ENTIRE model forward into a fused Metal graph. No cache state tracking needed because Python's nn.Module auto-captures state.

**Problem in Swift:** `compile()` crashes with "eval array without primitive" because:
- KVCacheSimple.update() does scatter-write that creates new array nodes
- The cache arrays aren't captured by the compile tracer

**Fix approach:** Restructure KVCache to work with compile:
- Option A: Make cache update return new arrays instead of mutating (functional cache)
- Option B: Use Module.innerState() to capture ALL state including cache
- Option C: Separate the forward pass from cache update — compile only the pure compute

**Branch:** `perf/compiled-forward`

## Phase 2: Dedicated Metal Stream (est. +10-20%)

**Source:** `vmlx_engine/mllm_batch_generator.py:1116,1835`

```python
MLLMBatchGenerator._stream = mx.new_stream(mx.default_device())
with mx.stream(MLLMBatchGenerator._stream):
    # ALL generation on dedicated stream
```

**Problem in Swift:** `Stream.$defaultStream` is internal to MLX module.

**Fix:** Fork mlx-swift (osaurus-ai/mlx-swift), expose `Stream.withDefaultStream(_:body:)`.
Already done in commit `2224855`. Metal library issue was stale build cache.

**Branch:** `perf/generation-stream`

## Phase 3: Paged KV Cache (est. +30-50% for multi-turn)

**Source:** `vmlx_engine/paged_cache.py` (428 lines)

Block-based KV cache with:
- Configurable tokens-per-block (default 64)
- Reference counting for shared prefix blocks
- Copy-on-Write for prefix sharing across requests
- O(1) LRU eviction via doubly-linked list
- Content-addressable hashing for block dedup

**Implementation plan:**
1. `PagedKVBlock` — fixed-size token block with ref counting
2. `PagedKVCache` — conforms to KVCache protocol, manages blocks
3. `BlockAllocator` — free list + LRU eviction
4. Block hash chain for prefix matching

**Branch:** `feat/paged-cache`

## Phase 4: Prefix Cache (est. +40-60% for chat)

**Source:** `vmlx_engine/prefix_cache.py` (438 lines) + `memory_cache.py` (521 lines)

Memory-aware LRU prefix cache:
- Stores computed KV state for prompt prefixes
- On cache hit: skip prefill entirely, resume from cached state
- Memory pressure detection: evicts when available RAM < 20%
- Forward + reverse prefix matching

**Depends on:** Phase 3 (paged cache) for block-level caching

**Branch:** `feat/prefix-cache`

## Phase 5: Disk Cache (L2/L3) (est. persistence across restarts)

**Source:** `vmlx_engine/disk_cache.py` + `block_disk_store.py`

- L2: Full prompt cache serialized to disk
- L3: Block-level disk store (64-token blocks)
- Lazy loading on cache miss
- Scoped per model (hash includes model path + quantization)

**Depends on:** Phase 3 + 4

**Branch:** `feat/disk-cache`

## Phase 6: Prompt Lookup Decoding (est. +30% for structured output)

**Source:** `vmlx_engine/prompt_lookup.py` + `scheduler.py:235-282`

N-gram based speculative decoding:
- Looks up recent output for repeating patterns
- K=2 drafts for hybrid SSM, K=5 for pure attention
- Auto-tuning with TCP slow-start inspired window

**Branch:** `feat/prompt-lookup-decoding`

## Phase 7: Continuous Batching (est. +2x for concurrent requests)

**Source:** `vmlx_engine/scheduler.py` + `engine/batched.py`

Process multiple requests in a single forward pass:
- Up to 256 concurrent requests batched
- Per-request KV cache management
- Dynamic batch admission/eviction

**Branch:** `feat/continuous-batching`

## Implementation Order

```
Phase 1 (compiled forward) ──► immediate impact, unblocks others
    │
Phase 2 (metal stream) ──► requires mlx-swift fork
    │
Phase 3 (paged cache) ──► foundation for all caching
    │
    ├── Phase 4 (prefix cache) ──► L1 in-memory
    │       │
    │       └── Phase 5 (disk cache) ──► L2/L3 persistence
    │
Phase 6 (prompt lookup) ──► independent, can parallel with 3-5
    │
Phase 7 (continuous batching) ──► final, requires all above
```

## Files to Create/Modify

### New files:
```
Libraries/MLXLMCommon/
  PagedCache/
    PagedKVBlock.swift
    PagedKVCache.swift
    BlockAllocator.swift
    BlockHashChain.swift
  PrefixCache/
    PrefixCacheManager.swift
    MemoryAwarePrefixCache.swift
  DiskCache/
    DiskCacheManager.swift
    BlockDiskStore.swift
  PromptLookup/
    PromptLookupDecoder.swift
    NgramIndex.swift
```

### Modified files:
```
Evaluate.swift — compiled forward, stream, fast-path generate
KVCache.swift — paged cache protocol conformance
ModelContainer.swift — prefix cache integration
ChatSession.swift — prefix cache for multi-turn
```
