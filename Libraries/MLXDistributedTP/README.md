# MLXDistributedTP

Tensor-parallel layer port for the multi-host distributed inference
rollout. Phase 5 of the spec at `docs/superpowers/specs/2026-05-02-distributed-inference-engine-design.md`.

## What's here

- **`Group`** — Swift wrapper around `mlx_distributed_group`.
- **`Collectives`** — `allSum`, `allGather`, `send`, `recvLike`,
  `sumScatter`. Each is identity on a size-1 group (matches the Python
  `mx.distributed.*` fallback semantics).
- **`AllToShardedLinear`** — TP linear with row-sharded weights; each
  rank produces a partition of the output. Mirror of Python's
  `mlx.nn.layers.distributed.AllToShardedLinear`.
- **`ShardedToAllLinear`** — TP linear with column-sharded weights;
  partial outputs are all-reduced. Mirror of Python's
  `ShardedToAllLinear`.
- **`shardLinear`** — convenience that takes a dense `Linear` and
  returns the requested TP variant. `Sharding` enum's raw values match
  the Python string tags exactly (`"all-to-sharded"`,
  `"sharded-to-all"`).

## What's not here yet

- **Quantized variants** (`QuantizedAllToShardedLinear`,
  `QuantizedShardedToAllLinear`) — Phase 5.5. The math is the same;
  the wrapper just needs to thread quant params through. JANGTQ's
  Hadamard kernel layer adds another wrinkle on top.
- **`shardInPlace`** — Python's in-place shard for a whole module
  tree. Requires a Swift equivalent of `tree_map_with_path`; deferred
  until a real model wrapper needs it (Phase 6).
- **Multi-rank correctness tests** — only single-rank reduction is
  exercised here. Real multi-rank verification needs Phase 6's
  TB5-cabled cluster.

## Architecture

```
   ClusterSession (MLXDistributedCore)
            │
            ▼
   ┌────────────────────────────────────┐
   │   MLXDistributedTP (this target)   │
   │                                    │
   │   shardLinear(Linear, .allToSharded)
   │                │                   │
   │                ▼                   │
   │   AllToShardedLinear (UnaryLayer)  │
   │   ShardedToAllLinear (UnaryLayer)  │
   │                │                   │
   │                ▼                   │
   │   Collectives.allSum / allGather  │
   │                │                   │
   │                ▼                   │
   │      Group (mlx_distributed_group) │
   └────────────────────────────────────┘
            │
            ▼
   CmlxDistributedShim (C bridge over mlx-c)
            │
            ▼
   mlx_distributed_* (compiled into Cmlx)
            │
            ▼
   /usr/lib/librdma.dylib  (when JACCL is available)
```

## Tests

19 tests pass on a single-rank build:

```sh
swift test --filter "GroupTests|CollectivesSingleRankTests|LinearLayersSingleRankTests|ShardingHelperTests"
```
