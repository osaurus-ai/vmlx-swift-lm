# Distributed Inference — Design

**Branch:** `feature/distributed-inference`
**Status:** Phase 0 design. No code yet.
**Goal:** Pipeline parallelism first, then tensor parallelism, across any set of
Apple Silicon Macs. Fast path on TB5/TB4; fallback on 10GbE and standard Ethernet.

## TL;DR

- **Transport is already solved** by mlx-core's `distributed` module (upstream
  `ml-explore/mlx`). Backends: `jaccl` (Apple-native collective communication),
  `ring` (generic TCP, works over any IP including TB Bridge), `mpi`, `nccl`
  (CUDA — not relevant here). **mlx-swift does not expose any of this today.**
- **Exo's approach** (github.com/exo-explore/exo — now Rust + Python):
  cycle/ring topology discovery via libp2p-style swarm, placement prefers
  RDMA-tagged cycles, delegates actual collectives to MLX's `jaccl` backend.
  Thunderbolt is handled by enabling macOS's built-in **Thunderbolt Bridge**
  (`bridge0`), which presents TB as an Ethernet-class interface — no kernel
  module required for a working first cut.
- **What vmlx needs to build:** Swift bindings for mlx-core distributed, a
  model-partitioning layer, a pipeline-parallel forward pass in BatchEngine,
  and later tensor-parallel variants of Linear/Attention/MoE ops.
- **What vmlx does NOT need to build:** the wire protocol, RDMA drivers, or
  swarm discovery for the first phase. Phase 0 uses static topology + mlx
  collectives; Phase 2+ adds auto-discovery.

## Research findings

### 1. mlx-core distributed API (C++)

Namespace `mlx::core::distributed`, headers in
`github.com/ml-explore/mlx/tree/main/mlx/distributed`:

```cpp
// distributed.h
MLX_API bool is_available();
MLX_API bool is_available(const std::string& bk);
struct MLX_API Group {
    int rank() const;
    int size() const;
    Group split(int color, int key = -1) const;
};
MLX_API Group init(bool strict = false, const std::string& bk = "any");

// ops.h
array all_sum  (const array& x, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array all_gather(const array& x, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array all_max  (const array& x, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array all_min  (const array& x, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array sum_scatter(const array& x, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array send     (const array& x, int dst, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array recv     (Shape shape, Dtype dtype, int src, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
array recv_like(const array& x, int src, std::optional<Group> group = std::nullopt, StreamOrDevice s = {});
```

Backends present in `mlx/distributed/`: `jaccl/`, `ring/`, `mpi/`, `nccl/`.
`init(bk: "any")` picks the first available. On a pair of Macs without MPI
installed, it will fall back to `ring` (TCP) — which works over any IP
interface including the TB Bridge.

**No Swift surface.** `.build/checkouts/mlx-swift/Source/` has MLX, MLXFFT,
MLXFast, MLXLinalg, MLXNN, MLXOptimizers, MLXRandom. Nothing distributed.

### 2. exo's architecture (github.com/exo-explore/exo)

Now a hybrid Rust + Python project (was Python-only earlier). Shape:

- `rust/networking/` — libp2p swarm for discovery (`discovery.rs`, `swarm.rs`).
  Includes a `RESEARCH_NOTES.txt` exploring NDRV sockets, BPF, and a potential
  macOS Thunderbolt kernel module — **not in production**. Current Mac path
  rides on the macOS Thunderbolt Bridge (`bridge0`).
- `src/exo/master/placement.py` — cycle-based placement. Picks the smallest
  topology cycle that fits the model by memory, prefers RDMA-tagged cycles,
  then orders by a download-progress score × available RAM.
- `src/exo/shared/topology.py` — `Topology` is a directed graph of `NodeId`s
  and connection types (`SocketConnection`, `RDMAConnection`). Each node
  carries `NodeNetworkInfo` with `interface_type`, `ip_address`, and
  `ThunderboltBridgeStatus`.
- `src/exo/worker/plan.py` — runtime task-dispatch for shard assignments.
  Actual layer-to-node mapping lives in `get_shard_assignments()`.
- Runtime collectives: `MlxJaccl` — mlx-core's `jaccl` backend. Exo does
  **not** reinvent the data plane; it picks the topology and delegates.
- Tensor-parallel gate: `hidden_size % len(cycle) == 0 && kv_heads % len(cycle) == 0`.
  Same constraint we'll adopt.

### 3. Apple Silicon transport landscape (2026-04-21)

| Transport | Bandwidth (practical) | User-space API | Ready today |
|---|---|---|---|
| Thunderbolt Bridge (`bridge0`) | ~10-20 Gbps, sometimes higher on TB5 | Standard BSD sockets / Network.framework over TCP | **Yes** — macOS enables with System Settings → Network → Thunderbolt Bridge |
| 10 GbE | ~9 Gbps | Same | Yes |
| TB5 "RDMA" (user-space DMA) | Would be higher | **Not exposed by macOS** | No — needs either kernel work (exo is researching) or Apple SDK support |
| Multipeer Connectivity | Low, high overhead | `MultipeerConnectivity` framework | Yes but poor fit for data plane |

**Decision:** Phase 0 and Phase 1 use `bridge0` + `ring` TCP backend. Not
inventing transport. Revisit TB5 RDMA when either exo ships a path or Apple
exposes one.

## vmlx constraints to preserve

Each must survive the distributed work unchanged or through a distributed-aware
variant. I've read every one of these.

| Component | Location | Distributed impact |
|---|---|---|
| `BatchEngine` actor | `BatchEngine/BatchEngine.swift` | Must evolve: the final stage owns sampling + `.chunk`/`.reasoning`/`.toolCall` emission; earlier stages just forward hidden states + manage their own slot KV. |
| `CacheCoordinator` | `Cache/CacheCoordinator.swift` | Each stage owns its own coordinator for its local layers. Paged/disk/SSM tiers are per-stage. Cross-stage cache is a Phase 4 concern. |
| `KVCache` variants (Simple, Rotating, Mamba, Arrays, CacheList, TurboQuant, Quantized) | `KVCache.swift` | All survive unchanged for pipeline parallel — each stage's layers allocate their own caches. Tensor parallel needs sharded attention on heads: `KVCacheSimple`'s `[B, H, L, D]` gets split along `H` by rank. |
| JANG / JANGTQ quantization | `JangLoader.swift`, `JANGTQKernels.swift` | Weights partition by layer → pipeline parallel is free. Tensor parallel within a quantized linear needs the dequantize step sharded across ranks. |
| TurboQuant KV compression | `BatchEngine/BatchQuantize.swift` | Local to each stage's KV. Unchanged. |
| Reasoning parser + tool-call processor | `ReasoningParser.swift`, `Tool/ToolCallProcessor.swift` | Only ever runs on the final stage. Zero cross-stage concern. |
| Sliding-window cache (Gemma-4 SWA, Mistral4 `maxKVSize`) | `RotatingKVCache` + SLIDING-1 serializer | Each stage's own — no cross-stage dependency. |
| Hybrid SSM (Qwen 3.6 MoE, Mistral 4, Nemotron Cascade) | `MambaCache`, `ArraysCache`, `CacheList` | SSM recurrence is path-dependent *within a layer*. Pipeline parallel is fine — the SSM layer runs on exactly one stage. Tensor-parallelizing across an SSM layer is a genuine research problem (deferred to future phase). |
| Speculative decoding | `SpecDec/` | Last-stage concern. Drafter can run fully on the last stage or on a dedicated node. |
| Coordinator-owned KV sizing contract (just shipped) | `KV-SIZING-CONTRACT.md` | Composes cleanly — each stage applies the same policy to the fraction of layers it owns. |

## Upstream blocker found during Phase 0 build

`osaurus-ai/mlx-swift` at `osaurus-0.31.3` deliberately excludes distributed
compilation in its `Package.swift`:

```
// line 193
// do not build distributed support (yet)
"mlx/mlx/distributed/mpi/mpi.cpp",
"mlx/mlx/distributed/ring/ring.cpp",
"mlx/mlx/distributed/nccl/nccl.cpp",
"mlx/mlx/distributed/nccl/nccl_stub",
"mlx/mlx/distributed/jaccl/jaccl.cpp",
"mlx/mlx/distributed/jaccl/mesh.cpp",
"mlx/mlx/distributed/jaccl/ring.cpp",
"mlx/mlx/distributed/jaccl/utils.cpp",
```

and on lines 116–117 additionally excludes the mlx-c bridge that our Swift
layer links against:

```
// example code + mlx-c distributed
"mlx-c/mlx/c/distributed.cpp",
"mlx-c/mlx/c/distributed_group.cpp",
```

Result: headers visible, symbols unresolved. Our Swift bindings link-fail
until the exclude list is shortened and at least one backend is compiled in.

**Phase 0 mitigation (shipped in commit on this branch):**
`Libraries/MLXLMCommon/Distributed/CFallback/MLXDistributedFallback.c`
provides **weak-alias** stubs for every `mlx_distributed_*` symbol the
Swift layer imports. Semantics are identity-on-size-1 — provably correct for
single-rank operation and serves as a complete dev-box target. Every stub is
`__attribute__((weak))` so when Phase 0.5 patches mlx-swift to include the
real implementations, the linker picks the real symbols automatically with
no Swift-side change.

**Phase 0.5 (upstream patch — owner: vmlx maintainer):**
Branch `osaurus-0.31.3-distributed-phase0` off `osaurus-ai/mlx-swift`:

1. Remove `"mlx-c/mlx/c/distributed.cpp"` and
   `"mlx-c/mlx/c/distributed_group.cpp"` from the `mlxSwiftExcludes` list.
2. Remove `"mlx/mlx/distributed/ring/ring.cpp"` from the excludes (keep the
   jaccl + mpi + nccl excludes — ring alone is sufficient for TCP over
   Thunderbolt Bridge / Ethernet).
3. Keep `no_jaccl.cpp`, `no_nccl.cpp` included (they're already included by
   default; they stub the non-compiled backends).
4. Verify `swift build -c release` in mlx-swift itself passes.
5. Update `vmlx-swift-lm/Package.swift` to track the patched branch. At that
   point our weak stubs fall out of the binary automatically.

Phase 0.5 is a mechanical change but requires a standalone round of testing
on the mlx-swift side (we must not regress upstream's existing build
matrix). Expect a half-day of work + CI run.

## Phased plan

Each phase ships with a concrete benchmark on **real** hardware before the next
phase starts. No speculative scope.

### Phase 0 — Transport bring-up and Swift bindings

**Ship:** `Libraries/MLXLMCommon/Distributed/MLXDistributed.swift` wrapping the
mlx-core distributed API via `Cmlx`. Functions:
`init(backend:)`, `rank()`, `size()`, `all_sum`, `all_gather`, `send`, `recv`.

**Measurement harness:** `scripts/measure-transport.sh` — two-Mac handshake,
send a `[N, D]` float16 tensor 100× for several `(N, D)` combos, print median
bandwidth + p50/p99 latency. Run over `bridge0` TB and over Ethernet.

**Shipping criterion:**
- `init` works on a 2-Mac cluster with both Macs on `bridge0`.
- Measured bandwidth on an M4 Max ↔ M4 Max TB4 pair is at least **8 Gbps** for
  1 MiB payloads. (If we see worse than that we have a problem before ML code
  enters the picture.)
- `send` / `recv` round-trip unchanged-bytes on both f16 and bf16.

**Complexity:** small. Mostly Cmlx binding plumbing plus one measurement test.

**Risk:** `jaccl` may not build on the machines we run today without extra
deps. Fallback: force `ring` backend in init and ship that.

### Phase 1 — Pipeline parallel, 2-way, single-request, text-only

**Ship:** a `PipelineEngine` that orchestrates a forward pass across N
stages. For N = 2. Text-only. `KVCacheSimple` only. Single request at a
time (continuous batching is Phase 3).

**Shipped in this phase so far:**
- `Libraries/MLXLMCommon/Distributed/ModelPartition.swift` — partition
  struct + uniform allocator. 8/8 unit tests cover even split,
  remainder distribution, full-coverage property, neighbor helpers,
  first/last flags.
- `Libraries/MLXLMCommon/Distributed/PipelineEngine.swift` —
  orchestrator. Rank 0 drives
  `runPrompt(tokens:parameters:maxNewTokens:stopTokenIDs:)`; non-zero
  ranks drive `runWorker(parameters:)`. Size-envelope + hidden-tensor
  send protocol between stages; int32 `[1]` sampled-token broadcast
  back to rank 0; zero-envelope terminator propagated through every
  middle stage before exit.
- `PipelineStageModel` protocol — the contract a model adapts to so
  the engine can call per-stage forward slices. Four requirements:
  `embedTokens(_:)`, `runLayers(hidden:partition:cache:)`,
  `finalizeLogits(_:)`, `newCache(parameters:)`, plus
  `totalLayerCount: Int`.

**What's NOT yet shipped (tracked for the next iteration):**
- A concrete `PipelineStageModel` conformance on a real model. Qwen 3
  is the first target. Adapting the model requires exposing a
  per-layer-range forward entry point in
  `Libraries/MLXLLM/Models/Qwen3.swift` — currently monolithic via
  `callAsFunction`.
- Byte-identical-logits regression test — needs the Qwen 3-0.6B
  adapter above AND a real-model fixture. Runs a single-device
  reference forward, then the two-process pipeline, asserts argmax
  (and top-k for fp tolerance) match.
- `PipelineRun` executable target — the launcher binary that reads
  `MLX_HOSTS` / rank, loads the model, builds the partition, and
  dispatches to `runPrompt` or `runWorker`.
- Cross-Mac TB4 real-bandwidth numbers. `TransportProbe` (Phase 0)
  is the tool; the data collection happens once a two-Mac rig
  exists.

**Shipping criterion (to fully close Phase 1):**
- End-to-end forward pass produces byte-identical logits vs. a
  single-device run on Qwen 3-0.6B for N=2 stages.
- Measured decode tok/s on a Qwen 3-4B 4-bit 2-Mac TB4 setup.
- Pipeline-engine unit tests + integration tests (mock model) green.

**Complexity:** medium. The per-layer forward exposure + selective
load are the real work; the orchestrator is ~330 lines.

**Risk:** chat-template / tokenizer / reasoning parser live only on
the final stage — aligns with existing BatchEngine which emits
`.chunk` / `.reasoning` / `.toolCall` from the sampling point (already
last-stage only). The Phase 1 MVP is raw-token-out; streaming
integration is Phase 3 when BatchEngine composes with the pipeline.

### Phase 2 — Pipeline parallel, N-way, heterogeneous partitioning

**Ship:** partitioning algorithm that takes
`[(NodeID, availableMemoryBytes, perfScore)]` and returns
`[(NodeID, [startLayer, endLayer))]`. Port exo's cycle + memory-weighted
approach. Include an env-var override for manual static assignments so operators
can pin on known hardware.

**Shipping criterion:**
- 3- or 4-Mac heterogeneous run: one Mac Studio + two MacBook Pros, each
  with a different memory size. The partitioner gives more layers to the
  Studio. End-to-end forward still byte-identical.

**Complexity:** medium.

### Phase 3 — Pipeline + BatchEngine continuous batching

**Ship:** pipeline-aware BatchEngine. The scheduler enqueues prefill chunks at
stage 0; the pipeline carries them forward; the final stage owns slots +
`stepBatchDecode` + emits tokens back to clients. Activations pipeline in a
micro-batch schedule.

**Shipping criterion:**
- Multi-slot throughput on 2-Mac pipeline exceeds single-Mac throughput for
  Qwen 3.5-35B-A3B — which requires the larger model to actually fit only
  when split. (On a 128 GB M4 Max alone, Qwen 3.5-35B fits, so the gain shows
  up in sustained batch throughput, not single-request latency.)

**Complexity:** large. This is the real integration point.

**Risk:** pipeline bubble. We mitigate with micro-batching but each token must
go through every stage. Plan for 1.2-1.5× single-stage latency, not parity.

### Phase 4 — Full cache tiers + JANG + TurboQuant + hybrid SSM on pipeline

**Ship:** stage-aware CacheCoordinator wiring, SSM-aware partitioning (SSM
layers always go to one node, never split), TurboQuant compression on each
stage's local KV.

**Shipping criterion:**
- 2-turn chat on a 2-Mac Qwen 3.6-35B-A3B (hybrid SSM + MoE) pipeline. KV
  cache hit on turn 2 at both stages. TurboQuant enabled, outputs match
  single-Mac TurboQuant reference within tolerance.
- Cache-coordinator KV-sizing contract respected per-stage.

**Complexity:** large. Hybrid SSM on pipeline is the most subtle part of this
phase — SSM layers are path-dependent and the partition must not put an SSM
layer on a pipeline-parallel boundary without preserving the sequence of
updates.

### Phase 5 — Tensor parallelism

**Ship:** sharded `Linear` (column- and row-parallel), sharded `GQA` attention
(split across KV-heads), sharded MoE (experts live per-rank), with
`all_gather` / `sum_scatter` glue. Runs *inside* a stage or across an entire
pipeline level.

**Shipping criterion:**
- Qwen 3-8B 4-bit split 2-way tensor-parallel on a 2-Mac ring. Byte-identical
  logits vs. single-Mac.
- Throughput improvement on TB4 vs. single Mac — tensor parallel *should* help
  latency (unlike pipeline) once the collective cost is hidden by compute
  overlap. Target: within 1.3× single-Mac latency per token.

**Complexity:** very large. Ship after Phase 4 is stable.

**Risk:** sharded JANG / TurboQuant is its own sub-project. SSM tensor-parallel
is unsolved in the literature for these architectures; gate those layers to
pipeline-only.

## Non-goals (for now)

- Dynamic rebalancing when a node joins/leaves mid-request. Phase 2+ can load
  new topologies between requests.
- Cross-OS (Linux) support. Apple Silicon only for everything through Phase 5.
- Replacing `jaccl` or `ring` with a custom transport. We *consume* mlx-core's
  distributed; we don't rewrite it.
- TB5 RDMA user-space DMA. Revisit when macOS or exo ships a path.

## Test matrix (every phase)

A distributed regression needs a real 2-Mac rig. At minimum:

- One Mac Studio M3/M4 Ultra.
- One MacBook Pro M4/M5 Max.
- TB4 cable connecting them.
- Both have `bridge0` enabled.
- `ssh` keys exchanged so CI / bench scripts can run end-to-end.

For each phase ship:
1. A correctness test (byte-identical vs. single-Mac reference).
2. A throughput benchmark (tok/s, latency percentiles).
3. Integration with every existing test suite that passes today — distributed
   paths must not break the single-device paths. Enforce via CI matrix.

## Open questions (resolve before writing Phase 0 code)

1. **Process model.** One process per Mac, launched independently via `ssh`
   by a coordinator script? Or one parent that fans out? exo does the former.
   Recommendation: **per-process-per-Mac, manual launch first**, then add
   a launcher script.
2. **Rank 0 role.** Does rank 0 own the client-facing API (receive submit,
   stream tokens back)? Or does every rank accept client connections and
   route internally? Recommendation: **rank 0 owns the client.** Matches
   single-Mac control-plane.
3. **Discovery.** Phase 0: hardcoded `MLX_HOSTS=ip1,ip2` env var. Phase 2:
   Bonjour service. Phase 4+: libp2p-style if we need dynamic membership.
4. **Cache coordinator cross-stage.** Stage boundaries can cache-hit on
   matching prompt prefixes *per stage*. Do we want a cross-stage coordinator
   that aggregates? Start with per-stage, add aggregation if measurement
   shows benefit.
5. **Tokenizer.** Lives on the final stage only (it decodes stage N's
   sampled tokens). Prompt tokenization happens at submission time on
   rank 0 — the stage 0 receives already-tokenized input from rank 0.

## Next steps (this branch)

1. Confirm this design doc reflects what you want. If yes, merge it into this
   branch as `docs/DISTRIBUTED-DESIGN.md` (or equivalent authoritative
   location) and lock it. No code lands without a doc commit first.
2. Phase 0 week 1: `MLXDistributed.swift` Cmlx bindings + `measure-transport`
   harness. Produce the first real TB4 bandwidth number on your rig.
3. Phase 0 week 2: byte-identical `send`/`recv` round-trip test on every
   dtype (f16, bf16, f32). Log as an integration test.
4. Then Phase 1 begins only after Phase 0 ships a measurement report.
