# Performance Checkpoint — 2026-04-04

## Current Speeds (Mac Studio M2 Ultra, 192GB)

| Model | Architecture | Format | Speed | Python vmlx | Gap |
|-------|-------------|--------|-------|-------------|-----|
| **Gemma4 26B MoE** | 128 experts | MLX 4-bit | **100 tok/s** | 89 tok/s | **EXCEEDED** |
| **Qwen3.5-35B MoE** | 256 experts + GDN | JANG 4K | **61 tok/s** | 74 tok/s | 17% |
| NemotronH 30B-A3B | Hybrid SSM/MoE | JANG 4M | **48 tok/s** | 143 tok/s | 67% |
| Gemma4 31B Dense | Dense | MLX 4-bit | 24 tok/s | — | bandwidth-bound |
| Gemma4 26B VLM | MoE | MLX 4-bit | 49 tok/s | — | vision overhead |

## What Fixed What

### Gemma4 26B: 25 to 100 tok/s (+300%)

| Fix | Impact | Commit |
|-----|--------|--------|
| bfloat16 MoE conversion | 25 to 34 (+38%) | pre-existing |
| Compiled SwiGLU/GeGLU | +5% | pre-existing |
| Memory.clearCache() every 256 tokens | 37 to 49 (+32%) | 91c3537 |
| mlx_set_wired_limit via Cmlx | baseline stability | d6e6d1a |
| **Remove 86 .asType() ops from forward** | **49 to 100 (+104%)** | 3dbec19 |
| Compiled logit_softcap (3 to 1 Metal op) | included above | 3dbec19 |

### Qwen3.5-35B: 42 to 61 tok/s (+45%)

| Fix | Impact | Commit |
|-----|--------|--------|
| bfloat16 MoE + compiled GLU | 42 to 59 (+40%) | pre-existing |
| Memory.clearCache() + wired limit | +3% | d6e6d1a |
| Symlink resolution (unblocked loading) | N/A to 61 | 1584724 |
| Compiled sigmoidMultiply + shared expert gate | ~0% | 2278e91 |

### NemotronH 30B: 25 to 48 tok/s (+92%)

| Fix | Impact | Commit |
|-----|--------|--------|
| bfloat16 MoE + compiled GLU | +40% | pre-existing |
| Memory.clearCache() + wired limit | +32% | d6e6d1a |

## What Doesn't Work (Proved By Benchmarking)

| Approach | Result | Why |
|----------|--------|-----|
| Flat decode loop (bypass TokenIterator) | 0% | GPU execution dominates, not Swift overhead |
| Pre-computed cache state arrays | 0% | Same |
| Double-buffered async pipeline | 0% | Same |
| Stream.withNewDefaultStream per loop | -24% | New Metal command queue per call |
| Stream.withDefaultStream per step | -6% | @TaskLocal allocation overhead per scope |
| Stream.setDefault per step | -10% | Stream boundary sync between step/item |
| stream.runWith (C-level) + restore for item | -10% | Per-token set/restore overhead > overlap gain |
| **Full stream-split TokenIterator** | **-11%** | **54 vs 61 tok/s on Qwen3.5-35B (confirmed 2026-04-04)** |
| asyncEval with cache state arrays | 0% | GPU pipeline already full from token-only asyncEval |
| evalLock removal | 0% | Lock uncontended in single-thread generation |
| compile with cache state tracking | -35% | State tracking overhead exceeds compile benefit |
| compile(shapeless:true) on model forward | CRASH | KV cache mutation breaks tracer |

## Root Cause of Remaining 17% Gap (Qwen3.5: 61 vs 74)

Stream-split approach DISPROVED — confirmed -11% overhead (54 vs 61 tok/s).
The gap is NOT caused by stream scheduling. Likely causes:

1. **Custom Metal GatedDelta kernels** — osa-jang has 4 vectorized Metal
   kernel variants for Qwen3.5's SSM layers. Our GatedDeltaNet uses
   standard MLX ops. Half of Qwen3.5's 64 layers are GatedDeltaNet.

2. **Auto-fused gate/up MoE projections** — osa-jang combines gate_proj +
   up_proj into gate_up_proj during weight loading (1 matmul instead of 2).

3. **Sorted expert indices in gatherMM** — osa-jang sorts expert indices
   for memory locality before gather operations (when indices.size >= 64).

4. **Python model code differences** — vmlx's Python model implementations
   may have fewer MLX ops per forward pass than our Swift implementations.

## osaurus-ai/mlx-swift Fork (osaurus-0.31.3) — NOT USED

Available at github.com/osaurus-ai/mlx-swift branch osaurus-0.31.3.
Contains Stream.setDefault, Stream.restoreDefault, Stream.runWith,
mlx_stream_run_with C API. Stream-split confirmed unhelpful; fork NOT
used by main branch (upstream ml-explore/mlx-swift 0.31.3).

## Files Modified (from upstream mlx-swift-lm baseline)

    Evaluate.swift       — asyncEval cache states, clearCache, wired limit
    Load.swift           — symlink resolution for mlxstudio model dirs
    LLMModel.swift       — clearCache after prefill chunks
    Gemma4Text.swift     — remove asType, compiled softcap (+300%)
    Qwen35.swift         — compiled sigmoid gate
    Qwen3Next.swift      — compiled sigmoidMultiply, remove asType
    Package.swift        — upstream ml-explore/mlx-swift 0.31.3
