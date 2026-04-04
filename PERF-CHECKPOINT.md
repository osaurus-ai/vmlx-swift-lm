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
| evalLock removal | 0% | Lock uncontended in single-thread generation |
| compile with cache state tracking | -35% | State tracking overhead exceeds compile benefit |
| compile(shapeless:true) on model forward | CRASH | KV cache mutation breaks tracer |

## Root Cause of Remaining 17% Gap (Qwen3.5: 61 vs 74)

Python runs model forward on a DEDICATED stream and item() on the DEFAULT stream:

    with mx.stream(generation_stream):     # gen stream
        logits = model(y, cache=cache)
        mx.async_eval(next_y)
    yield y.item()                         # default stream -- DIFFERENT thread

Swift runs everything on the SAME stream:

    let token = step(previous)             # default stream
    asyncEval(token)                       # default stream
    return previousY.tokens.item(Int.self) # default stream -- BLOCKS

item() on the same stream as asyncEval blocks until ALL pending work completes.
Python's item() on a different stream reads the PREVIOUS result without blocking.

## Stream-Split TokenIterator (IMPLEMENTED)

Matches Python's `generation_stream = mx.new_stream(mx.default_device())` pattern:

    // TokenIterator creates a dedicated generation stream once:
    let generationStream = MLX.Stream(Device.defaultDevice())

    // next() splits model forward (gen stream) from item() (default stream):
    mutating public func next() -> Int? {
        let previousY = y

        MLX.Stream.setDefault(generationStream)   // gen stream
        let token = step(previous: previousY)
        y = .init(tokens: token)

        MLX.Stream.restoreDefault()                // default stream
        asyncEval(token)                           // submit N+1

        return previousY.tokens.item(Int.self)     // read N (overlap!)
    }

    // prepare() also runs prefill on gen stream:
    MLX.Stream.setDefault(generationStream)
    model.prepare(input, cache: cache, windowSize: windowSize)
    // ... step + asyncEval ...
    MLX.Stream.restoreDefault()

No changes to generateLoopTask or public API. `for token in iterator` works
as before — the stream split is transparent inside next().

## osaurus-ai/mlx-swift Fork (osaurus-0.31.3) — NOW ACTIVE

Package.swift now depends on github.com/osaurus-ai/mlx-swift branch osaurus-0.31.3.
Contains Stream.setDefault, Stream.restoreDefault, Stream.runWith,
mlx_stream_run_with C API. Used by TokenIterator for stream-split.

## Files Modified (from upstream mlx-swift-lm baseline)

    Evaluate.swift       — stream-split TokenIterator, clearCache, wired limit
    Load.swift           — symlink resolution for mlxstudio model dirs
    LLMModel.swift       — clearCache after prefill chunks
    Gemma4Text.swift     — remove asType, compiled softcap (+300%)
    Qwen35.swift         — compiled sigmoid gate
    Qwen3Next.swift      — compiled sigmoidMultiply, remove asType
    Package.swift        — osaurus-ai/mlx-swift fork (osaurus-0.31.3)
