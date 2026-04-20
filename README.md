# MLX Swift LM

**by [Osaurus](https://osaurus.ai)** | Fork of [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

A Swift package for building applications with large language models (LLMs) and vision language models (VLMs) on Apple Silicon, powered by [MLX Swift](https://github.com/ml-explore/mlx-swift).

This fork adds native [JANG](https://jangq.ai) mixed-precision quantization, **TurboQuant KV cache compression** (4.7-5.0x memory savings), **Gemma 4**, **Mistral Small 4**, speculative decoding, VLM detection, and MoE performance optimizations on top of the full upstream library. Existing apps don't need to change anything -- all upstream APIs are preserved.

## For osaurus integrators (read first)

If you're integrating this fork from [osaurus-ai/osaurus](https://github.com/osaurus-ai/osaurus), three docs in this repo are the authoritative reference:

| Doc | Use it for |
|---|---|
| [`Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md`](Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md) | Per-symbol reference — every `MLXLMCommon` / `MLXLLM` / `MLXVLM` symbol osaurus consumes, with the exact osaurus file + line that calls it. |
| [`Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md`](Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md) | Quick integration notes — addresses `osaurus/docs/INFERENCE_RUNTIME.md` concerns, migration path to drop app-layer tool-call parsing, PR #893 consumer map. |
| [`Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md`](Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md) | Full continuous-batching architecture, every iter-log addendum, real-model verification matrix. |

Dedicated per-topic skill references live under [`skills/mlx-swift-lm/references/`](skills/mlx-swift-lm/references/) — in particular [`tool-calling.md`](skills/mlx-swift-lm/references/tool-calling.md) and [`reasoning-parser.md`](skills/mlx-swift-lm/references/reasoning-parser.md) document the streaming contract osaurus's `StreamAccumulator` + `StreamingDeltaProcessor` rely on.

**`Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift` is byte-identical with [ml-explore/mlx-swift-lm `main`](https://github.com/ml-explore/mlx-swift-lm)** — osaurus can pin to either repo without drift.

## What's New in This Fork

### New Model Architectures

**Gemma 4** -- Google's latest, with both MoE and dense variants:

| Variant | Params | Architecture | VLM |
|---------|--------|-------------|:---:|
| 26B (A4B) | 26B total, 4B active | MoE (128 experts, top-8) | Yes |
| 31B | 31B dense | Mixed sliding/full attention | Yes |

**Mistral Small 4** -- 119B MoE with Multi-head Latent Attention:

| Variant | Params | Architecture | VLM |
|---------|--------|-------------|:---:|
| 119B (A8B) | 119B total, 8B active | MLA + 128 experts + shared expert | Yes (Pixtral) |

### JANG Mixed-Precision Quantization

[JANG](https://jangq.ai) models use per-layer mixed-precision -- attention at 6-8 bit, MLP/experts at 2-4 bit -- for better quality at the same memory. Loaded natively with zero code changes:

```swift
// Loading a JANG model is identical to any other model
let container = try await loadModelContainer(
    from: URL(filePath: "/path/to/Gemma-4-26B-A4B-it-JANG_4M"),
    using: TokenizersLoader()
)
```

### Performance

MoE and hybrid SSM models run **2-4x faster** than both upstream mlx-swift-lm and Python mlx_lm:

| Model | Arch | Upstream Swift | **This Fork** | Python mlx_lm | Gain vs Upstream |
|-------|------|---------------:|------:|------:|-----:|
| Qwen 3.5-35B-A3B | SSM+MoE | 41 tok/s | **103 tok/s** | 94 | **+151%** |
| Gemma 4 26B-A4B | MoE | 27 tok/s | **87 tok/s** | -- | **+222%** |
| Gemma 4 E2B | Dense | 120 tok/s | **121 tok/s** | 128 | -- |
| Gemma 4 E4B | Dense | -- | **73 tok/s** | -- | -- |
| Mistral Small 4 119B | MLA+MoE | 16 tok/s | **70 tok/s** | 45-50 | **+338%** |
| Nemotron Cascade 30B | SSM+MoE | 45 tok/s | **110 tok/s** | 15.5 | **+144%** |
| MiniMax M2.5 172B | MoE (256 exp) | 14 tok/s | **46 tok/s** | 51 | **+229%** |

*All measurements on M4 Max 128GB. Python baselines from M3 Ultra 256GB (~1.5x more memory bandwidth), so Swift matching Python on M4 Max means Swift is actually faster per-bandwidth.*

**Key takeaway:** Dense models (Gemma 4 E2B) were already near-optimal upstream. The massive gains are on **MoE, MLA, and hybrid SSM** models where dozens of scalar operations per layer compound into thousands of unnecessary GPU dispatches. See [Why These Fixes Were Needed](#why-these-fixes-were-needed) below.

## 📊 Multi-Turn Benchmarks vs Other Apple-Silicon Runtimes

Single-turn micro-benchmarks miss the real workload. Below are full
multi-turn conversation benchmarks for the two architectures most representative
of modern open-weight models — a hybrid SSM+MoE (Qwen 3.5-35B-A3B) and a dense
MoE (Gemma 4 26B-A4B) — across **every Apple-Silicon LLM runtime that exists
today** for these models.

| Aspect | What it measures |
|---|---|
| **Decode tok/s** | Sustained generation speed once prefill is done |
| **Prefill tok/s** | How fast the runtime processes new prompt tokens |
| **TTFT** | Time to first token after submitting the prompt |
| **Overall tok/s** | End-to-end per-turn (256 generated / total wall time) — what the user actually feels |

### Qwen 3.5-35B-A3B (hybrid SSM + MoE) — long context, 10.9K tokens

5-turn conversation growing from 21 → 10,932 tokens, greedy decode, 256 tok/turn,
M4 Max 128 GB. **All other inference processes killed before each backend run.**
Full details in [`BENCHMARK-QWEN3.5-35B.md`](BENCHMARK-QWEN3.5-35B.md).

| Backend | Backend type | Decode T1 | Decode T5 | Prefill avg | TTFT T1 | Overall avg (T2-5) |
|---|---|---:|---:|---:|---:|---:|
| Python mlx_lm 0.31.2 | Python + mlx C++ (latest) | 122.1 | 106.1 | **1520** tok/s | 281 ms | **62.4** tok/s |
| **vmlx-swift-lm** (this fork) | Swift + mlx-swift osaurus-0.31.3 | 106 (peak 111)³ | **96.2** | 1335 tok/s | **53 ms** | ~58 tok/s |
| omlx 0.3.2 | Python + mlx C++, paged auto-cache | ~83 | ~57 | (broken)¹ | (broken)¹ | 47.0 tok/s |
| LM Studio 0.4.x | Swift + mlx-llm 1.5.0 | 107.5 | 25.5² | n/a | (broken)¹ | 42.4 tok/s |

¹ Streaming TTFT broken — backends buffer chunks until generation completes.
² LM Studio has no prefix cache; it re-prefills the full conversation every turn.
³ Swift median of 3 runs; peak of best run in parens. Run-to-run variance ±4 tok/s on T1.

**Updated 2026-04-13** — the compile micro-fusion work in commits
`cf55f6d` and `21176a4` lifts long-context (T5) decode from 91.3 → **96.2 tok/s
(+5.4%)** on Qwen 3.5-35B-A3B-4bit. T1 decode is essentially unchanged at the
median (104.7 vs prior 106.4) but peaks higher (111 vs prior 106) and
variance has collapsed — run-to-run spread on T5 went from ~10 tok/s to ~4.

**Headline:**
- **vmlx-swift-lm has the fastest cold-start TTFT (53 ms — 5× faster than Python)** thanks to wired memory + 8 K prefill batch.
- **vmlx-swift-lm is the fastest *Swift-binding* runtime — beats LM Studio by +37 % overall** and beats omlx by +23 % overall.
- Python mlx_lm currently leads on long-context decode by ~10 %. Investigation
  in [`docs/research/2026-04-13-decode-speed-to-120.md`](docs/research/2026-04-13-decode-speed-to-120.md)
  isolates the remaining gap to mlx-swift per-op eager overhead and compile-site
  coverage; further compile work is bounded to ~2 tok/s and closing the gap
  fully requires either mlx-swift bridge changes or custom Metal kernels.

### Gemma 4 26B-A4B (dense MoE) — short context, 5.5K tokens

5-turn conversation growing from 25 → 5,499 tokens, greedy decode, 256 tok/turn.
**This is the short-context test only**; a long-context version is pending more
op fusion work (router rms_norm, geglu) before publishing. Full details in
[`BENCHMARK-GEMMA-4-26B.md`](BENCHMARK-GEMMA-4-26B.md).

| Backend | Decode T1 | Decode T5 | Avg decode (T2-5) |
|---|---:|---:|---:|
| **vmlx-swift-lm** (this fork) | **98.2** | **86.5** | **88.4** |
| Python mlx_lm 0.31.2 | 71.6 | 78.1 | 77.4 |
| omlx 0.3.2 | 77.7 | 68.6 | 71.0 |
| LM Studio | — | — | — (Gemma 4 not yet supported by mlx-llm 1.5.0) |

vmlx-swift-lm wins every turn on the short-context Gemma 4 test (+14 % over
Python). The long-context picture will be republished after the next round of
op fusion lands.

### Llama 3.2 1B Instruct 4bit (pure dense baseline) — long context, 11K tokens

5-turn conversation growing from 47 → 11,022 tokens, greedy decode, 256 tok/turn.
This is the dense-Llama baseline — no MoE routing, no SSM layers, just plain
transformer. Full details in [`BENCHMARK-LLAMA-3.2-1B.md`](BENCHMARK-LLAMA-3.2-1B.md).

| Backend | Decode tok/s avg | Overall tok/s avg | Cold-start TTFT |
|---|---:|---:|---:|
| Python mlx_lm 0.31.2 | **282.5** | **159.9** | 79 ms |
| **vmlx-swift-lm** (this fork) | **271.4** *(-4 %)* | **156.2** *(-2 %)* | **21 ms** *(3.8× faster)* |
| LM Studio 0.4.x (mlx-llm 1.5.0) | n/a (no split) | 155.8 | n/a |
| omlx 0.3.2 | n/a (no split) | 147.2 | (broken) |

On a pure dense model **all four runtimes land within ~8 % of each other** —
the FFI-tax gap that hurts Swift on big MoE models barely matters when there's
no MoE routing or SSM state. vmlx-swift-lm is **within 4 % of Python on decode**
and **3.8× faster than Python on cold-start TTFT** thanks to wired memory and
8 K prefill batch.

Six root-cause fixes that eliminate Metal kernel dispatch overhead. See [Why These Fixes Were Needed](#why-these-fixes-were-needed) for the full explanation of how Python avoids these issues automatically.

1. **Float32 scalar elimination**: `MLXArray(Float)` without `dtype:` defaults to float32. When multiplied with bfloat16 tensors, MLX inserts AsType cast operations -- each a separate Metal kernel dispatch. Fixed across all 18 model files.
2. **Precise softmax**: Replace `softmax(x.asType(.float32))` with `softmax(x, precise: true)` -- computes in float32 internally but returns input dtype. Eliminated 988 AsType ops on Mistral4.
3. **Sigmoid cast removal**: `sigmoid(x.asType(.float32))` is unnecessary -- sigmoid is numerically stable in bfloat16.
4. **Universal bfloat16 conversion**: Convert ALL float16 parameters (including quantization scales/biases) to bfloat16. QuantizedMatmul output dtype follows scales dtype.
5. **Hot-path identity weight dtype**: `MLXArray.ones([n])` in per-token RMSNorm created float32 weights every call. Fixed with `dtype: input.dtype`.
6. **MoE gate zero-out dtype**: `putAlong(..., values: MLXArray(0.0))` needs `dtype: scores.dtype`.

Additional optimizations:
- **Compiled GLU activations**: Fused SwiGLU/GeGLU into single Metal dispatches via `compile(shapeless: true)`.
- **Periodic Metal cache cleanup**: `Memory.clearCache()` every 256 tokens reduces GPU allocator fragmentation.
- **bfloat16 MoE conversion**: Prevents Metal's automatic float16-to-float32 promotion on mixed-dtype operations.
- **Symlink resolution**: Properly follows symlinked model directories (mlxstudio compatibility).

### Continuous Batching

Serve multiple concurrent requests with automatic batching for **2-5x throughput improvement**:

```swift
let engine = await container.makeBatchEngine(maxBatchSize: 8)

// Submit multiple requests concurrently
let stream1 = await engine.generate(input: input1, parameters: params)
let stream2 = await engine.generate(input: input2, parameters: params)

// Each stream delivers tokens independently
for await generation in stream1 {
    print(generation.text, terminator: "")
}
```

- Works with all model families: LLM, VLM, MoE, hybrid SSM, JANG
- Per-request independent parameters (temperature, topP, maxTokens)
- Full KV cache isolation between concurrent sequences
- Request cancellation and graceful shutdown
- SSM/Mamba state merging for hybrid models (Qwen3.5, Nemotron-H)
- 3D RoPE support for Qwen VL models in batch mode

### Multi-Tier KV Cache (Prefix Caching)

Skip redundant prefill computation when the same prompt prefix is seen again. The cache system has three tiers:

- **L1 Paged Cache** -- In-memory, block-based (64 tokens/block), hash-chain prefix matching. Instant KV reuse for shared system prompts.
- **L2 Disk Cache** -- SQLite-indexed safetensors files. Survives process restarts.
- **SSM Companion** -- Stores cumulative SSM state for hybrid models (Qwen3.5, Nemotron-H) alongside KV cache. Deep-copied on fetch to prevent in-place corruption.

```swift
// Enable caching with auto-detection of hybrid models
let container = try await loadModelContainer(from: modelDir, using: TokenizersLoader())
await container.enableCachingAsync()  // auto-detects SSM layers, sets model key

// Or with custom config
let config = CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    diskCacheDir: URL(filePath: "/path/to/cache"),
    pagedBlockSize: 64,
    maxCacheBlocks: 1000
)
container.enableCaching(config: config)

// Generation automatically uses the cache -- no other changes needed.
// First request: full prefill, stores KV to cache.
// Second request with same prefix: restores KV, only prefills new tokens.
```

How it works:
1. On each request, the coordinator hashes prompt tokens into block-sized chunks with SHA-256 chain hashing
2. Matching blocks are found in L1 (instant) or L2 (disk load)
3. KV state from blocks is restored into the model's cache layers
4. `model.prepare()` only processes the remaining (uncached) tokens
5. After generation, prompt KV is extracted and stored for future requests
6. For hybrid SSM models, SSM companion state is stored/restored separately

### Speculative Decoding

Use a smaller draft model to speed up generation by 29-79% (cherry-picked from upstream [ml-explore#173](https://github.com/ml-explore/mlx-swift-lm/pull/173)):

```swift
let mainModel = try await loadModelContainer(
    from: HubClient.default, using: TokenizersLoader(),
    id: "mlx-community/Qwen3-14B-4bit")
let draftModel = try await loadModelContainer(
    from: HubClient.default, using: TokenizersLoader(),
    id: "mlx-community/Qwen3-0.6B-4bit")

let result = try await mainModel.generate(
    input: input, parameters: params, draft: draftModel)
```

### VLM Detection

Check at runtime whether a model supports vision input:

```swift
if await container.isVLM {
    // safe to pass images
}
```

Works from `MLXLMCommon` alone -- no need to import `MLXVLM`.

### TurboQuant KV Cache Compression

Compress the KV cache **4.7-5.0x** during inference with no quality loss on short outputs and minimal divergence on long outputs. Based on Google DeepMind's research ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

**One line to enable, works with every model -- no model changes needed:**

```swift
// 3-bit (recommended default -- best compression)
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 3, valueBits: 3))

// 4-bit (higher quality, less compression)
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 4, valueBits: 4))
```

**Works with ChatSession for multi-turn conversations:**

```swift
let session = ChatSession(
    container,
    generateParameters: GenerateParameters(
        kvMode: .turboQuant(keyBits: 3, valueBits: 3)))

let reply1 = try await session.respond(to: "What is the capital of Japan?")
// "Tokyo"
let reply2 = try await session.respond(to: "What country is that city in?")
// "Japan" -- context preserved across turns
```

**Works with speculative decoding:**

```swift
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 3, valueBits: 3))
let result = try await mainModel.generate(
    input: input, parameters: params, draft: draftModel)
```

#### How It Works

TurboQuant compresses the KV cache after prefill using three techniques:

1. **Randomized Hadamard rotation** -- spreads information uniformly across all dimensions so a single codebook works optimally for every component
2. **Lloyd-Max optimal codebook** -- minimizes quantization error for the statistical distribution of rotated vector components
3. **QJL residual correction** (keys only) -- 1-bit random projection that corrects the exponential error amplification in softmax attention scores

The compressed cache is decoded once into a float16 buffer. During generation, new tokens are scatter-written into a pre-allocated window. Models see normal float16 arrays from `update()` -- they never know compression happened.

#### Memory Savings

| Model | Context | Float Cache | TurboQuant-3 | Savings |
|-------|---------|-------------|-------------|---------|
| Gemma 4 26B MoE | 2K | 84 MB | 17 MB | **4.9x** |
| Qwen 3.5-35B | 32K | 655 MB | 135 MB | **4.9x** |
| Mistral Small 4 (119B) | 2K | 1,208 MB | 244 MB | **4.9x** |

#### Tested Configurations

| Model | Architecture | Format | Modes | Result |
|-------|-------------|--------|-------|--------|
| Gemma 4 26B | MoE (128 experts) | MLX 4-bit | LLM, VLM, multi-turn | Identical on short, near-identical on long |
| Gemma 4 31B | Dense | MLX 4-bit | LLM, multi-turn | Identical on short, near-identical on long |
| Gemma 4 31B | Dense | JANG 4M | LLM | Identical |
| NemotronH 30B-A3B | Hybrid SSM/attention | JANG 4M | LLM, multi-turn | Identical |
| NemotronH 30B-A3B | Hybrid SSM/attention | JANG 2L | LLM | Near-identical |

TurboQuant automatically skips non-KV cache layers (MambaCache for SSM, RotatingKVCache for sliding window). If `maxKVSize` is set (all RotatingKVCache), TurboQuant gracefully does nothing.

---

## Why These Fixes Were Needed

Both Python (`mlx-lm`) and Swift (`mlx-swift-lm`) use the **exact same C++/Metal backend**. Same library (`libmlx`), same Metal shaders, same GPU. The speed gap is 100% about how many kernel dispatches happen per token -- and that comes from the **computation graph**, not the compute.

### The Problem: Swift's Graph Was 2x Larger

We instrumented the MLX computation graph and found:

| Model | Upstream Swift | This Fork | Reduction |
|-------|---------------|-----------|-----------|
| Qwen3.5-35B graph nodes | 5,921 | 4,804 | -19% |
| Qwen3.5-35B AsType ops | **1,176** | **60** | -95% |
| Mistral4 119B AsType ops | **988** | **72** | -93% |
| MiniMax JANG AsType ops | **1,245** | **248** | -80% |
| Nemotron Cascade AsType ops | **562** | **161** | -71% |

Each `AsType` is a separate Metal kernel dispatch (~20µs). At 1,100+ extra dispatches per decode step: **1100 x 20µs = 22ms of pure overhead per token**.

### Root Cause: Python Has Implicit Dtype Inference, Swift Doesn't

Python's `pybind11` bindings **automatically infer scalar dtypes from context**:

```python
# Python: mx.array(0.5) sees the other operand is bfloat16 and matches it
inv_scale = mx.array(1.0 / math.sqrt(head_dim))
q = inv_scale * rms_norm(q)  # both bfloat16 -- no type cast needed

# Python: softmax has a 'precise' flag
scores = mx.softmax(gates, axis=-1, precise=True)  # float32 compute, bfloat16 output
```

Swift has **no implicit dtype inference**. `MLXArray(0.5)` is always `float32`:

```swift
// Swift: MLXArray(0.5) is ALWAYS float32
let invScale = MLXArray(1.0 / sqrt(Float(headDim)))  // float32
let q = invScale * rmsNorm(q, ...)  // bfloat16 * float32 = AsType inserted!

// Swift had no 'precise' option -- devs used explicit .asType(.float32)
let scores = softmax(gates.asType(.float32), axis: -1)  // everything goes float32
```

One hidden cast looks harmless. But a model like Qwen3.5-35B with 64 layers, each doing ~18 operations with untyped scalars: **64 x 18 = 1,152 extra Metal dispatches per token.** That's the entire difference between 41 and 103 tok/s.

### Why Upstream Didn't Catch It

1. **No graph-level profiling.** Models produce correct output. Without counting AsType operations, the symptom is just "slower than Python" with no obvious cause.
2. **Dense models hide the bug.** Llama, Phi, Gemma-E2B have minimal custom math -- few scalar operations, few casts. The bug only becomes catastrophic on MoE routing (dozens of expert-selection ops), GatedDeltaNet (custom normalizations), and hybrid SSM models (per-group RMSNorm).
3. **Python is the reference.** When Swift runs at 41 tok/s and Python at 94 tok/s, the assumption is "Swift overhead." Nobody traced it to 1,100 unnecessary Metal dispatches.
4. **JANG models are unique to us.** Apple's repo doesn't test JANG quantizations, which uniquely expose the scales/biases dtype cascade (QuantizedMatmul output dtype follows scales dtype).

### The Universal Rule (For Contributors)

**Every `MLXArray` scalar or constant created at runtime MUST specify `dtype:`**

```swift
// BAD -- creates float32, triggers AsType cascade
MLXArray(someFloat) * bfloat16Tensor
MLXArray(0.0)
MLXArray.ones([n])  // in a hot path
softmax(x.asType(.float32), ...)

// GOOD -- zero unnecessary casts
MLXArray(someFloat, dtype: tensor.dtype) * bfloat16Tensor
MLXArray(0.0, dtype: tensor.dtype)
MLXArray.ones([n], dtype: tensor.dtype)
softmax(x, axis: -1, precise: true)
```

See `docs/SPEED-FIXES.md` for the complete model-by-model breakdown, and `docs/STRESS-TEST-RESULTS.md` for the 199/199 server integration stress test.

---

## Multi-Tier KV Cache -- Detailed Usage

The cache system eliminates redundant prefill computation. Three tiers work together:

### Architecture

```
CacheCoordinator
 ├── L1: PagedCacheManager     (in-memory, block-aligned, instant)
 ├── L2: DiskCache             (SQLite + safetensors, survives restarts)
 └── SSM: SSMStateCache        (companion for hybrid models, deep-copy)
```

### Basic Usage

```swift
let container = try await loadModelContainer(from: modelDir, using: TokenizersLoader())

// One-line enable with auto-detection
await container.enableCachingAsync()
// ^ auto-detects hybrid models (MambaCache/ArraysCache)
// ^ sets modelKey from model config name
// ^ creates CacheCoordinator with default config

// Generate as normal -- cache is transparent
let session = ChatSession(container)
let reply1 = try await session.respond(to: "Explain quantum computing")
// First request: full prefill, stores KV to L1+L2 cache

let reply2 = try await session.respond(to: "Now explain it simpler")
// Second request: prefix match on shared system prompt + prior context
// Only prefills the new tokens, restores cached KV for the rest
```

### Custom Configuration

```swift
let config = CacheCoordinatorConfig(
    usePagedCache: true,          // Enable L1 in-memory (default: true)
    enableDiskCache: true,         // Enable L2 disk (default: true)
    pagedBlockSize: 64,            // Tokens per block (default: 64)
    maxCacheBlocks: 1000,          // L1 pool size (default: 1000)
    diskCacheMaxGB: 10.0,          // L2 max disk usage (default: 10 GB)
    diskCacheDir: URL(filePath: "/path/to/cache"),  // L2 location (default: tmp)
    ssmMaxEntries: 50,             // SSM companion cache size (default: 50)
    modelKey: "my-model-v1"        // Isolation key (default: auto from model)
)
container.enableCaching(config: config)
```

### With Continuous Batching

```swift
// Cache works with BatchEngine too
let engine = await container.makeBatchEngine(maxBatchSize: 8)
// Each slot independently checks cache on prefill
// Each slot stores its KV after generation completes
```

### With TurboQuant

```swift
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 3, valueBits: 3))

// Cache + TurboQuant work together:
// - Paged cache stores float KV (extracted from TQ via .state)
// - Disk cache stores TQ-native compressed format (26x smaller files)
// - On restore, TQ caches receive float KV and re-compress during generation
await container.enableCachingAsync()
```

### How It Works Under the Hood

1. **Store (after generation):**
   - `extractLayerData()` reads KV from each cache layer (KVCacheSimple, QuantizedKVCache, TurboQuantKVCache)
   - L1: Splits into 64-token blocks, computes SHA-256 chain hashes, stores in block pool
   - L2: Saves as safetensors file keyed by token hash. TQ-compressed layers use TQDiskSerializer (26x smaller)
   - SSM: For hybrid models, deep-copies MambaCache/ArraysCache states into companion cache

2. **Fetch (before prefill):**
   - L1 check: Walks token chunks, computes chain hashes, looks up blocks. O(1) per block.
   - L2 fallback: Tries exact token hash, then one-shorter (for partial matches).
   - On hit: Restores KV into model cache, restores SSM states, truncates input to remaining tokens.
   - Full hit: Feeds only the last token to seed decode (skips ALL prefill).

3. **Isolation:**
   - `modelKey` is included in all hashes -- different models can't read each other's cache
   - `RotatingKVCache` layers are excluded (partial restore would corrupt rotation state)
   - VLM requests with images skip cache (image data isn't in the token hash)
   - Float16 disk entries are auto-cast to bfloat16 on restore

### Cache Performance Impact

| Scenario | Without Cache | With Cache | Savings |
|----------|-------------|-----------|---------|
| 2K system prompt, repeated | ~200ms prefill | ~0ms (full hit) | **200ms** |
| Multi-turn, turn 5 | ~500ms prefill | ~50ms (partial hit) | **450ms** |
| App restart, same prompt | ~200ms prefill | ~20ms (disk load) | **180ms** |

### Monitoring

```swift
// Check cache stats
if let coordinator = container.cacheCoordinator {
    let pagedStats = coordinator.pagedCache?.stats
    print("L1 hits: \(pagedStats?.cacheHits ?? 0)")
    print("L1 misses: \(pagedStats?.cacheMisses ?? 0)")

    let diskHits = coordinator.diskCache?.hits ?? 0
    let diskMisses = coordinator.diskCache?.misses ?? 0
    print("L2 hits: \(diskHits), misses: \(diskMisses)")

    let ssmHits = coordinator.ssmStateCache.hits
    print("SSM hits: \(ssmHits)")
}
```

### Tested & Verified

The full cache stack has been stress-tested in the Osaurus server with **199/199 requests passing (100%)** across:

- Multi-turn session reuse (10 turns)
- Concurrent requests (10 simultaneous)
- Prefix cache hit verification (consistent `prefix_hash`)
- Rapid fire (50 sequential at 3.8 req/s)
- Streaming + non-streaming mixed concurrent
- Edge cases (empty input, single char, extreme temps, 5K char prompts)
- Session switching (5 sessions x 3 turns, rapidly alternating)
- Post-stress health check

Plus 15 unit tests covering cache eviction, SSM deep-copy isolation, batch engine, and concurrent coordinator access. See `docs/STRESS-TEST-RESULTS.md` for full results.

### Disabling Cache

```swift
container.disableCaching()  // Clears all tiers, releases memory
```

---

## Supported Models

### LLMs (50+ architectures)

Llama, Mistral, Phi, Phi-3, Phi-MoE, Gemma, Gemma 2, Gemma 3, Gemma 3n, **Gemma 4**, Qwen2, Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.5-MoE, DeepSeek-V3, Cohere, OpenELM, InternLM2, Starcoder2, MiniCPM, Granite, Granite-MoE-Hybrid, MiMo, MiMo-V2-Flash, MiniMax, GLM-4, GLM-4-MoE, Falcon-H1, Bitnet, SmolLM3, ERNIE 4.5, LFM2, LFM2-MoE, Baichuan-M1, Exaone4, GPT-OSS, Lille-130m, OLMoE, OLMo2, OLMo3, Bailing-MoE, NanoChat, Nemotron-H, AF-MoE, Jamba, **Mistral Small 4** (MLA + MoE), Mistral3, Apertus, and more.

### VLMs (17+ architectures)

PaliGemma, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3.5-MoE, Gemma 3, **Gemma 4**, SmolVLM2, FastVLM, Pixtral, **Mistral Small 4** (MLA + Pixtral), Mistral3, LFM2-VL, GLM-OCR, Idefics3, and more.

### Embedders

Sentence Transformers, BERT, and other popular embedding models.

---

## Quick Start

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", branch: "main"),
```

Then add tokenizer and downloader integrations:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.1.0"),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
        .product(name: "MLXLMHuggingFace", package: "swift-hf-api-mlx"),
    ]),
```

### Chat Session

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let model = try await loadModel(
    from: HubClient.default,
    using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
```

### Loading a Local Model

```swift
import MLXLLM
import MLXLMTokenizers

// Works for any model -- standard MLX, JANG, or unquantized
let container = try await loadModelContainer(
    from: URL(filePath: "/path/to/model"),
    using: TokenizersLoader()
)
```

JANG models are detected automatically. No special flags needed.

### Checking VLM Support

```swift
let container = try await loadModelContainer(from: modelDirectory, using: TokenizersLoader())

if await container.isVLM {
    // Model supports images -- can pass UserInput with .images
} else {
    // Text-only model
}
```

You can also check before loading, using the model type string from `config.json`:

```swift
import MLXVLM

// Synchronous -- no actor isolation needed
if VLMTypeRegistry.supportedModelTypes.contains(modelType) {
    // This model_type is a known VLM architecture
}
```

**VLM-capable families:** Gemma 4, Gemma 3, Qwen 3.5 VL, Qwen 3 VL, Qwen 2.5 VL, Mistral Small 4, Mistral 3, PaliGemma, Pixtral, SmolVLM2, FastVLM, Idefics3, LFM2-VL, GLM-OCR.

### Tokenizer and Downloader Integrations

MLX Swift LM focuses on model implementations. Tokenization and downloading are handled by separate packages:

| Downloader | Adapter |
|-|-|
| [huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface) | [DePasqualeOrg/swift-huggingface-mlx](https://github.com/DePasqualeOrg/swift-huggingface-mlx) |
| [DePasqualeOrg/swift-hf-api](https://github.com/DePasqualeOrg/swift-hf-api) | [DePasqualeOrg/swift-hf-api-mlx](https://github.com/DePasqualeOrg/swift-hf-api-mlx) |

| Tokenizer | Adapter |
|-|-|
| [DePasqualeOrg/swift-tokenizers](https://github.com/DePasqualeOrg/swift-tokenizers) | [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx) |
| [huggingface/swift-transformers](https://github.com/huggingface/swift-transformers) | [DePasqualeOrg/swift-transformers-mlx](https://github.com/DePasqualeOrg/swift-transformers-mlx) |

> **Note:** Adapters are optional. You can set up protocol conformance directly. See the adapter packages for examples.

---

## How JANG Loading Works

1. **Detection** -- Factory checks for `jang_config.json` in the model directory.
2. **Config parsing** -- `JangLoader` reads the JANG profile (bit widths, block size, source model info).
3. **Weight loading** -- Standard `.safetensors` files loaded normally (JANG v2 is MLX-native).
4. **Sanitize** -- Model-specific weight key remapping (VLM prefix stripping, expert key normalization).
5. **Gate dequantization** -- MoE gate weights restored to bfloat16 for routing precision.
6. **Quantization inference** -- Per-layer bit widths inferred from tensor shapes.
7. **Apply** -- Inferred per-layer quantization replaces uniform quantization from `config.json`.

If `jang_config.json` doesn't exist, the standard MLX loading path runs unchanged.

---

## Migrating from Upstream

Change your package URL:

```swift
// Before
.package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),

// After
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", branch: "main"),
```

Everything else stays the same. You gain JANG support, Gemma 4, Mistral Small 4, speculative decoding, `isVLM`, MoE performance boosts (2-4x), continuous batching, multi-tier KV cache, and TurboQuant compression for free.

If migrating from upstream 2.x, see the [version 3 migration guide](#migrating-to-version-3) below.

## Migrating to Version 3

Version 3 decouples tokenizer and downloader implementations.

### New dependencies

```swift
// Before (2.x)
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0"),

// After (3.x)
.package(url: "https://github.com/osaurus-ai/mlx-swift-lm/", branch: "main"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx/", from: "0.1.0"),
```

### New imports

```swift
// Before (2.x)
import MLXLLM

// After (3.x)
import MLXLLM
import MLXLMHuggingFace  // Downloader adapter
import MLXLMTokenizers   // Tokenizer adapter
```

### API changes

- `hub:` parameter is now `from:` (accepts any `Downloader` or local `URL`)
- `HubApi` is now `HubClient`
- `decode(tokens:)` is renamed to `decode(tokenIds:)`

```swift
// Before (2.x)
let container = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")

// After (3.x)
let container = try await loadModelContainer(
    from: HubClient.default,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

---

## Documentation

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Embedding model implementations

---

## Files Changed vs. Upstream

| File | Change | Purpose |
|------|--------|---------|
| `MLXLLM/Models/Gemma4Text.swift` | New | Gemma 4 text (MoE + Dense, dual attention, v_norm, K=V) |
| `MLXVLM/Models/Gemma4.swift` | New | Gemma 4 VLM (vision encoder, 2D RoPE, pooler, processor) |
| `MLXLLM/Models/Mistral4.swift` | New | Mistral Small 4 (MLA attention, 128-expert MoE, YaRN RoPE) |
| `MLXVLM/Models/Mistral4VLM.swift` | New | Mistral Small 4 VLM (MLA text + Pixtral vision) |
| `MLXLMCommon/JangLoader.swift` | New | JANG detection, config, per-layer quant, gate dequant |
| `MLXLMCommon/Load.swift` | Modified | JANG pipeline, VLM key remap, bfloat16 MoE conversion |
| `MLXLMCommon/SwitchLayers.swift` | Modified | Compiled SwiGLU/GeGLU activation kernels |
| `MLXLMCommon/LanguageModel.swift` | Modified | `VisionLanguageModelProtocol` for `isVLM` |
| `MLXLMCommon/ModelFactory.swift` | Modified | `ModelContext.isVLM` |
| `MLXLMCommon/ModelContainer.swift` | Modified | `ModelContainer.isVLM` |
| `MLXLMCommon/Tool/ToolCallFormat.swift` | Modified | Gemma 4, Gemma 3, MiniMax tool call formats |
| `MLXLLM/LLMModelFactory.swift` | Modified | gemma4, mistral4 registrations |
| `MLXVLM/VLMModelFactory.swift` | Modified | gemma4/mistral4 VLM + processor dispatch |
| `MLXLLM/Models/NemotronH.swift` | Modified | JANG key remap for Nemotron MoE |
| `MLXVLM/Models/Qwen35.swift` | Modified | JANG VLM sanitize fix |

## Roadmap

- **Compiled decode** -- Full model `compile(inputs: cache, outputs: cache)` for 10-20% additional decode speedup
- **VLM cache support** -- Image-hash in cache key for vision model prefix reuse
- **TQ compressed disk restore** -- Restore TurboQuant compressed representation directly from disk (currently restores as float, model re-compresses)
- **Native TurboQuant weights** -- Quantization-aware weight format for faster loading

## Known Limitations

- **Raw HuggingFace checkpoints** -- JANG and mlx-community pre-converted models are supported. Raw HF `transformers` checkpoints (with fused `gate_up_proj`) require conversion first.
- **Audio** -- Gemma 4 supports audio natively, but the audio encoder is not yet implemented.
- **Gemma 4 2B/4B** -- Per-layer input gating and KV sharing for smaller variants not yet implemented.
- **Speculative decoding + RotatingKVCache** -- Speculative decoding requires trimmable caches. Not compatible after cache wraps.

## License

MIT License. See [LICENSE](LICENSE) for details.

Based on [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) by Apple's ML Explore team.

## Acknowledgments

- [Apple ML Explore](https://github.com/ml-explore) for MLX and mlx-swift-lm
- [JANG](https://jangq.ai) mixed-precision quantization format
- [Google DeepMind](https://deepmind.google) for the Gemma 4 architecture
- [Mistral AI](https://mistral.ai) for the Mistral Small 4 architecture
