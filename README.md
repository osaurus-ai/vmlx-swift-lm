# MLX Swift LM

**by [Osaurus](https://osaurus.ai)** | Fork of [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

A Swift package for building applications with large language models (LLMs) and vision language models (VLMs) on Apple Silicon, powered by [MLX Swift](https://github.com/ml-explore/mlx-swift).

This fork adds native [JANG](https://jangq.ai) mixed-precision quantization, **TurboQuant KV cache compression** (4.7-5.0x memory savings), **Gemma 4**, **Mistral Small 4**, speculative decoding, VLM detection, and MoE performance optimizations on top of the full upstream library. Existing apps don't need to change anything -- all upstream APIs are preserved.

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

MoE and hybrid SSM models run **2-4x faster** than both upstream mlx-swift-lm and Python mlx_lm, matching or exceeding Python on every model family tested:

| Model | Before | This Fork | Python mlx_lm | Gain |
|-------|-------:|------:|------:|-----:|
| Qwen 3.5-35B-A3B (MoE) | 41 tok/s | **103 tok/s** | 94 | **+151%** |
| Gemma 4 26B (MoE) | 27 tok/s | **87 tok/s** | -- | **+222%** |
| Mistral Small 4 119B (MLA+MoE) | 16 tok/s | **70 tok/s** | 45-50 | **+338%** |
| Nemotron Cascade 30B (SSM+MoE) | 45 tok/s | **110 tok/s** | 130 | **+144%** |
| MiniMax M2.5 (256 experts) | 14 tok/s | **46 tok/s** | 51 | **+229%** |
| Gemma 4 E2B (dense) | 120 tok/s | **121 tok/s** | 128 | -- |

*Measured on M4 Max 128GB. Python baselines from M3 Ultra 256GB (1.5x more bandwidth).*

Six root-cause fixes that eliminate Metal kernel dispatch overhead:

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

Everything else stays the same. You gain JANG support, Gemma 4, Mistral Small 4, speculative decoding, `isVLM`, and MoE performance boosts for free.

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

- **Native TurboQuant** -- Quantization-aware weight format for faster loading
- **Paged KV Cache** -- Memory-efficient caching for long contexts
- **Prefix Caching** -- Reuse KV cache across prompts with shared prefixes
- **Async L2 Disk Cache** -- Spill KV cache to disk for very long contexts

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
