# Compiled Decode Benchmarks

Guide for measuring the performance impact of compiled decode on three target models.

## Build Requirements

Compiled decode requires Metal shaders to be compiled by Xcode's Metal compiler. `swift build` alone cannot do this. Use `xcodebuild`:

```bash
xcodebuild -scheme mlx-swift-lm -destination 'platform=macOS' build
```

If you get linker errors about missing Metal symbols, make sure the Metal framework is linked in the Xcode project settings.

---

## How to Enable Compiled Decode

Set `enableCompiledDecode: true` in `GenerateParameters`:

```swift
// Compiled decode ON
let params = GenerateParameters(
    enableCompiledDecode: true,
    compiledMaxCacheLength: 4096  // optional, defaults to 4096
)

// Compiled decode OFF (baseline)
let params = GenerateParameters(
    enableCompiledDecode: false
)
```

The parameter lives in `Libraries/MLXLMCommon/Evaluate.swift`.

### What happens internally

1. After the prefill step, every `KVCacheSimple` layer is converted to a `CompilableKVCache` (fixed-size buffer + boolean mask).
2. The model forward pass is wrapped in `compile(inputs: caches, outputs: caches)`, fusing hundreds of Swift-to-C-to-C++ FFI crossings into a single compiled Metal dispatch.
3. If the Metal JIT compiler fails at runtime, the system falls back to uncompiled execution automatically.

### Hardware requirements

Compiled decode only works on **M3 and later**. The `HardwareInfo.isCompiledDecodeSupported` gate auto-disables compilation on M1/M2 chips, where `compile(shapeless: true)` crashes due to an Apple Metal JIT bug in macOS Tahoe.

### What gets compiled

- `KVCacheSimple` layers are converted and compiled. This covers most attention-based transformer blocks.
- `RotatingKVCache` (sliding window) and `MambaCache` (SSM) layers are left unchanged. They continue running uncompiled.
- The speculative decoding path is NOT compiled. If you pass a draft model, compiled decode has no effect.

---

## Baseline Numbers

All numbers measured on **M4 Max 128GB**, greedy decode, from `SWIFT-PERF-FIXES.md`:

| Model | Architecture | Baseline (tok/s) | Python mlx_lm (tok/s) |
|-------|-------------|:---:|:---:|
| Qwen3.5-35B-A3B (4-bit) | MoE (128 experts) | 61 | 71 |
| Gemma4-26B (MLX 4-bit) | MoE (128 experts, top-8) | 97-102 | 80 |
| Qwen3.5-4B (4-bit) | Dense | 145 | ~123 |

For comparison, the VMLX engine (Python, compiled) gets 118 tok/s on Qwen3.5-35B. Closing that gap is the primary goal.

---

## Benchmark Procedure

Run each model twice per configuration: once with compiled decode off (baseline), once with compiled decode on. Use the same prompt for both runs and measure the average tokens/second over at least 256 generated tokens.

### Qwen3.5-35B MoE

```bash
# Baseline
mlx-swift-lm generate --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Explain the difference between TCP and UDP in detail." \
  --max-tokens 512

# Compiled
mlx-swift-lm generate --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Explain the difference between TCP and UDP in detail." \
  --max-tokens 512 \
  --enable-compiled-decode
```

**Target:** >= 71 tok/s (Python parity)

### Gemma4-26B MoE

```bash
# Baseline
mlx-swift-lm generate --model mlx-community/Gemma-4-26B-A4B-4bit \
  --prompt "Write a short essay about renewable energy." \
  --max-tokens 512

# Compiled
mlx-swift-lm generate --model mlx-community/Gemma-4-26B-A4B-4bit \
  --prompt "Write a short essay about renewable energy." \
  --max-tokens 512 \
  --enable-compiled-decode
```

**Target:** >= 97 tok/s (no regression)

### Qwen3.5-4B Dense

```bash
# Baseline
mlx-swift-lm generate --model mlx-community/Qwen3.5-4B-4bit \
  --prompt "What are three benefits of regular exercise?" \
  --max-tokens 512

# Compiled
mlx-swift-lm generate --model mlx-community/Qwen3.5-4B-4bit \
  --prompt "What are three benefits of regular exercise?" \
  --max-tokens 512 \
  --enable-compiled-decode
```

**Target:** >= 145 tok/s (no regression)

---

## Correctness Verification

Speed gains mean nothing if the output is wrong. For each model, verify that compiled decode produces the same tokens as the baseline.

### Steps

1. Run the same prompt with `enableCompiledDecode: false` (baseline). Record the full token ID sequence.
2. Run the same prompt with `enableCompiledDecode: true`. Record the full token ID sequence.
3. Compare the two sequences token by token.

```swift
// Pseudocode for verification
let baselineTokens = try await generate(
    input: prompt, parameters: GenerateParameters(enableCompiledDecode: false)
)
let compiledTokens = try await generate(
    input: prompt, parameters: GenerateParameters(enableCompiledDecode: true)
)
assert(baselineTokens == compiledTokens, "Token mismatch between baseline and compiled")
```

Minor divergence after many tokens is expected due to floating-point differences between fused and unfused operations. The first 50-100 tokens should match exactly.

---

## Results Template

Fill in your measured numbers here after running the benchmarks above.

| Model | Baseline (tok/s) | Compiled (tok/s) | Change | Target Met? |
|-------|:---:|:---:|:---:|:---:|
| Qwen3.5-35B MoE | 61 | | | >= 71 |
| Gemma4-26B MoE | 97-102 | | | >= 97 |
| Qwen3.5-4B Dense | 145 | | | >= 145 |

### Correctness

| Model | Tokens Match? | Notes |
|-------|:---:|-------|
| Qwen3.5-35B MoE | | |
| Gemma4-26B MoE | | |
| Qwen3.5-4B Dense | | |

---

## Reference

- `docs/research/PERFORMANCE-RESEARCH.md` -- full analysis of FFI overhead, compile blockers, and the Overflow Bin pattern
- `SWIFT-PERF-FIXES.md` -- original benchmark table and optimization history
- `Libraries/MLXLMCommon/Evaluate.swift` -- `GenerateParameters` definition and compiled decode integration
- `Libraries/MLXLMCommon/CompilableKVCache.swift` -- fixed-size KV cache implementation
- `Libraries/MLXLMCommon/HardwareInfo.swift` -- M3+ detection gate
