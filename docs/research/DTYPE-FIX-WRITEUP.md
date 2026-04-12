# Float32 Type Promotion Fix — Complete Writeup

**Date:** 2026-04-11
**Branch:** fix/swift-compile-performance
**Impact:** 2.5x speedup on Qwen3.5-35B (41 → 101 tok/s)

---

## The Problem

Swift mlx-swift-lm was 2.5x slower than Python mlx_lm on Qwen3.5-35B-A3B-4bit:
- Python: 98 tok/s (sync), 123 tok/s (pipelined)
- Swift: 41 tok/s (sync), 47 tok/s (pipelined)
- Same hardware (M4 Max 128GB), same MLX C++ 0.31.1, same Metal kernels

## Root Cause

**Swift graph: 5,921 nodes per decode step. Python graph: 2,693 nodes.**

The extra 3,228 nodes were dominated by **1,176 AsType (type cast) operations** — Python had ZERO.

### Why It Happened

When Swift code creates an `MLXArray` from a scalar without specifying `dtype:`, it defaults to float32:

```swift
// BEFORE (broken): creates float32 scalar
let qNormed = MLXArray(pow(invScale, 2)) * rmsNorm(q, ...)
```

When this float32 scalar multiplies a bfloat16 tensor, MLX's type promotion rules force an AsType cast. This cascade propagates through the entire GatedDeltaNet computation:

1. q and k become float32 (upcast)
2. The recurrent state loop runs in float32
3. Output projection downcasts back to bfloat16 for weight matching
4. Each cast is a separate Metal kernel dispatch

With ~24 GatedDeltaNet layers × ~18 casts per layer = 1,176 extra Metal dispatches.

### Why Python Doesn't Have This Problem

Python's scalar arithmetic dynamically infers the scalar's dtype to match the array:

```python
# Python: scalar adopts array's dtype automatically
q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
```

## The Fix

Explicitly specify `dtype:` when creating scalar MLXArrays:

```swift
// AFTER (fixed): scalar inherits tensor dtype
let qNormed = MLXArray(pow(invScale, 2), dtype: q.dtype) * rmsNorm(q, ...)
```

### Files Changed

| File | Lines | What |
|---|---|---|
| Libraries/MLXLLM/Models/Qwen35.swift | 274, 277 | invScale scalar dtype for GDN q/k norms |
| Libraries/MLXLLM/Models/Qwen3Next.swift | 295, 298 | Same fix for Qwen3Next variant |

## Results

### Qwen3.5-35B-A3B-4bit (the target model)

| Metric | Before | After | Delta |
|---|---|---|---|
| Graph nodes | 5,921 | 4,804 | -19% |
| AsType ops | 1,176 | 60 | -95% |
| Uncompiled tok/s | 41 | **99** | **+141%** |
| Compiled tok/s | 44 | **114** | **+159%** |
| Pipelined tok/s | 47 | **134** | **+185%** |

### Other Models Tested

| Model | Nodes | AsType | tok/s | Status |
|---|---|---|---|---|
| Llama 3.2 1B | 980 | 0 | 396-472 | Clean |
| Gemma4 E2B 4-bit | 2,877 | 89 | 153-194 | Has AsType — needs audit |
| Gemma4 26B 4-bit | 4,969 | 32 | 26-28 | REGRESSION from 97-102 — investigate |
| Qwen3.5 35B JANG_2S | 5,929 | 1,105 | 59-73 | JANG weights not converted — needs fix |

## Remaining Work

### Models with AsType issues (need same fix pattern)
1. **Gemma4 E2B** — 89 AsType. Audit Gemma4Text.swift for float32 scalars.
2. **Gemma4 26B** — 32 AsType + regression from 97 to 27 tok/s. Likely different root cause (RotatingKVCache? chat template?).
3. **Qwen3.5 JANG** — 1,105 AsType. The JANG weight loader may store some params as float16 instead of bfloat16, causing the dtype fix to not apply.
4. **All MoE models** — BailingMoe, DeepseekV3, MiniMax, etc. need same audit.
5. **VLM models** — Gemma4 VLM, Qwen35 VLM, etc.

### The fix pattern (apply everywhere)

```swift
// WRONG: creates float32 scalar, causes AsType cascade
MLXArray(someFloat) * bfloat16_tensor

// RIGHT: scalar inherits tensor's dtype
MLXArray(someFloat, dtype: tensor.dtype) * bfloat16_tensor
```

### Framework-level fix (long-term)

The root cause is in mlx-swift's `ScalarOrArray` protocol. When Swift does `array * Float`, the `toArrays` function correctly promotes the scalar to array's dtype. But when code manually creates `MLXArray(Float)`, there's no array context to infer dtype from. A framework-level solution could:
1. Add a `MLXArray.scalar(_:like:)` convenience that creates a scalar matching another array's dtype
2. Lint rule to flag `MLXArray(Float)` without `dtype:` in model code
