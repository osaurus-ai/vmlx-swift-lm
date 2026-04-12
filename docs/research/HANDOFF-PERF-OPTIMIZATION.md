# Handoff: mlx-swift-lm Performance Optimization

## Session Date: 2026-04-10

## What We Were Working On

Closing the **64% performance gap** between mlx-swift-lm and Inferencer.app/vmlx Python on MoE models (Qwen3.5 35B, MiniMax MoE).

## The Critical Finding

**Inferencer.app achieves 100 tok/s on Qwen3.5 35B MoE. mlx-swift-lm achieves 61 tok/s.**

**Root cause**: vmlx Python compiles the **entire model forward pass** into ONE Metal dispatch. mlx-swift-lm only compiles individual activation functions inside SwitchGLU.

### Evidence

vmlx Python (single line):
```python
self._compiled_forward = mx.compile(self.model.__call__)
```

mlx-swift-lm MoE forward (Qwen35SparseMoeBlock.callAsFunction):
```swift
var gates = gate(x)                        // FFI #1
gates = MLX.softmax(gates, ...)           // FFI #2
let inds = MLX.argPartition(...)           // FFI #3
var scores = MLX.takeAlong(...)            // FFI #4
scores = scores / scores.sum(...)          // FFI #5
let y = switchMLP(x, inds)                 // FFI #6
let combined = (y * scores[...]).sum()    // FFI #7
let gatedSharedY = compiledSigmoidGate(...) // FFI #8
return combined + gatedSharedY
```

Each FFI crossing costs ~1-2μs. When GPU kernels are tiny (softmax, argPartition, takeAlong all run in ~1μs), the bridge overhead equals the kernel time. CPU becomes bottleneck, GPU starves.

### Same Problem in MiniMax MoE

```swift
let gates = gate(x.asType(.float32))    // FFI #1
var scores = sigmoid(gates)               // FFI #2
let inds = argPartition(...)              // FFI #3
scores = takeAlong(...)                   // FFI #4
scores = scores / scores.sum(...)         // FFI #5
scores = scores.asType(x.dtype)           // FFI #6
let y = switchMLP(x, inds)              // FFI #7
return (y * scores[...]).sum(axis: -2)   // FFI #8
```

---

## What We Reverse-Engineered from Inferencer.app

**Inferencer.app** (`/Applications/LLM/Inferencer.app`, v1.10.5, MLX Swift 0.13.0)

### Key Optimizations

1. **Fragment Preloading** — Pre-JITs Metal shader fragments before first inference, eliminating ~500ms-2s JIT compilation delay. Not used in mlx-swift-lm.

2. **Wired GPU Memory** (`mlx_set_wired_limit`) — Prevents model weights from being paged to SSD between tokens. mlx-swift-lm tried auto-applying this and it crashed on small GPUs. Inferencer requires it to be set explicitly via `sudo sysctl iogpu.wired_limit_mb=<size>`.

3. **Full Model compile()** — The same approach as vmlx Python. Compiles entire model forward including MoE routing.

4. **280+ MoE model types** — All major MoE architectures supported with optimized dispatch.

### Steel Attention (macOS 26.2+)

Inferencer's binary contains `get_steel_attention_kernel` and `get_steel_attention_nax_kernel`.

**NAX path** (macOS 26.2+, M3+ gen >= 17): Uses `MetalPerformancePrimitives` tensor_ops — Apple's new MPU API.

**Standard path**: Works on all Apple Silicon.

Swift's `MLX.scaledDotProductAttention()` auto-dispatches to the right kernel. Not directly controllable.

---

## Steel Attention ≠ Tahoe Crash Fix

**The macOS Tahoe Metal JIT crash** (MLX issues #3329, #3201, #3256) is from `compile(shapeless: true)` returning zero results on M1/M2 GPUs. It's completely unrelated to steel attention.

Steel attention uses `MetalPerformancePrimitives` — different code path entirely.

The Tahoe fix is already partially in `HardwareInfo.swift`: `isCompiledDecodeSupported = false` on M1/M2. The remaining work is removing `compile(shapeless: true)` from activation functions.

---

## What Exists in the Codebase

### CompilableKVCache (feature/overflow-bin-compile branch)
- Fixed-size KV cache using Overflow Bin pattern
- Pre-allocates `[B, H, maxLength, D]` buffer
- Returns FULL buffer from `update()` — enables compile() to trace through
- Uses `DynamicSliceUpdate` for compile-traceable writes
- `offsetArray` as MLXArray (tracked by compile tracer)

### DynamicSlice.swift
- C API wrappers for `mlx_slice_dynamic` and `mlx_slice_update_dynamic`
- Enables compile-traceable slice operations

### RoPEApplication.swift
- MLXArray offset path for CompilableKVCache
- `rope(x, offset: compilable.offsetArray)` when using Overflow Bin

### Evaluate.swift — setupCompiledDecode()
- `enableCompiledDecode` flag in GenerateParameters
- Converts KVCacheSimple → CompilableKVCache after prefill
- Compiles forward with `compile(inputs: cache, outputs: cache)`
- **BLOCKED**: Requires ALL layers to use CompilableKVCache. Qwen3.5 has heterogeneous caches (MambaCache + KVCacheSimple).

### SwitchLayers.swift — compiledSwiGLU, compiledGeGLU
- Already compiling activation functions inside SwitchGLU
- These work correctly — they're the reference pattern

### HardwareInfo.swift
- `isCompiledDecodeSupported` — false on M1/M2 (Tahoe crash hardware)
- Machine identifier parsing for GPU generation detection

---

## The Gap: Full Model compile() vs Partial

**vmlx Python / Inferencer**: `mx.compile(model.__call__)` — entire forward compiled, all ops fused into one Metal dispatch

**mlx-swift-lm**: `compiledSwiGLU`, `compiledGeGLU` — only activation functions inside SwitchGLU are compiled

**Result**: vmlx collapses 7+ ops into 1 Metal dispatch. mlx-swift-lm has 7 separate FFI crossings per token.

---

## The Fix: Extend setupCompiledDecode for Heterogeneous Caches

### Current Problem

`setupCompiledDecode()` in Evaluate.swift requires ALL caches to be CompilableKVCache:

```swift
let hasCompilableLayers = cacheRef.contains { $0 is CompilableKVCache }
guard hasCompilableLayers else { return }
```

But Qwen3.5 has heterogeneous caches:
- `Qwen35GatedDeltaNet` layers use `MambaCache`
- Standard attention layers use `KVCacheSimple`

### Fix

**Don't require all layers to be compilable.** Only compile what can be compiled. Leave MambaCache uncompiled.

```swift
// Pseudocode for the fix:
mutating func setupCompiledDecode(maxCacheLength: Int) throws {
    guard HardwareInfo.isCompiledDecodeSupported else { return }
    
    eval(cache)
    
    // Convert only KVCacheSimple to CompilableKVCache
    for i in 0..<cache.count {
        if cache[i] is KVCacheSimple {
            cache[i] = CompilableKVCache(from: cache[i], maxLength: maxCacheLength)
        }
        // MambaCache and RotatingKVCache stay as-is
    }
    
    // Try compile with mixed cache types
    let capturedModel = model
    let cacheRef = cache
    
    self.compiledForward = compile(
        inputs: cacheRef, outputs: cacheRef
    ) { (args: [MLXArray]) -> [MLXArray] in
        let result = capturedModel(
            LMInput.Text(tokens: args[0])[text: .newAxis],
            cache: cacheRef.isEmpty ? nil : cacheRef,
            state: nil)
        return [result.logits]
    }
}
```

### Why This Should Work

Qwen3.5's GatedDeltaNet (`Qwen35GatedDeltaNet`) does NOT read `cache.offset` as Int inside its forward. It only uses `cache` for state access. This means it's compatible with compile().

The current `setupCompiledDecode` is overly conservative — it fails if ANY cache can't be converted. We need to make it only convert what's possible and compile what's convertible.

---

## Implementation Plan (see IMPLEMENTATION-PLAN.md)

Priority order:
1. **Fix heterogeneous cache handling in setupCompiledDecode** (P0 — enables full compile)
2. **Add warmup pass** (P1 — eliminates first-token JIT delay)
3. **Document wired memory setup** (P2 — opt-in performance tip)
4. **Compile MoE routing in MiniMax** (P2 — same pattern as Qwen3.5 fix)

---

## Files to Change

| File | What to Change |
|------|----------------|
| `Evaluate.swift` | Fix `setupCompiledDecode` to handle heterogeneous caches. Add warmup support. |
| `GenerateParameters.swift` | Add `enableWarmup: Bool` parameter |
| `MiniMax.swift` | Test compile compatibility, add compiled routing wrapper if needed |
| `Qwen35.swift` | Verify compile compatibility (should work after Evaluate.swift fix) |
| `README.md` | Document wired memory setup |
| `docs/research/IMPLEMENTATION-PLAN.md` | Full implementation guide (already written) |

---

## Files NOT to Change (Constraint)

Per your explicit instructions:
- Do NOT modify model implementations (GatedDelta.swift, Gemma4Text.swift, SwitchLayers.swift, etc.)
- Do NOT touch VLM code paths
- Do NOT add dependencies
- No `.item()` calls inside compiled closures
- No `compile()` on speculative decoding path

---

## Git State

```
research/custom-metal-kernels (current branch)
├── SWIFT-PERF-FIXES.md          (comprehensive research log)
├── docs/research/IMPLEMENTATION-PLAN.md  (implementation guide)
├── Libraries/MLXLMCommon/HardwareInfo.swift  (GPU detection)
├── Libraries/MLXLMCommon/CompilableKVCache.swift  (overflow bin, untracked)
├── Libraries/MLXLMCommon/DynamicSlice.swift  (C API wrappers, untracked)
├── Libraries/MLXLMCommon/Evaluate.swift  (+56 lines uncommitted)
├── Libraries/MLXLMCommon/KVCache.swift  (+1 line uncommitted)
├── Libraries/MLXLMCommon/RoPEApplication.swift  (committed 5b22be5)
└── Tests/MLXLMTests/EvalTests.swift  (benchmark test, uncommitted)
```

---

## Next Steps

1. **Fix setupCompiledDecode** — make it handle heterogeneous caches (MambaCache + KVCacheSimple mixed)
2. **Test on Qwen3.5 35B** — enable `enableCompiledDecode = true` and benchmark
3. **If it works, extend to MiniMax MoE**
4. **Add warmup pass** — force JIT compilation before first token
5. **Document wired memory** — explain how to set `iogpu.wired_limit_mb`

---

## Key Learnings

1. **The gap is about compile SCOPE, not individual ops.** vmlx compiles the entire model forward. mlx-swift-lm compiles individual activation functions. The bridge overhead multiplies with each separate op.

2. **Heterogeneous caches are the blocker.** Qwen3.5 mixes MambaCache and KVCacheSimple. The fix isn't to make all caches the same — it's to compile what can be compiled and handle the rest.

3. **FFI bridge ~1-2μs per op.** When GPU kernels are tiny (~1μs), bridge overhead = kernel time. CPU becomes bottleneck.

4. **Swift already beats Python on large-kernel models** (Gemma4 +28%). The issue is only on fragmented-op models (MoE, GatedDeltaNet).

5. **Don't auto-apply wired memory.** It crashed on small GPUs. Make it opt-in with documentation.

---

*Session handoff — comprehensive research complete, implementation plan written*
