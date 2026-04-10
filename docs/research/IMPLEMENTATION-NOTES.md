# Implementation Notes: compile() Integration and Custom Kernels

Practical notes for the engineering work. See `PERFORMANCE-RESEARCH.md` for the
research context and `SWIFT-PERF-FIXES.md` for benchmark numbers.

---

## 1. applyRotaryPosition: The Universal compile() Blocker

### The Problem

`RoPEApplication.swift` line 26:

```swift
return rope(x, offset: cache?.offset ?? 0)
```

`cache?.offset` returns `Int`. This triggers `.item(Int.self)` on the underlying
MLXArray, which is a synchronous GPU readback. Inside a compiled closure this
crashes or produces wrong results because the compile tracer cannot track an Int
value through the computation graph.

### Why It's Universal

`applyRotaryPosition` is called by every attention layer in every model (Qwen3.5,
Gemma4, Llama, Mistral, Phi, etc.). The blocker is not model-specific. Fixing it in
`RoPEApplication.swift` unblocks compile() for all attention models at once.

### The Fix

Detect `CompilableKVCache` and use its `offsetArray` (MLXArray [1] int32) directly.
The `offsetArray` is tracked by the compile tracer as part of the graph.

```swift
public func applyRotaryPosition<R: RoPELayer>(_ rope: R, to x: MLXArray, cache: KVCache?)
    -> MLXArray
{
    if let compilable = cache as? CompilableKVCache {
        return rope(x, offset: compilable.offsetArray)  // ArrayOffsetLayer -- compile-traceable
    }
    return rope(x, offset: cache?.offset ?? 0)  // OffsetLayer -- standard Int path
}
```

### Constraints

- Do NOT add `.item()` calls in the MLXArray code path. That defeats the purpose.
- Do NOT modify model implementations. The fix goes in `RoPEApplication.swift` only.
- Standard `KVCache` (non-compilable) must continue using the Int path unchanged.
- The `RoPELayer` protocol needs both `OffsetLayer` and `ArrayOffsetLayer` conformance
  for this to work. Verify the protocol supports `MLXArray` offsets.

---

## 2. CompilableKVCache Architecture

### Overview

`CompilableKVCache` is a drop-in replacement for `KVCacheSimple` that returns fixed-size
buffers, enabling `compile()` to trace through the entire decode step.

### Key Components

| Component | Type | Purpose |
|-----------|------|---------|
| `keys`, `values` | `MLXArray?` | Pre-allocated `[B, H, maxLength, D]` zero buffers |
| `offsetArray` | `MLXArray` | Position tracker as `[Int32]` (compile-traceable) |
| `maxLength` | `Int` | Buffer capacity (default 4096) |
| `step` | `Int` | Pre-allocation chunk size (default 256) |
| `maskRinds` | `MLXArray` | Pre-computed column indices `[0..maxLength)` |

### Data Flow

```
update(newKeys, newValues):
  1. Lazy-init buffers on first call (zeros [B, H, maxLength, D])
  2. Write via dynamicSliceUpdate(self.keys, newKeys, start=offsetArray, axes=[2])
  3. Write via dynamicSliceUpdate(self.values, newValues, start=offsetArray, axes=[2])
  4. offsetArray += nTokens
  5. Return (self.keys, self.values)  // FULL buffer, constant shape
```

### Conversion from KVCacheSimple

After prefill completes, convert standard caches to compilable ones:

```swift
let compilableCache = standardCache.map { c in
    CompilableKVCache(from: c, maxLength: 2048)
}
```

The `init(from:)` convenience initializer copies existing cache state into the
fixed-size buffer at position 0 and sets `offsetArray` to the current sequence length.

### What NOT to Convert

- `RotatingKVCache`: has special rotation/temporal ordering semantics, must stay as-is
- `MambaCache`: uses Int offset from BaseKVCache; GatedDeltaNet doesn't read it, but
  `compactMap` in `innerState()` may return variable-size arrays that break compile()

### Current Status

Code-complete on `feature/overflow-bin-compile` branch. Not yet wired into
`Evaluate.swift` on main. The wiring involves:
1. `GenerateParameters.enableCompiledDecode: Bool = false`
2. `GenerateParameters.compiledMaxCacheLength: Int? = nil`
3. After prefill: `KVCacheSimple` -> `CompilableKVCache` conversion
4. `compile(inputs: caches, outputs: caches) { model forward }` wrapping decode

### Memory Overhead

Overflow Bin pre-allocates `[B, H, maxLength, D]` buffers. For a 35B model at maxLength=4096
this is significant. Validate memory usage at target context lengths (2K, 8K, 32K)
before enabling by default.

---

## 3. DynamicSlice API and Gotchas

### API

```swift
// Write at a dynamic position (compile-traceable)
dynamicSliceUpdate(_ src: MLXArray, update: MLXArray, start: MLXArray, axes: [Int32]) -> MLXArray

// Read at a dynamic position (compile-traceable)
dynamicSlice(_ src: MLXArray, start: MLXArray, axes: [Int32], sliceSize: [Int32]) -> MLXArray
```

Both use `mlx_slice_dynamic` / `mlx_slice_update_dynamic` from Cmlx. The `start`
parameter is MLXArray (not Int), so the compile tracer can track it.

### Gotchas

1. **`start` must be 1D array, not scalar.** Use `MLXArray([Int32(3)])`, not `MLXArray(Int32(3))`.
   Scalar start crashes at the C API boundary.

2. **`sliceSize` must be static.** `DynamicSlice` only supports `std::vector<int>` for
   slice sizes, not MLXArray. This is why the Overflow Bin returns the full buffer
   (constant sliceSize = maxLength) rather than a dynamic slice.

3. **`dynamicSliceUpdate` returns a new array.** It does not mutate in place. Always
   assign the return value: `self.keys = dynamicSliceUpdate(self.keys!, ...)`

4. **Cmlx import required.** These functions call C-level APIs through `import Cmlx`.

### File Location

`Libraries/MLXLMCommon/DynamicSlice.swift` (71 lines). Code-complete on
`feature/overflow-bin-compile` branch.

---

## 4. Issue #3263: Unique Kernel Names

MLX's Metal kernel registry uses the kernel `name` string as a key. If two call sites
register kernels with the same name, the second silently overwrites the first or
causes undefined behavior.

**Rule:** Every `MLXFast.metalKernel(name:)` call must use a globally unique name.

Existing names in the codebase:
- `"gated_delta_step"` and `"gated_delta_step_mask"` (GatedDelta.swift)
- `"bitnet_linear"` (Bitnet.swift)
- `"ssm_selective_scan"` (SSM.swift)
- Two interpolation kernels (InterpolationUtils.swift)

When writing new kernels, include the operation name and variant to avoid collisions.
Example: `"moe_dispatch_block"` not just `"dispatch"`.

---

## 5. Speculative Decoding: Do NOT Compile

Speculative decoding uses `trim()` on the KV cache, which calls `.item(Int.self)` to
determine how many tokens to remove. This synchronous readback inside a compiled
closure will crash.

The compile() integration must explicitly skip speculative decoding paths. The
`SpeculativeTokenIterator` should never wrap its forward pass in `compile()`.

---

## 6. Reference: Existing Compiled Closures

These already work on M3+ and prove the `compile()` + `shapeless: true` pattern:

| Closure | Location | Ops Fused | Impact |
|---------|----------|-----------|--------|
| `_compiledComputeG` | GatedDelta.swift:17 | exp+exp+softplus+mul+neg -> 1 | +11% Qwen3.5 decode |
| `_compiledStepOps` | GatedDelta.swift:205 | ~10 SSM ops -> 1 | Prefill only |
| `compiledLogitSoftcap` | SwitchLayers.swift | tanh+div+mul -> 1 | Minor |
| `compiledSigmoidGate` | SwitchLayers.swift | sigmoid -> 1 | Minor |
| `_compiledSigmoidMultiply` | GatedDelta.swift (implied) | sigmoid+mul -> 1 | Minor |
