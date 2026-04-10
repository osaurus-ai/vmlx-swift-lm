# Performance Research: Swift MLX Inference

Synthesized from `SWIFT-PERF-FIXES.md`, `DECODE-ANALYSIS.md`, and `FRAMEWORK-FIXES.md`.
Read those files for full detail. This document summarizes the findings and maps each
performance gap to its root cause, existing mitigations, and remaining work.

---

## 1. FFI Bridge Architecture

Every MLX operation in Swift crosses the Swift-to-C-to-C++ boundary. This costs
~1-2us per call due to `std::shared_ptr` atomic refcounting, micro-heap allocations
for `mlx_array` handles, and ARC synchronization when wrapping returned C handles.

The bridge is structurally required. Swift cannot import MLX's C++20 code directly
(advanced templates, `std::variant`, `constexpr if`). Apple's MLX team assessed
and rejected direct Swift-C++ interop for MLX.

**When it matters:**
- Large GPU kernels (Gemma4 gather_mm, large projections): kernel runs 10-100us,
  the 1-2us bridge cost is invisible. Swift wins here (no GIL, deterministic memory).
- Tiny GPU kernels (GatedDeltaNet sigmoid/Hadamard, MoE mx.take): kernel runs ~1us,
  bridge cost equals or exceeds kernel cost, GPU starves waiting for next command buffer.

Python avoids this via lazy graph construction. pybind11 builds a DAG of pointers
fast; only the explicit `evaluate` dispatches to Metal. The overhead amortizes across
the entire graph. Swift's C-wrapper path disrupts lazy graph building more aggressively.

**Key insight:** the FFI bridge cannot be optimized, only bypassed. The path forward is
reducing the NUMBER of crossings via `compile()`, op fusion, and custom Metal kernels.

See `DECODE-ANALYSIS.md` for a per-token breakdown of every overhead source.

---

## 2. Metal Kernel API (MLXFast.metalKernel)

Swift has full custom Metal kernel support through `MLXFast.metalKernel`, backed by
the same C++ runtime as Python's `mx.fast.metal_kernel`. The API:

```swift
let kernel = MLXFast.metalKernel(
    name: "unique_kernel_name",       // MUST be globally unique (Issue #3263)
    inputNames: ["q", "k", "v"],       // kernel argument names
    outputNames: ["y", "state_out"],   // output names
    source: "/* Metal C++ source */"
)

let outputs = kernel(
    inputs,
    template: [("InT", inputType), ("Dk", Dk)],  // compile-time substitutions
    grid: (32, Dv, B * Hv),                       // thread grid
    threadGroup: (32, 4, 1),                       // threadgroup size
    outputShapes: [[B, T, Hv, Dv], state.shape],   // output tensor shapes
    outputDTypes: [inputType, inputType]            // output dtypes
)
```

Pattern: wrap kernel creation in a `Sendable` singleton manager so compilation happens
once. See `GatedDelta.swift` for the canonical example (`GatedDeltaKernelManager`).

### Existing Kernel Inventory

| Kernel | File | Purpose | Notes |
|--------|------|---------|-------|
| GatedDelta | `GatedDelta.swift:108` | Linear recurrence (SSM) | 2 variants: masked (prefill) and unmasked (decode). Template params: InT, Dk, Dv, Hk, Hv |
| Bitnet | `Bitnet.swift:56` | 1.58-bit quantized linear | Replaces standard matmul for BitNet biformer models |
| SSM | `SSM.swift:62` | Structured state space model | Selective scan with discretization |
| Interpolation | `InterpolationUtils.swift:139,186` | Multi-mode interpolation | 2 kernels: nearest and bilinear modes for VLM image preprocessing |

---

## 3. Overflow Bin Pattern

The standard `KVCacheSimple.update()` returns `keys[..<offset]`, a dynamically sized
slice that changes shape every decode step. MLX's `compile()` tracer bakes the offset
as a constant on first trace, producing wrong results on subsequent steps.

`shapeless=True` has a bug: reductions (sum, mean) on dynamic-shaped outputs replay
stale cached results instead of re-executing. Not usable for KV cache compilation.

The Overflow Bin pattern solves this:

```
Standard:     keys[..<offset]       -> shape changes every step -> compile breaks
Overflow Bin: keys[0..<maxLength]   -> shape constant           -> compile works
              + mask[j <= offset]   -> marks valid positions
```

Pre-allocate a fixed `[B, H, maxLength, D]` buffer of zeros. Write new keys/values via
`DynamicSliceUpdate` (compile-traceable because start position is MLXArray, not Int).
Return the FULL buffer from `update()`, shape is constant. A boolean attention mask
marks which positions are valid.

The implementation lives in `CompilableKVCache.swift` (see `IMPLEMENTATION-NOTES.md` for
architecture details). The tradeoff is marginal redundant compute over masked-zero
positions in exchange for enabling `compile()` to fuse hundreds of FFI crossings into
a single compiled call.

Test results from `feature/overflow-bin-compile` branch: Gemma4 E2B with Overflow Bin
but no compile produced **identical output** over 39 tokens with a 5.7% speedup. With
compile enabled, it **crashes** because Gemma4's KV sharing layer reads `cache.offset`
as Int inside the forward pass.

---

## 4. compile() Blockers

### 4.1 Universal Blocker: applyRotaryPosition Int Readback

`applyRotaryPosition` in `RoPEApplication.swift` reads `cache?.offset ?? 0` as Int.
This is called by ALL models with attention (Qwen3.5, Gemma4, Llama, etc.), not just
Gemma4. Inside a compiled closure, `.item(Int.self)` triggers synchronous GPU readback,
which crashes compile() or produces wrong results.

The fix: detect `CompilableKVCache` and use its `offsetArray` (MLXArray) instead of
the Int `offset` property. This unblocks compile() for every attention model at once.

### 4.2 M1/M2 Metal JIT Crash

`compile(shapeless: true)` crashes on M1/M2 macOS Tahoe due to an Apple Metal JIT bug.
The existing compiled closures in `SwitchLayers.swift` and `GatedDelta.swift` use
`shapeless: true` and will crash on M1/M2. Solution: gate compile() on M3+ at runtime,
log a message when auto-disabled, fall back to uncompiled path.

### 4.3 KV Cache Shape Changes

Without Overflow Bin, the KV cache grows by 1 token each decode step. Without
`shapeless: true`, compile() retraces every step (overhead exceeds benefit). With
`shapeless: true`, some ops crash (e.g. `argPartition` in MoE routing).

### 4.4 shapeless=True Bugs

Reductions on dynamic-shaped outputs replay stale cached results. Not reliable for
KV cache paths that involve sum/mean on variable-length slices.

---

## 5. Platform Compatibility

| Platform | compile() | Custom Metal Kernels | Notes |
|----------|-----------|---------------------|-------|
| M1/M2 | CRASH (shapeless) | Works | Gate compile() on M3+. Metal JIT bug in macOS Tahoe |
| M3+ | Works | Works | Full support |
| Intel Mac | N/A | N/A | No Metal, MLX requires Apple Silicon |

---

## 6. Performance Gap Summary

| Gap | Root Cause | Impact | Mitigation | Status |
|-----|-----------|--------|------------|--------|
| Existential dispatch | `any LanguageModel` vtable lookup | ~0.5-0.7ms/token (3-5%) | Genericize `TokenIterator<M>` | Planned |
| Generation stream | No dedicated Metal command queue | ~1-2ms/token (7-10%) | Requires upstream MLX C++ change | Deferred |
| Full model compile | KV cache shape changes | ~0.5-1ms/token (3-5%) | Overflow Bin + RoPE fix | In progress |
| FFI crossings | Swift-C-C++ bridge overhead | Architecture-dependent | compile() fusion + Metal kernels | In progress |
| O(n^2) detokenizer | Re-tokenizes entire history | Blocks async pipeline | Fix StreamingDetokenizer | Not started |
| MoE routing | Granular mx.take loops | Model-dependent | Block-diagonal gather_mm | Not started |

All model-level compile fusions are exhausted (compiled softcap, sigmoid gate,
computeG, sigmoid multiply, step ops). Further gains require framework-level changes.

See `FRAMEWORK-FIXES.md` for detailed fix recommendations and `DECODE-ANALYSIS.md`
for the per-token overhead breakdown.
