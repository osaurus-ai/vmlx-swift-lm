# Swift MLX Inference Performance: Complete Research Log

## Current Production Numbers (M4 Max 128GB, greedy decode)

| Model | Python mlx_lm | Inferencer.app | Swift mlx-swift-lm | Gap (Swift vs Inferencer) | Notes |
|-------|:---:|:---:|:---:|:---:|:---:|
| Gemma4 26B MLX-4bit | 80 tok/s | ~100 tok/s | 97-102 tok/s | **-0%** | Monolithic GPU kernels, Swift competitive |
| Qwen3.5 35B-A3B 4-bit | 71 tok/s | **100 tok/s** | 61 tok/s | **-39%** | GatedDeltaNet fragmented ops |
| MiniMax-M2.5 MoE | Superior | ~100 tok/s | ~10 tok/s | **-90%** | Granular expert routing |
| Gemma4 E2B 4-bit | -- | -- | 65-144 tok/s | -- | New model, PLE+KV sharing |
| Gemma4 E4B 4-bit | -- | -- | 64-86 tok/s | -- | New model, PLE+KV sharing |

**Key finding**: Inferencer.app achieves **100 tok/s** on Qwen3.5 35B MoE — same model, same hardware, 64% faster than mlx-swift-lm.

---

## Root Cause Analysis (Gemini Deep Research + our investigation)

### The Critical Gap Found: Inferencer.app vs mlx-swift-lm

**Inferencer.app achieves 100 tok/s on Qwen3.5 35B MoE. mlx-swift-lm achieves 61 tok/s. 64% gap on identical model.**

**Root cause**: Inferencer/vmlx Python **compiles the entire MoE routing** into ONE Metal dispatch. mlx-swift-lm only compiles individual activation functions inside SwitchGLU.

**Qwen35SparseMoeBlock — our current approach (7+ separate FFI crossings per token)**:
```swift
var gates = gate(x)
gates = MLX.softmax(gates, ...)           // FFI #1
let inds = MLX.argPartition(...)           // FFI #2
var scores = MLX.takeAlong(...)            // FFI #3
scores = scores / scores.sum(...)           // FFI #4
let y = switchMLP(x, inds)                // FFI #5 (activation fused internally)
let combined = (y * scores[...]).sum()     // FFI #6
let gatedSharedY = compiledSigmoidGate(...) // FFI #7 (compiled sigmoid*)
return combined + gatedSharedY
```

**vmlx Python / Inferencer approach (1 FFI crossing for entire routing)**:
```python
# vmlx_engine/model_runner.py:
mx.compile(self.model.__call__)  # Entire forward including routing fused
```

**Result**: vmlx/Inferencer collapse 7+ separate GPU kernel dispatches into 1. mlx-swift-lm pays 7x the bridge overhead.

**MiniMax MoE has the same problem**:
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

**Same for Qwen3.5 GatedDeltaNet**: The `gatedDeltaUpdate` state update loop has many tiny ops (sigmoid, multiply, add) all as separate FFI crossings.

### Why Swift beats Python on some models

The speed gap is **model-dependent and architecture-dependent**, not a general Swift deficiency.

**Swift-to-C-to-C++ FFI bridge overhead**: ~1-2us per MLX operation due to:
1. `std::shared_ptr` atomic ref counting at the C API boundary
2. Micro-heap allocations for `mlx_array` handles
3. ARC synchronization when Swift wraps returned C handles

**When GPU kernels are large** (Gemma4: gather_mm, large projections): GPU executes for tens-hundreds of us per op. The 1-2us bridge cost is invisible. Swift wins because of no GIL, deterministic memory management.

**When GPU kernels are tiny** (Qwen3.5 MoE routing, MiniMax expert dispatch): Each Metal kernel executes in ~1us. Bridge cost equals kernel cost, CPU becomes bottleneck, GPU starves waiting for next command buffer.

### Why Python doesn't have this problem (as badly)

Python's pybind11 bridge also has overhead, but it's decoupled from GPU execution via lazy graph construction. Python rapidly builds a DAG of pointers; only the explicit evaluate call dispatches to Metal. The pybind11 cost is amortized across the entire graph. Swift's deeper C-wrapper path disrupts lazy graph building more aggressively.

### Direct Swift-to-C++ interop is NOT viable

Swift's C++ interoperability can't handle MLX's C++20 code (advanced templates, std::variant, constexpr if). The MLX team assessed this and rejected it. No short-term fix exists.

---

## All Optimizations Applied (committed to main)

### Model-Level Compile Fusions
| Optimization | Impact | Commit |
|---|---|---|
| `.asType()` removal in Gemma4 router/layerScalar | +300% Gemma4 | early commits |
| compiled logit_softcap (tanh+div+mul fused) | minor | committed |
| compiled computeGatedDeltaG | +11% Qwen3.5 (55-to-61) | 6bf2cf1 |
| compiled gatedDeltaStepOps | prefill only, neutral decode | b186468 |
| compiled sigmoidGate for MoE shared expert | minor | committed |
| compiled sigmoidMultiply for GatedDeltaNet output | minor | committed |
| VLM ports (Gemma4.swift, Qwen35.swift in MLXVLM) | mirrors LLM | 6dfeef9 |

### Framework-Level Optimizations
| Optimization | Impact | Commit |
|---|---|---|
| processor nil fast-path in convertToToken | skip chain when nil | 5a44ba1 |
| Memory.clearCache every 256 tokens | prevents OOM | committed |
| needsCacheQuantization guard | prevents re-quant per step | committed |

### Removed (0% benefit or harmful)
| What | Why Removed | Commit |
|---|---|---|
| Generation stream (dedicated MLX.Stream) | 0% impact, no concurrent Metal load | removed |
| asyncEval-of-cache-states | 0% benefit, adds per-token overhead | 5de7d15 |
| Auto wired memory (mlx_set_wired_limit) | **CRASH** on small GPU working set | 847a8c7 |
| Auto cache limit (Memory.cacheLimit = physMem/4) | Match upstream, no auto-guessing | 847a8c7 |

**All model-level compile fusions are exhausted.** Further gains require framework-level changes.

---

## Inferencer.app Deep Reverse-Engineering (v1.10.5, macOS 26 SDK)

**Inferencer.app** (`/Applications/LLM/Inferencer.app`) is Apple's reference Swift MLX inference app. Despite being built with an OLDER MLX Swift version (0.13.0 vs mlx-swift-lm's 0.31.x), it achieves 100 tok/s on Qwen3.5 35B MoE.

### Architecture
Inferencer is a **client-server app**, NOT pure local inference. Uses `xnet` (distributed compute) to connect to inference servers. Runs its own HTTP + NIO server for distributed compute. For local single-request inference, its speed advantage comes from compilation and memory management, not the server architecture.

### Key Optimizations Discovered

| Optimization | What It Does | Impact for Local Inference |
|---|---|---|
| `fragmentPreloadedLibraries` | Pre-JIT compiles Metal shader fragments before first inference | Eliminates JIT compilation delay on first token |
| `mlx_set_wired_limit` | Wires GPU memory to prevent paging to SSD | Prevents weight eviction between tokens |
| `CachedModelWrapper` | Persistent KV cache state across requests | Faster cold-start on subsequent requests |
| Continuous batching | Batches multiple concurrent requests | Server-only, no local benefit |

### Fragment Preloading — Critical for First-Token Latency
```swift
// Inferencer uses this to preload shader fragments:
MetalLibrary fragmentPreloadedLibraries
setFragmentPreloadedLibraries(...)
// Pre-JITs all Metal kernels before model load completes
```

This eliminates the ~500ms-2s JIT compilation delay on first inference. mlx-swift-lm doesn't use this.

### mlx_set_wired_limit — Wired GPU Memory
```swift
// Same C API mlx-swift-lm tries to use:
sudo sysctl iogpu.wired_limit_mb=<size>
// Prevents macOS from paging model weights to SSD during inference
```

**Important**: This crashed on small GPU machines when mlx-swift-lm auto-applied it. Inferencer requires user/admin to set it explicitly. It's an opt-in tuning knob, not automatic.

### MoE Patterns in Inferencer (280+ model types)
All MoE variants fully supported:
- `MiniMaxSparseMoeBlock`
- `DeepseekV3MoE` with `block_sparse_moe` pattern
- `BailingMoeSparseMoeBlock`
- `Qwen35SparseMoeBlock`
- `Qwen3MoESparseMoeBlock`

### MLX Primitives in Inferencer Binary
All core MLX ops compiled in: AddMM, QuantizedMatmul, GatherMM, GatherQMM, QQMatmul, Softmax (multiple dtypes), RMSNorm, LayerNorm, RoPE, ScaledDotProductAttention, DynamicSlice, DynamicSliceUpdate, Convolution.

### Steel Attention Kernels (macOS 26.2+ Metal Features)
Inferencer's binary contains `get_steel_attention_kernel` and `get_steel_attention_nax_kernel` — the same macOS 26 Metal kernels in MLX main.

**Standard steel_attention**: Works on all Apple Silicon GPUs
**NAX steel_attention** (macOS 26.2+): Uses `MetalPerformancePrimitives` tensor_ops — Apple's new MPU API for high-performance attention on M3+ (gen >= 17).

```cpp
// is_nax_available() check in MLX C++:
if (__builtin_available(macOS 26.2, ...)) {
    can_use_nax = true;
}
can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
// NAX requires GPU gen >= 17 (or gen >= 18 for Powderder)
```

**Swift public API for attention**: `MLX.scaledDotProductAttention()` — auto-dispatches to steel_attention/NAX internally. Not directly controllable.

### Key Insight: Why Inferencer is Fast for Local Inference
1. **Fragment preloading**: No JIT delay on first token
2. **Wired memory**: No weight paging between tokens
3. **Full model compile** (inferred): The compiled forward approach (same as vmlx Python)

NOT from: client-server architecture (server workload), continuous batching (multiple concurrent requests).

---

## macOS 26 / Xcode 26.2 Metal SDK Findings

### Steel Attention in MLX C++ (`mlx/backend/metal/kernels.h`)
```cpp
MTL::ComputePipelineState* get_steel_attention_kernel(
    metal::Device& d, const std::string& kernel_name,
    const std::string& hash_name, const metal::MTLFCList& func_consts,
    const array& q, int bq, int bk, int bd, int wm, int wn, const array& m);

MTL::ComputePipelineState* get_steel_attention_nax_kernel(
    metal::Device& d, const std::string& kernel_name,
    const std::string& hash_name, const metal::MTLFCList& func_consts,
    const array& q, int bq, int bk, int bd, int wm, int wn, const array& m);
```

### Steel Attention NOT Exposed in Swift Public API
The `steel_attention` and `steel_attention_nax` are **internal kernel names** used by `scaled_dot_product_attention.cpp`. They're dispatched automatically by the C++ backend when conditions are met. Swift's `MLX.scaledDotProductAttention()` uses them internally but there's no Swift API to select NAX vs standard path.

### NAX Kernel Selection Logic (C++ scaled_dot_product_attention.cpp):
```cpp
if (metal::is_nax_available() && q.shape(3) != 80 &&
    (env::enable_tf32() || q.dtype() != float32)) {
    return sdpa_full_self_attention_nax(...);  // MPU path (macOS 26.2+)
}
// else: standard steel_attention via get_steel_attention_kernel(...)
```

### MLX Swift Fast API Surface — Gaps vs Python

**Deprecated**: `MLXFast` module — functions moved to main `MLX` module. Use `MLX.rmsNorm()`, `MLX.RoPE()`, etc. directly.

| Operation | Swift | C++ | Gap |
|-----------|-------|-----|-----|
| RMSNorm | ✅ `MLX.rmsNorm()` | ✅ `mx.fast.rms_norm` | **Parity** |
| LayerNorm | ✅ `MLX.layerNorm()` | ✅ `mx.fast.layer_norm` | **Parity** |
| RoPE | ✅ `MLX.RoPE()` | ✅ `mx.fast.rope` | **Parity** |
| SDPA | ✅ `MLX.scaledDotProductAttention()` | ✅ `mx.fast.scaled_dot_product_attention` | **Parity** |
| GELU (fast) | ✅ `MLX.geluFastApproximate()` | ✅ `mx.fast.gelu_fast_approx` | **Parity** |
| SiLU | ✅ `MLX.silu()` | ✅ `mx.silu` | **Parity** |
| **Custom Metal Kernel** | ❌ NOT exposed | ✅ `mx.fast.metal_kernel` | **GAP** |
| **Custom CUDA Kernel** | ❌ NOT exposed | ✅ `mx.fast.cuda_kernel` | **GAP** |
| **Precompiled CUDA** | ❌ NOT exposed | ✅ `mx.fast.precompiled_cuda_kernel` | **GAP** |

### RMSNorm Nil Weight Gap
Python: `mx.fast.rms_norm(x, None, eps)` — allows nil/None weight
Swift: `MLXFast.rmsNorm(weight:)` — weight parameter REQUIRED

Models work around this:
- **NanoChat.swift**: Manual `functionalRMSNorm` using `mean(x.square(), axis: -1, keepDims: true)`
- **Gemma4Text.swift**: Manual `rmsNormNoScale` — `MLXFast.rmsNorm` doesn't support nil weight

The C++ `rms_norm` function accepts `const array& weight = array()` (optional). Swift binding requires explicit weight.

---

## Overflow Bin Pattern (feature/overflow-bin-compile)

### The Core Problem: compile() can't trace dynamic KV cache reads

Standard `KVCacheSimple.update()` returns `keys[..<offset]` -- a dynamically sized slice. The offset changes every decode step. MLX's compile() tracer bakes the offset as a constant on first trace, producing wrong results on subsequent steps.

`shapeless=True` has a bug: reductions (sum, mean) on dynamic-shaped outputs replay stale cached results instead of re-executing.

### The Solution: Fixed-size buffers + boolean masking

Instead of returning `keys[..<offset]` (dynamic size), return the ENTIRE pre-allocated buffer (static size). Use a boolean attention mask to mark which positions are valid.

```
Standard:     keys[..<offset]       -> shape changes every step -> compile breaks
Overflow Bin: keys[0..<maxLength]   -> shape constant           -> compile works
              + mask[j <= offset]   -> marks valid positions
```

### Implementation (on feature/overflow-bin-compile branch)

**CompilableKVCache.swift** -- Fixed-size KV cache with Overflow Bin pattern:
- Pre-allocates `[B, H, maxLength, D]` buffer (zeros)
- `update()`: writes via DynamicSliceUpdate (compile-traceable), returns FULL buffer
- `makeMask()`: generates boolean mask using `offsetArray` (MLXArray, not Int)
  - Decode (n=1): `mask[0, j] = (j <= offset)` -- allows valid + new position
  - Prefill (n>1): `mask[i, j] = (j <= offset + i)` -- causal within full buffer
- `offsetArray`: MLXArray [1] int32 -- compile tracer sees offset flow through graph
- Window size support: `mask & (rinds >= linds - windowSize + 1)`

**DynamicSlice.swift** -- C API wrappers for compile-traceable slice operations:
- `dynamicSliceUpdate(src, update, start, axes)` -- write at MLXArray position
- `dynamicSlice(src, start, axes, sliceSize)` -- read at MLXArray position
- Uses `mlx_slice_dynamic` / `mlx_slice_update_dynamic` from Cmlx
- **GOTCHA**: `start` must be 1D array (`MLXArray([Int32(3)])`), NOT scalar -- scalar crashes

**RoPEApplication.swift** -- MLXArray offset path for CompilableKVCache:
```swift
if let compilable = cache as? CompilableKVCache {
    return rope(x, offset: compilable.offsetArray)  // ArrayOffsetLayer
}
return rope(x, offset: cache?.offset ?? 0)  // OffsetLayer (standard)
```

**Evaluate.swift** -- Integration:
- `GenerateParameters.compiledMaxCacheLength: Int?` -- enables Overflow Bin
- `GenerateParameters.enableCompiledDecode: Bool` -- enables compile() wrapping
- After prefill, converts `KVCacheSimple` to `CompilableKVCache`
  - Only converts KVCacheSimple -- leaves RotatingKVCache, MambaCache unchanged
  - Forces materialization of all cache state before conversion
- Compiled forward: `compile(inputs: caches, outputs: caches) { model forward }`

**KVCache.swift** -- Added `Updatable` conformance to `BaseKVCache` (needed for compile inputs/outputs)

### Test Results

**Gemma4 E2B (correctness verified):**
- Overflow Bin (no compile): **IDENTICAL output** over 39 tokens, +5.7% speed
- Overflow Bin (with compile): **CRASHES** -- Gemma4 KV sharing reads `cache.offset` as Int inside forward, triggers a synchronous readback inside compile transformation

**Qwen3.5 35B:** Could not test -- background codebook quantization was consuming GPU, model generated garbage even on baseline. Must retest when GPU is free.

### Known Issues and Blockers

1. **compile() crashes on models that read `cache.offset` as Int inside forward:**
   - Gemma4 E2B/E4B: KV sharing layer returns `cache?.offset ?? 0` as Int tuple element
   - Fix: change model code to use `CompilableKVCache.offsetArray` or return MLXArray offset
   - Qwen3.5: does NOT read cache.offset in forward -- should be compatible

2. **Speed without compile is comparable, not faster:**
   - Full-buffer attention computes over `maxLength` positions instead of `offset`
   - Without compile, this is more work for no benefit
   - The gain ONLY comes from compile() fusing hundreds of FFI crossings into one call

3. **MambaCache compatibility:**
   - MambaCache uses Int offset from BaseKVCache but GatedDeltaNet doesn't read it
   - MambaCache.innerState() uses compactMap which may return variable-size arrays
   - Need to verify compile() handles MambaCache state correctly

4. **RotatingKVCache must NOT be converted:**
   - Has special rotation/temporal ordering semantics
   - Our filter correctly skips it (only converts KVCacheSimple)

---

## Strategic Recommendations

### Priority 1: Fix Heterogeneous Cache Handling in setupCompiledDecode (P0 — HIGHEST)
**The fix**: Modify `setupCompiledDecode()` in Evaluate.swift to handle mixed cache types (MambaCache + CompilableKVCache). Don't require all layers to be CompilableKVCache.

**Why this is the fix**: vmlx Python / Inferencer compile the entire model forward — ALL ops fused. Our `setupCompiledDecode` only works when ALL caches are the same type. Qwen3.5 has mixed types, so it falls back to uncompiled path.

**Status**: Ready to implement. Qwen3.5 GatedDeltaNet doesn't read `cache.offset` as Int — compile-compatible. Fix is in Evaluate.swift only.

### Priority 2: Add Warmup Pass (P1)
Force JIT compilation of all Metal kernels during model load (after weights, before first inference). Eliminates ~500ms-2s first-token latency.

**Status**: Not started. Single function addition to ModelContainer or TokenIterator.

### Priority 3: Document Wired Memory Opt-In (P2)
Add documentation explaining `sudo sysctl iogpu.wired_limit_mb=<size>` — same pattern as Inferencer. Not auto-applied (crashed before), opt-in only.

**Status**: Documentation only. No code changes needed.

### Priority 4: Compile MiniMax MoE Routing (P2)
Same pattern as Qwen3.5 fix — extend setupCompiledDecode to handle MiniMax's MoE routing.

**Status**: After Qwen3.5 fix validates the approach.

---

## Key Learnings

1. **The gap is about compile SCOPE, not individual ops.** vmlx Python compiles `model.__call__` — entire forward. mlx-swift-lm compiles only activation functions inside SwitchGLU. This is why vmlx: 100 tok/s, mlx-swift-lm: 61 tok/s.

2. **Heterogeneous caches are the blocker for full compile().** Qwen3.5 mixes MambaCache + KVCacheSimple. `setupCompiledDecode()` fails unless ALL caches are the same type. Fix: don't require uniformity — compile what can be compiled.

3. **FFI bridge ~1-2μs per op.** When GPU kernels are tiny (~1μs for softmax, argPartition, takeAlong), bridge overhead = kernel time. CPU becomes bottleneck, GPU starves. Only compile() bypasses this.

4. **shapeless=True has bugs on dynamic shapes.** Reductions (sum, mean) on dynamic-shaped outputs replay stale results. Not usable for KV cache compilation. Use CompilableKVCache (static shapes) instead.

5. **Swift already beats Python on large-kernel models.** Gemma4: Swift 97-102 tok/s vs Python 80 tok/s (+28%). Only fragmented-op models (MoE, GatedDeltaNet) have the gap.

6. **Don't auto-apply wired memory.** It crashed on small GPUs (847a8c7). Make it opt-in with documentation, like Inferencer does.

7. **The FFI bridge can't be optimized — only bypassed.** Swift can't import C++20 MLX backend. The bridge is structural. Only compile() or op fusion reduces crossings.

---

## File Reference

| File | Branch | Purpose |
|------|--------|---------|
| `Libraries/MLXLMCommon/CompilableKVCache.swift` | feature/overflow-bin-compile | Overflow Bin KV cache |
| `Libraries/MLXLMCommon/DynamicSlice.swift` | feature/overflow-bin-compile | C API slice wrappers |
| `Libraries/MLXLMCommon/Evaluate.swift` | main (clean) / feature (extended) | Generation loop |
| `Libraries/MLXLMCommon/KVCache.swift` | main (clean) / feature (Updatable) | Cache implementations |
| `Libraries/MLXLMCommon/RoPEApplication.swift` | main (clean) / feature (MLXArray offset) | RoPE helper |
| `Libraries/MLXLMCommon/HardwareInfo.swift` | research/custom-metal-kernels | M1/M2 GPU detection |
| `Libraries/MLXLLM/Models/Qwen35.swift` | main | MoE model (Qwen3.5 35B) |
| `Libraries/MLXLLM/Models/MiniMax.swift` | main | MoE model (MiniMax) |
| `Libraries/MLXLMCommon/SwitchLayers.swift` | main | compiledSwiGLU, compiledGeGLU |
| `TestRunner/Sources/main.swift` | local only | Test harness with --overflow-bin --compiled flags |
| `docs/research/IMPLEMENTATION-PLAN.md` | research/custom-metal-kernels | **Full implementation guide** |
| `docs/research/HANDOFF-PERF-OPTIMIZATION.md` | research/custom-metal-kernels | **Session handoff** |

---

## Implementation Plan

**See `docs/research/IMPLEMENTATION-PLAN.md` for detailed implementation steps.**

### Quick Summary

| Priority | Fix | Expected Impact |
|----------|-----|----------------|
| **P0** | Fix heterogeneous cache handling in `setupCompiledDecode` | 64% speedup (61→100 tok/s on Qwen3.5) |
| **P1** | Add warmup pass (force JIT compilation) | Eliminates ~500ms-2s first-token JIT delay |
| **P2** | Document wired memory opt-in | Prevents SSD paging between tokens |
| **P2** | Compile MiniMax MoE routing | Closes gap on MiniMax MoE models |

### The Fix (P0 — Heterogeneous Caches)

**Current blocker**: `setupCompiledDecode()` requires ALL caches to be CompilableKVCache. Qwen3.5 mixes:
- `MambaCache` (for GatedDeltaNet layers)
- `KVCacheSimple` → `CompilableKVCache` (for standard attention layers)

**The fix**: Don't require all caches to be the same type. Compile what CAN be compiled, leave the rest.

```swift
// In Evaluate.swift — setupCompiledDecode fix:
// Before: fails if ANY cache is not CompilableKVCache
// After: converts only KVCacheSimple, leaves MambaCache as-is, compiles anyway
```

Qwen3.5's GatedDeltaNet does NOT read `cache.offset` as Int inside forward — it's compile-compatible.

### Performance Targets

| Model | Current | Target |
|-------|:---:|:---:|
| Qwen3.5 35B MoE | 61 tok/s | **90-100 tok/s** |
| MiniMax MoE | ~10 tok/s | **40-50 tok/s** |
| Gemma4 26B | 97-102 tok/s | **100+ tok/s** |
