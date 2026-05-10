# vmlx-swift-lm Speed Fixes — Complete Technical Reference

> Every fix that makes Osaurus faster than Python mlx_lm, model by model.

## Summary: Before & After

All measurements on **M4 Max 128GB**. Python baselines from M3 Ultra 256GB (~1.5x more bandwidth), so Swift matching Python = Swift is actually faster per-bandwidth.

| Model | Before | After | Python Baseline | Speedup |
|-------|--------|-------|----------------|---------|
| Qwen3.5-35B-A3B MLX 4-bit | 41 tok/s | **103 tok/s** | 94 | 2.5x |
| Gemma4-26B-A4B 4-bit | 27 tok/s | **87 tok/s** | — | 3.2x |
| Gemma4-E2B 4-bit | 120 tok/s | **121 tok/s** | 128 | — |
| Gemma4-E4B 4-bit | — | **73 tok/s** | — | — |
| Mistral-Small-4 119B JANG 2L | 16 tok/s | **70 tok/s** | 45-50 | 4.4x |
| MiniMax-M2.5 JANG 2L | 14 tok/s | **46 tok/s** | 51 | 3.3x |
| Nemotron-Cascade-2 30B JANG 2L | 45 tok/s | **110 tok/s** | 15.5 | 2.4x |

---

## The Five Root Causes (and their fixes)

### 1. Float32 Scalar Contamination

**The bug:** Swift's `MLXArray(someFloat)` defaults to float32. When this scalar multiplies a bfloat16 tensor, MLX inserts an `AsType` cast — a separate Metal kernel dispatch. Python's `mx.array()` infers dtype from context automatically; Swift does not.

**The fix:** Always specify `dtype:` when creating scalar MLXArrays that interact with model tensors.

```swift
// BEFORE (broken): float32 scalar × bfloat16 tensor = AsType cascade
let qNormed = MLXArray(pow(invScale, 2)) * rmsNorm(q, ...)

// AFTER (fixed): scalar inherits tensor dtype
let qNormed = MLXArray(pow(invScale, 2), dtype: q.dtype) * rmsNorm(q, ...)
```

**Impact:** 1,176 → 60 AsType ops on Qwen3.5. Each AsType is a separate Metal kernel dispatch (~20µs). At 1,100+ extra dispatches per decode step: 1100 × 20µs = **22ms of pure overhead** — that's the entire difference between 41 and 100 tok/s.

**Files fixed:**
| File | What | Model Family |
|------|------|-------------|
| `Qwen35.swift` (LLM) | `invScale` scalars for GatedDeltaNet q/k norms | Qwen 3.5 |
| `Qwen3Next.swift` (LLM) | Same `invScale` pattern | Qwen 3 Next |
| `Qwen35.swift` (VLM) | Same `invScale` pattern | Qwen 3.5 VL |
| `Gemma4Text.swift` | `sqrt(hiddenSize)` embedding scale | Gemma 4 |
| `Gemma4.swift` (VLM) | Embedding scale + vision encoder masks | Gemma 4 VL |
| `Gemma3.swift` (VLM) | Embedding scale + softcap scalar | Gemma 3 VL |
| `Gemma3nText.swift` | `activationSparsity` + `altupCorrectScale` + AltUp `clip()` scalars | Gemma 3n |
| `GPTOSS.swift` | `clip()` scalars in swiglu | GPT-OSS |
| `NanoChat.swift` | `applySoftcap` scalar | NanoChat |

---

### 2. MoE Gate putAlong Zero-Out

**The bug:** MoE routing zeroes out non-selected expert groups with `putAlong(..., values: MLXArray(0.0))`. The `0.0` literal creates a float32 scalar. When the scores tensor is bfloat16, every `putAlong` call inserts an AsType.

**The fix:** Pass dtype: `MLXArray(0.0, dtype: groupScores.dtype)`.

```swift
// BEFORE: float32 zero into bfloat16 tensor
scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)

// AFTER: zero matches tensor dtype
scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0, dtype: groupScores.dtype), axis: -2)
```

**Files fixed:** BailingMoe, DeepseekV3, GLM4MOE, GLM4MOELite, AfMoE, MiMoV2Flash, NemotronH — **all 7 MoE model families**.

---

### 3. Explicit `.asType(.float32)` Before Softmax/Sigmoid

**The bug:** MoE gate routing in Swift used `softmax(gates.asType(.float32))`, forcing the entire routing path into float32. All downstream operations (argPartition, takeAlong, score normalization, expert weighting) stay float32, creating a cascade of AsType ops when results interact with bfloat16 expert outputs.

Python uses `mx.softmax(gates, axis=-1, precise=True)` which computes in float32 internally but **returns the input dtype**. No cascade.

**The fix:** Replace `.asType(.float32)` with `precise: true` for softmax. Remove `.asType(.float32)` entirely for sigmoid (sigmoid is numerically stable in bfloat16).

```swift
// BEFORE: forces entire routing path to float32
let scores = softmax(gates.asType(.float32), axis: -1)

// AFTER: computes precisely but returns bfloat16
let scores = softmax(gates, axis: -1, precise: true)
```

```swift
// BEFORE: unnecessary float32 promotion
let scores = sigmoid(gates.asType(.float32))

// AFTER: bfloat16 sigmoid is stable
let scores = sigmoid(gates)
```

**Impact on Mistral4 119B:** 988 → 72 AsType. **16 → 70 tok/s (+338%).**

**Files fixed:**
| File | Change | Model Family |
|------|--------|-------------|
| `Mistral4.swift` | softmax precise:true | Mistral Small 4 |
| `Mistral4VLM.swift` | softmax precise:true | Mistral Small 4 VL |
| `GLM4MOE.swift` | softmax precise:true + remove sigmoid .asType | GLM-4 MoE |
| `GLM4MOELite.swift` | remove sigmoid .asType | GLM-4 MoE Lite |
| `AfMoE.swift` | softmax precise:true + remove sigmoid .asType | AfMoE |
| `BailingMoe.swift` | remove sigmoid .asType | BailingMoe |
| `NemotronH.swift` | remove sigmoid .asType | Nemotron-H |
| `MiMoV2Flash.swift` | remove sigmoid .asType | MiMo V2 Flash |
| `LFM2MoE.swift` | softmax precise:true (was `gate(x).asType(.float32)` → softmax) | LFM-2 MoE |
| `MiniMax.swift` | remove `gate(x.asType(.float32))` → `gate(x)`, typed epsilon | MiniMax M2.5 |

---

### 4. JANG Float16 → BFloat16 Conversion (Including Scales/Biases)

**The bug:** JANG quantized models store scales and biases as float16. The original `convertToBFloat16()` function **skipped** quantization scales/biases with the comment "they're used directly by Metal kernels." This was wrong.

`QuantizedMatmul` uses the scales dtype to determine its output dtype. Float16 scales → float16 output → AsType when multiplied with bfloat16 norm weights. This created 1,000+ AsType ops on JANG models.

**The fix (two stages):**

Stage 1: Convert ALL parameters (including scales and biases) to bfloat16. No more skipping.

Stage 2: Check `model.parameters()` instead of the original weights dict. The quantize step creates new QuantizedLinear modules with `MLX.quantized()` which can produce float32 scales even when the original weights were bfloat16. Checking the original `weights` dict missed these.

```swift
// BEFORE (v1): skipped scales/biases, only triggered on MoE
if key.hasSuffix(".scales") || key.hasSuffix(".biases") { continue }
let isMoE = weights.keys.contains { $0.contains("switch_mlp") ... }

// BEFORE (v2): checked original weights dict — missed post-quantize float32 scales
let hasFloat16 = weights.values.contains { $0.dtype == .float16 }
if hasFloat16 { convertToBFloat16(model: model) }

// AFTER (v3): checks model's live parameters AFTER quantize step
let allParams = model.parameters().flattened().map { $0.1 }
let hasNonBFloat16 = allParams.contains { $0.dtype == .float16 || $0.dtype == .float32 }
if hasNonBFloat16 { convertToBFloat16(model: model) }
```

**Why v2 was insufficient:** JANG_2S models have `format: mlx` metadata and store weights as bfloat16 in safetensors. `weights.values.contains { .float16 }` returned `false`. But the quantize step at line 89-127 calls `MLX.quantized()` which produces float32 scales inside QuantizedLinear. These float32 scales cause the same AsType cascade — 1,105 ops on Qwen3.5 JANG_2S.

**Impact on MiniMax JANG:** 1,245 → 248 AsType. **14 → 46 tok/s (+229%).**

**File:** `Load.swift`

---

### 5. Identity Weight Dtype in Hot-Path RMSNorm

**The bug:** `NemotronHRMSNormGated` created a float32 identity weight (`MLXArray.ones([groupSize])`) for per-group RMS normalization on **every forward call**. With 26 Mamba layers per step, each calling this norm, the float32 weight caused ~400 AsType ops per step.

**The fix:** Create the identity weight with the input tensor's dtype.

```swift
// BEFORE: float32 identity weight every call
let identityWeight = MLXArray.ones([groupSize])
let normed = MLXFast.rmsNorm(unflattened, weight: identityWeight, eps: eps)

// AFTER: matches input dtype
let normed = MLXFast.rmsNorm(unflattened, weight: MLXArray.ones([groupSize], dtype: unflattened.dtype), eps: eps)
```

**Impact on Nemotron Cascade:** 562 → 161 AsType. **45 → 110 tok/s (+144%).**

**File:** `NemotronH.swift`

---

### 6. SSM computeDt Optimization

**The bug:** `computeDt` used `softplus(dt + dtBias)` which expands to `log(1 + exp(x))` — three separate ops. Python uses `mx.logaddexp(x, 0)` — a single fused op that's also compiled with `@mx.compile(shapeless=True)`. The `clip()` call also used bare Float scalars which default to float32.

**The fix:** Use `logAddExp` and typed clip scalars.

```swift
// BEFORE: 3 ops + float32 clip scalars
let dt = softplus(dt + dtBias)
return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)

// AFTER: 1 fused op + typed scalars
let dt = logAddExp(dt + dtBias, MLXArray(0, dtype: dt.dtype))
return MLX.clip(dt, min: MLXArray(timeStepLimit.0, dtype: dt.dtype), max: MLXArray(timeStepLimit.1, dtype: dt.dtype))
```

**File:** `SSM.swift`

---

## The Embedding Scale Double-Cast Pattern

**The bug (Gemma family specific):** Gemma models scale embeddings by `sqrt(hiddenSize)`. The Swift code created the scalar as float32, cast it to bfloat16, then cast it AGAIN to the tensor's dtype — two AsType ops per forward pass.

```swift
// BEFORE: float32 → bfloat16 → h.dtype (double cast)
h = h * MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16).asType(h.dtype)

// AFTER: direct to target dtype (zero casts)
h = h * MLXArray(sqrt(Float(config.hiddenSize)), dtype: h.dtype)
```

**Files:** `Gemma4Text.swift`, `Gemma4.swift` (VLM), `Gemma3.swift` (VLM)

---

## macOS Tahoe (26.x) — compile(shapeless: true) Crash Guard

**The bug:** `compile(shapeless: true)` crashes on M1/M2 GPUs (A14/A15, Metal GPU family g7x) running macOS 26 (Tahoe). The compiled Metal kernel returns zero results, then Swift does `compileState.call([a])[0]` on an empty array → `Index out of range` → `EXC_BREAKPOINT`.

This is an upstream MLX Metal JIT bug (MLX #3329, #3201, #3256). M3+ chips (A16+, g8x family) are unaffected. No upstream fix released.

**The fix:** All `compile(shapeless: true)` calls are guarded by `HardwareInfo.isCompiledDecodeSupported`. On M1/M2: falls back to the plain (non-compiled) closure. On M3+: uses compiled path.

```swift
// BEFORE: crashes on M1/M2 + Tahoe
private let compiledSwiGLU = compile(shapeless: true) { silu(gate) * x }

// AFTER: guarded — plain closure on M1/M2, compiled on M3+
private let compiledSwiGLU = {
    let body = { silu(gate) * x }
    return HardwareInfo.isCompiledDecodeSupported
        ? compile(shapeless: true, body) : body
}()
```

**All guarded closures (9 total across 7 files):**

| File | Closure | Used By |
|------|---------|---------|
| `SwitchLayers.swift` | `safeGeluApproximate` | ALL models using SafeGELU |
| `SwitchLayers.swift` | `compiledSwiGLU` | ALL MoE models using SwitchGLU with SiLU |
| `SwitchLayers.swift` | `compiledGeGLU` | ALL MoE models using SwitchGLU with GELU |
| `Gemma4Text.swift` | `compiledLogitSoftcap` | Gemma 4 LLM |
| `Gemma4.swift` (VLM) | `compiledLogitSoftcap` | Gemma 4 VLM |
| `Qwen35.swift` (LLM) | `compiledSigmoidGate` | Qwen 3.5 LLM shared expert |
| `Qwen35.swift` (VLM) | `compiledSigmoidMultiply`, `compiledSigmoidGate`, `_vlmCompiledComputeG` | Qwen 3.5 VLM |
| `GatedDelta.swift` | `_compiledComputeG` | GatedDeltaNet (Qwen 3.5/3 Next) |
| `Qwen3Next.swift` | `_compiledSigmoidMultiply`, `_compiledPreciseSwiGLU` | Qwen 3 Next |

**Also guarded:** `setupCompiledDecode` in `Evaluate.swift` (full-model compile for decode) — this was already guarded before this sweep.

**Impact on M1/M2 users:** ~10% slower decode (no fused activation kernels), but no crashes. All models work correctly.

**Related issues:** osaurus-ai/osaurus#808 (JANG bit-width mismatch), osaurus-ai/osaurus#813 (same), osaurus-ai/osaurus#814 (Tahoe compile crash). #808/#813 were JANG loading bugs (fixed in commit `10d547e`). #814 was the Tahoe crash (fixed by this guard).

See `HANDOFF-JANG-CRASH-813-814.md` for the full investigation and evidence chain.

---

## Full Audit — Verified Clean (2026-04-12)

Complete sweep of all 174 .swift files in Libraries/:

**No unauthorized hardware checks.** Only `HardwareInfo.swift` queries `hw.machine`. No `ProcessInfo.physicalMemory`, no RAM detection, no chip name branching.

**No memory limit hacks.** No `mlx_set_wired_limit`, no `Memory.cacheLimit` overrides. Wired memory is managed by the caller via `WiredMemoryTicket` (opt-in, never auto-set).

**No sleeps or polling.** Zero `sleep()`, `Thread.sleep`, or `usleep` calls.

**All shared state properly locked.** `OSAllocatedUnfairLock` on ModelContainer, DiskCache, CacheCoordinator, PagedCacheManager, SSMStateCache. `NSLock` on registries. `AsyncMutex` on SerialAccessContainer.

**Memory.clearCache() calls are all legitimate:**
- `Evaluate.swift:909` — every 256 decode tokens (prevents GPU allocator fragmentation)
- `BatchEngine.swift:336` — same periodic cleanup in batch mode
- `LLMModel.swift:35` — between prefill chunks (prevents OOM on large prompts)
- `BenchmarkHelpers.swift` — benchmark measurement (not production code)

**Remaining `.asType(.float32)` calls are all legitimate:**
- `GatedDelta.swift:20` — exp(exp()) decay computation requires float32 for numerical correctness
- `Qwen3Next.swift:35-36` — precise SwiGLU (compiled fused kernel, matches Python `_precise_swiglu`)
- `Gemma3nText.swift:429,436,454,465` — AltUp coefficient matmul requires float32 precision
- `Mistral3Text.swift:25` — RoPE position scaling (integer math)
- `BaichuanM1.swift:281` — weight normalization at load time (not hot path)
- `NanoChat.swift:94` — frequency index creation at init (not hot path)
- `GraniteMoeHybrid.swift:97`, `NemotronH.swift:140` — heads range creation at init
- `Phi.swift:66` — queries in attention (upstream pattern)

---

## Why Swift is Faster Than Python

Python and Swift use the same MLX C++ backend and Metal kernels. The speed difference comes from **graph structure**, not compute:

1. **Fewer graph nodes** = fewer Metal kernel dispatches = less CPU-GPU synchronization overhead
2. **No AsType cascades** = operations fuse better in the Metal command buffer
3. **Double-buffered decode** = CPU builds next token's graph while GPU evaluates current token
4. **Compiled decode** = `compile(inputs: cache, outputs: cache)` traces the full step once, replays it

Python's advantage was that its scalar dtype inference eliminated AsType automatically. With our fixes, Swift's graph is now as clean as Python's — and Swift's compiled decode path gives an additional 10-20% boost that Python's `generate()` doesn't use.

---

## How to Verify

```bash
cd TestRunner && swift build -c release
.build/release/TestRunner <model-path> --raw-bench
```

Check:
- **AsType count** < 100 for MLX models, < 200 for JANG models
- **Graph node count** should be within 50% of Python's 2693 (model-dependent)
- **tok/s** should match or exceed Python baseline (adjusted for M4 Max vs M3 Ultra bandwidth: ×0.68)

---

## Model-Specific Notes

### MiniMax M2.5
- Routing uses **sigmoid + bias** (NOT softmax) — `sigmoid(gates)` directly, no `.asType(.float32)`
- `group_size=128` mandatory for 192+ expert models (conversion-time fix, not runtime)
- `temp=1.0` required (greedy decoding = infinite thinking loops)

### Qwen 3.5 (all sizes)
- Layer pattern: [SSM, SSM, SSM, FA] repeating (GatedDeltaNet linear attention)
- The `invScale` scalar in GatedDeltaNet is the #1 AsType source — MUST have `dtype: q.dtype`
- EOS tokens: `[248046, 248044]` (NOT Qwen 2.5's `[151645, 151643]`)
- All 5 norm types need +1 shift for JANG models

### Gemma 4 (all variants)
- Embedding scale `sqrt(hiddenSize)` must use `dtype: h.dtype`, NOT `.bfloat16` intermediate
- MoE router (26B) uses `argPartition(MLXArray(0) - expertScores, ...)` — the integer 0 is fine
- VLM vision encoder masks need typed scalars for `MLX.where`

### Mistral Small 4 (119B)
- DeepSeek V3-style MLA attention — 7 QuantizedLinear projections per layer
- `softmax(precise: true)` is critical — `.asType(.float32)` before softmax caused 988 AsType
- Python mlx_lm doesn't even support this model natively (needs custom `mistral4_mlx.py`)

### Nemotron-H / Nemotron Cascade
- Hybrid Mamba-2 SSM + MoE — 52 layers with `hybrid_override_pattern`
- The `NemotronHRMSNormGated` identity weight was the #1 issue: float32 on every call
- SSM Metal kernel handles the core state update efficiently
- `fc1_latent_proj`/`fc2_latent_proj` needed for Nemotron-H Super 120B (LatentMoE)

### GPT-OSS 120B
- mxfp4 native quantization (NOT post-training)
- FP16 overflow at L30 → MUST use BF16 (residual stream exceeds fp16 max 65504)
- `clip()` scalars in swiglu need `dtype: x.dtype`

---

## The Universal Rule

**Every `MLXArray` scalar or constant created at runtime must carry the tensor's dtype.**

If you see any of these patterns in a model file, it's a bug:

```swift
// BUG: float32 scalar
MLXArray(someFloat) * tensor
MLXArray(0.0)
MLXArray(1.0)
MLXArray.ones([n])  // in a hot path
softmax(x.asType(.float32), ...)
sigmoid(x.asType(.float32))

// FIX: inherit dtype
MLXArray(someFloat, dtype: tensor.dtype) * tensor
MLXArray(0.0, dtype: tensor.dtype)
MLXArray(1.0, dtype: tensor.dtype)
MLXArray.ones([n], dtype: tensor.dtype)
softmax(x, axis: -1, precise: true)
sigmoid(x)
```
