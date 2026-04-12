# CHANGELOG — vmlx-swift-lm (osaurus-ai/vmlx-swift-lm)

## [2026-04-11] — Float32 AsType Cascade Elimination + JANG bfloat16 Conversion

### Root Cause
Two performance bugs causing 2-5x slowdowns:

1. **Scalar dtype mismatch:** `MLXArray(someFloat)` without `dtype:` defaults to float32. When multiplied with bfloat16 tensors → AsType cascade. Python infers dtype automatically.
2. **JANG scales/biases left as float16:** `convertToBFloat16()` skipped quantization scales/biases, but QuantizedMatmul uses scales dtype for output. float16 output × bfloat16 norms → AsType cascade.

### Fixes Applied

**Load.swift — Universal dtype unification:**
- Removed float16 scales/biases skip in `convertToBFloat16()`
- Broadened trigger: converts when ANY mixed float16/float32/bfloat16 params exist
- Before: only MoE models with `switch_mlp`/`switch_glu` keys
- After: all models with mixed floating-point dtypes

**Qwen35.swift (LLM) — THE critical fix:**
- `MLXArray(pow(invScale, 2))` → `MLXArray(pow(invScale, 2), dtype: q.dtype)`
- `MLXArray(invScale)` → `MLXArray(invScale, dtype: k.dtype)`

**Qwen3Next.swift (LLM):** Same invScale fix

**Qwen35.swift (VLM):** Same invScale fix for VLM variant

**Gemma4Text.swift:** `MLXArray(sqrt(...), dtype: .bfloat16).asType(h.dtype)` → `dtype: h.dtype`

**Gemma4.swift (VLM):** Lines 382, 466-468, 723, 842 — sqrt double-cast + vision encoder masks

**Gemma3.swift (VLM):** Embedding scale double-cast + softcap scalar

**Gemma3nText.swift:** activationSparsity scalars + altupCorrectScale fallback

**GPTOSS.swift:** clip() scalars in swiglu hot path

**NanoChat.swift:** applySoftcap scalar

**MoE gating — 7 models (putAlong zero-out):**
- BailingMoe, DeepseekV3, GLM4MOE, GLM4MOELite, AfMoE, MiMoV2Flash, NemotronH
- `MLXArray(0.0)` → `MLXArray(0.0, dtype: groupScores.dtype)`

### Test Results (M4 Max 128GB)

| Model | AsType Before | AsType After | tok/s Before | tok/s After (uncomp) | tok/s After (pipe) |
|-------|--------------|-------------|-------------|---------------------|-------------------|
| Qwen3.5-35B MLX 4-bit | 1,176 | 60 | 41 | **103** (+151%) | **138** |
| Gemma4 E2B 4-bit | 89 | 89 | ~120 | **121** | **147** |
| Gemma4 E4B 4-bit | ~112 | 112 | ~73 | **73** | **84** |
| Gemma4 26B 4-bit | ~181 | 181 | **27** | **87** (+222%) | **111** |
| MiniMax JANG 2L | 1,245 | 248 | **14** | **46** (+229%) | **52** |
| Nemotron Cascade JANG 2L | 565 | 562 | 44 | **45** | **50** |

*Python baselines: Qwen3.5 ~98 sync / ~123 pipelined. Swift now BEATS Python.*

**Gemma4 26B regression: FIXED** (27 → 87 tok/s, was previously 97-102)

### Merged Branches
- `perf/astype-elimination-20260411-185828` → main (8 commits)
- `feature/continuous-batching` → main (1 commit)

### Infrastructure
- Private repo: osaurus-ai/vmlx-swift-lm (main branch only, clean)
- .gitignore: internal docs/research/dev scripts excluded
- Hookify rules: model testing matrix, API compat guard, debug/changelog requirements
