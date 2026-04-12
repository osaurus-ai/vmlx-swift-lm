# CHANGELOG — vmlx-swift-lm (osaurus-ai/vmlx-swift-lm)

## [2026-04-12] — AsType Cascade Elimination (Complete)

### Fix 1: Float32 Scalar Contamination
`MLXArray(someFloat)` without `dtype:` → float32 scalar × bfloat16 tensor = AsType cascade.
- **Qwen35.swift** (LLM + VLM): invScale scalars
- **Qwen3Next.swift**: invScale scalars
- **Gemma4Text.swift**: sqrt embedding scale double-cast
- **Gemma4.swift** (VLM): sqrt embedding + vision encoder masks
- **Gemma3.swift** (VLM): embedding scale + softcap scalar
- **Gemma3nText.swift**: activationSparsity + altupCorrectScale
- **GPTOSS.swift**: clip() scalars in swiglu
- **NanoChat.swift**: applySoftcap scalar

### Fix 2: MoE Gate putAlong Zero-Out
`MLXArray(0.0)` → `MLXArray(0.0, dtype: groupScores.dtype)`.
- **7 MoE models**: BailingMoe, DeepseekV3, GLM4MOE, GLM4MOELite, AfMoE, MiMoV2Flash, NemotronH

### Fix 3: Precise Softmax + Remove Sigmoid Float32 Cast
`softmax(x.asType(.float32))` → `softmax(x, precise: true)`.
`sigmoid(x.asType(.float32))` → `sigmoid(x)`.
- **Mistral4, Mistral4VLM**: softmax precise
- **GLM4MOE, GLM4MOELite, AfMoE**: softmax precise + sigmoid
- **BailingMoe, NemotronH, MiMoV2Flash**: sigmoid

### Fix 4: Universal Float16 → BFloat16 Conversion
All float16 params (including scales/biases) → bfloat16.
- **Load.swift**: no more skipping scales/biases, trigger on any float16

### Fix 5: NemotronH Identity Weight Dtype
`MLXArray.ones([groupSize])` → `MLXArray.ones([groupSize], dtype: unflattened.dtype)`.
- **NemotronH.swift**: RMSNormGated identity weight in hot path

### Fix 6: SSM computeDt Optimization
`softplus(x)` → `logAddExp(x, 0)`, typed clip scalars.
- **SSM.swift**

### Results (M4 Max 128GB)

| Model | Before | After | Python (M3 Ultra) | Key Fix |
|-------|--------|-------|-------------------|---------|
| Qwen3.5-35B MLX Q4 | 41 | **103** | 94 | #1 (invScale) |
| Gemma4 26B Q4 | 27 | **87** | — | #1 (sqrt double-cast) |
| Gemma4 E2B Q4 | 120 | **121** | 128 | #1 (sqrt) |
| Gemma4 E4B Q4 | — | **73** | — | #1 (sqrt) |
| Mistral4 119B JANG 2L | 16 | **70** | 45-50 | #3 (precise softmax) |
| MiniMax JANG 2L | 14 | **46** | 51 | #2 + #4 (putAlong + scales) |
| Nemotron Cascade 30B JANG 2L | 45 | **110** | 130 | #5 (identity weight) |

### VLM Status
- Gemma4 E2B VLM: **9/9 tests pass** (image, multi-turn, TokenIterator)
- Qwen3.5 VL: chat template image token issue (pre-existing)
- Mistral4 VL: multi-turn reshape crash (pre-existing)

### Batching Status
- BatchEngine: 0 files changed by speed fixes, fully intact
- Unit tests: 14 tests (7 unit + 7 integration), documented 2.59x throughput
- Speed fixes make each batch slot 2-4x faster (AsType elimination)

### Infrastructure
- Private repo: osaurus-ai/vmlx-swift-lm (main only)
- Complete technical reference: docs/SPEED-FIXES.md (gitignored)
- Hookify rules: model testing, API compat, speed targets, debug/changelog
