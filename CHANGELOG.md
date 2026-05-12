# CHANGELOG — vmlx-swift-lm (osaurus-ai/vmlx-swift-lm)

## [2026-05-12] — DSV4 Flash JANGTQ-K routing bits and MoE top-k parity

- DeepSeek-V4 JANGTQ-K now preserves `routed_expert_bit_plan` from bundle metadata instead of flattening all routed expert layers to one bit width.
- `DeepseekV4JANGTQModel` exposes the effective routed expert bit width per layer for regression tests and diagnostics.
- DSV4 routed MoE `num_experts_per_tok` now participates in the existing lower-only runtime top-k override helper, while DSV4 compressor `index_topk` remains untouched as NSA/compression selection.
- Factory dispatch now preserves mixed routed layer bit plans when loading `weight_format=mxtq` DSV4 bundles.

**Verification:**
- `swift build --target MLXLLM`
- `swift build --target MLXLMCommon`
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter 'DeepseekV4|RuntimeMoETopKConfigWiringTests' --no-parallel`
- Focused tests cover DSV4 config decode/encode, factory dispatch, chat/reasoning/tool parser flows, cache topology, disk round-trip, TurboQuant bypass, and MoE top-k wiring.

## [2026-05-10] — B=1 lifecycle, streaming experts, and VLM cache offsets

- `BatchEngine.generate` now clears the B=1 solo fast-path lifecycle before the returned stream finishes. This removes the race where callers could receive completion while the engine still reported active solo work.
- `JANGTQStreamingExperts` now indexes pre-stacked `switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}` tensors in addition to legacy per-expert tensors, and slices the selected expert rows at load time.
- Qwen3.5-VL gated-delta layers now advance `MambaCache.offset` when writing recurrent state, so session startup and subsequent SSM masks see the true token position.
- ZAYA1-VL source-contract coverage now asserts the canonicalized LoRA module names (`down`/`up`, `expert_N`) and the sanitizer rewrites from shipped sidecar keys.

**Verification:**
- 6/6 solo fast-path lifecycle tests pass.
- 8/8 focused regression tests pass for streaming experts, Qwen3.5-VL gated-delta cache offset, and ZAYA1-VL source contracts.
- Broad serialized runtime matrix passes: 603 tests in 89 suites, covering batch/cache/KV/TurboQuant/SSM/JANGTQ/runtime/reasoning/templates/tools/VLM/ZAYA/Hy3/Gemma/Kimi/Laguna/Bailing/DeepSeek/Nemotron.

## [2026-04-21] — Coordinator-owned KV sizing + BatchEngine decode perf

### Feature: `CacheCoordinator` now owns KV sizing end-to-end
Driver: ferebee's Osaurus 0.17.0 report — 200K-char / 55K-token translation reliably OOM'd the app. Root cause was a contract gap: osaurus 0.17.0 removed its per-request `maxKVSize` UI knob with the comment "KV cache sizing is owned end-to-end by vmlx-swift-lm's CacheCoordinator", but vmlx had no such logic. Every request arrived `maxKVSize: nil, kvMode: .none` → unbounded `KVCacheSimple` + no quantization fallback.

- New fields on `CacheCoordinatorConfig`:
  - `defaultKVMode: KVQuantizationMode = .none` — applied to slots whose request `kvMode` is `.none`
  - `defaultMaxKVSize: Int? = nil` — applied when `maxKVSize` is nil AND prompt > `longPromptMultiplier × defaultMaxKVSize`
  - `longPromptMultiplier: Double = 2.0` — gate keeping short chat turns out of the rotating-window cap
- `CacheCoordinatorConfig.resolveKVPolicy(kvMode:maxKVSize:promptTokenCount:)` — pure function encoding the rules. Explicit caller values always win; defaults only fill gaps.
- `BatchEngine.admitPendingRequests` calls `resolveKVPolicy` before `context.model.newCache(...)`.
- `BatchPendingRequest.parameters` changed from `let` to `var` to allow the admission-path gap-fill.
- New docs: `Libraries/MLXLMCommon/BatchEngine/KV-SIZING-CONTRACT.md` with the full resolution table, osaurus wiring example, and out-of-scope items.
- New tests: `Tests/MLXLMTests/CacheCoordinatorKVPolicyTests.swift` — 10 unit tests covering every path (explicit-wins, gap-fill, short/long prompt gate, ferebee scenario, custom multiplier).

**Verification (no regressions, 94 tests green):**
- 10/10 `CacheCoordinatorKVPolicyTests`
- 7/7 `BatchEngineIntegrationTests`
- 6/6 `BatchEngineTurboQuantIntegrationTests`
- 6/6 `BatchEngineMultiTurnTests`
- 75/75 reasoning/harmony/tool-call tests (`ReasoningParser`, Harmony Gemma-4 streaming, `StopStringMatcher`, startInReasoning Qwen3.6 enable_thinking, `Generation.reasoning`, `ToolCallFormat`)

**Recommended osaurus wiring** (one-time at model load):
```swift
CacheCoordinatorConfig(
    defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),
    defaultMaxKVSize: 8192,
    longPromptMultiplier: 2.0
)
```

### Perf: `BatchEngine` decode — +1-2 tok/s across all variants
- B=1 cache bypass in `stepBatchDecode` — skip `BatchKVCache` wrappers at batch size 1 (48+ Swift allocations per token saved on Qwen 35B-A3B MoE)
- Conditional `Task.yield()` — only yield when work pending
- `asyncEval(logits)` + `asyncEval(sampledTokens)` — mirrors `TokenIterator.next()` idiom

**Measured (M4 Max 128GB, median of 3, 128-token decode):**

| Variant | Before | After | Δ |
|---|---|---|---|
| Qwen3.5-35B-A3B-4bit | 87.1 | 89.4 | +2.3 |
| Qwen3.6-35B-A3B-MXFP4 | 86.2 | 88.4 | +2.2 |
| Qwen3.6-35B-A3B-JANGTQ2 | 76.3 | 77.8 | +1.5 |
| Qwen3.6-35B-A3B-JANGTQ4 | 70.5 | 71.4 | +0.9 |
| Gemma-4 E2B 4bit | 120.6 | 122.0 | +1.4 |
| Gemma-4 E4B 4bit | 71.9 | 73.5 | +1.6 |

### Out of scope (pinned in KV-SIZING-CONTRACT.md)
1. **Wired-memory sticky state across reboot** — `mlx-swift`'s `WiredMemoryManager` calls `mlx_set_wired_limit` (kernel sysctl `iogpu.wired_limit_mb`), which persists across process restart until reboot. Fix belongs upstream in mlx-swift (signal handler) or in osaurus (conservative ticket sizing + startup reset). vmlx-swift-lm only consumes tickets.
2. **Gemma looping** — 75/75 parser tests green post-fix. Needs real-model byte-level repro before a parser change can land.

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
