# DeepSeek-V4 / Kimi K2.6 — vmlx-swift-lm port status

Living tracker for bringing the **DSV3 / DSV4 / Kimi K2.x family**
to production on Swift. Paired with the jang research docs at:

- `../../jang/research/DSV-FAMILY-RUNTIME-GUIDE.md` — exhaustive
  runtime guide (read first)
- `../../jang/research/DSV4-FLASH-IMPLEMENTATION.md` — DSV4
  per-layer architecture
- `../../jang/research/KIMI-K2.6-VMLX-INTEGRATION.md` — Kimi-specific
  Swift porting plan
- `../../jang/research/KIMI-K2.6-IMPLEMENTATION.md` — Kimi master
- `../../jang/research/JANGTQ-VMLX-SWIFT-PLAN.md` — JANGTQ kernel
  parity plan

**DO NOT delete `DeepseekV3.swift` or `DeepseekV3JANGTQ.swift` when
adding DSV4** — they continue to serve DSV3 / DSV3.2 / Kimi K2.6 /
Kimi K2 / GLM-5.1 / Nemotron-family bundles. DSV4 is **architecturally
distinct** enough to warrant its own `DeepseekV4.swift` +
`DeepseekV4JANGTQ.swift` alongside.

---

## Family matrix

| model_type      | Reference               | Status in vmlx-swift-lm                    | Attention kind       | Notes |
|-----------------|-------------------------|---------------------------------------------|----------------------|-------|
| `deepseek_v3`   | DSV3 / DSV3.2 / GLM-5.1 | ✅ `DeepseekV3.swift` + `DeepseekV3JANGTQ.swift` | MLA (prefill-style)  | Routes via `dispatchDeepseekV3Family` — `weight_format=mxtq` → JANGTQ variant |
| `kimi_k2`       | Pre-K2.6 Kimi           | ✅ alias of `deepseek_v3`                    | MLA (prefill-style)  | Reasoning stamp `kimi` → `<think>` parser; tool parser `kimiK2` |
| `kimi_k25`      | Kimi K2.6               | ✅ alias of `deepseek_v3`                    | MLA (prefill-style)  | Same — native-template + JANGTQ both work. VL needs KimiVLM (follow-up). |
| `kimi_k26`      | Future Kimi K2.6 post-REAP-50 | ✅ (handled by `kimi` prefix in stamp helper) | TBD                  | |
| `deepseek_v4`   | DSV4-Flash / DSV4-Pro   | ⚠️ **NOT YET WIRED** — needs `DeepseekV4.swift` | MLA + mHC + CSA/HCA | See §"What's missing" below |

---

## Already in place (this session + prior)

- [x] `kimi_k25` / `kimi_k2` / `deepseek_v3` all route through
      `dispatchDeepseekV3Family` in `LLMModelFactory.swift`.
- [x] `deepseek_v3` alias chain also catches GLM / Nemotron / Mistral
      bundles that use DSV3 backbone.
- [x] `DeepseekV3JANGTQ.swift` — TurboQuantSwitchGLU-backed MoE for
      `weight_format == "mxtq"` bundles.
- [x] `KimiK2ToolCallParser.swift` — `<|tool_calls_section_begin|>...`
      parser for Kimi's TS-style tool output.
- [x] `ReasoningParser.fromCapabilityName("kimi"/"kimi_k2"/"kimik2")`
      → `<think>` parser.
- [x] `reasoningStampFromModelType(_:)` allowlist resolves
      `kimi*` / `deepseek*` / `qwen3*` / `glm4_moe*` / `glm5*` /
      `minimax*` / `nemotron_h` / `holo*` to `think_xml`;
      `gemma4*` → `harmony`; everything else → `none`.
      **Regression-locked** in `ReasoningStampFromModelTypeTests`.
- [x] `ToolCallFormat.infer(from: "kimi*")` → `.kimiK2`.
- [x] `ToolCallFormat.fromCapabilityName("kimi"/"kimi_k2"/"kimik2")`
      → `.kimiK2`.

---

## What's missing for DSV4-Flash (tracked; not this session)

Source of truth: `jang/research/DSV-FAMILY-RUNTIME-GUIDE.md` §2, §12,
§13, §14, §15, §21.

### Architecture — all NEW vs DSV3

| Concern                                  | Needs                                          |
|------------------------------------------|------------------------------------------------|
| **mHC (Manifold-Constrained Hyper-Conns)** | `_hc_pre` / `_hc_post` with **correct axis alignment** (§12 Bug #1) |
| **Sinkhorn iteration (20 iters)**         | Row/col normalize `comb` matrix for mHC        |
| **`_DSV4SwiGLU(limit=10.0)`**             | Clamp gate/up to ±10 before silu (§2 Bug #2)   |
| **Learned `attn_sink` per head**          | Prepend sink column before softmax, drop after. ON by default (§2 Bug #3) |
| **Inverse RoPE on attention output**      | Rotate last 64 dims of O backward via `conj(freqs_cis)` |
| **Grouped low-rank O projection**         | `o_groups=8`, reshape O to `(B, L, 8, 4096)` then einsum `wo_a` + `wo_b(flatten(...))` |
| **sqrtsoftplus scoring**                  | Replace `sigmoid` gate with `sqrt(softplus(x))`. Bias → topk → gather UNBIASED weights |
| **Hash routing (layers 0-2)**             | `num_hash_layers=3`: load `tid2eid` int64 hash table, bypass topk |
| **YaRN RoPE with DSV4 parameters**        | `rope_factor=16`, `original_seq_len=65536`, `beta_fast=32`, `beta_slow=1` |
| **`_hc_head_reduce` at top**              | Pre-norm mHC reduction using `hc_head_*` params |
| **Sliding window + compressed attention** | `sliding_window=128`, `compress_ratios[i] in {4, 128, 0}`, `Compressor` + `Indexer`. OPTIONAL for short prompts (see §9) |

### Runtime / load

| Concern                                  | Needs                                          |
|------------------------------------------|------------------------------------------------|
| FP4 (e2m1fn) routed-expert dequant       | `fp4_codec.swift` port of `fp4_codec.py`       |
| FP8 (e4m3fn) + UE8M0 128×128 dequant     | `fp8_ue8m0_codec.swift` port                   |
| Skip MTP keys (`mtp.0.*`) in sanitize    | Filter in `DeepseekV4Model.sanitize`           |
| Skip `compressor.*` / `indexer.*` keys   | Filter in sanitize (until sparse_attn ported) |
| Stack routed experts into `switch_mlp.*` | Mirror DSV3JANGTQ's stacking pattern           |

### Chat + tokenizer

| Concern                                  | Notes                                          |
|------------------------------------------|------------------------------------------------|
| **No `chat_template` in tokenizer_config** | §6: DSV4-Flash ships `encoding/encoding_dsv4.py` separately. Swift needs a port OR fallback template. |
| Special tokens `<｜User｜>` / `<｜Assistant｜>` / `<think>` | Must be treated as specials in `NaiveStreamingDetokenizer` (shrinkage-safe already after commit `3e0ec66`). |
| EOS token id = 1 (decodes to `<｜end▁of▁sentence｜>`) | TokenizerBridge auto-sets via `eos_token_id` in tokenizer_config. |

---

## Port plan (recommended sequence)

### Phase 1a — Landed 2026-04-24 (chat template + config + math helpers)

Canonical ref: `jang/research/DSV4-RUNTIME-ARCHITECTURE.md`.

- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` —
      full config struct mirroring Python `ModelArgs` (mHC, attn_sink,
      compress_ratios, o_groups, hash_layers, swiglu_limit, YaRN,
      sliding_window). Provides `isHashLayer` / `hasCompressor` /
      `ropeTheta(forLayer:)` helpers.
- [x] `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift` —
      verbatim Swift port of `encoding_dsv4.py` (744 LOC Python).
      Handles `chat` vs `thinking` modes, reasoning_effort=max
      preface, drop_earlier_reasoning multi-turn rule (forced off
      when tools present), DSML tool-call encoding, tool_result
      merging into user contentBlocks, sort_tool_results_by_call_order,
      latest_reminder role, developer role. 15 round-trip tests green.
- [x] `Libraries/MLXLLM/Models/DeepseekV4MathHelpers.swift` —
      pure-math kernels usable without model weights: `hcSplitSinkhorn`
      (20-iter row/col normalize on `comb`), `applyPartialRoPE` (with
      inverse branch for attention output), `dsv4SwiGLU(gate, up, limit)`,
      `sqrtSoftplus`, `sqrtSoftplusSelect` (bias-for-selection /
      unbiased-for-weighting per §6 bug fix), `yarnInvFreq` with
      `high = min(..., dim-1)` clamp.

### Phase 1b — Full forward + Compressor/Indexer (landed 2026-04-24)

- [x] `DeepseekV4.swift` — `DeepseekV4Attention` with corrected
      numerics: q_norm on qLoraRank (not headDim), per-head fp32
      variance-rsqrt rescale after wq_b (prevents exponential drift on
      middle layers), wo_a shape (numHeads*headDim // oGroups →
      oGroups*oLoraRank) with `einsum bsgd,grd→bsgr` for grouped O,
      per-layer compress_ratio detection from config.compressRatios
      (with DSV4-Flash default fallback), per-layer rope_theta
      (10000 / 160000) + YaRN on compressRatio>0 layers.
- [x] `DeepseekV4Compressor.swift` — `DeepseekV4Cache` (composite of
      RotatingKVCache + compressor/indexer buffer + pooled state),
      `DeepseekV4Compressor` (wkv/wgate projection + APE + window
      accumulation + overlap transform for ratio=4 + softmax
      pooling + RMSNorm + partial RoPE), `DeepseekV4Indexer` (per-
      query top-k over pooled via wq_b + weights_proj + its own
      inner Compressor).
- [x] `DeepseekV4Attention` forward wires Compressor/Indexer:
      compressor state pulled from DeepseekV4Cache or short-prompt
      fast-path when `L < compress_ratio` without persistent cache;
      Indexer top-k selected indices gather pooled keys which are
      concatenated onto local KV for SDPA; mask extended to cover
      the extra pooled columns.
- [x] `DeepseekV4Model.newCache` + `DeepseekV4JANGTQModel.newCache`
      return per-layer `DeepseekV4Cache` so Compressor/Indexer get
      persistent buffer state across calls.
- [x] `sanitize` KEEPS `compressor.*` / `indexer.*` keys, remapping
      under `model.layers.L.self_attn.{compressor,indexer}.*`.
- [x] Factory dispatch live: `deepseek_v4` → `DeepseekV4Model`;
      `weight_format=mxtq` → `DeepseekV4JANGTQModel` with
      `TurboQuantSwitchGLU` for routed experts.
- [x] All 13 EXHAUSTIVE-VARIABLES-GUIDE §1 bug fixes encoded.

### Phase 2 — JANGTQ refinements (follow-up)

Known approximation in Phase 1b JANGTQ variant: the fused
`fusedGateUpSwiGLU` kernel applies plain silu(gate)*up without the
`swiglu_limit=10` clamp. MXTQ codebook's natural boundedness
regularizes routed experts; shared experts still clamp via
DeepseekV4MLP. Adding limit support to the fused kernel or switching
to per-projection gather (3× dispatch cost) is tracked for Phase 2
if coherence checks fail at Phase 3.

### Phase 1c — Minimal text forward (prefill-style, no optimization)
1. `DeepseekV4Configuration` — parse all new fields (mHC, attn_sink,
   compress_ratios, hash_layers, sliding_window, o_groups).
2. `DeepseekV4SwiGLU` with `swiglu_limit=10.0` clamp.
3. `DeepseekV4Attention` — inherit DSV3 MLA contract but add:
   - `attn_sink` per-head prepended logit
   - inverse RoPE on output
   - grouped low-rank O projection via `einsum`
   - fp32 SDPA cast at L==1 (§2 Bug #3 + GLM-5.1 MLA fix)
4. `DeepseekV4MoEGate` — sqrtsoftplus + noaux_tc bias + hash routing
   for first 3 layers (requires loading `tid2eid`).
5. `DeepseekV4MoE` — SwitchGLU with DSV4SwiGLU activation.
6. `DeepseekV4DecoderLayer` — mHC `_hc_pre` / `_hc_post` wrapping
   attention and MoE.
7. `DeepseekV4Model` — ties layers + `_hc_head_reduce` at top.
8. Factory dispatch: `"deepseek_v4"` → `DeepseekV4Model`.
9. Port chat encoder OR ship a minimal `DeepseekV4.jinja` fallback in
   `ChatTemplates/`.

### Phase 2 — JANGTQ variant
10. `DeepseekV4JANGTQ.swift` — swap SwitchGLU for
    `TurboQuantSwitchGLU(..., activation: DSV4SwiGLU(10.0))`.
11. Factory dispatch: `"deepseek_v4"` + `weight_format == "mxtq"`
    → `DeepseekV4JANGTQModel`.

### Phase 3 — Validation
12. Port `diff_one_block.py` → Swift test with synthetic random
    weights. Expect max abs diff < 0.05 vs torch reference.
13. Real-model bench on Mac Studio 256 GB — JANGTQ4-Flash (185 GB).
14. Gate on `KimiK2.6-JANGTQ_1L` coherence harness equivalent.

### Phase 4 — Optimization (follow-up)
15. MLA L==1 absorb branch (optional ~1.5× decode speedup).
16. Compressor + Indexer for long-context (beyond window=128).
17. Sparse attention Metal kernel port.

---

## Explicit non-goals this session

- DSV4 Python fp32 SDPA patch — **not our concern**; runs in
  `jang_tools.kimi_prune.runtime_patch`, refuses to touch
  `vmlx-swift-lm/`.
- DSV4 VL / MoonViT — separate port, gated on text coherence.
- `Compressor` / `Indexer` sparse attention — prompts < 128 tokens
  use full attention safely per §9.

---

## Test coverage

| Concern                           | Test suite                                     |
|-----------------------------------|------------------------------------------------|
| Reasoning stamp allowlist         | `ReasoningStampFromModelTypeTests` (13 tests) |
| Tool format inference             | `KimiK25RoutingTests` + `ToolTests`           |
| Kimi parser grammar               | `ToolTests` (KimiK2 character-stream)         |
| JANGTQ mxtq dispatch              | `KimiK25RoutingTests.testKimiK25Mxtq*`        |
| DSV4 arch — TBD                   | follow-up (synthetic diff harness port)       |

---

## When in doubt

1. **Don't delete anything** in `DeepseekV3*.swift` — DSV3 family is
   production-active.
2. **Use `DeepseekV4*.swift` for DSV4-specific code.** Share via
   helpers or protocols at module level, don't subclass.
3. **Follow `DSV-FAMILY-RUNTIME-GUIDE.md` §18 checklist** before
   blaming model quality on the Swift port.
4. **Run the Python reference first** (`diff_one_block.py`
   `DIFF_REAL=1`) to rule out quantization before chasing arch bugs.
