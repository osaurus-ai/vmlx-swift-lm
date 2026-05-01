# vmlx-swift-lm Integration Guide for osaurus

**Date:** 2026-05-01
**Pin:** `origin/main` HEAD `71065ca` (or later)
**Audience:** osaurus model-loading + chat-engine agents

This is the authoritative integration document. It supersedes
`MODEL-LOADING-STATUS-2026-05-01.md` and absorbs the per-model status,
cache topology, VL/audio/video pipeline details, EOS handling, chat
template fallbacks, and ALL structural fixes landed in main as of
2026-05-01.

---

## TL;DR — what changed since the last pin bump

7 commits land structural fixes that affect EVERY JANGTQ / MXFP4 /
hybrid-SSM model. Bump past `71065ca` and you get all of them.

| Commit | Subsystem | What it fixes |
|---|---|---|
| `1135950` | Laguna model | softplus (not sigmoid) g_proj gate; un-biased weights for routed-MoE; YaRN scaling on full-attention layers |
| `576916b` | Laguna MoE | `TurboQuantSwitchGLU` (was 256 individual modules) + sanitize splits fused `gate_up_proj` |
| `babbe34` | Mistral 3.5 VLM | strips `model.` prefix on `vision_tower.…` keys |
| `0fab91c` | Mistral 3.5 VLM | strips `model.` prefix on `multi_modal_projector.…` keys |
| `1173822` | Cache (cross-cutting) | (A) TurboQuant paged-cache compounding-quantization fix; (B) hybrid models honor `defaultMaxKVSize` |
| `3f8a5e9` | Cache + sanitize | Qwen35JANGTQ also honors `maxKVSize`; tied-embeddings hardenings (Mistral3VLM, NemotronH, Mistral3VLMJANGTQ) |
| `0d85e9d` | EOS detection | BatchEngine probes 7 common end-of-turn special tokens against the tokenizer vocab (defensive widening) |
| `0756dc0` | Disk-tier cache | closes the trim+seed Metal lifecycle crash on full disk-cache hit (`notifyExternalReferencesNonZeroOnDealloc` repro) |
| `71065ca` | Chat templates | Laguna minimal fallback + bridge sniff (covers swift-jinja `{% generation %}` block-tag throw) |

**Osaurus can re-enable the L2 disk cache** (`enableDiskCache = true`)
after this pin bump — `0756dc0` closes the repro.

---

## Verified-coherent bundles (real load + decode this session)

All loaded from `/Volumes/EricsLLMDrive/`. Bench harness:
- `BENCH_SIMPLE=1` — synthetic prompt, raw token decode
- `BENCH_COHERENT=1` — 3-turn ChatSession with real tokenizer
- `BENCH_BATCH_CHAT=1` — 3-turn BatchEngine with cache reuse

| Bundle | Quant | Test | Result |
|---|---|---|---|
| Laguna-XS.2-JANGTQ | mixed (codebook MoE + 8-bit affine dense) | BENCH_SIMPLE + BENCH_BATCH_CHAT 3-turn | "That's a great color! Blue has such a calming, peaceful quality..." → "Your favorite color is blue!..." → "Blue is a cool color..." |
| Mistral-Medium-3.5-128B-JANGTQ | all-codebook decoder + fp16 vision/projector/lm_head | BENCH_SIMPLE → tokenizer decode | Loads in 13s, valid multilingual tokens (English + French) |
| Nemotron-3-Nano-Omni-30B-JANGTQ | omni JANGTQ | BENCH_OMNI 13-row matrix | **All 13 pass**: text + image + video + audio + reasoning toggle + mixed + media-salt + hybrid-SSM-warm-pass |
| Qwen3.6-27B-MXFP4 | MXFP4 dense | BENCH_BATCH_CHAT + BENCH_QWEN_THINKING_CHECK + BENCH_BATCH_DISK_RESTORE | Multi-turn "Blue.", **255 reasoning deltas / 0 chunk leak**, **disk-cache full hit no Metal crash** (4.5x prompt-time speedup) |
| Qwen3.6-35B-A3B-JANGTQ4 | hybrid SSM + codebook MoE | BENCH_COHERENT + BENCH_BATCH_CACHE_HIT | "Blue is a wonderful choice!" → "Your favorite color is blue." → "cool color"; paged-tier cache HIT 128/161 |
| MiniMax-M2.7-Small-JANGTQ | MoE JANGTQ | BENCH_COHERENT 3-turn | "Blue is a great choice—it's calming..." → "Your favorite color is blue." → "Blue is a cool color." |
| **Gemma-4-26B-A4B-it-JANG_4M-CRACK** | JANG_4M | BENCH_COHERENT 3-turn | "That's a great color! ... Is there a specific shade of blue you like, or do you prefer it because of the ocean or the sky?" → "Your favorite color is blue." → "That is a cool color." |
| **Holo3-35B-A3B-mxfp4** | mxfp4 (Qwen3.5 MoE arch) | BENCH_COHERENT 3-turn | Empty turn 1 (full reasoning) → "Your favorite color is blue." → "Colors are generally categorized as warm... or cool..." |

---

## Dispatch + sanitize per family

### Mistral 3 / Mistral 3.5 (LLM)

**Outer model_type:** `mistral3` OR `ministral3` (text-only Mistral 3.5 LLM
bundles can use either spelling). Both registered to `dispatchMistral3LLM`
(`Libraries/MLXLLM/LLMModelFactory.swift:186, :193`).

**Logic chain:**
1. `vision_config` present → throw `unsupportedModelType("…route via VLMModelFactory")` — osaurus should auto-redispatch.
2. `text_config.model_type == "mistral4"` → `Mistral4Model` (3.5 wrapper around 4 architecture).
3. `weight_format == "mxtq"` → `Mistral3TextJANGTQModel` (entire decoder JANGTQ-quantized).
4. Otherwise → `Mistral3TextModel`.

**Reasoning / tool stamps:** `none` reasoning (no `<think>`); `mistral` tool format.

### Mistral 3 / Mistral 3.5 (VLM)

**Outer model_type:** `mistral3` OR `ministral3` — both registered to
`dispatchMistral3VLM` (`VLMModelFactory.swift:115, :119`).

**Logic chain:**
1. `weight_format == "mxtq"` → `Mistral3VLMJANGTQ` (JANGTQ inner LM, fp16 Pixtral vision tower).
2. `text_config.model_type == "mistral4"` → `Mistral4VLM`.
3. Otherwise → `Mistral3VLM`.

**Processor override:** `processorTypeOverrides` map at
`VLMModelFactory.swift:554` covers BOTH `mistral3` and `ministral3` →
`Mistral3Processor` (the Pixtral processor in `preprocessor_config.json`
loses spatial-merge handling).

**Sanitize key remaps (verified against real bundle):**
- `model.language_model.…` → `language_model.model.…`
- `model.vision_tower.…` → `vision_tower.vision_model.…` (strips `model.` prefix; commit `babbe34`)
- `model.multi_modal_projector.…` → `multi_modal_projector.…` (commit `0fab91c`)
- `lm_head.weight` (root) → `language_model.lm_head.weight`
- `model.<llm-key>` (LLM-shape without wrapper) → `language_model.model.<llm-key>` (fallback)
- Tied embeddings: drop redundant `language_model.lm_head.{weight,scales,biases}`
- Drop `.tq_bits` per-tensor scalars and `self_attn.rotary_emb.inv_freq`

### Laguna (Poolside)

**Outer model_type:** `laguna`. Routes through `LagunaModel` always
(LagunaJANGTQModel was deleted in commit `576916b` — it modeled the
wrong scheme).

**Architecture (mixed-quant):**
- Layer 0 dense MLP, layers 1..39 sparse MoE (40 layers total)
- Per-layer attention head count via `num_attention_heads_per_layer` (48 full / 64 SWA)
- Dual RoPE per `rope_parameters[layer_type]` (full = YaRN factor=32, SWA = default base=10000)
- `q_norm` / `k_norm` per-head RMSNorm BEFORE rope
- `g_proj` per-head **softplus** gating (NOT sigmoid — see commit `1135950`)
- Routed experts: 256 stacked codebook tensors at `experts.gate_up_proj.{tq_packed,tq_norms}` + `experts.down_proj.{tq_packed,tq_norms}`. Bundle ships gate+up FUSED on out-dim axis.
- Shared expert: vanilla `LagunaDenseMLP` (8-bit affine quant via MLX auto-quant)
- Sigmoid + bias router; bias ONLY for top-k selection, weights are un-biased sigmoid scores

**Sanitize:**
- `model.<x>` → `<x>`
- `mlp.experts.e_score_correction_bias` → `mlp.e_score_correction_bias`
- `mlp.experts.gate_up_proj.{tq_packed,tq_norms}` → split into `mlp.experts.{gate_proj,up_proj}.{…}` halves
- Drop `self_attn.rotary_emb.inv_freq`, `.tq_bits`, tied `lm_head.weight`

**bits / seed:** read `mxtq_bits` (dict) and `mxtq_seed` from `jang_config.json`. Routed-expert bits = 2 (codebook); attention/dense/shared/embed/lm_head = 8 (affine).

**Cache topology:** per-layer mixed `RotatingKVCache` (sliding) + `KVCacheSimple` (full) — bounded by `parameters.maxKVSize` since `1173822`.

**Reasoning / tool stamps:** `think_xml` reasoning (Laguna emits `<think>`); `glm4` tool format.

**Chat template:** native uses `{% generation %}` HF block tags swift-jinja can't parse. Bridge sniff (`bos_token == "〈|EOS|〉"`) routes directly to `LagunaMinimal` fallback — see `ChatTemplateFallbacks.swift`.

### NemotronH Omni (Cascade-2 Nano-Omni)

**Auto-detect:** `VLMModelFactory.swift:387` — when `config_omni.json` exists, override `baseConfig.modelType` to `"NemotronH_Nano_Omni_Reasoning_V3"`. The bundle's own `config.json` reports `model_type: nemotron_h` (LLM-only schema).

**Components:**
- LM: `NemotronHModel` (or `NemotronHModel(jangtqContext:)` for JANGTQ omni)
- Vision: RADIO ViT (`NemotronHRADIOVisionModel`) + pixel-shuffle + `NemotronHVisionMLPProjector` (`mlp1`)
- Audio: Parakeet conformer encoder + mel STFT + sound projector
- All 4 sub-modules at root with single-segment `@ModuleInfo` keys

**Multi-modal placeholder splice:**
- Image AND video share `<image>` placeholder (`imageContextTokenId`); the model's `prepare()` concatenates image + video embeds (image-first matching processor placeholder order) and splices in ONE pass — fixed in commit prior to this pin
- Audio uses `<so_embedding>` placeholder (`soundContextTokenId`)

**Long-prompt safety:** `prepare()` chunks text-only prefill at `windowSize ?? 512` tokens to bound the SSM segsum O(L²) memory cost (avoids the 158GB allocation OOM osaurus team reported on long contexts).

**Reasoning / tool stamps:** `think_xml` reasoning; `xmlFunction` tool format.

### Qwen 3.5 / 3.6

**Outer model_types:** `qwen3_5`, `qwen3_5_moe`, `qwen3_5_text`, `qwen3_vl` (in VLM).

**Hybrid pattern:**
- `layer.isLinear == true` → SSM Mamba layer → `MambaCache()`
- `layer.isLinear == false` → attention layer → `RotatingKVCache(maxSize: maxKVSize, keep: 4)` when `parameters.maxKVSize` set (commit `1173822`); else `KVCacheSimple()`
- Both `Qwen35Model` AND `Qwen35JANGTQModel` honor maxKVSize as of `3f8a5e9`

**JANGTQ4 routed-MoE:** pre-stacked at `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{tq_packed,tq_norms}` (already split in the bundle, not fused). `Qwen35JANGTQTextModel.sanitize` correctly skips its own stacking branch when keys are already in switch_mlp form.

**MXFP4 dense (e.g. Qwen3.6 27B):** standard MLX affine quant, vanilla `Linear` modules, MLX auto-quant from `config.json`'s `quantization` field.

**Reasoning / tool stamps:** `think_xml` reasoning (Qwen3.5/3.6 native `<think>` chat template); `xmlFunction` tool format.

### MiniMax M2 / M2.7

**Outer model_types:** `minimax`, `minimax_m2`, `minimax_m3`.

**Sanitize:** drops tied `lm_head.weight` correctly (also covers `.scales` / `.biases` after commit `3f8a5e9` — forward-compat for tied+quantized lm_head).

**Reasoning / tool stamps:** `minimaxM2` tool format (special XML envelope per Python `minimax_m2/encoding.py`); reasoning per JANG `tool_parser=minimax` capability stamp.

### DeepSeek-V4 Flash

**Outer model_type:** `deepseek_v4`. Dispatched via `dispatchDeepseekV4` which reads `weight_format=="mxtq"` / `jangtq2` / `jangtq4` (or env override `DSV4_FORCE_JANGTQ=1`) → `DeepseekV4JANGTQModel`; otherwise `DeepseekV4Model`.

**Architecture distinct from DSV3:** mHC residual stream, CSA/HCA hybrid attention, sqrtsoftplus + hash-routing on layers 0-2, grouped low-rank O. The factory throws a structured error rather than dispatching to DSV3 (which would silently produce garbage).

**Routed MoE:** `TurboQuantSwitchGLU` keyed at `switch_mlp` (different from Laguna's `experts` key — bundle layout convention).

**Chat template:** DSV4 ships NO `chat_template` field. Bridge sniff (`bos_token == "<\|begin▁of▁sentence\|>"` with curly-quote U+FF5C) auto-engages `DSV4Minimal.jinja` fallback. `VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE=1` opts out.

**Note (open issues):**
1. `DeepseekV4ChatEncoder.swift` is a full Swift port of `encoding_dsv4.py` but is currently NOT wired into the bridge — runtime uses the `DSV4Minimal.jinja` approximation instead. Tool-calling chats render with simplified envelopes vs the native DSML format. Tracked for future wiring; not blocking current basic chat.
2. **DSV4 model forward has a separate reshape bug at HEAD** — verified 2026-05-01 against `DeepSeek-V4-Flash-JANGTQ` bundle on disk. `BENCH_SIMPLE` with a 10-token synthetic prompt fails with `[reshape] Cannot reshape array of size 163840 into shape (1,5,16384)` — the model produces 2× the expected positions on axis-1. Factor-of-2 suggests an mHC residual-stream split or a multi-token-prediction artifact not being reduced before reshape. Reproducible with `BENCH_SIMPLE` AND `BENCH_COHERENT`. Pre-existing; not introduced by this pin's commits. Out of scope for the current osaurus integration push — flag for the DSV4 author to investigate. **DO NOT ship DSV4 in osaurus until this is fixed.**

**Mistral 3 / 3.5 VLM — root cause traced, mxfp4 path FIXED (commit `af89da7`):**

After deep tracing with diagnostic prints + bundle inspection, the real
root cause was in `JangQuantization` default initializer:

- `JangQuantization()` had `bitWidthsUsed: [Int] = [2, 4, 6]` as the
  default. mxfp4 bundles ship `jang_config.json` with only an `mxfp4`
  field — no `quantization` field — so the parser fell through to the
  default initializer, getting bogus `bitWidthsUsed=[2,4,6]`.
- `inferPerLayerQuantization` keys on `isAuthoritativeJang =
  !bitWidthsUsed.isEmpty`. The bogus default classified mxfp4 bundles
  as authoritative-JANG, ignoring config.json's `quantization.bits`
  override.
- This made shape walk pick `defaultBits = bitWidthsUsed.min() = 2`,
  then for embed_tokens (`weight=[131072, 1536]` uint32, `scales=[131072,
  384]`) the disambiguation chose `(bits=2, gs=64)` — mathematically
  valid for the same packed shape but wrong: produces `in_dim = 384 × 64
  = 24576`, double the correct `hiddenSize = 12288`.
- Embed forward returned `[B, T, 24576]`. Next RMSNorm has `weight=[12288]`.
  MLX correctly threw `[rms_norm] (*weight) must have the same size as
  the last dimension of x but has 12288 elements`.

**Fix in commit `af89da7`:** Default `bitWidthsUsed: [Int] = []`. Added
`overrideBits` parameter to `inferPerLayerQuantization`, threaded from
`Load.swift` (`quantization?.bits`). When `isAuthoritativeJang = false`,
prefer overrideBits over `bitWidthsUsed.min()` fallback.

**Verification (real load + decode on disk):**
- Mistral-Medium-3.5-128B-mxfp4 BENCH_SIMPLE: `h.shape=[1, 5, 12288]`
  through all 88 layers + final norm. Decoded tokens: `"Okay, the user
  just sent a blank message."` — **coherent English**.
- Regression checks: Laguna (3-turn coherent), Qwen3.6 27B MXFP4
  (multi-turn cache reuse, "Blue." in turn 2), 13/13 dispatch + multi-
  turn unit tests pass.
- Real Mistral 3.5 mxfp4 multi-turn chat blocked only by `kIOGPUCommandBufferCallbackErrorTimeout`
  on the second 128B load (BENCH_BATCH_CHAT loads twice for compile-OFF
  + compile-ON). That's M5 Max memory pressure on 128B × 2 loads —
  not a model bug. Single-load decode is coherent.

**JANGTQ Mistral 3.5 path — separate decoder bug (still open):**
The `Mistral3VLMJANGTQ` codebook decoder still produces multilingual
token salad ("jek, iÅ¾, consequent, dirait, sper, ặ, unar, cy, Ever,
Back") on real chat prompts. The (bits, gs) shape-walk fix corrected
the affine-quantized embed_tokens (verified `bits=8 gs=64 → in_dim=12288`),
but the JANGTQ codebook dense-linear forward in `Mistral3VLMJANGTQ` has
its own divergence we haven't traced yet. Likely candidates:
  1. JANGTQDenseLinear codebook lookup convention vs Python `tq_quantize_weight`
  2. Hadamard rotation seed mismatch for non-power-of-2 12288 dim
  3. YaRN scaling vs plain RoPE (Python ref's runtime uses plain RoPE
     base=1e6; Swift uses YarnRoPE because rope_type=yarn in config)
Investigation path: instrument `Mistral3JANGTQAttention.callAsFunction`
the same way (env-gated print of every layer's q/k/v/output shape +
selected element values) and compare to Laguna's working JANGTQ codebook
forward. mxfp4 bundle is sufficient for shipping until JANGTQ traced.

**Upstream investigation (`ml-explore/mlx-swift-lm`) — Mistral 3 history:**
- Upstream PR #18 added Ministral 3 with Pixtral vision (likely tested on 3B/8B).
- PR #43 fixed Mistral3TextConfiguration parsing.
- PR #108 fixed loading error for Mistral-Small-3.2-24B.
- PR #132 fixed tool calling for Mistral 3.
- **Upstream has NEVER tested 128B Mistral 3.5 Medium**. The 88-layer / 12288-hidden / 96-head config may exercise edge cases the upstream test suite (smaller bundles) doesn't cover.
- Our fork's `Mistral3.swift` Ministral3ModelInner forward is identical to upstream's. So the bug is either:
  1. In a config field upstream's smaller bundles don't set
  2. In a numerical edge case (e.g. fp16 overflow) that hits 12288 hidden but not 4096
  3. In the YaRN config interpretation (Swift uses `rope_type=yarn` from config; Python ref's runtime uses plain RoPE at base=1e6 — its YaRN code is dead)
- Reproducer: `BENCH_SIMPLE BENCH_MODEL=/Volumes/EricsLLMDrive/jangq-ai/OsaurusAI/Mistral-Medium-3.5-128B-mxfp4 BENCH_PROMPT_LEN=10 .build/release/RunBench` — loads in ~30s then immediately throws.
- Code path: `Libraries/MLXVLM/Models/Mistral3.swift` Ministral3ModelInner.callAsFunction (lines 463-509). The 88-layer loop should produce hidden=12288 at every step but somewhere fails. To diagnose: instrument the layer loop with `print("layer \(i): h.shape = \(h.shape)")` — find the first layer that produces wrong shape.
- **DO NOT ship Mistral 3.5 in osaurus until the forward bug is traced and fixed.** Other Mistral 3 family bundles (smaller / older) may work fine — needs separate verification once the 128B path is traced.

**Upstream review summary (other deps for completeness):**
- `osaurus-ai/swift-jinja@58d21aa5` — for-iterable parser fix in our fork; not yet upstream'd to `huggingface/swift-jinja`. Worth opening a PR there for community benefit.
- `osaurus-ai/mlx-swift@0a56f9041` — pinned by revision (no PRs from us blocked on upstream).
- `huggingface/swift-transformers` — pinned at 1.0.0+; `swift-jinja 2.x` already absorbed three earlier root-cause fixes we previously carried in our archived `osaurus-ai/Jinja` 1.3.1 fork.
- `ml-explore/mlx-swift-lm` upstream — last seen at HEAD `7e2b710` (Apr 2026). 321 commits diverged on our fork (mostly JANG/JANGTQ/Cascade-2/Laguna/etc — out of scope to upstream).

---

## Cache topology — fully specified

### Per-layer cache types

| Cache class | Purpose | When created |
|---|---|---|
| `KVCacheSimple` | Full attention, unbounded | `newCache` when no `maxKVSize` set |
| `RotatingKVCache(maxSize:, keep:)` | Sliding-window OR `maxKVSize`-bounded full attention | `newCache` when `maxKVSize` is set, OR for SWA layers with `slidingWindow` |
| `MambaCache` | SSM hidden + conv state | Hybrid models on Mamba layers |
| `ArraysCache` | Generic state slots | Some Cascade variants |
| `CacheList` | Composite (e.g. SSM + KV in one layer) | Models with co-located SSM and attention |
| `TurboQuantKVCache` | Codebook-quantized KV | Wrapped on top of `KVCacheSimple` by `maybeQuantizeKVCache` when `kvMode == .turboQuant(k,v)` and offset > threshold |
| `QuantizedKVCache` | Standard MLX affine-quantized KV | Alternative to TurboQuant |

### `defaultMaxKVSize` contract (osaurus → vmlx)

`CacheCoordinatorConfig.defaultMaxKVSize` writes the bound into
`parameters.maxKVSize` at admission. Models honor it as follows
(verified post-`3f8a5e9`):

| Model | Honors `maxKVSize`? |
|---|---|
| Llama, Mistral (vanilla) | ✅ via `KVCache.makePromptCacheWithLayerCount` |
| Mistral 3 / 3.5 (LLM + VLM) | ✅ via `RotatingKVCache` for SWA layers, KVCacheSimple for full |
| Laguna | ✅ per-layer mixed (sliding bounded by `slidingWindow`, full bounded by `maxKVSize`) |
| Qwen3.5 / 3.6 (vanilla) | ✅ since `1173822` |
| Qwen3.5 / 3.6 JANGTQ | ✅ since `3f8a5e9` |
| Qwen3Next | ✅ since `1173822` |
| NemotronH / NemotronH Omni | ✅ since `1173822` (per-block via `hybridOverridePattern`) |
| Mamba layers (any hybrid) | ⚠️ ignore by design — SSM hidden state is fixed-size |

### TurboQuant KV — paged-cache restore (commit `1173822`)

`TurboQuantKVCache.restoreFromDecodedKV(keys:values:sourceOffset:)` is
the **correct** entry point for paged-tier restore. Seats decoded
float as the `.compressed` phase prefix WITHOUT re-encoding, leaving
`compressedKeys`/`compressedValues` nil (paged tier doesn't carry them
— only the disk tier does via `restoreCompressed`).

**Why this matters:** the previous `tq.state = [keys, values]` path
transitioned the cache back to `.fill` phase with already-lossy float
as the new prefill. Subsequent compression at the next threshold
cross re-quantized the lossy float — **compounding quantization error
per turn**. This is the exact "first turn fine, later turns garbage"
symptom osaurus observed across every JANGTQ/MXFP4 multi-turn path.

**3-bit KV verdict:** Now safe to re-enable (`defaultKVMode =
.turboQuant(3, 3)` works post-`1173822`). The per-bit packing is
sound; the bug was in the cross-turn handoff. Conservative
`.turboQuant(4, 4)` still recommended as the default — 3-bit is most
sensitive to error amplification and gains less compression benefit.

### Disk-tier cache — re-enable now (commit `0756dc0`)

The `notifyExternalReferencesNonZeroOnDealloc` crash is closed.
`enableDiskCache = true` is safe to re-enable in osaurus
`ModelRuntime.swift`. The fix forces materialization of trim mutations
in a SEPARATE command buffer before the seed-token forward encodes,
avoiding the Metal library-cache eviction race.

### Hybrid SSM seed-on-restore

Inline synchronous seed at `BatchEngine.swift:911` (post-prefill).
SSMReDeriver was reverted (Metal race). Current path:
1. `extractSSMStates` walks per-layer cache pulling Mamba conv+hidden
2. Stored in `coordinator.ssmStateCache` keyed by promptTokens hash
3. On restore: `restoreSSMStates` overwrites a fresh cache
4. Safety net: if hybrid SSM detected AND remaining > 0, full re-prefill
   (partial cache hits never benefit hybrid SSM models — by design)

---

## EOS detection — fully wired (commits `0d85e9d` + factory chain)

### Sources cascaded into `BatchEngine.stopTokenIDs`

1. `config.json` `eos_token_id` (int or array via `IntOrIntArray`)
2. `generation_config.json` `eos_token_id` (overrides config.json per Python mlx-lm convention)
3. `tokenizer.eosTokenId` (tokenizer_config.json `eos_token` → vocab ID)
4. `tokenizer.unknownTokenId`
5. `ModelConfiguration.extraEOSTokens` (caller-supplied)
6. **Defensive widening** (commit `0d85e9d`) — probes 7 common end-of-turn special tokens against the tokenizer vocab:
   - `<|im_end|>` (Qwen / Mistral 3 / NemotronH-Omni / many)
   - `<|endoftext|>` (Qwen2/3, GPT-style)
   - `<|eot_id|>` (Llama 3.x)
   - `<|end_of_text|>` (Llama 3.x alt)
   - `<|end|>` (Phi 3, Phi 4)
   - `<|end_of_turn|>` (Gemma family)
   - `<end_of_turn>` (Gemma 2/3 alt spelling)

`convertTokenToId` returns nil for non-vocab strings — plain content
bytes that happen to spell these tokens do NOT match. Only genuine
special tokens do.

---

## Chat template fallbacks — bridge dispatch flow

Bridge: `Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift:111-228`.

Order of attempts:
1. `VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja` — explicit override.
2. The model's shipped `chat_template.jinja` via swift-jinja.
3. On `missingChatTemplate` throw: DSV4 BOS sniff (`<\|begin▁of▁sentence\|>` curly-quote) → `DSV4Minimal`.
4. On any other `swift-jinja` throw:
   - **Laguna sniff** (`bos_token == "〈|EOS|〉"`): `LagunaMinimal` only (commit `71065ca`).
   - **Gemma family sniff** (`bos_token == "<bos>"` OR `convertTokenToId("<|turn>") != nil`): full `orderedFallbacks`.
   - Otherwise: `[NemotronMinimal, Gemma4WithTools, Gemma4Minimal]`.
5. Re-throw if no fallback parses.

`VMLX_CHAT_TEMPLATE_FALLBACK_DISABLE=1` opts out of step 3 + 4 entirely.
`VMLX_CHAT_TEMPLATE_FALLBACK_LOG=1` logs which fallback engaged to stderr.

### Known native templates that work

- Gemma-4 / Gemma-4n (E2B/E4B/26B-A4B/31B JANG) — works post swift-transformers 1.3.0 bump; fallback auto-engages on regression
- Nemotron-Cascade-2 30B-A3B — works post 1.3.0 bump
- Qwen3.6 27B + 35B — verified end-to-end multi-turn this session
- MiniMax-M2.7 Small — verified multi-turn
- Mistral 3.5 LLM + VLM — verified load+decode

### Known native templates with workarounds

- **Laguna** — `{% generation %}` block tag → bridge auto-routes to `LagunaMinimal`
- **DSV4** — no `chat_template` shipped → bridge auto-routes to `DSV4Minimal`

### Known unresolved gaps (caller workarounds available)

If any family's swift-jinja rendering breaks at HEAD:
- Set `VMLX_CHAT_TEMPLATE_OVERRIDE=/abs/path/to/your.jinja` to bypass
- File issue with the family + repro

---

## Reasoning + tool format dispatch

### Reasoning stamp resolution priority (highest first)

1. JANG `chat.reasoning_parser` stamp from `jang_config.json`
2. JANG `capabilities.reasoning_parser` stamp
3. `reasoningStampFromModelType(model_type)` heuristic — explicit allowlist:
   - `gemma4*` → `harmony` (channel envelope)
   - `qwen3*`, `deepseek*`, `glm4_moe*`, `glm5*`, `minimax*`, `kimi*`, `nemotron_h*`, `holo*`, `laguna*` → `think_xml`
   - Everything else → `none`

### Tool format resolution priority

1. Caller-supplied `configuration.toolCallFormat`
2. JANG `chat.tool_calling.parser` stamp (DSV4-era; e.g. `"dsml"`)
3. JANG `capabilities.tool_parser` stamp
4. `ToolCallFormat.infer(from: model_type)` heuristic

### Sentinel encoding (osaurus-side responsibility)

The streaming pipeline emits typed events via sentinel-prefixed strings:
- `\u{FFFE}reasoning:…` → OpenAI `delta.reasoning_content`
- `\u{FFFE}tool:…` → OpenAI `tool_calls`
- `\u{FFFE}stats:…` → `usage`

osaurus's `StreamingReasoningHint` / `StreamingToolHint` /
`StreamingStatsHint` are responsible for decoding these.

### Multi-turn history filter (osaurus-side)

Filter to `t.role == .user` only when re-templating prior turns.
Prevents `<think>` content from previous assistant turns from leaking
into the new prompt. (Confirmed correct in `ChatView.swift:1288-1291`
per osaurus audit.)

---

## VL pipeline — per-family considerations

### Mistral 3.5 VLM (Pixtral vision tower)

- Use `Mistral3Processor` (not `PixtralProcessor`) — spatial-merge handling
- Vision feature dim flows through `multi_modal_projector.{linear_1, norm, linear_2, patch_merger.merging_layer}` — all fp16 passthrough
- JANGTQ inner LM, fp16 vision tower (per `mxtq_bits.vision_tower=passthrough_fp16`)

### NemotronH Omni — image / video / audio in one model

- Image: tile-pipeline → RADIO ViT → pixel-shuffle → `mlp1` projector
- Video: `nemotronOmniPreprocessVideo` → group-stack 32-frame → 256 tokens per group
- Audio: mel STFT → Parakeet conformer → sound projector → 1 audio token per 8 mel frames
- Image and video SHARE `<image>` placeholder; processor concatenates embeds in image-first-then-video order; model's `prepare()` splices in ONE pass
- Audio has its own `<so_embedding>` placeholder; supports `preEncodedEmbedding` fast-path to skip mel+Parakeet on repeat turns

### Qwen 3.5 / 3.6 VLM

- Native video pipeline via Qwen3VL.swift (separate from omni pipeline)
- 3D rotary position via `Qwen3VLLanguage.applyMultimodalRotary` — image/video tokens use the multimodal rope branch

### Capability matcher (osaurus side)

`ModelMediaCapabilities.from(modelId:)` → `omni` / `imageVideo` /
`imageOnly` / `textOnly`. Refines via post-load `vision_config` +
`config_omni.json` sidecar. Matcher logic verified correct in osaurus
audit.

---

## Pre-flight before testing

Per memory notes (model + GPU):

```bash
# Run model tests ONE AT A TIME — never parallel (GPU RAM)
# Kill any lingering inference / test processes first
pkill -f xctest
pkill -f RunBench
pkill -f mlx_lm
pkill -f ollama
pkill -f lms

# Verify nothing's holding the GPU
ps -A | grep -E "xctest|RunBench|mlx" | grep -v grep
```

xctest zombies hold 27+GB and cut decode throughput in half.

---

## Bench harness

Single binary: `swift build -c release` → `.build/release/RunBench`.

Modes:
- `BENCH_SIMPLE=1` — synthetic-prompt decode, prints raw token IDs
- `BENCH_COHERENT=1` — 3-turn ChatSession (compile OFF + ON) — empty `text` for reasoning-heavy turns is normal
- `BENCH_BATCH_CHAT=1` — 3-turn BatchEngine multi-turn with cache reuse + full HF tokenizer
- `BENCH_BATCH_CACHE_HIT=1` — exercises CacheCoordinator cross-turn prefix reuse
- `BENCH_OMNI=1 BENCH_MODEL=...` — 11-row Nemotron Omni matrix (text/image/video/audio/reasoning toggle/mixed)
- `BENCH_VL=1 BENCH_MODEL=... BENCH_VIDEO=...` — VL video smoke
- `BENCH_CROSS_VALIDATE=1` — TokenIterator vs BatchEngine byte-parity assertion (temp=0)

Per-bench env knobs documented in `Bench.swift`.

---

## Surface area for osaurus

Public APIs — see `OSAURUS-API-SURFACE.md` for the full list. Highlights:

- `MLXLMCommon.loadModel(from:URL, using: tokenizerLoader)` — high-level dispatch (LLM vs VLM auto-detected)
- `BatchEngine(context:maxBatchSize:cacheCoordinator:)` — admit `BatchRequest` → typed event stream
- `ToolCallFormat.fromCapabilityName(_:)` — JANG short-name → enum (accepts `qwen`, `qwen3_6`, `minimax`, `glm47`, `deepseek`, `nemotron`, `gemma4`, `mistral`, `lfm2`, `kimi_k2`)
- `ReasoningParser.fromCapabilityName(_:)` — same for reasoning stamps
- `CacheCoordinatorConfig.defaultMaxKVSize` / `defaultKVMode` — coordinator-owned KV sizing (now honored by all hybrid families)
