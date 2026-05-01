# Model Loading Status — Laguna / Mistral 3.5 / NemotronH Omni

**Date:** 2026-05-01
**Branch:** `main`
**Audience:** osaurus model-loading agent

This document is the ground truth for what's wired up vs what's still
broken for the three models the osaurus integration cares about most.
Read this BEFORE claiming any of these are "ready" — the prior agent
made that claim while the JANGTQ schemas were fundamentally wrong.

---

## TL;DR

| Family | Outer `model_type` | Status | Notes |
|--------|-------------------|--------|-------|
| Laguna XS.2 / S.3 (JANGTQ2) | `laguna` | ✅ Loads + decodes | Mixed quant: vanilla `LagunaModel` + `TurboQuantSwitchGLU` MoE + affine-quant dense |
| Mistral 3.5 LLM | `ministral3` (or `mistral3`) | ✅ Loads + decodes | Routes via `dispatchMistral3LLM` → `Mistral3TextModel` / `Mistral3TextJANGTQModel` |
| Mistral 3.5 VLM | `ministral3` (or `mistral3`) | ✅ Loads + decodes | Routes via `dispatchMistral3VLM` → `Mistral3VLM` / `Mistral3VLMJANGTQ` / `Mistral4VLM`. Processor override now covers BOTH spellings |
| NemotronH Omni Nano | `nemotron_h_omni` (auto-detected via `config_omni.json`) | ✅ Loads + decodes | Image, video, audio (Parakeet) all wired. Image+video splice bug fixed |

All four families pass dispatch + multi-turn + reasoning-stamp + tool-format unit tests.

---

## What was actually broken before 2026-05-01

The prior agent claimed Laguna/Mistral/Omni were "ready" but two
fundamental architectural mistakes meant they would **never load on a
real bundle**:

### Bug 1 — Laguna JANGTQ schema completely wrong

The original `LagunaJANGTQModel` modeled every dense `Linear` as
`JANGTQDenseLinear` (codebook schema: `tq_packed` + `tq_norms`). Real
Laguna mxtq bundles are **mixed-quant**:

- Dense paths (attention Q/K/V/O, layer-0 dense MLP, shared expert,
  router gate): MLX standard affine quant — `weight` packed uint32 +
  `scales` + `biases` of shape `[out, in/group_size]`. Vanilla `Linear`
  works here because the MLX loader auto-substitutes
  `QuantizedLinear` from `config.json`'s top-level `quantization`
  field.
- Routed MoE experts: JANGTQ codebook, **stacked across all 256
  experts into a single tensor**, with gate and up **fused** on the
  out-dim axis at `experts.gate_up_proj.{tq_packed, tq_norms}`.

Symptom in osaurus:
```
Error: Unable to set layers.0.mlp.down_proj.biases on
LagunaJANGTQModel.LagunaJANGTQLayer.LagunaJANGTQDenseMLP.JANGTQDenseLinear:
none not compatible with [2048, 128]
```

Then after a partial fix:
```
Error: Unable to set layers.1.mlp.experts on LagunaModel.LagunaLayer.LagunaMoE:
[256 × LagunaDenseMLP { down_proj, gate_proj, up_proj }]
not compatible with [
  down_proj:    { tq_norms: [256, 2048], tq_packed: [256, 2048, 32] },
  gate_up_proj: { tq_norms: [256, 1024], tq_packed: [256, 1024, 128] }
]
```

**Fix:** `LagunaModel.LagunaMoE` now uses `TurboQuantSwitchGLU` (same
codebook MoE primitive DSV4 / Mistral 4 / NemotronH JANGTQ already
use). `LagunaModel.sanitize` splits the bundle's fused
`experts.gate_up_proj.{tq_packed,tq_norms}` into the two halves
`TurboQuantSwitchGLU` expects (`gate_proj.*` + `up_proj.*`), keyed by
`moeIntermediateSize`. `LagunaJANGTQModel` is **deleted** (was dead
code after this rewrite).

### Bug 2 — Mistral 3.5 VLM `ministral3` outer dispatch + sanitize

1. The VLM factory's `processorTypeOverrides` only knew the
   `"mistral3"` outer key, not `"ministral3"`. A `ministral3`-keyed
   bundle silently fell back to `processor_class` from
   `preprocessor_config.json` (typically `PixtralProcessor`), which
   loses Mistral3's spatial-merge handling.
2. `Mistral3VLM.sanitize` (and its JANGTQ twin) didn't have a fallback
   for plain `model.<llm-key>` keys (no `language_model.` wrapper).
   Bundles shipping `model.embed_tokens.*`, `model.layers.<i>.*`
   without the wrapper got `Unhandled keys ["model"]`.

**Fix:** override map covers both spellings; sanitize now re-prefixes
unrouted `model.*` (non-vision, non-projector) with `language_model.`
in both `Mistral3VLM` and `Mistral3VLMJANGTQ`.

### Bug 3 — NemotronH Omni image + video splice

`prepare()` called `spliceAtToken` twice with the same
`imageContextTokenId` (image and video share the placeholder per
Python `model.py`). First call's `nReplace == replacement.dim(0)`
precondition would crash on the combined image+video placeholder
count.

**Fix:** concatenate image + video embeds (image-first, matches
processor placeholder order) and splice in one pass.

---

## Per-family wiring details

### Laguna (`laguna`)

**Dispatch:** `LLMModelFactory.swift:139` (`additionalModels()`).
Reads `mxtq_bits` / `mxtq_seed` from config.json (merged in by the
factory pre-decode from `jang_config.json`), passes through to
`LagunaModel(cfg, bits:, seed:)`.

**Architecture:**
- 40 hybrid layers: layer 0 dense MLP, layers 1..39 sparse MoE.
- Per-layer attention head count via `num_attention_heads_per_layer`
  (48 full-attention / 64 sliding-attention).
- Dual RoPE per `rope_parameters[layer_type]` (mixed-shape config
  decoder is permissive — see `LagunaConfiguration.init(from:)`).
- 256 routed experts top-8 via `TurboQuantSwitchGLU` (codebook MoE);
  shared expert is affine-quant `LagunaDenseMLP`; sigmoid+bias
  routing per DeepSeek-V3 recipe.
- Per-layer mixed cache: `RotatingKVCache` (sliding) + `KVCacheSimple`
  (full).

**Reasoning / tool stamps:**
- `reasoningParserName = "think_xml"` (Laguna emits `<think>…</think>`
  via `laguna_glm_thinking_v5/chat_template.jinja`)
- `toolCallFormat = .glm4` (GLM-family function-calling tags)

**Bundle key remaps in sanitize:**
- `model.<x>` → `<x>`
- `mlp.experts.e_score_correction_bias` → `mlp.e_score_correction_bias`
- `mlp.experts.gate_up_proj.{tq_packed,tq_norms}` → split into
  `mlp.experts.{gate_proj,up_proj}.{tq_packed,tq_norms}` halves
- Drop `self_attn.rotary_emb.inv_freq`, `.tq_bits`, tied `lm_head.weight`

### Mistral 3.5 LLM (`ministral3` outer / `mistral3` outer)

**Dispatch:** `LLMModelFactory.swift:186` + `:193` both route to
`dispatchMistral3LLM`. Logic chain:

1. **Vision gate** — if `vision_config` present, throw
   `ModelFactoryError.unsupportedModelType` with a clear "route via
   VLMModelFactory" message. The error message is the SOURCE OF TRUTH
   for osaurus's "wrong factory" diagnostic.
2. **Mistral 4 text decoder wrapper** — if `text_config.model_type ==
   "mistral4"`, decode as `Mistral4Configuration` and return
   `Mistral4Model` (Mistral 3.5 wrapper around the Mistral 4 architecture).
3. **JANGTQ** — `weight_format == "mxtq"` → `Mistral3TextJANGTQModel`
   (uses `JANGTQDenseLinear` for entire decoder — Mistral family is
   genuinely all-codebook, distinct from Laguna's mixed-quant scheme).
4. **Vanilla** — `Mistral3TextModel`.

**Reasoning / tool stamps:**
- `reasoningParserName = "none"` (Mistral 3.5 doesn't emit `<think>`)
- `toolCallFormat = .mistral` (Mistral inline format)

### Mistral 3.5 VLM (`ministral3` outer / `mistral3` outer)

**Dispatch:** `VLMModelFactory.swift:115` + `:119` both route to
`dispatchMistral3VLM`. Logic chain:

1. **JANGTQ** — `weight_format == "mxtq"` → `Mistral3VLMJANGTQ`
   (JANGTQ inner LM; vanilla Pixtral vision tower per
   `mxtq_bits.vision_tower=passthrough_fp16`).
2. **Mistral 4 text-decoder wrapper** — `text_config.model_type ==
   "mistral4"` → `Mistral4VLM`.
3. **Vanilla** — `Mistral3VLM`.

**Processor:** override map at
`VLMModelFactory.swift:554` maps **both** `"mistral3"` and
`"ministral3"` to `"Mistral3Processor"` (Pixtral processor is the
wrong choice — it skips spatial-merge handling).

### NemotronH Omni (`nemotron_h_omni`)

**Auto-detect:** `VLMModelFactory.swift:387` checks for
`config_omni.json` in the model directory and overrides
`baseConfig.modelType` to `"nemotron_h_omni"` (the bundle's own
`config.json` reports `model_type: nemotron_h` since it's an LLM-only
schema; the omni multimodal config lives in the sidecar).

**Architecture (NemotronHOmni.swift):**
- LM: `NemotronHModel` (or `NemotronHModel(jangtqContext: ...)` for
  JANGTQ omni bundles)
- Vision: RADIO ViT (`NemotronHRADIOVisionModel`)
- Vision projector: `NemotronHVisionMLPProjector` (with pixel-shuffle)
- Audio: Parakeet conformer encoder (`NemotronHParakeetEncoder`) +
  mel STFT + sound projector
- Multi-modal placeholder splice: image and video share
  `<image>` (= `imageContextTokenId`), audio uses `<so_embedding>`
  (= `soundContextTokenId`)

**Reasoning / tool stamps:**
- `reasoningParserName = "think_xml"` (NemotronH emits `<think>` per
  `reasoningStampFromModelType` allowlist)
- `toolCallFormat = .xmlFunction`

**Long-prompt safety:** `prepare()` chunks text-only prefill at
`windowSize ?? 512` tokens to bound the SSM segsum O(L²) memory cost
(see comment at NemotronHOmni.swift:222).

---

## What still needs work

### Stress matrix bodies missing

`Tests/MLXLMStressTests/StressMatrix.swift:177` — `runCell` is a stub
that always returns `passed: true`. The whole exhaustive matrix sweep
(cache mode × disk × L2 × KV quant × arch × cap × prompt len ×
workload) is skeleton-only.

### Old Laguna non-codebook bundles

If anyone has a Laguna bundle that pre-dates the JANGTQ codebook MoE
schema (per-expert `experts.<i>.weight` instead of stacked
`experts.gate_up_proj.tq_packed`), it will fail to load against the
new wiring. There are no such bundles in production today — all real
Laguna distributions ship as JANGTQ codebook MoE. If one shows up,
the failure mode is a clean `tq_packed missing` error pointing at the
exact key — not a silent garbage-output trap.

### Image+video stress test missing

The image+video splice fix (Bug 3) doesn't have a unit test because
splice requires a model instance. End-to-end coverage is gated behind
`BENCH_NEMOTRON_OMNI_BUNDLE`.

---

## How osaurus should detect "wrong factory" errors

When `LLMModelFactory.loadContainer` is called on a Mistral 3.5 VLM
bundle (config.json has `vision_config`), vmlx throws:

```
ModelFactoryError.unsupportedModelType(
  "mistral3/ministral3 (has vision_config — route via VLMModelFactory)")
```

Match the exact substring `"route via VLMModelFactory"` in the error
string and re-dispatch via `VLMModelFactory.shared`. The same pattern
applies to the inverse direction — if the user requests a text-only
LLM bundle through VLMModelFactory, the inner config decoder will
fail with a clear `vision_config required` message.

The cleanest osaurus-side approach is the standard one: peek
`config.json` for `vision_config` (or `image_processor_type`) BEFORE
picking a factory, and route VLM bundles directly. The error path is
a backstop for misrouted requests; routing correctly upfront avoids
the throw entirely.
