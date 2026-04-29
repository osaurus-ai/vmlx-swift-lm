# Nemotron-Omni × Osaurus Hookup Guide

**Audience**: osaurus integrators consuming `vmlx-swift-lm` ≥ commit `b4eec09`.

**Scope**: everything the osaurus runtime layer needs to know to safely
serve **Nemotron-3-Nano-Omni-30B-A3B-{MXFP4, JANGTQ4, JANGTQ2}** bundles
end-to-end (text + image + audio + video × multi-turn × reasoning toggle ×
all three quants), without surprising regressions on neighbour VL families
(Qwen 2/2.5/3/3.5/3.6 VL, Kimi VL, Gemma 3/4 VL, Mistral 3 VL, etc.).

This is the **omni** companion to `OSAURUS-INTEGRATION.md`. Read that first
for the LLM-tier contracts (BatchEngine flag, KV-sizing contract, reasoning
stream events, stop-string contract, Gemma-4 sliding-window crash). This
file covers everything those docs don't:

1. Bundle detection + factory dispatch
2. The four-tower wrapper layout (LLM + RADIO + Parakeet + projectors)
3. Multimodal embed splice through `inputsEmbeds` (image + video work via
   `LMInput`; **audio currently does NOT** — it's the open seam)
4. Hybrid Mamba/Attention/MoE cache topology (52 layers: 23M + 23E + 6\*)
   — what works under coordinator, disk cache, BatchEngine; what doesn't
5. TurboQuant KV interaction with the hybrid pattern
6. JANGTQ vs MXFP4 routing (vision/audio always fp16; LLM-only differs)
7. EVS (Efficient Video Sampling) — embedding-level, after the projector
8. Wired-memory + long-context envelope (30B + 1.6 GB vision + 0.4 GB audio
   ≈ 18 GB MXFP4, 16 GB JANGTQ4, 9 GB JANGTQ2, plus KV)
9. The cross-VLM `VLMVideoUtils.swift` shared library (CLIP/SigLIP norms,
   uniform sampler, T-frame channel stack, generic EVS)
10. Known gaps + osaurus-side TODOs

If anything in here disagrees with code, **trust the code**. Open an issue
that links to the exact symbol + this doc and I'll reconcile.

---

## TL;DR — what osaurus needs to do

```swift
// 1. Auto-detect: VLMModelFactory does this for you.
let context = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(directory: omniBundle))

// 2. Tag the cache coordinator hybrid (auto-flips on first admission, but
//    do it eagerly to avoid the first-turn no-op admission edge case).
coordinator.setHybrid(true)
coordinator.setMediaSalt(computeMediaSalt(for: input))   // image/video

// 3. Build LMInput as usual — image goes through input.image.pixels, video
//    through input.video.pixels. NemotronHOmniProcessor handles tile
//    selection + chat-template splice + 256-tokens-per-tile expansion.

// 4. Audio: TODO — see §3. Today osaurus has TWO options:
//      a. Pre-encode via `omni.extractAudioEmbeds(waveform:)` and splice
//         manually before calling `prepare`. Audio is NOT in LMInput yet.
//      b. Punt audio to a future turn behind a feature flag.

// 5. Run BatchEngine OR Evaluate. Both work, with caveats:
//      - BatchEngine: heterogeneous cache (M/*/E) → uncompiled decode,
//        TQ KV applies only to the 6 attention layers, mamba slots use
//        BatchArraysCache automatically.
//      - Evaluate: single-slot, fully supported, all features work.

// 6. Reasoning toggle: chat template kwarg `enable_thinking=true|false`.
//    Reasoning parser stamp = `deepseek_r1`. Tool parser = `nemotron`.
//    Both auto-resolve from jang_config.capabilities (already wired).
```

---

## 1. Bundle layout + dispatch

### 1.1 Files in an omni bundle (verified inventory of MXFP4, JANGTQ4, JANGTQ2)

| File | Required | Role |
|---|---|---|
| `config.json` | yes | LLM-only config (`model_type: nemotron_h`). 52 layers, hybrid pattern, 2688 hidden, 32q × 2kv heads, 64 mamba heads, 128 SSM state, conv_kernel=4, MoE 128 experts top-6, ReLU² mlp, partial_rotary_factor=1.0. |
| `config_omni.json` | **yes — also the VLM trigger** | Multimodal extras: `force_image_size=512`, `downsample_ratio=0.5`, `vit_hidden_size=1280`, `projector_hidden_size=20480`, `sound_config.{hidden_size=1024, num_attention_heads=8, num_hidden_layers=24, intermediate_size=4096, conv_kernel_size=9, num_mel_bins=128, sampling_rate=16000}`, plus the three placeholder token IDs (`img_context_token_id=18`, `video_context_token_id=131081`, `sound_context_token_id=27`). Also `min_num_patches=1024`, `max_num_patches=13312`. |
| `jang_config.json` | yes (osaurus-AI bundles) | `weight_format ∈ {mlx, mxtq}`, `capabilities.{reasoning_parser="deepseek_r1", tool_parser="nemotron", cache_type="hybrid", modality="omni", supports_thinking=true, supports_tools=true}`. |
| `model-*.safetensors` (sharded) | yes | All four towers in one sharded safetensors set. LLM weights are quantized per `weight_format`; **vision/audio/projector weights are always fp16/bf16**, never quantized. |
| `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` | yes | Mistral-style sentencepiece vocab=131072, bos=1, eos=11, pad=0. |
| `chat_template.jinja` | yes | NVLM 1-D placeholder convention + `<think>` reasoning block. Driven by `enable_thinking` kwarg. Compatible with current swift-jinja 1.3.0+ — no template patch needed. |
| `generation_config.json` | optional | Sampling defaults: temp=0.6, top_p=0.95. |
| `preprocessor_config.json` | yes | `image_processor_type="NemotronH_Nano_Omni_Reasoning_V3ImageProcessor"`. **No `processor_class` field** — this is intentional (see §1.2). |
| `feature_extractor_config.json` | optional | Audio mel STFT params (n_fft=512, hop=160, win=400, n_mels=128, sr=16000). Hardcoded as defaults in `NemotronHOmniConfiguration`; field exists if you want to override. |
| `audio_model.py`, `image_processing.py`, `video_processing.py`, `processing.py`, `modeling.py`, `evs.py` | optional | Original Python reference. Ignored at load — Swift implementation is in `Libraries/MLXVLM/Models/NemotronHOmni/`. |

### 1.2 Factory dispatch — three model-type strings, one trigger file

```swift
// VLMTypeRegistry registrations (Libraries/MLXVLM/VLMModelFactory.swift):
"nemotron_h_omni":                       create(NemotronHOmniConfiguration.self, NemotronHOmni.init),
"NemotronH_Nano_Omni_Reasoning_V3":      create(NemotronHOmniConfiguration.self, NemotronHOmni.init),

// VLMProcessorTypeRegistry:
"NemotronHOmniProcessor": create(NemotronHOmniProcessorConfiguration.self,
                                 NemotronHOmniProcessor.init),
```

Bundles ship `model_type: nemotron_h` in `config.json` (LLM only — they
predate any multimodal naming standardization). `VLMModelFactory._load`
detects omni by the **presence of `config_omni.json`** in the model
directory and rewrites `dispatchModelType` to
`NemotronH_Nano_Omni_Reasoning_V3` before calling the type registry. Same
mechanism flips the processor lookup to `NemotronHOmniProcessor`, even
though `preprocessor_config.json` lacks a `processor_class` field. (We
also made `BaseProcessorConfiguration.processorClass` optional so the
decode doesn't throw on bundles using `image_processor_type` instead.)

**Osaurus impact**: zero. The auto-detect lives entirely inside
`VLMModelFactory._load`, which osaurus already calls. There is **no**
osaurus-side dispatch table to update. Just point `loadContainer` at the
bundle directory and you get a `ModelContext` whose `model` is
`NemotronHOmni` and whose `processor` is `NemotronHOmniProcessor`.

### 1.3 Why both `nemotron_h_omni` and `NemotronH_Nano_Omni_Reasoning_V3`?

The Python reference uses the long name as the registered HF auto-class.
The short name is a forward-compatibility alias for any bundle that
ever stamps `config_omni.json::model_type` differently. Both routes go to
the same Swift type — pick whichever shows up first in your config
inspectors.

---

## 2. Four-tower module layout

```
NemotronHOmni  (VLMModel, KVCacheDimensionProvider, LoRAModel)
├── language_model:                       NemotronHModel  (existing in MLXLLM)
│   • 52-layer hybrid: 23 Mamba (M) + 23 MoE (E) + 6 Attention (*)
│   • hybrid_pattern: "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
│   • 32q × 2kv attention heads, head_dim=128, partial_rotary_factor=1.0
│   • Mamba2 mixer: 64 heads × 64 head_dim, 128 SSM state, conv_kernel=4
│   • MoE: 128 routed + 1 shared, top-6, routed_scaling=2.5, ReLU² gate
│   • EOS=11, vocab=131072
│
├── vision_model.radio_model:             NemotronHRADIOVisionModel
│   • 32-block ViT-Huge with patch=16, embed=1280, heads=16, MLP=5120
│   • CPE patch generator: 10 cls/register tokens + bilinear pos_embed interp
│     from stored 128×128 grid down to actual gy×gx
│   • video_embedder: separate Linear(T*3*P*P → 1280) for T=2 frame stacking
│   • Always fp16/bf16 — never quantized
│
├── mlp1.vision_mlp:                      NemotronHVisionMLPProjector
│   • LayerNorm → Linear(5120→20480) → GELU → Linear(20480→2688)
│   • Bias-optional; on-disk keys: mlp1.{0,1,3}.{weight,bias}
│
├── sound_encoder:                        NemotronHParakeetEncoder
│   • Subsampling: 5-conv stack (1→256→256→256→256→256) factor=8 + Linear(4096→1024)
│   • 24 × ConformerBlock: ½FF (silu) + LN + Rel-Pos MHA (8 heads, 128 head_dim,
│                          bias_u/bias_v, Transformer-XL skewing) + LN +
│                          Conv module (pointwise GLU + 9-tap depthwise + BN
│                          + silu + pointwise) + ½FF + final LN
│   • Inference-only BatchNorm1d using stored running stats
│   • Always fp16/bf16 — never quantized
│
└── sound_projection:                     NemotronHSoundProjector
    • RMSNorm(1024) → Linear(1024→4096) → SquaredReLU → Linear(4096→2688)
```

**Weight remap helpers** are wired into `NemotronHOmni.sanitize`. Osaurus
gets correct loading for free — no per-bundle key rewriting needed.

**Public encoders** (callable from the runtime layer, not just internally):

```swift
// Returns flat (totalTokens, llmHidden=2688) embeddings, tile-row-major.
public func extractImageEmbeds(pixelValues: MLXArray, video: Bool = false) -> MLXArray

// Returns flat (frames, llmHidden=2688) embeddings.
// Audio: 16 kHz mono Float32 waveform, computes mel STFT internally.
public func extractAudioEmbeds(waveform: [Float]) -> MLXArray
```

---

## 3. Audio pipeline — the open seam

**`LMInput` today carries text + image + video. There is no audio field.**

This means:

- `input.image?.pixels` and `input.video?.pixels` are honored by
  `NemotronHOmni.prepare(_:cache:windowSize:)` — they get encoded by the
  vision tower + mlp1 and spliced at `<image>`-token positions
  (`imageContextTokenId=18`).
- **Audio embeds must be computed and spliced manually** before
  `prepare` — there's no built-in path. Two options, depending on how much
  you're willing to modify osaurus:

### 3.1 Option A — pre-encode in osaurus, splice manually

```swift
// 1. Decode + resample audio to 16 kHz mono Float32:
let waveform = try nemotronOmniLoadAudioFile(audioURL)
//  (the helper handles AVAudioConverter resample + downmix; if you already
//   have a Float32 array at 16 kHz mono, skip this and pass it directly)

// 2. Run mel STFT + Parakeet + sound_projection:
let audioEmbeds = omni.extractAudioEmbeds(waveform: waveform)
//  shape: (n_frames/8, 2688)

// 3. Build the prompt with `<so_embedding>` placeholder tokens, one per
//    audio embed row. Source convention:
//      "<sound>" + ("<so_embedding>" * audioEmbeds.dim(0)) + "</sound>\n"
//    + the rest of the user message.

// 4. Tokenize via the chat template; you'll see `sound_context_token_id=27`
//    repeated audioEmbeds.dim(0) times.

// 5. Splice manually — call NemotronHOmni's helper if you make a thin
//    extension, or do the embed-buffer write yourself:
//      a. text_embeds = omni.languageModel.embedTokens(input.text.tokens)
//      b. positions = where input.text.tokens == 27
//      c. text_embeds[0, positions, :] = audioEmbeds
//      d. logits = omni.languageModel(inputsEmbeds: text_embeds, cache: cache)
```

### 3.2 Option B — extend `LMInput` with `ProcessedAudio`

Cleaner long-term. Touchpoints (vmlx-side, **not yet done**):

```swift
// MLXLMCommon/LanguageModel.swift
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?
    public let audio: ProcessedAudio?       // NEW

    public struct ProcessedAudio {
        public let waveform: MLXArray       // (1, samples) Float32 @ 16 kHz
        public let embedding: MLXArray?     // optional pre-computed (frames, hidden)
    }
}

// NemotronHOmni.prepare(...) gains a third splice branch for input.audio.
// Existing image/video paths are unchanged.

// MediaSalt.computeMediaSalt() must hash audio.waveform too, otherwise
// disk-cache hits will be wrong on "same prompt, different voice" turns.
```

**This is the recommended next step** if osaurus wants first-class audio.
Drop a request issue and I'll land it as a follow-up commit. Until then,
osaurus must use Option A for any audio turn.

### 3.3 Audio media-salt gap (today)

`computeMediaSalt(for:)` in `Cache/MediaSalt.swift` only hashes image and
video pixels. For an audio turn keyed via Option A, **the disk cache will
return false-positive hits across different waveforms with identical text
prefixes**.

Workaround until LMInput.audio lands:
- Disable disk cache (`enableDiskCache: false`) on audio turns, OR
- Compute your own salt over the audio bytes and pass it through the
  coordinator's `mediaSalt` parameter manually:

```swift
// Add the audio salt to whatever computeMediaSalt returned.
var salt = computeMediaSalt(for: input) ?? ""
salt += "audio:" + sha256OfFloat32Array(waveform)
coordinator.setMediaSalt(salt)
```

---

## 4. Image + video — works through standard `LMInput`

### 4.1 Image flow (text + N images, single turn)

```swift
let input = UserInput(
    prompt: "What's in this image?",
    images: [.init(URL(fileURLWithPath: "cat.jpg"))]
)
let lmInput = try await processor.prepare(input: input)
// lmInput.image.pixels: (totalTiles, 3, 512, 512) Float32, CLIP-normalized
// lmInput.text.tokens contains 256 × `imageContextTokenId=18` tokens per
// tile (the NVLM 1-D 16×16 post-pixel-shuffle grid).
```

NemotronHOmniProcessor handles:
- NVLM 1-D dynamic tile selection (1..12 tiles based on aspect ratio + a
  bicubic-resized thumbnail)
- CIImage rasterization (sRGB working space, RGBAf → strip alpha → CHW)
- CLIP normalization (mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
- Tile-row-major stack → (totalTiles, 3, 512, 512)
- Chat-template splice with `<img>…<image>…</img>\n` markers, one
  `<image>` token per post-pixel-shuffle position (256 per tile)

### 4.2 Video flow (single video, T=2 frame stacking)

```swift
// Video preprocessing returns the (groups, T*3, 512, 512) tensor directly
// for the RADIO video_embedder path:
let pixelValues = try await nemotronOmniPreprocessVideo(
    url: videoURL,
    imageSize: 512,
    targetFrames: 32,
    videoTemporalPatchDim: 2)
// shape: (16, 6, 512, 512) — 32 frames padded to even, paired into 16
// 2-frame channel-stacked groups.

// Wrap as a ProcessedVideo for LMInput:
let lmInput = LMInput(
    text: .init(tokens: tokenized, mask: mask),
    video: LMInput.ProcessedVideo(
        pixels: pixelValues,
        frames: [THW(16, 512, 512)]))

// NemotronHOmni.prepare detects input.video.pixels and runs:
//   feats = radioModel(pixels, video: true)   // uses video_embedder
//   feats = mlp1(pixel_shuffle(strip_cls(feats), 0.5))
//   spliced into <image>-token slots in the prompt.
```

**Optional EVS pruning** (drops ~70% redundant inter-frame tokens at the
embedding level — already projected to LLM hidden):

```swift
let raw = omni.extractImageEmbeds(pixelValues: pixelValues, video: true)
//  raw shape: (groups*tokensPerGroup, 2688)
let pruned = vlmApplyEVS(
    raw.reshaped([groups, tokensPerGroup, 2688]),
    pruningRate: 0.7,
    keepFirstFrame: true)
//  pruned shape: (1, kept, 2688)
```

EVS lives in `Libraries/MLXVLM/VLMVideoUtils.swift` and is **generic** —
applies to any (groups, tokens, hidden) tensor, so Qwen 3.6 VL / Kimi VL /
future video VLMs can use the same primitive.

### 4.3 The cross-VLM `VLMVideoUtils.swift` shared library

All five primitives are public and stable. Other VLMs in this repo
(Qwen 2/2.5/3/3.5/3.6, Kimi, future) can adopt them as a one-line refactor:

| Primitive | Purpose | Used by |
|---|---|---|
| `vlmExtractFramesUniform(url:targetFrames:)` | Uniform sample via `MediaProcessing.asCIImageSequence` | NemotronHOmni today; safe drop-in for Qwen/Kimi |
| `vlmResizeAndNormalize(image:target:mean:std:)` | Bicubic resize + per-channel normalize → planar `[Float]` | NemotronHOmni today |
| `vlmStackFramesIntoChannels(_:imageSize:temporalPatchDim:mean:std:)` | T-frame channel stacking → (groups, T*3, H, W) MLXArray | NemotronHOmni (T=2); usable with T=1 for non-stacked VLMs |
| `vlmApplyEVS(_:pruningRate:keepFirstFrame:)` | Cosine-similarity-based token retention | NemotronHOmni today; opt-in for Qwen 3.6 VL etc. |
| `CLIP_NORM_MEAN/STD`, `SIGLIP_NORM_MEAN/STD` | Standard normalization presets | Any VLM that uses the corresponding mean/std |

---

## 5. Cache topology — hybrid Mamba/Attention/MoE

This is the part osaurus must **most carefully** review. Nemotron-Omni's
hybrid pattern is unusual:

```
Pattern (52 layers):
M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E M E M * E M E M E M E M * E M E M E M E M E

Counts:
  M (Mamba2)           = 23 layers  → MambaCache (size=2: conv state + hidden state)
  E (MoE)              = 23 layers  → no cache (FFN/MoE only)
  * (Attention, GQA)   = 6 layers   → KVCacheSimple or RotatingKVCache
```

`NemotronHModel.newCache(parameters:)` already returns the correct
heterogeneous list (one entry per layer; nil for MoE/MLP slots are
elided by the iter pattern).

### 5.1 Coordinator interaction

| Surface | Behaviour for Nemotron-Omni |
|---|---|
| **`CacheCoordinator.isHybrid`** | **MUST be true** for omni. Auto-flips on first BatchEngine admission via `BatchEngine.swift:622-635` when any layer is `MambaCache` or `ArraysCache`. Eager-set is harmless and avoids a one-frame stale-flag window if osaurus admits requests via Evaluate first. |
| **`CacheCoordinator.diskCache`** (L2) | Works. `Cache/CacheCoordinator.swift:storeAfterGeneration` iterates per-layer caches and stores `KVCacheSimple` and `MambaCache` round-trip-capable arrays via `TQDiskSerializer` (`LayerKind.simple=0`, `LayerKind.mamba=2` — and `LayerKind.rotating=6` was added 2026-04-15). MoE layers have no cache so they're skipped. |
| **`CacheCoordinator.ssmStateCache`** | Used. When `isHybrid=true`, the coordinator additionally fetches/stores SSM companion state (Mamba conv/hidden) keyed by token sequence. This is the same SSMReDeriver path that ships for Qwen 3.5 / 3.6 hybrid models. |
| **`PagedCacheManager` (paged KV)** | Works for the 6 attention layers. The 23 Mamba layers go through the SSM state cache instead — they're recurrent, not block-paged. |
| **`MediaSalt`** (image+video) | Works. **Audio is missing** — see §3.3. |
| **`CacheCoordinatorConfig.defaultKVMode`** | Applies only to the 6 attention layers. Mamba layers ignore the kv-mode field; they're always full-precision. Recommended: `.turboQuant(keyBits: 3, valueBits: 3)` for memory-bounded omni serving. |
| **`defaultMaxKVSize`** | Applies only to the attention layers. Reasonable default: 8192 (matches `KV-SIZING-CONTRACT.md` recommendation). The `longPromptMultiplier=2.0` lets it stretch to 16K for long video / audio prompts. |

### 5.2 Concrete coordinator config for omni serving

```swift
let coordinator = CacheCoordinator(
    config: CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: true,
        modelKey: "Nemotron-3-Nano-Omni-30B-A3B-MXFP4",   // include quant in key
        defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),
        defaultMaxKVSize: 8192,
        longPromptMultiplier: 2.0))
coordinator.setHybrid(true)   // do this before the first turn
```

### 5.3 Multi-turn cache reuse

The standard image multi-turn flow works:

- Turn 1: `<img>…<image>…</img>\n Describe this.` → image encoded once,
  cached at the appropriate token positions.
- Turn 2: `Follow up: what color?` → same coordinator, same modelKey;
  attention KV from turn 1 is restored (paged + disk hit), Mamba SSM state
  is restored from `ssmStateCache`. Vision tower is **not** re-run.

For audio (Option A above):
- Manual splice means **the audio embed is part of the prompt embedding**.
- If you ALSO disable disk cache on audio turns (per §3.3), you sacrifice
  L2 reuse. Coordinator's in-memory paged cache still works.
- If you compute an audio salt yourself and pass via `setMediaSalt`,
  multi-turn reuse with audio works correctly.

---

## 6. TurboQuant KV — what applies, what doesn't

`BatchEngine/BatchQuantize.swift` documents the rules; the omni-specific
distillation is:

| Layer kind | Behaviour under `kvMode: .turboQuant(k, v)` |
|---|---|
| `KVCacheSimple` (the 6 `*` layers) | **Promoted** to `TurboQuantKVCache(keyBits: k, valueBits: v, sinkTokens: 4)` at admission via `wrapNewCacheIfNeeded`. ~5× KV memory savings on these slots. |
| `RotatingKVCache` | Preserved (TQ doesn't replace rotating). If osaurus uses a maxKVSize for the attention layers, the wrapper picks up `RotatingKVCache(maxSize: maxKVSize, keep: 4)` at LLM-cache-creation time and TQ sits on top transparently. |
| `MambaCache` (the 23 `M` layers) | **Preserved unchanged**. Mamba state is conv+hidden recurrent state, not KV — TQ would corrupt the SSM dynamics. `BatchQuantize.wrapNewCacheIfNeeded` already type-gates against this (line 100-101). |
| MoE / MLP / `nil` slots | Skipped (no cache to wrap). |

**Affine KV (`kvMode: .affine`, `kvBits` legacy)**: explicitly NOT
supported under BatchEngine — would require quantized-tuple attention
sites out of Stage-0 scope. Logged as a warning at admission; the request
runs with float KV and continues. Same behavior as for any other model.

**Compile path** (`BatchCompile.swift`): omni's heterogeneous topology
classifies as `.heterogeneous` (mixed `.mamba` + `.simple`/`.turboQuant`).
**Compiled decode is NOT taken**. This is correct-by-design — the Mamba
trace grouping is its own spec (`Stage 4 pending`) and not yet in. Decode
runs uncompiled on omni regardless of TQ enable. Throughput envelope:
~80–110 tok/s on M3 Max at B=1 depending on quant and KV mode.

---

## 7. JANGTQ vs MXFP4 routing

| Component | MXFP4 | JANGTQ4 | JANGTQ2 |
|---|---|---|---|
| LLM embeddings + lm_head | affine 4 | affine 8 + JANG sidecar | affine 8 + JANG sidecar |
| LLM dense layers (M / `*` projections, attn QKV, mlp gate/up) | affine 4 | affine 8 | affine 8 |
| LLM routed experts (E layers' switch_mlp.fc1/fc2) | affine 4 | TurboQuant 4-bit, Hadamard-rotated | TurboQuant 2-bit, Hadamard-rotated |
| RADIO ViT (`vision_model.radio_model.*`) | **fp16** | **fp16** | **fp16** |
| Parakeet (`sound_encoder.*`) | **fp16** | **fp16** | **fp16** |
| mlp1, sound_projection | **fp16** | **fp16** | **fp16** |
| Runtime KV cache (attention layers) | float32/bf16 default; coordinator may TQ-wrap | same | same |
| Mamba conv/SSM state | float32/bf16 always | same | same |

**Critical**: `NemotronHOmni.sanitize` routes weights into four buckets
before dispatching:
- LLM keys (everything not vision/audio/projector) → `NemotronHModel.sanitize`
  (handles conv1d transpose, JANG expert remap from `down_proj`/`up_proj` →
  `fc2`/`fc1`, expert weight stacking, JANGTQ sidecar codebook attachment)
- RADIO keys (`vision_model.radio_model.*`) → `remapRadioWeights`
- Parakeet keys (`sound_encoder.encoder.*`) → `remapParakeetWeights`
  (handles Conv2d OIHW→OHWI and Conv1d OIK→OKI transpose)
- Projector keys (`mlp1.*`, `sound_projection.*`) → `remapMlp1Weights`,
  `remapSoundProjectionWeights`

So the **same Swift type** loads all three quant variants. Osaurus does
not need to dispatch differently per quant — `JangLoader.loadConfig`
auto-resolves the weight format from `jang_config.json::weight_format`,
and the existing JANGTQ runtime patches (P3 Hadamard rotation, P17 thread
tiling, P18 QKV-fusion skip for nemotron_h, P19 mxtq sidecar codebook
load) all flow through.

If you see `loadWeights` fail with a key shape mismatch, **first** check
that `jang_config.json::weight_format` matches the actual safetensors
contents (commit `fa77575` adds auto-correction when a sidecar is present
but config says mlx — but it can't fix the inverse). See
`JANGTQ-RUNTIME-PATCH-GUIDE.md` for the full table.

---

## 8. Wired memory + long-context envelope

Bundle sizes (RAM-resident weights, no KV):

| Bundle | Total disk | LLM weights | Vision (fp16) | Audio (fp16) | Projectors (fp16) |
|---|---|---|---|---|---|
| MXFP4 | 18 GB | ~16 GB | ~1.6 GB | ~0.4 GB | ~0.05 GB |
| JANGTQ4 | 16 GB | ~14 GB | ~1.6 GB | ~0.4 GB | ~0.05 GB |
| JANGTQ2 | 9.1 GB | ~7.0 GB | ~1.6 GB | ~0.4 GB | ~0.05 GB |

**Wired-memory policy (`Libraries/MLXLMCommon/WiredMemoryPolicies.swift`):**

DO NOT manually call `mlx_set_wired_limit` or
`MLX.GPU.set_cache_limit` based on bundle size — the explicit set caused
a regression that crashed Mac mini 16 GB systems on omni load. The
existing `WiredMemoryPolicies.applyDefault()` (or whatever osaurus's
Cache/Memory plumbing already uses) is correct. See
`memory/wired_memory_crash_fix.md` for the post-mortem (2026-04 commit
`847a8c7`). Net: **let MLX manage wired memory automatically**; the
omni bundle is just a normal "large model" from the runtime's point of
view.

**KV envelope** (the 6 attention layers only):
- Sequence length 8K, kv_heads=2, head_dim=128, fp16 → 8K × 2 × 128 × 2 ×
  6 × 2 (K+V) ≈ 50 MB per turn at B=1.
- TurboQuant 3-bit → ~10 MB.
- The 23 Mamba layers contribute fixed-size SSM state (~2 MB total
  regardless of context length — that's the Mamba advantage).

---

## 9. Reasoning + tool capability stamps

Both auto-resolve from `jang_config.capabilities` already, no osaurus
config needed:

```swift
// From jang_config.json:
"capabilities": {
    "reasoning_parser": "deepseek_r1",
    "tool_parser":      "nemotron",
    "supports_thinking": true,
    "supports_tools":    true,
    "cache_type":        "hybrid",
    "modality":          "omni"
}
```

VLMModelFactory.\_load reads these (commit `e5fb015` and earlier) and
sets `mutableConfiguration.reasoningParserName` and
`mutableConfiguration.toolCallFormat` accordingly. The streaming reasoning
events (`Generation.reasoning(String)`, see `REASONING-STREAM-EVENT.md`)
fire correctly. The `enable_thinking` chat-template kwarg flows through
the standard Jinja template path — pass it via `additionalContext` if
you want to toggle reasoning per-turn (see `OSAURUS-API-SURFACE.md`
§ChatTemplates).

**Iter 66 tool-call parsing**: covered by `nemotron` parser registered in
`Libraries/MLXLMCommon/Tool/`. Osaurus `BatchEngine.generate()` and
`Evaluate.generate()` both emit authoritative `.toolCall(ToolCall)`
events. No app-layer parsing required.

---

## 10. BatchEngine — what works at B>1, what doesn't

Omni in BatchEngine is supported but with the same heterogeneous-cache
caveats as Qwen 3.5 hybrid:

| Surface | Status |
|---|---|
| Multi-slot admission | ✅ works. `BatchEngine.admitPendingRequests` auto-flips coordinator hybrid. |
| Mamba slot handling | ✅ via `BatchArraysCache` (subclass of `MambaCache`). Per-slot SSM states are merged on admit, written back on detach. |
| KV slot quantization | ✅ TQ on the 6 attention layers (`BatchQuantize.maybeCompress`). |
| Compiled decode promotion | ❌ **NOT** taken — heterogeneous topology classifies as `.heterogeneous`, falls to uncompiled (`BatchCompile.swift:85`). Correct-by-design; Mamba trace grouping is `Stage 4 pending`. |
| Disk cache (multi-slot) | ✅ each slot's per-token store after generation goes to `DiskCache.store` under the per-slot lock (iter 61 fix). Hybrid slots also store SSM companion state. |
| Multimodal embed splice (per-slot) | ⚠️ **OPEN**. Today the BatchEngine prefill path takes raw token IDs, not pre-spliced `inputsEmbeds`. Multimodal works through `Evaluate` but **not** through `BatchEngine` for the omni image/video/audio splice. See §10.1. |
| Reasoning + tool events at B>1 | ✅ same as text-only — per-slot streams. |

### 10.1 BatchEngine + multimodal splice — open seam

`BatchEngine.admitPendingRequests` currently expects each slot to provide
text tokens; the prefill loop calls `model(inputs:cache:)` not
`model.callAsFunction(inputsEmbeds:cache:)`. For omni, this means an image
or video request submitted to BatchEngine **would forward the literal
placeholder tokens with no replacement** — not a crash, but the model
produces nonsense.

**Workaround for now** (osaurus-side): **do not admit omni multimodal
turns through BatchEngine**. Route them through `Evaluate.generate()`
which already calls `model.prepare(_:cache:windowSize:)`, which is the
overridden path that does the splice. Text-only omni turns can go through
BatchEngine without restriction.

**Fix path** (vmlx-side, future):
- Either extend `BatchEngine`'s admit/prefill to call `model.prepare(...)`
  for VL models (mirror of Evaluate's path).
- Or expose a `inputsEmbeds`-aware admission API on BatchEngine.

This is a non-trivial change because it affects every VLM, not just omni.
File issue if osaurus needs this prioritized; the current Evaluate fallback
for VL turns is the sane default and matches what every other VLM in the
repo does today.

---

## 11. Audio decode envelope (AVAudioConverter quirks)

`nemotronOmniLoadAudioFile(url:targetSampleRate:)` handles:
- Any AVFoundation-decodable input (WAV, AAC, MP3 via system codec, M4A, FLAC).
- Auto-resample to 16 kHz mono Float32 via AVAudioConverter (single-pass
  block-conversion).
- Fast path: 16 kHz mono Float32 input bypasses the converter entirely.

**Known quirk**: AVAudioConverter is single-shot in our wiring (we set
`consumed=true` after the first input buffer). For very long audio
(>100 MB raw) you may need to chunk yourself. For the typical Nemotron-Omni
use case (≤30 s clips for chat) this is fine. If osaurus serves long-form
ASR-style audio (>1 minute), pull the converter loop out into a
multi-block driver.

**Per-sample mel normalize** is applied by default in
`nemotronOmniExtractMelFeatures`. **Do not disable it** — without per-sample
normalize the model produces nonsense ("sound of a door opening" in lieu
of speech text). The Python port lost ~30 minutes to this. The Swift port
flags it as CRITICAL in the source.

---

## 12. Migration checklist for osaurus

Use this as a PR checklist when wiring omni support into the osaurus runtime:

- [ ] `loadContainer(configuration: .init(directory: omniBundle))` returns a
      `ModelContext` whose `model` is `NemotronHOmni`. Verify by running
      a text-only smoke turn first.
- [ ] `coordinator.setHybrid(true)` called eagerly before first admission.
- [ ] `defaultKVMode = .turboQuant(keyBits: 3, valueBits: 3)`,
      `defaultMaxKVSize = 8192` (or your house defaults).
- [ ] Image turn through standard `UserInput(prompt:images:)` path —
      `NemotronHOmniProcessor` is auto-selected. No osaurus dispatch
      changes needed.
- [ ] Video turn — call `nemotronOmniPreprocessVideo` ahead of
      `LMInput`, wrap as `ProcessedVideo`.
- [ ] Audio turn — Option A (manual pre-encode + splice + custom salt)
      OR wait for `LMInput.audio` (Option B) to land.
- [ ] Multi-turn cache reuse — verify image disk cache hits across turns
      with same image + extended prompt. (`mediaSalt` should auto-resolve.)
- [ ] BatchEngine — text-only OK, multimodal turns must route through
      `Evaluate.generate()` until the BatchEngine inputsEmbeds path lands.
- [ ] Reasoning toggle — `enable_thinking=false` produces no `<think>`
      block; `=true` (default) emits `Generation.reasoning(String)`
      events, captured by your stream consumer.
- [ ] Tool calls — same flow as Qwen / DSV4 / Gemma; emit
      `.toolCall(ToolCall)` events. Parser stamp resolves to `nemotron`
      automatically.
- [ ] All three quant variants (MXFP4, JANGTQ4, JANGTQ2) load via the
      same code path. Verify with the `00_verify_all` equivalent — three
      bundles, three text-only smoke turns.

---

## 12.5 Real-bundle state — what actually loads + runs (2026-04-28)

Updated after `BENCH_OMNI=1` against the local bundles + the JANG
quant-inference fix in `ae526a3`.

| Bundle | Load | E2E multi-turn | Notes |
|---|---|---|---|
| `Nemotron-3-Nano-Omni-30B-A3B-MXFP4` (21 GB) | ✅ 1.1 s | ✅ **7/7 PASS** | text single + multi-turn cache reuse, image single + multi-turn (MediaSalt), video encoder smoke (T=2 channel-stack), audio encoder smoke (Parakeet + sound_projection), reasoning OFF parity. Decode 79–121 tok/s @ B=1 / M3 Max. **Production-ready.** |
| `Nemotron-3-Nano-Omni-30B-A3B-JANGTQ4` (19 GB) | ❌ | n/a | needs `NemotronHJANGTQ.swift` — see below |
| `Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2` (12 GB) | ❌ | n/a | needs `NemotronHJANGTQ.swift` — see below |

### What `ae526a3` fixed

The original `[rms_norm] (*weight) must have the same size as the last
dimension of x but has 2688 elements` trap — first hit by tpae's
osaurus run on Cascade-2 JANG_4M, then independently caught by
`BENCH_OMNI=1` on Nemotron-Omni MXFP4 — was a JANG quant-inference
shape-ambiguity bug, NOT an omni-wrapper bug.

Root cause: `(bits=8, gs=32)` and `(bits=4, gs=64)` produce the SAME
packed tensor shape for any `numGroups`. The primary path of
`inferBitWidthAndGroupSize` accepts whichever matches the prior
`gs`; for bundles with mixed-gs layers + a single prior, the wrong
half got loaded with double-bits / half-gs and dequant reconstructed
wrong row vectors. Trap fired mid-prefill.

Three plumbing fixes in `ae526a3`:
1. `JangLoader.inferPerLayerQuantization` — prefer
   `jangConfig.blockSize` (authoritative) over `overrideGroupSize`
   when `bit_widths_used` is non-empty (real JANG conversion signal).
2. `VLMModelFactory._load` — pass `baseConfig.quantization` through
   to `loadWeights` so omni / VL bundles' top-level
   `quantization.group_size` lands as the prior. (LLM factory already
   did this; VLM was missing it.)
3. Both factories — pass `perLayerQuantization` through even when
   JANG; `loadWeights` retains shape walk for the JANG path but the
   plumbing is now uniform.

Affected bundles: anything with a JANG-converted `bit_widths_used:
[…, …]` and mixed gs across layers (Cascade-2 JANG_4M, Nemotron-Omni
MXFP4). Unaffected: JANG_2L bundles whose prior gs matches every
layer (Cascade-2 JANG_2L unchanged at 125 tok/s).

### Still open: NemotronHJANGTQ wrapper for omni JANGTQ2 / JANGTQ4

JANGTQ omni still fails at `unhandledKeys: experts`. Bundle ships
per-expert TurboQuant tensors:

```
backbone.layers.{l}.mixer.experts.{e}.{up,down}_proj.{tq_packed,
                                                      tq_norms,
                                                      tq_bits}
```

`NemotronHModel.sanitize` only stacks `experts.{e}.{up,down}_proj.weight`
(plain affine), so the TQ tensors propagate through and the model
rejects with `unhandledKeys: experts`.

Fix path: write `NemotronHJANGTQ.swift` mirroring the
`DeepseekV3JANGTQ.swift` pattern — ~300 LOC subclass that:
- Stacks per-expert TQ tensors into
  `switch_mlp.{fc1,fc2}.{tq_packed, tq_norms, tq_bits}`
- Substitutes `TurboQuantSwitchLinear` for the MoE routed-expert
  switch under `backbone.layers.{l}.mixer.switch_mlp`
- Wires JANGTQ runtime cache (signs / codebook) for the routed
  experts at first forward

Tracked in §13 below as a HIGH-severity gap. Until it lands, JANGTQ
omni doesn't load; MXFP4 omni serves as the production path.

### Osaurus posture (revised, 2026-04-28)

- **Ship MXFP4 omni now.** All seven bench rows pass on real bundle.
  The factory + processor + cache + reasoning + tool plumbing are
  validated. No known gaps for MXFP4.
- **Don't ship JANGTQ omni yet** — wait for `NemotronHJANGTQ.swift`.
- **All non-omni paths unaffected** by either change. Cascade-2 JANG_2L
  + JANG_4M text-only also benefit from the `ae526a3` fix.

Reproduce locally:

```bash
BENCH_OMNI=1 \
  BENCH_MODEL=~/.mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4 \
  BENCH_MAX_TOKENS=24 \
  swift run -c release RunBench
# Expected: "7 passed, 0 failed | load 1.11s"
```

---

## 13. Known gaps + tracking

| Gap | Severity | Owner | Tracking |
|---|---|---|---|
| ~~MXFP4 omni first-forward crash (rms_norm 2688)~~ | ~~HIGH~~ | ~~vmlx-side~~ | **CLOSED in `ae526a3`** — see §12.5 |
| **`NemotronHJANGTQ.swift` missing** | **HIGH** | vmlx-side | §12.5 — blocks JANGTQ4 / JANGTQ2 omni serving; MXFP4 unaffected |
| `LMInput` has no audio field | medium | vmlx-side | this doc §3.2; unblock once osaurus signals demand |
| `MediaSalt` skips audio | medium | vmlx-side | this doc §3.3; trivial fix once §3.2 lands |
| BatchEngine prefill uses tokens, not inputsEmbeds | medium | vmlx-side | this doc §10.1; affects ALL VLMs not just omni |
| Compiled decode for hybrid (Stage 4) | low | vmlx-side | called out in `BatchCompile.swift:85`; perf optimization, not correctness |
| EVS keep-first-frame edge case | low | vmlx-side | `vlmApplyEVS` first-group always-kept logic; matches Python ref but Python defaults `dissimilarity[0] = 255` directly. Re-verify if you see token-count drift between Python + Swift. |
| Long-form audio (>1 min) chunking | low | osaurus-side | AVAudioConverter loop in §11; escalate if you serve long ASR clips |

---

## 14. Cross-VLM video sharing (don't forget!)

`VLMVideoUtils.swift` is a peace offering for the rest of the VLM
families. It's not omni-specific. If osaurus ever wants to add video
support to **any** of these — Qwen 2/2.5/3/3.5/3.6 VL, Kimi VL, Gemma 4 VL,
Mistral 3 VL — the primitives are there:

```swift
import MLXVLM

let frames = try await vlmExtractFramesUniform(url: videoURL, targetFrames: 16)
let pixelValues = vlmStackFramesIntoChannels(
    frames, imageSize: 448,
    temporalPatchDim: 1,                      // 1 = no stacking, per-frame patch
    mean: SIGLIP_NORM_MEAN, std: SIGLIP_NORM_STD)
let embeds = qwenModel.extractImageEmbeds(pixelValues: pixelValues)
let pruned = vlmApplyEVS(embeds.reshaped([groups, P, hidden]),
                          pruningRate: 0.5,
                          keepFirstFrame: true)
```

This unifies the video path so only the per-model ViT body and
per-model placeholder-token convention vary.

---

## Appendix A — Full module index

```
Libraries/MLXVLM/Models/NemotronHOmni/
├── NemotronHOmni.swift            (431 LOC) — VLMModel wrapper, processor, splice
├── RADIOVision.swift              (323 LOC) — RADIO ViT with CPE bilinear interp
├── Parakeet.swift                 (403 LOC) — 24-layer Conformer, Transformer-XL rel-pos
├── Projectors.swift                (96 LOC) — mlp1 + sound_projection + remap helpers
└── Preprocessors.swift            (739 LOC) — NVLM tiling, mel STFT, video frames + EVS

Libraries/MLXVLM/
├── VLMVideoUtils.swift            (228 LOC) — shared video primitives (cross-VLM)
└── VLMModelFactory.swift                   — registry + omni dispatch + processor override

Libraries/MLXLLM/Models/
└── NemotronH.swift                         — embedTokens(), callAsFunction(inputsEmbeds:)

Tests/MLXLMTests/
└── NemotronHOmniSmokeTests.swift   (160 LOC) — 12 smoke tests covering all towers
```

## Appendix B — Quick verify

```bash
# Build all targets
swift build -c release

# Smoke tests (no bundle needed, ~1.5 s)
swift test --filter NemotronHOmniSmokeTests

# Real-bundle e2e (requires ~/.mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4)
# — gated future work; will land as BENCH_NEMOTRON_OMNI_BUNDLE harness.
```

---

**Last updated**: 2026-04-28. Tracks `feat(omni): native Swift port` (commit
`b4eec09`) plus the Stage 3 Swift video closeout discussed in
`research/NEMOTRON-OMNI-SWIFT-VIDEO-2026-04-28.md`.

If you're integrating omni into osaurus and hit something this doc didn't
warn you about, file an issue tagged `omni-integration` and link the
specific section so I can sharpen the doc + close the gap.
