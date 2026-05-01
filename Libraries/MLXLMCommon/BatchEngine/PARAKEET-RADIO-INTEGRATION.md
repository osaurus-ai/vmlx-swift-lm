# Parakeet (audio) + RADIO (vision) host-integration spec

Status: spec draft for the host team UI hookup of Nemotron-3-Nano-Omni's
two non-text modalities. Pinned to vmlx revision `13abe40`.

The Nemotron-3 omni bundle ships THREE encoders (text + Parakeet audio +
RADIO vision) that share one MoE+SSM hybrid backbone. The host already
exercises the text path; this doc spells out exactly what the audio +
vision paths require so the UI can wire them up without round-tripping
through this repo for clarification.

---

## 1. API surface — what the host calls

Single entry point. Same `applyChatTemplate` + `BatchEngine.generate`
pipeline as text-only, with audios/videos populated on `UserInput`:

```swift
import MLXLMCommon

let userInput = MLXLMCommon.UserInput(
    chat: chatMessages,                  // existing text path
    tools: tools,
    additionalContext: ["enable_thinking": true],
    audios: [.url(audioURL)],            // NEW — audio attachments
    videos: [.url(videoURL)],            // NEW — video attachments
    images: [.url(imageURL)]             // pre-existing image path
)

// BatchEngine.generate is unchanged — the processor handles modality
// dispatch internally based on which fields are populated.
for await event in await batchEngine.generate(input: userInput, ...) {
    // ...
}
```

`UserInput.Audio` and `UserInput.Video` are unions that accept either a
file URL or a pre-decoded float array. Both paths converge on the same
preprocessor.

| Surface | Type | Where |
|---|---|---|
| `UserInput.audios` | `[UserInput.Audio]` | `Libraries/MLXLMCommon/UserInput.swift:207` |
| `UserInput.videos` | `[UserInput.Video]` | `Libraries/MLXLMCommon/UserInput.swift:207` |
| Audio decoder | `nemotronOmniLoadAudioFile` | `Libraries/MLXVLM/Models/NemotronHOmni/Preprocessors.swift:465` |
| Video frame extractor | `nemotronOmniExtractVideoFrames` | `Libraries/MLXVLM/Models/NemotronHOmni/Preprocessors.swift:577` |
| Audio mel preproc | `NemotronHOmniProcessor.preprocess(audios:)` | `Libraries/MLXVLM/Models/NemotronHOmni/NemotronHOmni.swift:467` |
| Video preproc | `NemotronHOmniProcessor.preprocess(videos:)` | `Libraries/MLXVLM/Models/NemotronHOmni/NemotronHOmni.swift:495` |
| Parakeet encoder | `NemotronHParakeetEncoder` | `Libraries/MLXVLM/Models/NemotronHOmni/Parakeet.swift:328` |
| RADIO vision encoder | `NemotronHRADIOVisionTransformer` | `Libraries/MLXVLM/Models/NemotronHOmni/RADIOVision.swift` |

---

## 2. Audio path — Parakeet ASR encoder

### What it accepts

- **File URL**: any container AVAudioConverter can decode — wav, mp3,
  m4a, flac, ogg/opus, aac. Sample rate doesn't matter (resampled to
  16 kHz mono Float32 internally).
- **Raw `[Float]`**: must already be 16 kHz mono. No resampling done.

### What the host UI must enforce

Nothing on the host side beyond accepting the file. Don't try to decode
audio yourself — `nemotronOmniLoadAudioFile` does sinc-quality
resampling via AVAudioConverter; rolling your own would lose fidelity.
Just hand the file URL through.

osaurus's `extractAudioSources` already handles this correctly via
`materializeMediaDataUrl` (writes base64 `data:audio/...` content parts
to a temp `.wav`/`.mp3`/`.m4a` file, hands the URL to vmlx).

### Format extension hints

The file extension drives AVAudioConverter dispatch. The host's
canonical extensions:

| Client format string | Temp file ext | Notes |
|---|---|---|
| `wav` / `wave` / `x-wav` | `.wav` | |
| `mp3` / `mpeg` / `x-mpeg` | `.mp3` | |
| `m4a` / `x-m4a` | `.m4a` | osaurus audio path canonicalizes `mp4 → m4a` (audio container only, NOT video — see `materializeMediaDataUrl` audio-mime guard) |
| `flac` | `.flac` | |
| `ogg` / `opus` | `.ogg` | |

### Streaming vs one-shot

Parakeet runs as a **non-streaming encoder** — the full audio is encoded
into a token sequence before decode begins. Parakeet itself does not
emit incremental events. The host UI should:

- Show a "transcribing audio…" spinner while `BatchEngine.generate` is
  blocked on audio preprocessing (typically <500 ms for clips ≤ 30 s on
  M5 Max).
- Display TTFT (time-to-first-token) including the audio-encode latency
  rather than from-prefill — `runtime_start` → `first_token` already
  measures this end-to-end via `TTFTTrace`.

### Length limits

- Soft cap: 30 s per audio attachment. Longer clips work but mel frame
  count grows proportionally and consumes hybrid SSM cache.
- Hard cap: bounded by `defaultMaxKVSize` if set on the
  `CacheCoordinator`. **Do not set `defaultMaxKVSize` for hybrid
  models** until Bug 2 lands (over-cap allocation triggers a 100+ GB
  metal::malloc on some hosts).

### UI input recommendation

```swift
// SwiftUI drag-drop or .fileImporter for audio
let allowedTypes: [UTType] = [.audio, .mp3, .wav]

// Pass URL straight through to the OpenAI input_audio shape:
{
  "type": "input_audio",
  "input_audio": { "data": <base64>, "format": "wav" }
}
```

---

## 3. Vision path — RADIO encoder (also serves images for Nemotron-3)

### What it accepts

- **Image** (single frame): JPEG / PNG / HEIC / TIFF / BMP / WebP via
  `CGImageSource`. AVAsset for HEIC/HEIF.
- **Video**: any AVAsset-readable container (mp4, mov, m4v). Frames
  extracted at fixed FPS via `nemotronOmniExtractVideoFrames`.

### Frame budget

| Modality | Frame count | Notes |
|---|---|---|
| Single image | 1 | RADIO encodes once, ~80–110 patch tokens after pixel-shuffle |
| Video | 8 frames default | Sampled uniformly across clip duration |
| Long video | up to 16 frames | Set via `processing.frameStride` on `UserInput.Processing` |

Each RADIO frame produces ~256 vision tokens after the spatial-merge
2×2 pixel-shuffle. Eight frames → ~2K vision tokens before the
multi-modal projector compresses them onto the language hidden size.

### What the host must NOT do

- Don't pre-resize. RADIO's resize layer + bilinear interpolation
  (`nemotronOmniBilinearResize2D` at `RADIOVision.swift:35`) handles
  arbitrary input shapes and is deterministic across image sizes.
  Resizing on the host loses information at non-square aspect ratios
  because the host doesn't know RADIO's per-axis target.
- Don't normalize pixel values. RADIO normalizes internally (mean/std
  on RGB channels match the trained encoder).
- Don't extract video frames on the host. `nemotronOmniExtractVideoFrames`
  uses AVAssetImageGenerator with the correct timestamping (avoids
  off-by-one on first/last frame).

### RADIO checkpoint quirks

The RADIO ViT in Nemotron-3 is the AM-RADIO/CRADIO variant. It does NOT
share weights with CLIP / SigLIP / Pixtral. If a future Nemotron bundle
ships a different vision encoder, the model_type→processor map at
`VLMModelFactory.swift:138-140` (`nemotron_h_omni` /
`NemotronH_Nano_Omni_Reasoning_V3`) is the dispatch point.

### UI input recommendation

```swift
// SwiftUI .fileImporter or drag-drop for video
let allowedTypes: [UTType] = [.image, .movie, .video, .quickTimeMovie]

// OpenAI shape — image_url for stills, video_url for clips:
{ "type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."} }
{ "type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."} }
```

osaurus's `extractVideoSources` + `materializeMediaDataUrl` already
shepherd both shapes onto file URLs that vmlx consumes.

---

## 4. Cache + memory considerations

### Hybrid SSM behavior under audio/video

Nemotron-3's per-layer cache list contains 23 `MambaCache(size=2)` SSM
slots. Audio + video tokens flow through the same SSM layers as text
tokens — the SSM state cumulates. For multi-turn conversations with
repeated media attachments:

- The disk L2 cache (`CacheCoordinator` with `enableDiskCache: true`)
  keys on `(model, normalized_prefix_hash, media_salt)` where
  `media_salt` is computed from pixel + audio sample bytes. Different
  attachments → different salt → no false sharing across requests.
- Tested via `CacheCoordinatorMediaSaltTests` (8 tests, all pass at pin
  `13abe40`).

### `setHybrid(true)` requirement

Hybrid models require the SSM-state companion cache to round-trip. On
osaurus, `ModelRuntime.installCacheCoordinator` eager-flips this for
any name matching `isKnownHybridModel(name:)` (substring `nemotron-3`).
BatchEngine also auto-flips on first slot admission, so the eager set
is belt-and-suspenders rather than required.

### Memory profile (M5 Max, 30B Nemotron-3 MXFP4)

Empirical baselines from OmniBench:

| Workload | Peak RAM | Decode tok/s |
|---|---|---|
| Text-only, 4K prompt | ~22 GB | ~65 |
| + 1 image (1024×1024) | ~24 GB | ~62 |
| + 8-frame video (256×256) | ~28 GB | ~58 |
| + 30 s audio clip | ~25 GB | ~60 |
| All three modalities together | ~32 GB | ~55 |

---

## 5. Streaming + cancellation

`BatchEngine.generate` returns `AsyncThrowingStream<GenerationEvent, Error>`.
For omni inputs the stream emits the same event types as text-only:

- `.chunk(String)` — visible content tokens
- `.reasoning(String)` — `<think>` block tokens (when `enable_thinking=true`)
- `.toolCall(...)` — structured tool calls
- `.usage(...)` — final token counts on close

Cancellation: dropping the stream task cancels prefill mid-encode.
Audio/video preprocessing IS pre-prefill, so cancelling during the
"transcribing audio..." spinner is safe — the file decode is in
`Task.detached` and respects cooperative cancellation.

---

## 6. Error surfaces

| Failure | Where it surfaces | Recovery |
|---|---|---|
| Unsupported audio format | `nemotronOmniLoadAudioFile` throws | Show user "audio format unsupported"; vmlx process stays up because `MLX.setErrorHandler` traps non-fatal MLX errors (osaurus `MLXErrorRecovery.installGlobalHandler`) |
| Truncated/corrupt video | `nemotronOmniExtractVideoFrames` throws | Same |
| Wrong pixel layout / dimensions | RADIO ViT throws on shape mismatch | Same — server stays up, request gets a 4xx |
| Out-of-memory on large media batch | `metal::malloc` may exceed budget | Set `OSAURUS_MLX_MALLOC_TRACE=1` to log allocation site; clamp batch size |

---

## 7. End-to-end coverage (already passing on the pin)

OmniBench at vmlx pin `13abe40` runs **13 / 13** scenarios:

1. Text-only chat
2. Single-image VQA
3. Multi-image grounding
4. Single-frame video
5. Multi-frame video (8 frames)
6. Audio transcription (30 s clip)
7. Audio + image co-grounding
8. Multi-turn with image carry-over
9. Multi-turn with audio carry-over
10. Reasoning (`enable_thinking=true`) over image
11. Tool call dispatch with audio attachment
12. Hybrid SSM warm-pass after disk-cache hit
13. Media-salt isolation (same prompt + different image → distinct cache entries)

Each scenario's expected event counts and content invariants are
asserted in `RunBench/OmniBench.swift`. To run on the host:

```bash
cd vmlx-swift-lm
swift run -c release RunBench   # default: OmniBench
```

The runner needs `OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4` weights
on disk at the standard `~/MLXModels` path or `MLX_MODELS_DIR` override.

---

## 8. Versioning

This spec is pinned to:

- vmlx-swift-lm `13abe40` (osaurus PR #967 pin)
- mlx-swift `osaurus-0.31.3` at `e0b6111` (osaurus PR #967 pin)
- Nemotron-3-Nano-Omni-30B-A3B-MXFP4 weights revision shipped with the
  bundle on HF.

Any future bundle whose `config.json.model_type` is NOT one of
`{nemotron_h_omni, NemotronH_Nano_Omni_Reasoning_V3}` will route through
the standard VLM factory dispatch and these Parakeet/RADIO encoder
classes will not be loaded.
