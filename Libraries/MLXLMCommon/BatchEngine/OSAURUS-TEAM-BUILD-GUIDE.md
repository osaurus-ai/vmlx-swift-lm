# Osaurus team build + UI integration guide

Audience: osaurus host engineers wiring vmlx-swift-lm into the host app
chat / API / streaming / UI layers. This is the consolidated doc — read
this first, then dive into the sibling files for specifics.

## Table of contents

1. [What vmlx-swift-lm provides](#1-what-vmlx-swift-lm-provides)
2. [Build matrix](#2-build-matrix)
3. [Public API surface](#3-public-api-surface)
4. [Multimodal — audio, video, image](#4-multimodal-audio-video-image)
5. [Cache coordinator + KV sizing](#5-cache-coordinator--kv-sizing)
6. [Streaming + cancellation](#6-streaming--cancellation)
7. [Reasoning tokens](#7-reasoning-tokens)
8. [Tool calls](#8-tool-calls)
9. [Stop sequences](#9-stop-sequences)
10. [Memory + performance](#10-memory--performance)
11. [Debug + telemetry knobs](#11-debug--telemetry-knobs)
12. [UI connection points](#12-ui-connection-points)
13. [Testing recipe](#13-testing-recipe)
14. [Known issues + workarounds](#14-known-issues--workarounds)
15. [Pin bump checklist](#15-pin-bump-checklist)

---

## 1. What vmlx-swift-lm provides

A self-contained MLX-on-Apple-Silicon LLM/VLM/Omni inference engine.
Public products (consumed via `Package.swift` / `Package.resolved`):

| Product | Role |
|---|---|
| `MLXLMCommon` | Shared types — `LMInput`, `BatchEngine`, `CacheCoordinator`, `KVCache`, `GenerateParameters`, JANGTQ runtime, chat-template overrides, reasoning parsers, tool-call processors, stop-string matchers |
| `MLXLLM` | Text-only LLM model implementations + factory |
| `MLXVLM` | Vision-language + omni model implementations + factory |

Osaurus already imports these via `OsaurusCore/Package.swift` (lines
166-169). Pin convention: revision (commit SHA), not branch — see
the host-side pin policy from commit `e2e13b4f` (2026-04-26).

## 2. Build matrix

| Component | Required | Notes |
|---|---|---|
| Xcode | 26.4+ | Swift 6.1+, swift-tools-version 6.2 |
| macOS deployment target | 14.0+ | Apple Silicon only (M1+) |
| Metal Toolchain | 17E188+ | Separate download on Xcode 26: `xcodebuild -downloadComponent MetalToolchain` |
| Code signing | Optional for local | Use `CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO` for unsigned local builds |
| GPU family | M1 / M2 / M3 / M4 / M5 (all) | No GPU-family-specific code paths |

CLI wrapper for the host app build:

```bash
make app   # builds CLI + app, embeds CLI into Helpers/
```

Unsigned local override (when no Mac Development cert on the box):

```bash
xcodebuild -project App/osaurus.xcodeproj -scheme osaurus -configuration Release \
  -derivedDataPath build/DerivedData \
  CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO \
  DEVELOPMENT_TEAM="" build
```

After build, ad-hoc sign for clean Gatekeeper:

```bash
codesign --force --deep --sign - build/.../osaurus.app
```

## 3. Public API surface

The shape osaurus calls into. All of these are stable across the
1c62d21+ pin range; no breaking changes are planned for this release.

```swift
// Load a model bundle (factory auto-dispatches by config_omni.json /
// model_type / weight_format).
let context = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(directory: bundle))
// or for text-only LLMs:
let context = try await LLMModelFactory.shared.loadContainer(
    configuration: .init(directory: bundle))

// Build the cache coordinator. Recommended config for any
// memory-bounded inference workload:
let coord = CacheCoordinator(config: CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    diskCacheDir: ~/.osaurus/cache/kv_v2,
    modelKey: bundle.lastPathComponent,
    defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),  // ~5x KV savings
    defaultMaxKVSize: 8192,                                // ring window
    longPromptMultiplier: 2.0                              // gate
))

// Optional but recommended for Nemotron-3 / Qwen3.5/3.6 / hybrids:
coord.setHybrid(true)

// Build a batch engine.
let engine = BatchEngine(
    context: context,
    maxBatchSize: 4,         // tune per device
    cacheCoordinator: coord
)

// Build LMInput (multimodal-ready).
let input = LMInput(
    text: .init(prompt: prompt),
    image: image.flatMap { LMInput.Image.pixels($0) },
    video: video.flatMap { LMInput.Video.pixels($0) },
    audio: audio.flatMap { LMInput.Audio.pcm16k($0) }
)

// Generate.
for try await event in engine.generate(input: input,
                                        parameters: parameters) {
    switch event {
    case .text(let chunk): /* append to UI */
    case .reasoning(let chunk): /* show in <think> chip */
    case .toolCall(let call): /* dispatch the tool */
    case .completion(let info): /* unclosedReasoning, finishReason, perf */
    }
}
```

The full event types are in `MLXLMCommon/BatchEngine/BatchEngine.swift`
and the reasoning/tool/finish payloads in `MLXLMCommon/Generation.swift`.

## 4. Multimodal — audio, video, image

`LMInput` is the single struct for ALL modalities. As of pin
1c62d21:

| Modality | Field | Loader | Models that consume |
|---|---|---|---|
| Image | `LMInput.Image` | `pixels(MLXArray)`, `imageFile(URL)` | All VLMs (Qwen3-VL, Qwen3.5-VL, Mistral3, Gemma3/4, Idefics3, Pixtral, Paligemma, SmolVLM2, Nemotron-Omni, etc.) |
| Video | `LMInput.Video` | `pixels([MLXArray])`, `videoFile(URL)` | Nemotron-Omni RADIO, Qwen3-VL native video pipeline |
| Audio | `LMInput.Audio` | `pcm16k(MLXArray)`, `wavFile(URL)` | Nemotron-Omni Parakeet (STT + voice I/O) |

For Nemotron-3-Nano-Omni specifically, mix-and-match works:
- Image-only turn: `LMInput(text:image:)` — RADIO ViT path
- Audio-only turn: `LMInput(text:audio:)` — Parakeet STT + LM
- Image + audio in same turn: `LMInput(text:image:audio:)` — both
  embeddings spliced before LM forward
- Video turn: `LMInput(text:video:)` — RADIO video frames + EVS

Mediasalt fingerprinting: `CacheCoordinator.computeMediaSalt(for: input)`
emits a stable fingerprint mixed into every cache tier's hash so
"same text + same media" hits and "same text + different media"
misses correctly.

For the FULL omni hookup (4-tower wrapper, EVS, JANGTQ vs MXFP4, the
hybrid Mamba/Attention/MoE cache topology), see
`OMNI-OSAURUS-HOOKUP.md` in this folder. As of 1c62d21 the audio
"open seam" called out in §3 of that doc is **closed** — see
`OSAURUS-RELEASE-HANDOFF.md`.

## 5. Cache coordinator + KV sizing

The coordinator is osaurus-side-owned: you pass the config in, vmlx
honours it. Key behaviours:

- `usePagedCache: true` enables block-aligned prefix matching for
  cross-turn KV reuse.
- `enableDiskCache: true` enables L2 disk persistence via
  `TQDiskSerializer` v2 schema. Round-trips kvSimple, tqCompressed,
  qkv, mamba, rotating layers. Disk dir defaults to system temp;
  set `diskCacheDir` to put it under `~/.osaurus/cache/kv_v2`.
- `defaultKVMode: .turboQuant(3, 3)` — caller's `parameters.kvMode` of
  `.none` gets filled with TQ3,3 → ~5x KV memory savings.
- `defaultMaxKVSize: 8192` — caller's `parameters.maxKVSize` of `nil`
  gets filled with 8192 ONLY when `promptCount > 8192 *
  longPromptMultiplier`. Short turns pass through unbounded.
- **TO BE BUILT (Bug 2 still observable):** `prefillStepSize` is NOT
  yet auto-clamped when overcap. The `[metal::malloc] 154 GB` fatal
  on hybrid Qwen-3.6-27B-MXFP4 with a 56k-token prompt is still
  reproducible. Empirical pin + fix is the next item — see
  `OSAURUS-RELEASE-HANDOFF.md` "NOT in this PR" section.
  Workaround until landed: don't set `defaultMaxKVSize` for hybrid
  models when prompts may exceed the cap, OR keep prompts under the
  cap on the host side.

Explicit per-request `kvMode` / `maxKVSize` always win — the
coordinator only fills gaps.

Callers MUST call `coord.setHybrid(true)` for hybrid SSM + attention
models OR rely on the auto-flip at first slot admission. Without it,
SSM companion states aren't fetched/stored across turns and hybrid
cache reuse breaks silently.

## 6. Streaming + cancellation

`BatchEngine.generate(...)` returns an `AsyncSequence`. Behaviour:

- Each event arrives as soon as the producer task emits it. Backpressure
  is handled by SwiftConcurrency.
- `Task.isCancelled` propagates to the BatchEngine slot via
  `engine.cancel(requestId)` automatically. As of 2026-04-29 (commit
  `a7db6e5`), the slot reaper picks up cancelled iterators promptly
  — no orphan-slot crash.
- Client-side disconnect (TCP RST) on osaurus's HTTP layer reaches
  the engine through `bugfix/http-model-runtime-fix` PR's NIO
  channelInactive handler.

Recommended HTTP wiring (already done in osaurus):

```swift
let stream = engine.generate(input: input, parameters: parameters)
stream.onTermination = { _ in
    Task { try await engine.cancel(requestId: id) }
}
```

## 7. Reasoning tokens

vmlx routes `<think>...</think>` blocks into a separate `.reasoning`
event (vs `.text`). The completion event includes
`unclosedReasoning: Bool` so the UI can show "thinking didn't close"
when a reasoning model hits `max_tokens` mid-thought.

Reasoning parser auto-resolves from the model's `jang_config.json`
`capabilities.reasoning_parser` field:
- `deepseek_r1` (Nemotron-3, DSV4)
- `qwen` / `qwen3` (Qwen3.5/3.6)
- `harmony` (Gemma-4 channels)
- `gpt_oss`
- See `MLXLMCommon/Generation.swift` for the full enum.

Toggle at request time via `enable_thinking` chat-template kwarg or
the `/think` / `/no_think` magic strings in the user message
(Nemotron-Omni template handles both).

## 8. Tool calls

vmlx ships parsers for most major formats:
- XML function (Qwen3, Nemotron)
- Mistral inline + EOS-bracket
- Gemma-4 Harmony (with custom escape markers)
- JSON (default + custom tags)
- Pythonic (with/without brackets)
- Kimi K2, MiniMax M2 (interleaved-thinking)
- DSML (DeepSeek)

Auto-dispatch via `jang_config.json` `capabilities.tool_parser`. The
`ToolCallProcessor` consumes the streaming token feed and emits
`.toolCall` events as soon as a complete invoke is parsed.

Edge cases handled (see `Tests/MLXLMTests/ToolCallEdgeCasesTests.swift`):
- Stray `</think>` in content mode
- Closer-before-opener (literal content)
- Multi-line value trimming
- Nested `<|channel>` (first closer wins)
- maxTokens truncation mid-opener (partial = content)

## 9. Stop sequences

`GenerateParameters.extraStopStrings: [String]` accepts caller-provided
stop sequences. `StopStringMatcher` (in `MLXLMCommon`) handles
multi-byte UTF-8 boundaries correctly — never truncates mid-codepoint.

Per-model auto-stop tokens are baked in via the tokenizer config and
chat template. Don't add EOS to `extraStopStrings`; it's already
handled.

## 10. Memory + performance

| Metric | Knob | Notes |
|---|---|---|
| Wired memory hint | `parameters.wiredMemoryGB` | Sets MLX wired-memory floor |
| Prefill chunk size | `parameters.prefillStepSize` | Default 512. Auto-clamp on overcap is still TODO (see Bug 2 — NOT in this PR). |
| Batch size | `BatchEngine(maxBatchSize:)` | Higher = more concurrent slots; tune per device |
| KV quantization | `parameters.kvMode` / `defaultKVMode` | TQ(3,3) ~5x smaller than fp16 |
| Sliding window | `parameters.maxKVSize` / `defaultMaxKVSize` | Caps absolute KV memory regardless of prompt length |
| Disk cache size | `CacheCoordinatorConfig.diskCacheMaxGB` | Default 50 GB |

For Nemotron-Omni specifically:
- MXFP4: ~18 GB total + KV
- JANGTQ4: ~16 GB total + KV
- JANGTQ2: ~9 GB total + KV
- Multimodal addon: 2.7 GB (RADIO ViT 1.6 GB + Parakeet 0.4 GB +
  projectors)

## 11. Debug + telemetry knobs

Environment variables that affect runtime behaviour:

| Env | Effect |
|---|---|
| `MLX_CLEAR_LIBRARY_RELEASE=1` | Restore eager pipeline release in mlx Device::clear_library (DEBUG only — Bug 1 returns) |
| `OSAURUS_MLX_CLEAR_LIBRARY_TRACE=1` | stderr-log every clear_library trigger with kernel name + source diff |
| `OSAURUS_STRESS_RUN=1` | Enable the L1 stress matrix in MLXLMStressTests |
| `OSAURUS_STRESS_HYBRID_MODEL=<path>` | Path to hybrid model used by the L1 stress matrix |

Logging via `os.log`:
- `MLXLMCommon.BatchEngine` — admission, cache hits, prefill chunks
- `MLXLMCommon.CacheCoordinator` — fetch outcomes (miss / paged / disk)
- `MLXLMCommon.JangLoader` — JANGTQ bundle detection + sidecar load
- `MLXLMCommon.MLXErrorRecovery` — global MLX error handler events

## 12. UI connection points

What the host UI layer needs to wire up (this is the osaurus team's
work; vmlx side is API-stable for these):

### Drag-and-drop / file-picker for attachments

For each attachment, dispatch by UTType:

```swift
if utType.conforms(to: .image) {
    let pixels = try MediaProcessing.loadImagePixels(from: url)
    attach(LMInput.Image.pixels(pixels))
} else if utType.conforms(to: .movie) || utType.conforms(to: .video) {
    let frames = try MediaProcessing.loadVideoFrames(from: url, fps: 1)
    attach(LMInput.Video.pixels(frames))
} else if utType.conforms(to: .audio) {
    let waveform = try MediaProcessing.loadAudioPCM(from: url, sampleRate: 16000)
    attach(LMInput.Audio.pcm16k(waveform))
}
```

`MediaProcessing` lives in `Libraries/MLXVLM/MediaProcessing.swift`.

### Streaming display

- `.text(chunk)` → append to message body
- `.reasoning(chunk)` → append to "thinking" sub-bubble (collapsible)
- `.toolCall(call)` → render tool-call card, dispatch tool, append
  result as tool-result message
- `.completion(info)` → finalize message, show stats / unclosedReasoning
  warning if applicable

### Cancellation

When the user closes a tab / cancels a generation:

```swift
streamTask.cancel()
// BatchEngine + osaurus HTTP layer handle the rest
```

### Multi-turn cache reuse

For cache reuse to work, the host MUST:
1. Reuse the same `CacheCoordinator` instance across turns of the
   same conversation.
2. Pass the same `modelKey` for cache scoping.
3. Pass `mediaSalt` (computed via `CacheCoordinator.computeMediaSalt`)
   if the conversation has any image/video.
4. Ensure `coord.setHybrid(true)` is set for hybrid models (auto-flip
   on first admission also works).

Without this, every turn does a full prefill — no L1 paged cache
hit, no L2 disk hit, no SSM-state reuse.

## 13. Testing recipe

Three test layers, all run before pin bump:

### L1 — vmlx-swift-lm `swift test`

```bash
cd vmlx-swift-lm
swift test                                    # full suite (~700 tests)
swift test --filter CacheCoordinator          # cache-only subset
swift test --filter MLXLMStressTests          # stress matrix
```

Note: full-suite runs may flake on M5 / macOS 26.4 due to a Metal
validation race in concurrent tests (`AGXG17XFamilyCommandBuffer
tryCoalescing...` assert OR `EvalTests.testConcurrentSampling`
SIGSEGV). Both are environment-level, hit baseline AND patched
identically. Use `--filter` to focus on cache + bug-fix surfaces if
you need a clean signal.

### L2 — osaurus HTTP stress

```bash
# Server already running
python3 scripts/eval_http_stability.py                     # host-side S1-S6 stability suites
python3 investigation/repros/stress_extras.py --only S7    # Bug 1 repro
python3 investigation/repros/stress_extras.py --only S8    # Bug 2 repro
python3 investigation/repros/stress_extras.py --only S9    # 20-turn agent
python3 investigation/repros/stress_extras.py --only S10   # 100-burst
```

### L3 — manual UI smoke

For each modality:
- Drop image, ask question → response references the image
- Drop audio, ask transcription → text matches audio content
- Drop video, ask description → response references frames

Run on at least one M-series chip; ideally M1 + M4 to cover both
ends of the GPU family range.

## 14. Known issues + workarounds

| Issue | Workaround | Owner |
|---|---|---|
| Bug 1 (Metal pipeline-evict on warm-disk-cache 2nd request) | mlx-swift fork commit `fa3a9616` — keeps pipelines alive across `clear_library`. Set `MLX_CLEAR_LIBRARY_RELEASE=1` to revert for testing. | vmlx (FIX READY) |
| Bug 2 (`metal::malloc 154 GB` on hybrid + over-cap prompt) | None yet — empirical pin + fix is the next item. Workaround: don't set `defaultMaxKVSize` for hybrid models when prompts may exceed cap. | vmlx (TO BE BUILT) |
| swift-embeddings `branch: main` pin | Convert to `revision:` per the host-side revision-pin policy | osaurus |
| swift-crypto 3.15.1 vs 4.5.0 split | OK for now; bump to 4.x when PR #958 (HPKE) lands | osaurus |
| Full-suite test flake on M5 | Use `--filter` for clean signal; investigate later | engine team |
| signing certs on this M5 | Use ad-hoc `codesign -s -` for local; Apple Dev cert needs to be re-issued | osaurus ops |

## 15. Pin bump checklist

Before bumping vmlx-swift-lm pin in `OsaurusCore/Package.swift`:

- [ ] `swift test --filter CacheCoordinator` passes against new vmlx pin
- [ ] `swift test --filter MLXLMStressTests` (when bodies land) passes
- [ ] L2 S1-S10 all green against the new build
- [ ] `STRESS_REPORT.md` regenerated and committed
- [ ] Comment block in `Package.swift` updated to cite new commits
  (the existing comment claiming `a7db6e5` closes Bug 1 is stale —
  must be rewritten to reflect the new mlx-swift fork pin + vmlx
  Bug 2 clamp)
- [ ] mlx-swift dep converted from `branch:` to `revision:` SHA at
  the same time (host-side pin policy)


---

## 16. End-to-end coverage matrix (verified 2026-04-30 on M5 Max)

Driven from `RunBench` against
`OsaurusAI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4` from the local model
cache. Every row was exercised on real weights with the
`cf8c525` vmlx-swift-lm pin and the patched mlx-swift fork
(`osaurus-ai/mlx-swift osaurus-0.31.3` @ `e0b6111` referencing the
`osaurus-ai/mlx fix/clear-library-no-release` submodule branch).
Full pass/fail grid in `STRESS_REPORT.md`.

| Surface | Verified row | Notes |
|---|---|---|
| **Parakeet (audio STT / voice I/O)** | OmniBench 6a + 6b | encoder shape `[T, 2688]`, LMInput.audio end-to-end at ~66 tok/s |
| **RADIO (image + video vision)** | OmniBench 3, 4, 5, 5b, 9 | image single + multi-turn, video frames + LMInput end-to-end |
| **NemotronHJANGTQ routing** | OmniBench loads `NemotronHOmni` via `VLMModelFactory` — factory dispatch by `weight_format` works for MXFP4 / JANGTQ4 / JANGTQ2 with no caller change |
| **Hybrid SSM (Mamba/Attn/MoE)** | OmniBench 11 + StabilityBench S10 | `setHybrid(true)` auto-flips on first admission; SSM state companion cache round-trips across two engine instances |
| **Reasoning toggle** | OmniBench 7 + 8 | `enable_thinking: false` parity, ON→OFF→ON swap |
| **Tool calls** | tested via SampleTests + ToolTests in `swift test` baseline (1190/1190 OsaurusCore + 53/53 vmlx focused) |
| **Stop sequences** | StopStringMatcher unit tests + GenerateParameters.extraStopStrings honored — multi-byte UTF-8 boundary safe |
| **L1 paged cache hit (shared prefix)** | StabilityBench S3 | 1st 0.5s → 2nd 0.3s warm-cache speedup |
| **L2 disk cache restore (full prefix)** | StabilityBench S2, S8, S10 | `restored N tokens from disk, prefilling 0 remaining` log fired without crash |
| **L2 disk cache restore (partial)** | StabilityBench S5 (paged hit + remaining tail) | `restored 256 tokens, prefilling 81 remaining` etc |
| **TurboQuant KV mode + disk** | StabilityBench S8 | `defaultKVMode: .turboQuant(3,3)` round-trips through TQDiskSerializer v2 |
| **mediaSalt isolation** | OmniBench 10 | audio A vs B with same text prompt → cache scopes don't alias |
| **Concurrent batched B=2** | OmniBench B2 + StabilityBench S6 | batched decode through hybrid topology |
| **Cancel + recovery** | StabilityBench S7 | break stream after 3 events, next request completes cleanly |
| **Memory.clearCache mid-run** | StabilityBench S9 | engine survives forced eviction |
| **Multi-turn agent loop (8 turns)** | StabilityBench S4 | grow context with system + tool-result style, no slot leak |
| **OpenAI `input_audio` / `video_url` API surface** | host-side branch `feat/openai-multimodal-audio-video` (commit `e7f68045`) routes to `LMInput.audios` / `.videos` via `MessageContentPart` — vmlx side already supports these via `LMInput`'s audio/video fields (commits `ae49c7c` + `3b78db4` in `cf8c525`) |
| **Long prompt (~16k)** | StabilityBench S5 | chunked prefill engages, no cap clamp needed at this length |
| **Over-cap prompt (60k+, 154 GB territory)** | observed on M5 Max — peak 418 GB, escapes only via 128 GB unified memory + macOS swap. **Bug 2 fix still TODO**: needs `mx::malloc > 1 GB` tracer to identify the exact allocation site before designing the right-shape clamp. M4 Pro / smaller machines remain susceptible. |

## 17. API surface (host-facing)

What the host calls into. All stable across the `cf8c525` vmlx pin range.

### Inputs

```swift
// LMInput is the single struct for ALL modalities.
public struct LMInput: Sendable {
    public var text: Text
    public var image: Image?    // RADIO ViT (VLMs) — 256 tokens/tile
    public var video: Video?    // RADIO frames + EVS (Nemotron-Omni, Qwen3-VL)
    public var audio: Audio?    // Parakeet STT / voice (Nemotron-Omni)
}

// Convenience constructors:
LMInput.Image.pixels(MLXArray)   // pre-decoded HWC pixels
LMInput.Image.imageFile(URL)     // read + decode for the caller
LMInput.Video.pixels([MLXArray]) // pre-extracted frames
LMInput.Video.videoFile(URL)     // read + frame-extract
LMInput.Audio.pcm16k(MLXArray)   // 16 kHz mono PCM
LMInput.Audio.wavFile(URL)       // read + resample for the caller
```

The `additionalContext` map carries chat-template kwargs:
```swift
var ui = UserInput(prompt: prompt)
ui.additionalContext = ["enable_thinking": false]   // or true
let lmInput = try await context.processor.prepare(input: ui)
```

### Cache coordinator config — every knob

```swift
CacheCoordinatorConfig(
    usePagedCache:           true,                                      // L1 in-memory paged
    enableDiskCache:         true,                                      // L2 ~/.osaurus/cache/kv_v2 etc.
    pagedBlockSize:          256,                                       // tokens per L1 block
    maxCacheBlocks:          1024,                                      // upper bound on resident L1 blocks
    diskCacheMaxGB:          50,                                        // L2 LRU cap
    diskCacheDir:            URL(fileURLWithPath: "~/.osaurus/cache/kv_v2"),
    ssmMaxEntries:           64,                                        // SSM-state companion cache size
    modelKey:                bundle.lastPathComponent,                  // disk-cache + paged-cache scope
    defaultKVMode:           .turboQuant(keyBits: 3, valueBits: 3),     // ~5x KV memory savings vs fp16
    defaultMaxKVSize:        8192,                                      // ring window
    longPromptMultiplier:    2.0                                        // gate: only clamp once prompt > cap*multiplier
)
```

After construction:
```swift
coord.setHybrid(true)                              // hybrid SSM/Mamba models — idempotent
let salt = coord.computeMediaSalt(for: input)     // VL salt, mixed into cache hash for image/video
```

### Stream events

```swift
for try await event in engine.generate(input: input, parameters: parameters) {
    switch event {
    case .chunk(let text):       // visible content (after reasoning extraction)
    case .reasoning(let chunk):  // <think>...</think> block content
    case .toolCall(let call):    // parsed tool invocation (model + parser dependent)
    case .info(let info):        // GenerateCompletionInfo: finishReason, unclosedReasoning, perf
    }
}
```

### OpenAI HTTP body shapes (osaurus-side, already wired)

| OpenAI field | vmlx surface | Models |
|---|---|---|
| `messages[].content` (string) | `UserInput(prompt:)` | all |
| `messages[].content[].type=text` | text channel | all |
| `messages[].content[].type=image_url` | `LMInput.Image` | VLMs |
| `messages[].content[].type=video_url` | `LMInput.Video` | Nemotron-Omni RADIO, Qwen3-VL |
| `messages[].content[].type=input_audio` | `LMInput.Audio` | Nemotron-Omni Parakeet |
| `tools` | tool-call processor (chat-template controlled) | model-dependent |
| `temperature`, `top_p`, `max_tokens` | `GenerateParameters` | all |
| `repetition_penalty` | `GenerateParameters.repetitionPenalty` (1.0 treated as no-op per `cf8c525`) | all |
| `presence_penalty`, `frequency_penalty` | additive penalties (0 = no-op) | all |
| `stream: true` | SSE — host writes `data: ...` per event | all |

Host-side mapping lives in `feat/openai-multimodal-audio-video` (osaurus PR
target). vmlx side accepts `LMInput.audios` / `.videos` collections so the
host can pass multiple audio or video parts without flattening.

