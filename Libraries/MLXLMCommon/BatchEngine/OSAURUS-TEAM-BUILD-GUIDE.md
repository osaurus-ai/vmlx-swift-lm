# Osaurus team build + UI integration guide

Audience: tpae + osaurus engineers wiring vmlx-swift-lm into the host app
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
tpae's 2026-04-26 commit `e2e13b4f`.

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
python3 scripts/eval_http_stability.py                     # tpae's S1-S6
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
| swift-embeddings `branch: main` pin | Convert to `revision:` per tpae's 2026-04-26 policy | osaurus |
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
  the same time (tpae's policy)

