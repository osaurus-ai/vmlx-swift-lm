# Nemotron-Omni Voice + Audio + Video Integration Guide

**Audience**: osaurus app/UI engineers wiring the multimodal capabilities
exposed by `Libraries/MLXVLM/Models/NemotronHOmni/` into product
features (voice mode, drag-drop audio/video, push-to-talk, etc.).

**Companion doc**: `OMNI-OSAURUS-HOOKUP.md` covers the bundle/factory/cache/
HTTP-API contracts. This file picks up where that one ends — once the API
surface lands content into `UserInput.audios`/`videos`, **how do you
build a great voice/multimodal product on top?**

**Scope boundary**: the modality contract is honest. Nemotron-Omni is a
**speech-IN, text-OUT** model. There is no neural vocoder in the bundle.
This doc tells you exactly what's available natively in Swift, what
needs to be a system-TTS fallback, and what hooks to leave open if you
want to layer a third-party neural TTS later.

---

## TL;DR for product engineers

```swift
import MLXVLM   // for NemotronHOmniMicRecorder + NemotronHOmniSpeaker

// === Voice IN (mic → omni) ===
let mic = NemotronHOmniMicRecorder()
try mic.start()                             // user holds button
// ... user speaks ...
let pcm = try mic.stop()                    // [Float] @ 16 kHz mono
let userInput = UserInput(
    prompt: "(spoken)",
    audios: [.samples(pcm, sampleRate: 16_000)])

// === Voice OUT (system TTS) ===
let speaker = NemotronHOmniSpeaker(voiceLanguage: "en-US")
for try await chunk in chatStream {
    text += chunk
    // optional: speak completed sentences as they finish
    if endsSentence(text) { speaker.speak(currentSentenceBuffer) }
}
```

That's the entire surface area. Six lines of glue plus your own
"sentence completed" detector. Apple's TTS handles 70+ languages
on-device — no network calls, no model download, no licensing.

---

## 1. Architecture map — what each file owns

```
osaurus-ai/vmlx-swift-lm  (this repo, what your pin points at)
├── MLXVLM/Models/NemotronHOmni/
│   ├── NemotronHOmni.swift        — the model + processor (HTTP-side wired here)
│   ├── RADIOVision.swift          — vision encoder (32-block ViT-Huge, CPE bilinear)
│   ├── Parakeet.swift             — audio encoder (24-layer Conformer, rel-pos attn)
│   ├── Projectors.swift           — mlp1 + sound_projection (vision/audio → LM)
│   ├── Preprocessors.swift        — NVLM tile, mel STFT, video frames + EVS
│   └── AudioIO.swift              — ⇐ THIS IS WHERE THE VOICE TOOLS LIVE
│
└── MLXVLM/VLMVideoUtils.swift     — cross-VLM (Qwen / Kimi / Gemma 4 reusable)

osaurus-ai/osaurus  (your app)
├── ChatEngine                      — drives chat session + media intake
├── ChatUI                          — SwiftUI / AppKit views (drop zones, mic button)
└── VoiceMode                       — NEW (or wherever your voice feature lives)
```

The four classes in `AudioIO.swift` are the only public symbols the
voice-mode UI needs to know about:

| Symbol | Purpose | Used by |
|---|---|---|
| `NemotronHOmniMicRecorder` | live mic → 16 kHz mono PCM | push-to-talk, voice mode |
| `NemotronHOmniSpeaker` | text → speech via AVSpeechSynthesizer | voice mode (TTS replies) |
| `nemotronOmniLoadAudioFile(_:targetSampleRate:)` | file URL → 16 kHz PCM | drag-drop audio file |
| `linearResamplePCM(_:fromRate:toRate:)` | in-memory rate conversion | 3rd-party audio formats |

All four are thread-safe in the obvious way: `MicRecorder` synchronizes
on its own queue, `Speaker` wraps `AVSpeechSynthesizer` (which is
already main-actor-safe), and the two functions are pure transforms.

---

## 2. Voice IN — recording from the mic

### 2.1 Permissions (caller owns these)

The vmlx code does NOT request mic permission. Your app must:

1. **Info.plist**: `NSMicrophoneUsageDescription` with the user-facing
   reason string (App Store will reject without it).
2. **Runtime**: request permission BEFORE calling `mic.start()`.

```swift
import AVFAudio

func ensureMicPermission() async -> Bool {
    switch AVAudioApplication.shared.recordPermission {
    case .granted: return true
    case .denied:  return false
    case .undetermined:
        return await AVAudioApplication.requestRecordPermission()
    @unknown default: return false
    }
}
```

If your app is sandboxed (Mac App Store / hardened runtime), also add
the `com.apple.security.device.audio-input` entitlement.

### 2.2 Push-to-talk pattern

```swift
@MainActor
final class PushToTalkController {
    private let mic = NemotronHOmniMicRecorder()
    private var startedAt: Date?

    func onPressDown() {
        Task {
            guard await ensureMicPermission() else {
                showPermissionDeniedSheet()
                return
            }
            do {
                try mic.start()
                startedAt = Date()
                showRecordingIndicator()
            } catch {
                showError(error)
            }
        }
    }

    func onPressUp() async throws -> [Float] {
        let pcm = try mic.stop()
        let duration = Date().timeIntervalSince(startedAt ?? Date())
        startedAt = nil
        hideRecordingIndicator()
        // Soft floor — too short = misclick. ~150ms.
        guard pcm.count > 2_400 else { return [] }
        // Soft ceiling — Parakeet handles long audio but UX-wise
        // anything over ~30s should be a separate "long form" UI.
        if duration > 30 { showLongRecordingWarning() }
        return pcm
    }
}
```

### 2.3 Continuous voice-mode pattern (always-listening)

Less common, more complex. Use VAD (voice activity detection) to chunk
the stream. A working starting point is `MLXEmbedders` for VAD or just
energy-thresholding the PCM samples — pick what fits your UX:

```swift
// Energy-threshold VAD (simple — not robust to noise, good enough for
// quiet desktop environments).
extension Array where Element == Float {
    var rms: Float {
        sqrt(reduce(0) { $0 + $1 * $1 } / Float(count))
    }
}

// Inside your AVAudioEngine tap (after NemotronHOmniMicRecorder
// extracts 16 kHz PCM in chunks), watch RMS — sustained low RMS
// over ~600 ms = end-of-utterance, flush accumulated PCM to omni.
```

For production-quality VAD, layer Silero or Web Audio VAD on top.
That's product-specific; vmlx doesn't ship a VAD.

### 2.4 The audio actually fed to the model

```swift
// Whatever you captured with NemotronHOmniMicRecorder OR loaded from
// a file goes through UserInput.audios as one of three shapes:

// Live mic capture or in-memory PCM:
UserInput(prompt: "...", audios: [.samples(pcm, sampleRate: 16_000)])

// User-dropped audio file (.m4a, .wav, .mp3, .aac, .flac, .mov):
UserInput(prompt: "...", audios: [.url(fileURL)])

// Pre-loaded MLXArray waveform (rare; used for replay buffers):
UserInput(prompt: "...", audios: [.array(mlxArray, sampleRate: 16_000)])
```

The processor handles resampling automatically. Don't try to
pre-process; let `AVAudioConverter` (file path) or `linearResamplePCM`
(in-memory path) do the work — they're already tuned and tested.

---

## 3. Voice OUT — text → speech

### 3.1 The honest constraint

**Nemotron-3-Nano-Omni has NO audio decoder.** The model's output is
text tokens. There is no diffusion model, no vocoder, no waveform
generator in the bundle. Voice OUT requires either:

| Option | Quality | Latency | Cost | Ships with bundle |
|---|---|---|---|---|
| `NemotronHOmniSpeaker` (system TTS) | OK | ~50 ms first audio | free | ✅ yes |
| Coqui XTTS (BYOM) | great | ~500 ms | free | ❌ separate model |
| F5-TTS / VITS (BYOM) | great | varies | free | ❌ separate model |
| ElevenLabs API | great | ~300 ms (network) | $$ | ❌ network call |
| OpenAI TTS API | great | ~400 ms (network) | $ | ❌ network call |

`NemotronHOmniSpeaker` is the **default-ship** option — runs entirely
on-device, ~70 languages, zero install. For "wow factor" voice
features ship a neural TTS too, but `NemotronHOmniSpeaker` should be
what falls out of `osaurus chat --voice` with no extra config.

### 3.2 Streaming-text → streaming-speech

The omni LLM streams tokens. You want speech to start as soon as the
first sentence completes, not after the whole response is finished.
Pattern:

```swift
@MainActor
final class StreamingSpeaker {
    private let speaker = NemotronHOmniSpeaker(voiceLanguage: "en-US")
    private var buffer = ""

    func feed(_ chunk: String) {
        buffer += chunk
        // Flush at sentence boundaries — speech feels natural, ~50 ms
        // first-audio latency from the LLM emitting the period.
        while let endIdx = nextSentenceEnd(in: buffer) {
            let sentence = String(buffer[..<endIdx])
            speaker.speak(sentence)
            buffer.removeSubrange(buffer.startIndex..<endIdx)
        }
    }

    func flushTail() {
        // End of generation — speak the remainder (often the last
        // sentence has no terminating period).
        if !buffer.isEmpty {
            speaker.speak(buffer)
            buffer.removeAll()
        }
    }

    private func nextSentenceEnd(in s: String) -> String.Index? {
        // Find the FIRST `[.?!]` followed by a space or string end.
        // Don't break on inline `.` like "Dr." or "i.e." — naive
        // version below; replace with a real sentence-tokenizer for
        // production (NaturalLanguage framework's
        // NLTokenizer(unit: .sentence) is one option).
        var idx = s.startIndex
        while idx < s.endIndex {
            if [".", "?", "!"].contains(s[idx]) {
                let next = s.index(after: idx)
                if next == s.endIndex || s[next].isWhitespace {
                    return next
                }
            }
            idx = s.index(after: idx)
        }
        return nil
    }
}
```

For better sentence detection use `NaturalLanguage` framework's
`NLTokenizer(unit: .sentence)`. That's a five-line replacement of
`nextSentenceEnd(in:)`.

### 3.3 Voice selection + rate

```swift
// Pick a voice — see AVSpeechSynthesisVoice.speechVoices() for the full list.
let speaker = NemotronHOmniSpeaker(voiceLanguage: "en-US")

// Override per-utterance:
speaker.voiceLanguage = "en-GB"        // British English
speaker.speak("Right then, off we pop.", rate: 0.5)  // slower

// Default rate constants:
//   AVSpeechUtteranceMinimumSpeechRate    (very slow)
//   AVSpeechUtteranceDefaultSpeechRate    (Apple's default)
//   AVSpeechUtteranceMaximumSpeechRate    (fast)
```

For a "settings → voice" UI dropdown:

```swift
let voices = AVSpeechSynthesisVoice.speechVoices()
    .filter { $0.language.hasPrefix("en") }   // or no filter
    .sorted { $0.name < $1.name }

ForEach(voices, id: \.identifier) { v in
    Button(v.name) { speaker.voiceLanguage = v.language }
}
```

---

## 4. Drop-zone UI for audio + video

### 4.1 SwiftUI drag-drop on macOS / iPadOS

```swift
import SwiftUI
import UniformTypeIdentifiers

struct ChatComposer: View {
    @State private var attachments: [URL] = []
    @State private var text: String = ""

    var body: some View {
        VStack {
            // ... message bubbles ...

            ChatBar(
                text: $text,
                onSend: send,
                attachments: $attachments
            )
            .onDrop(of: [.audio, .movie, .image], isTargeted: nil) { providers in
                Task {
                    for p in providers {
                        if let url = await loadURL(from: p) {
                            attachments.append(url)
                        }
                    }
                }
                return true
            }
        }
    }

    private func send() {
        let images = attachments.filter { isImage($0) }
        let videos = attachments.filter { isVideo($0) }
        let audios = attachments.filter { isAudio($0) }
        Task { try await chatEngine.send(
            text: text,
            images: images.map { .url($0) },
            videos: videos.map { .url($0) },
            audios: audios.map { .url($0) })
        }
    }
}
```

The four `is{Image,Video,Audio}(_:)` predicates use the file's
`UTType` (`UTType(filenameExtension:)?.conforms(to: .audio)` etc).

### 4.2 File-format guidance for users

What works (decoded by AVFoundation):

- **Audio**: `.wav`, `.aac`, `.m4a`, `.mp3`, `.flac`, `.aiff`, `.caf`,
  `.mov` audio track. Anything `AVAudioFile(forReading:)` accepts.
- **Video**: `.mov`, `.mp4`, `.m4v`. Anything `AVURLAsset` opens.
- **Image**: `.png`, `.jpg`, `.heic`, `.tiff`, `.gif` (first frame),
  `.webp`. Anything `CIImage(contentsOf:)` accepts.

What to reject in the UI before upload (don't let it reach vmlx):

- DRM-protected media (FairPlay-encrypted .m4v from iTunes Store).
- Live-streaming URLs (HLS .m3u8) — no support; would need
  pre-recorded download.
- Audio over ~5 minutes — works but UX-painful (long prefill); offer
  "split into chunks" UI for long-form.

---

## 5. Voice-mode UX patterns

### 5.1 Push-to-talk button (simplest)

Keyboard or on-screen button. While held: capture mic. On release:
flush PCM → vmlx → stream tokens back → flush sentences to TTS.

```swift
// Bind to a long-press gesture or keyboard key (e.g. spacebar).
@GestureState private var isTalking = false

Button { } label: {
    Image(systemName: isTalking ? "mic.fill" : "mic")
}
.simultaneousGesture(
    LongPressGesture(minimumDuration: 0.05)
        .updating($isTalking) { _, state, _ in state = true }
        .onChanged { _ in pttController.onPressDown() }
        .onEnded { _ in
            Task {
                let pcm = try await pttController.onPressUp()
                if !pcm.isEmpty { await chatEngine.sendVoice(pcm) }
            }
        }
)
```

### 5.2 Conversation mode (always-on)

Toggle button. While on: continuous capture with VAD-based chunking;
each detected utterance fires a chat turn; LLM response streams to
TTS. User can interrupt by speaking again (mic still hot — clip the
in-progress TTS):

```swift
func onUserSpoke() {
    speaker.stop()                          // interrupt current TTS
    Task { await processUtterance() }
}
```

Apple's `AVSpeechSynthesizer.stopSpeaking(at: .immediate)` is the
interruption hook. `NemotronHOmniSpeaker.stop()` calls it.

### 5.3 Voice replay

Cache the LLM text response — replaying via TTS is free
(`speaker.speak(savedText)`). No need to call the model twice.

---

## 6. Mixed-modality UX

The model accepts image + audio + video in one message. Real-world UX:

> User drops a photo of a recipe + records 3 seconds of voice asking
> "What's the second step?"

Both go into the same `UserInput`:

```swift
let input = UserInput(
    prompt: "(see photo and recording)",
    images: [.url(photoURL)],
    audios: [.samples(voicePCM, sampleRate: 16_000)])
```

The processor stitches placeholders correctly: `<img>...</img>` first,
then `<sound>...</sound>`, then text. Verified by Row 9 of
`BENCH_OMNI=1` ("mixed image + audio one turn"): output is coherent
text combining what's seen and heard.

UX-wise: show both attachments in the composer, label them clearly
("📸 recipe.jpg" + "🎤 0:03"), let the user remove either before
sending.

---

## 7. RADIO + Parakeet — implementation references

For engineers who want to understand or modify the encoders:

### RADIO (vision)

**File**: `Libraries/MLXVLM/Models/NemotronHOmni/RADIOVision.swift`

- 32-block ViT-Huge body (1280-dim, 16 heads, MLP 5120)
- CPE patch generator with bilinear pos_embed interpolation from a
  stored 128×128 max grid down to actual `gy×gx`
- 10 cls/register tokens (1 cls + 9 registers, per radio_v2.5-h)
- `video_embedder` Linear handles the T=2 channel-stacked video path
- No final norm (timm sets `model.norm = nn.Identity` for RADIO)

**Python reference**: `~/jang/jang-tools/jang_tools/nemotron_omni/radio.py`
(372 lines, Swift port is byte-identical math).

**Forward shape contract**:
- Input:  `(B, 3, H, W)` Float32, CLIP-normalized
- Output: `(B, num_cls + num_patches, 1280)`
- After cls-strip + pixel-shuffle + mlp1: `(B, 256_per_tile, 2688)`

### Parakeet (audio)

**File**: `Libraries/MLXVLM/Models/NemotronHOmni/Parakeet.swift`

- 24-layer Conformer encoder (1024-dim, 8 heads, FFN 4096, conv 9)
- Subsampling stack: 5 Conv2d layers + Linear(4096 → 1024), factor=8
- Each ConformerBlock: macaron ½FF + Transformer-XL rel-pos MHA (with
  `bias_u`/`bias_v` + skewing trick) + GLU-conv + ½FF + final LN
- Inference-only `BatchNorm1d` using stored running stats
- Manual depthwise conv (kernel=9) per channel

**Python reference**: `~/jang/jang-tools/jang_tools/nemotron_omni/parakeet.py`
(409 lines).

**Forward shape contract**:
- Input:  `(1, n_frames, 128)` mel features (after STFT + per-sample
  normalize, NOT raw waveform)
- Output: `(1, n_frames/8, 1024)` after subsampling + 24 conformer
  blocks
- After sound_projection: `(1, n_frames/8, 2688)` ready to splice at
  `<so_embedding>` token positions

### Mel STFT (audio preprocess)

**File**: `Libraries/MLXVLM/Models/NemotronHOmni/Preprocessors.swift`
(`nemotronOmniExtractMelFeatures`)

- vDSP-based STFT (n_fft=512, hop=160, win=400, hann periodic=False)
- Slaney-norm mel filterbank, 128 mels, fmin=0, fmax=8 kHz
- Pre-emphasis 0.97 + log + **per-sample zero-mean unit-variance
  normalize with Bessel-corrected variance** ← CRITICAL; without this
  the model produces nonsense (the Python port wasted ~30 minutes
  debugging this same bug)

If you're modifying anything in this file: keep the per-sample
normalize. Verified across MXFP4 / JANGTQ4 / JANGTQ2 — drop it and
audio decode produces "the sound of a door opening and closing" for
every input.

### NVLM tile preprocessor (image)

**File**: `Preprocessors.swift` (`nemotronOmniDynamicPreprocess`)

- Aspect-ratio-preserving tile selection (cols × rows, cols×rows ∈
  [min_num, max_num])
- Bicubic resize to (cols × image_size, rows × image_size) via
  `MediaProcessing.resampleBicubic`
- Top-down crop into individual 512×512 tiles (CIImage origin is
  bottom-left; we translate to match Python's row-major iter order)
- Optional thumbnail at the END if cols×rows > 1
- CLIP normalization: mean=[0.481, 0.458, 0.408], std=[0.269, 0.261,
  0.276]

---

## 8. What's intentionally NOT in vmlx

The osaurus team should plan for these as separate scopes:

| Feature | Why not in vmlx | What you'd need |
|---|---|---|
| Voice-activity detection (VAD) | UX-specific; many libs to choose from | Silero / WebRTC VAD wrapper, threshold tuning |
| Whisper-quality ASR (alt to Parakeet) | Different model, different bundle | mlx-swift-examples Whisper port (exists) |
| Neural TTS (vocoder) | Bundle has no decoder | Coqui XTTS / F5-TTS / network API of choice |
| Anthropic `document.audio` mapping | API adapter is osaurus-side | Add to your Anthropic adapter module |
| Speaker diarization | Different model entirely | Pyannote port or third-party |
| Real-time interruption / barge-in | UX glue | `speaker.stop()` + mic-while-speaking detection |

All of these can be layered on top — they don't need anything from
vmlx to ship. Bring your own model / library / UI policy for each.

---

## 9. Performance envelope (B=1, M3 Max, real-bundle bench)

The numbers below are from `BENCH_OMNI=1` against the real bundles
(`MXFP4` / `JANGTQ4` / `JANGTQ2`). Use to set product-side
expectations:

| Operation | MXFP4 | JANGTQ4 | JANGTQ2 |
|---|---|---|---|
| Bundle load | 3.3 s | 3.1 s | 0.8 s |
| Text decode | 79–128 tok/s | 84–101 tok/s | 99–107 tok/s |
| Image preprocess (single 224×224) | <100 ms | <100 ms | <100 ms |
| Image first-token (single tile) | ~1 s | ~1 s | ~1 s |
| Audio mel STFT (1 s clip) | <100 ms | <100 ms | <100 ms |
| Audio first-token (1 s clip) | ~700 ms | ~500 ms | ~500 ms |
| Video first-token (32 frame, 1080p) | ~20 s | ~22 s | ~21 s |
| Decode (post-prefill) | 51–128 tok/s | 36–101 tok/s | 38–107 tok/s |

Voice-mode UX target: under 1 s from end-of-utterance to first audio.
With Parakeet + LLM prefill ≈ 500 ms + first-sentence-then-TTS ≈
200 ms, you land at ~700 ms on JANGTQ2. MXFP4 is ~1 s. Tune by:

- Reducing chat history (faster prefill).
- Using JANGTQ2 (smallest, fastest decode).
- Streaming text → TTS at sentence boundaries (don't wait for full
  reply).
- Pre-warming the speaker with `NemotronHOmniSpeaker(voiceLanguage:
  "en-US")` at app launch (first call has ~100 ms init cost).

---

## 10. Common pitfalls + fixes

| Pitfall | Symptom | Fix |
|---|---|---|
| Using `[.url(audioURL)]` for in-memory PCM | Crash or silent drop | Use `.samples(pcm, sampleRate: ...)` |
| Forgetting permissions | `mic.start()` throws or returns silent | Add `NSMicrophoneUsageDescription`, request at runtime |
| Mid-recording app suspended | Empty PCM returned | Activate audio session category `.record`; vmlx doesn't manage AVAudioSession |
| Speaking while TTS playing | Both voices overlap | Call `speaker.stop()` on user mic-down event |
| Sending raw 44.1 kHz PCM as `.samples(_, 44_100)` | Works (resampler runs) but slower | Resample to 16 kHz client-side if you have many turns; processor accepts any rate but pays the cost per turn |
| Leaving `MicRecorder` started across app background | iOS suspends it; unpredictable | `applicationWillResignActive` → `try mic.stop()` |
| Mixed audio + image, expecting alphabetical order | Confused output order | Vmlx serializes image-first then audio in the prompt; describe modalities in the prompt text accordingly ("the photo shows... the recording says...") |

---

## 11. Quick "ship it" checklist for a voice-mode PR

When you're building the osaurus voice-mode feature, your checklist:

- [ ] Add `NSMicrophoneUsageDescription` to Info.plist
- [ ] Request mic permission at first voice-mode launch (not at app
      launch — bad UX)
- [ ] Add `com.apple.security.device.audio-input` entitlement if
      sandboxed
- [ ] Wire `NemotronHOmniMicRecorder.start()` / `.stop()` to your
      push-to-talk or VAD-driven flow
- [ ] Pass captured PCM via `UserInput.audios = [.samples(pcm, sampleRate: 16_000)]`
- [ ] Stream LLM tokens to `NemotronHOmniSpeaker.speak(_:)` at sentence
      boundaries
- [ ] Handle interruption: `speaker.stop()` on mic-down
- [ ] Show recording indicator (red mic icon, vu meter, etc.)
- [ ] Show speaking indicator while TTS active
- [ ] Test on BOTH Mac (macOS audio session is permissive) AND iOS
      (sandboxed; permission UX differs)
- [ ] Test the audio file drop path too — `audios: [.url(fileURL)]`
- [ ] Mix-test: image + audio in one turn (Row 9 of bench validates
      this works model-side; verify your UI doesn't drop one)

That's it. Five new ObservableObjects, a couple of SwiftUI views, your
existing chat-engine plumbing.

---

## 12. Honest scope boundary — what this doc doesn't cover

- **Anthropic API adapter** — `document.audio` shape mapping. PR #2
  doesn't touch the Anthropic adapter; that's a separate file in
  osaurus, separate PR. Map `document.audio` the same way PR #2
  maps `input_audio` (base64 → temp file → `UserInput.audios`).
- **Server-side voice-mode HTTP streaming** — sending audio chunks
  back over the wire (e.g., for an osaurus headless server). Out of
  scope; covered by your existing OpenAI streaming response shape if
  you go text-only over the network and synthesize speech at the
  client.
- **Multi-speaker conversations** — Parakeet doesn't speaker-diarize.
  If you want "who said what" you need a separate diarization model.

Last updated: 2026-04-29. References commit `a5c02a0` and later on
`osaurus-ai/vmlx-swift-lm` main.
