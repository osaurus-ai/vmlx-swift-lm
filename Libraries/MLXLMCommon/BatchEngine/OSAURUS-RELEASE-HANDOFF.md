# Osaurus release hand-off — vmlx-swift-lm 1c62d21+

Audience: tpae + osaurus team. Read this before bumping the vmlx pin.
Companion to `OSAURUS-INTEGRATION.md` (LLM tier) and `OMNI-OSAURUS-HOOKUP.md`
(multimodal). This file consolidates only the deltas relevant to the
upcoming release.

## What's new in 1c62d21 vs the previous a7db6e5 pin

Five vmlx-swift-lm commits land between a7db6e5 and 1c62d21:

| SHA | What | Why osaurus cares |
|---|---|---|
| `537e386` | `feat(omni): NemotronHJANGTQ — JANGTQ4 + JANGTQ2 omni 7/7 PASS` | NemotronHJANGTQ wraps the standard NemotronHOmni so JANGTQ4 / JANGTQ2 omni bundles route correctly. Use the same `VLMModelFactory.shared.loadContainer(...)` call — factory dispatches by `weight_format`. |
| `ae49c7c` | `feat(omni): full audio LMInput integration — STT + voice I/O` | Audio is no longer the "open seam" called out in `OMNI-OSAURUS-HOOKUP.md`. Pass audio through `LMInput.audio` and the Parakeet path runs end-to-end. |
| `3b78db4` | `feat(omni): close audio + video gaps the osaurus agent flagged` | Multiple audio + video edge cases closed. Together with `ae49c7c` this is what makes the OpenAI `input_audio` + `video_url` content parts (osaurus PR `feat/openai-multimodal-audio-video`) round-trip correctly. |
| `a5c02a0` | `docs+stress(omni): API hand-off contract + 4 stress rows green on all 3 quants` | Stress matrix rows that pass on MXFP4, JANGTQ4, JANGTQ2. |
| `d020e76` | `docs+fix(omni): voice integration guide + BatchEngine text-only .logits` | BatchEngine now exposes text-only `.logits` for voice integration. |
| `1c62d21` | `docs(omni): correct §10 — empty-stream was test methodology, not runtime` | Doc-only correction. |

## API surface — drop-in for osaurus

```swift
// 1. Load (factory auto-detects MXFP4 / JANGTQ4 / JANGTQ2 / standard)
let context = try await VLMModelFactory.shared.loadContainer(
    configuration: .init(directory: bundle))

// 2. Coordinator (this is YOUR side, osaurus already has buildCacheCoordinatorConfig)
//    Recommended config for hybrid SSM models like Nemotron-Omni:
let coord = CacheCoordinator(config: CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    diskCacheDir: ~/.osaurus/cache/kv_v2,
    modelKey: bundle.lastPathComponent,
    defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),
    defaultMaxKVSize: 8192,
    longPromptMultiplier: 2.0
))

// 3. Hybrid flag (idempotent — auto-flips on first admission too,
//    set explicitly to avoid the empty-stream methodology trap)
coord.setHybrid(true)

// 4. Build LMInput. ALL of these now work on the same struct:
let input = LMInput(
    text: .init(prompt: prompt),
    image: image.flatMap { LMInput.Image.pixels($0) },     // 256 toks/tile
    video: video.flatMap { LMInput.Video.pixels($0) },     // RADIO
    audio: audio.flatMap { LMInput.Audio.pcm16k($0) }      // Parakeet
)

// 5. Feed BatchEngine. Nothing special — same as text-only.
let output = try await batchEngine.generate(
    input: input,
    parameters: parameters,
    coordinator: coord
)
```

## Two stability fixes shipping with this release

### A. `notifyExternalReferencesNonZeroOnDealloc` (Bug 1)

**Reproduced**: tpae 2026-04-30 on M4 Pro with Qwen-3.6 35B A3B MXFP4
hybrid + warm KV disk cache. Two identical prompts, second crashes.

**What we tried that didn't fix it**: vmlx-swift-lm `98289d9` added
`MLX.eval(slot.cache)` after disk-restore. Eager-eval helps but the
race re-opens during the model's own prefill custom-kernel dispatches,
not just the disk-restore phase.

**Root cause**: `mlx::core::metal::Device::clear_library` (in
mlx-swift's bundled `mlx` C++ submodule, `device.cpp:706`) drops the
Swift-side ref count on `MTLComputePipelineState` immediately. If the
pipeline is still encoded in an in-flight `MTLCommandBuffer`, Metal's
debug validation kills the process.

**Fix**: patch in `osaurus-ai/mlx-swift` mlx submodule — turn
`Device::clear_library` into a "remove from lookup, do not release".
Stale pipelines remain alive until command-buffer completion drops
their last ref naturally. Cost: bounded memory residue (one
specialised pipeline per unique source string). Reverter env:
`MLX_CLEAR_LIBRARY_RELEASE=1`.

**What osaurus needs to do**: bump the mlx-swift pin to the new SHA
(once we push the fork branch). Convert
`Packages/OsaurusCore/Package.swift:16` from `branch:` to a
`revision:` SHA — matches the pin policy from `e2e13b4f`.

## NOT in this PR — to be built next, one at a time

### Bug 2 — `[metal::malloc] 154 GB allocation` on hybrid + over-cap prompt

**Reported**: tpae 2026-04-30 on hybrid Qwen-3.6-27B-MXFP4 with a
56,797-token prompt and `defaultMaxKVSize=8192`. Process fataled with
`Attempting to allocate 154843162032 bytes which is greater than the
maximum allowed buffer size of 30150672384 bytes`.

**What we know:** the math `154,843,162,032 / 56,797^2 = 48 bytes/pair`
is consistent with a `[1, 24, L, L]` fp16 attention scores matrix. So
the fault is most likely a non-chunked prefill or an SDPA fallback
from Flash-Attn to manual softmax under a hybrid attention mask.

**What we have not yet verified empirically:**
- Whether chunked prefill is engaging at all for hybrid Qwen-3.6
- The exact line that allocates the 154 GB tensor
- Whether `MLXFast.scaledDotProductAttention` falls back for the
  observed mask shape

**Why this isn't in the PR:** without the empirical pin, any clamp
or fail-fast guard might paper over the wrong layer or break a
non-overcap workload. A wrong fix here has worse failure modes
(silent wrong outputs) than the original crash.

**Plan to build it (in order):**
1. Spin up osaurus app on M5 pointing at the ext-drive
   `Nemotron-3-Nano-Omni-30B-A3B-MXFP4` bundle.
2. Add allocation tracing in `mlx::core::metal::Device::malloc` for
   any allocation > 1 GB — log the requested shape + caller stack.
3. Run a 60k-token prompt, capture the over-1GB allocation log line.
4. Identify which layer / op produces the 154 GB shape.
5. Either (a) clamp `prefillStepSize` to `defaultMaxKVSize` at
   admission, (b) add a pre-alloc fail-fast check in the offending
   op, or (c) fix the SDPA fallback (force Flash-Attn engagement) —
   pick the right shape after step 4.
6. Add a unit test in `Tests/MLXLMTests/CacheCoordinatorKVPolicyTests.swift`
   exercising the over-cap path.

**Workaround until then**: don't set `defaultMaxKVSize` for hybrid
models when prompts may exceed cap, OR keep prompts under the cap on
the host side.

## Stress matrix — must pass before pin bump

Two layers, both in this PR's investigation/ folder:

- L1 — `Tests/MLXLMStressTests/StressMatrix.swift` (vmlx side, swift test)
- L2 — `osaurus-staging/investigation/repros/stress_extras.py` (HTTP)

The L2 suites are drop-in companions to tpae's
`scripts/eval_http_stability.py`. Run order:

```bash
# L1 (vmlx-swift-lm)
export OSAURUS_STRESS_HYBRID_MODEL=/Volumes/EricsLLMDrive/jangq-ai/Nemotron-3-Nano-Omni-30B-A3B-MXFP4
swift test --filter MLXLMStressTests

# L2 (osaurus, server already running)
python3 scripts/eval_http_stability.py                              # tpae's S1-S6
python3 investigation/repros/stress_extras.py --only S7             # Bug 1 repro
python3 investigation/repros/stress_extras.py --only S8             # Bug 2 repro
python3 investigation/repros/stress_extras.py --only S9             # 20-turn agent loop
python3 investigation/repros/stress_extras.py --only S10            # 100-request burst
```

All cells must be green before the pin bump.

## What we explicitly DID NOT change

- No new public API on vmlx-swift-lm.
- No new model families. (NemotronH variants pre-existed.)
- No changes to BatchEngine semantics. (The Bug 2 hybrid clamp was
  considered then deferred — see "NOT in this PR" above.)
- No new dependency. The mlx-swift fork patch is a leaf C++ change.
- No HTTP-layer changes. tpae's `bugfix/http-model-runtime-fix` owns
  HTTP cancellation; vmlx-side just makes the hot path crash-free.
- No UI work. tpae owns the chat-window file-import paths.

## Open questions (need a decision before merge)

1. The mlx-swift patch ships as a C++ submodule change. Do we publish
   our mlx fork as `osaurus-ai/mlx` (separate repo) or vendor the
   commit inside `osaurus-ai/mlx-swift`'s submodule pin? Recommendation:
   vendor — fewer moving repos.
2. Should we lock `swift-embeddings` (currently `branch: main` in
   osaurus's Package.swift)? Recommendation: yes — pin to a SHA per
   the convention from `e2e13b4f`.

## Stress-test scaffolds in this PR (NOT yet executed)

`Tests/MLXLMStressTests/StressMatrix.swift` and `StressTestPlan.md`
land in this PR as planning artifacts. The cells / axes / cartesian
product are real Swift code; the per-cell `runCell` body is a TODO
skeleton. Gated by `OSAURUS_STRESS_RUN=1` and
`OSAURUS_STRESS_HYBRID_MODEL=<path>` env vars so the test suite
default behaviour is unchanged.

To-do (one at a time):
1. Wire BatchEngine + CacheCoordinator setup in `runCell` for each
   workload pattern.
2. Connect to a real model at the env-var path.
3. Capture per-cell pass/fail + memory/perf to STRESS_REPORT.md.
4. Run the targeted S7 + S8 cells against the patched build (Bug 1
   fixed; Bug 2 still observable until the empirical fix lands).

## Reference docs in this folder

- `OSAURUS-INTEGRATION.md` — LLM-tier contract (BatchEngine flag, KV
  sizing, reasoning stream events, stop-string contract)
- `OMNI-OSAURUS-HOOKUP.md` — Nemotron-Omni multimodal contract
  (sections 3 + 10 should be re-read AFTER this hand-off — the audio
  "open seam" called out in §3 is now closed; left for accuracy in
  the next pass)
- `OMNI-VOICE-INTEGRATION.md` — voice path
- `KV-SIZING-CONTRACT.md` — KV sizing math
- `STOP-SEQUENCES-CONTRACT.md` — stop strings
- `REASONING-STREAM-EVENT.md` — reasoning event payload
