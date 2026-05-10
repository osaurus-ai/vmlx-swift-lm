# ZAYA1-VL Production Gap Ledger — 2026-05-10

This is a local coordination note. It is not release text.

Purpose: keep ZAYA1-VL work proof-first and prevent fake "support" switches,
placeholder guards, or cache claims that do not execute the real runtime path.

## Current Hard Boundary

ZAYA1-VL is component-proven, not native-generation-ready.

It is correct for `VLMModelFactory` to recognize `model_type=zaya1_vl` and then
fail with a precise native-adapter-pending error. It would be incorrect to
return a dummy model, route through text-only `zaya`, use a KV-only cache class,
or claim TurboQuant KV support for this family.

## Proven Now

| Surface | Proven evidence |
|---|---|
| Family recognition | `Zaya1VLRegistrationTests.registryRecognizesZaya1VL` proves `zaya1_vl` is recognized and does not silently route to text-only ZAYA. |
| Factory fail-closed behavior | `factoryLoadFailsAtRecognitionGateBeforeWeights` proves the recognition gate fires before tokenizer/weight loading, preventing fake runtime support. |
| Processor identity | `Zaya1VLProcessor` is a separate processor using Qwen2.5-VL image geometry while emitting ZAYA `<image>` placeholders. |
| Image resolution / aspect ratios | `processorPatchifiesImageInput`, `processorPatchifiesNonSquareImageGrids`, `processorPreservesMultipleImageFrames`, and `processorRejectsTooSmallImages` cover legal square/non-square grids, multi-image placeholder expansion, and too-small rejection. |
| Video fail-closed behavior | `processorRejectsVideoInput` proves ZAYA1-VL rejects video explicitly until a native video path exists, instead of silently dropping video attachments. |
| Finite image extents | `QwenVLIntExtentTests` cover the centralized finite-positive size guard and source adoption across Qwen2VL/Qwen25VL/Qwen3VL/Gemma4/Zaya1VL/GlmOcr, preventing infinite/NaN CIImage extents from reaching `Int(...)`. |
| RGB normalization | `processorNormalizesSolidBlackAndWhitePixels` checks Qwen2.5-VL-style normalization constants. |
| Media/request cache salt | `cacheSaltCombinesMediaAndRequestScope`, `CacheCoordinatorMediaSaltTests`, `VLMProcessorCacheScopeSaltTests`, and `VLMDefaultContextProcessorTests` prove source/component-level salt behavior. |
| Image-token merge | `Zaya1VLRuntimeSupport.mergeImageFeatures` and `Zaya1VLInputEmbeddingAdapterTests` prove replacement only at image-token positions, shape rejection, batch-shaped ids, dtype preservation, and image-mask return. |
| Vision tower bridge | `Zaya1VLVisionTowerBridgeTests` proves Qwen2.5-VL `VisionModel` can be instantiated standalone under the ZAYA1-VL namespace. |
| Image-mask LoRA helper | `Zaya1VLImageMaskLoRATests` proves masked add touches only image-token rows, preserves text-only nil-mask path, supports flat routed-expert shape, and rejects incompatible masks. |
| Native LoRA namespace shells | `Zaya1VLNativeAdapterSourceCoverageTests` pins shipped attention/output/expert LoRA module names and local-expert container coverage. |
| Real bundle metadata/indexes | `Zaya1VLRegistrationTests.realBundleMetadataDecodes` and `realBundleStructureMatchesNativeAdapterRequirements` cover MXFP4/JANGTQ2/JANGTQ4 config, preprocessor, quant metadata, vision tower, 40 text layers, LoRA surfaces, and sidecars. |
| Non-thinking policy | `realBundleCapabilityStampsAreNonThinking`, `realBundleTemplatesMatchNonThinkingPolicy`, and `ZayaParserDispatchTests` prevent stale `supports_thinking` / `think_in_template` drift. |
| Weight sanitizer prerequisite | `Zaya1VLWeightSanitizerTests` covers local MXFP4/JANGTQ2/JANGTQ4 index rewrites without coercing into text-only 80-layer ZAYA. |
| MLX test harness | The 2026-05-10 expanded default-parallel verifier passed 132 Swift Testing tests in 20 suites plus 23 XCTest cases after `mlx.metallib` aliasing and unified MLX test serialization. |

## Not Proven / Must Not Be Claimed

| Missing surface | Why it matters | Required proof before claiming production |
|---|---|---|
| Runnable 40-block trunk | ZAYA1-VL has 40 layers where every block runs attention then MLP/MoE. Text-only ZAYA has an 80-layer alternating topology. Reusing text-only trunk would be a topology bug. | Native trunk code with tests proving all 40 blocks own both attention and MLP/MoE calls and no text-only 80-layer coercion. |
| Trunk-integrated LoRA | Source LoRA deltas are `image_mask` gated inside attention and MLP/expert paths. Helper-only coverage does not prove they affect forward. | Forward-path tests proving image-mask rows receive LoRA deltas and text-only rows are unchanged for attention output and routed experts. |
| Model-level sanitizer call | `Zaya1VLWeightSanitizer` exists, but no runnable model calls it yet. | Native model `sanitize(weights:)` or loader hook invoking the sanitizer, with a test proving real bundle keys are rewritten before module assignment. |
| VLM factory runnable dispatch | Current dispatch correctly fail-closes. | `VLMModelFactory` returns a real `Zaya1VL` model only after the trunk, sanitizer, processor, and cache contracts are wired. |
| Real image generation | Component tests do not prove coherent decode. | Real model row: image question, enough `maxTokens` for a complete answer, full output saved, no loop/leak/empty-content, speed recorded. |
| Image -> text-only multi-turn cache | Media salt and cache helpers are component-tested, but the live model path has not stored/restored a multimodal turn. | Real row: Turn 1 image+text, Turn 2 text-only follow-up, same session and fresh L2 restore, full outputs inspected, cache hit/miss counters recorded. |
| Same text + different image isolation | Component salt tests prove hash separation, not live decode behavior. | Real row with same text and two different images proving no stale image state is reused. |
| B>1 media isolation | Continuous batching must not cross-contaminate image masks, CCA state, or LoRA-gated tokens. | B=2/B=4 real rows with simultaneous image/text slots, active-slot overlap proof, distinct outputs, cancellation isolation, and cache-on overlap. |
| ZAYA1-VL scheduler helper | The scheduler must know this is path-dependent CCA state plus media salt, not paged-KV-only. | Shared helper or policy row proving prefix-paged cache is disabled or disk-backed until KV+CCA+media restore parity is proven. |
| ZAYA1-VL L2 disk store | ZAYA CCA disk serializer exists; ZAYA1-VL must add media/request identity and use CCA companion state. | Disk restore row proving `(tokens, mediaSalt, cacheScopeSalt, KV, conv_state, prev_hs)` all match before reuse. |
| TurboQuant KV | Not applicable as a ZAYA1-VL cache mode. JANGTQ2/4 are weight formats; ZAYA1-VL cache topology is CCA/path-dependent. | Do not add a `Zaya1VLTurboQuantKVCache`. Generic TurboQuant KV tests only cover compatible KV families. |
| Per-quant speed/coherence | MXFP4/JANGTQ2/JANGTQ4 have different weight paths. | Separate real rows for all three local bundles with full outputs and speed. |

## Scheduler / Cache Policy To Implement

ZAYA1-VL should use the same conservative policy as path-dependent ZAYA CCA
until live rows prove otherwise:

1. `cacheRequiresDiskBackedCoordinatorRestore(_:)` must classify ZAYA1-VL's
   cache list as path-dependent because it contains `ZayaCCACache`.
2. Generic paged-prefix restore must stay disabled for ZAYA1-VL until the
   restore path can prove KV plus CCA companion state plus media salt all match.
3. L2 disk restore may be enabled only when it serializes and validates the full
   ZAYA CCA state (`KV`, `conv_state`, `prev_hs`) and folds `mediaSalt` plus
   request `cacheScopeSalt` into the key.
4. Continuous batching must gather/scatter per-slot CCA state and per-token
   image masks. It cannot share image masks across slots and cannot treat
   vision-token LoRA as a prompt-only side effect.
5. Async rederive is not a free substitute for media/CCA restore. If used, it
   needs a warm-pass parity test over image+text and text-only follow-up turns.

## Implementation Order

1. Build the native 40-block text trunk with attention and MLP/MoE in each
   block. Keep it separate from text-only `ZayaModel` until topology parity is
   proven.
2. Wire model-level sanitizer and prove real local MXFP4/JANGTQ2/JANGTQ4 keys
   assign into modules.
3. Integrate image-token merge into embedding input and pass image masks through
   the trunk.
4. Apply vision-gated LoRA at attention and MLP/expert call-sites.
5. Add `newCache` / batch-cache policy using ZAYA CCA state and media salt.
6. Enable factory runnable dispatch.
7. Run real rows: image one-turn, image->text multi-turn, different-image miss,
   L2 restore, B>1 media isolation, cancellation, and per-quant speed/coherence.

Any step that cannot produce a failing test first should be documented as a
design note, not hidden behind a production switch.
