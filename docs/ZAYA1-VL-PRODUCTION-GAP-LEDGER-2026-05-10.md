# ZAYA1-VL Production Gap Ledger — 2026-05-10

This is a local coordination note. It is not release text.

Purpose: keep ZAYA1-VL work proof-first and prevent fake "support" switches,
placeholder guards, or cache claims that do not execute the real runtime path.

## Current Hard Boundary

ZAYA1-VL is now native-generation-capable on the local Swift runtime, but it is
not fully production-covered yet.

The shipped path is a real `Zaya1VL` model, not a dummy wrapper and not
text-only `zaya`: Qwen2.5-VL vision tower output is merged into image-token
embeddings, then a 40-block Zaya CCA/MoE trunk runs with image-mask-gated LoRA
call sites and one `ZayaCCACache` per block.

The cache policy is intentionally conservative: ZAYA1-VL uses disk-backed
TQDiskSerializer v2 restore for path-dependent `ZayaCCACache` state. Do not
claim paged-prefix support or TurboQuant KV for this family. JANGTQ2/JANGTQ4
and MXFP4 are weight formats, not KV-cache formats.

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
| Runnable 40-block trunk | `Zaya1VLRegistrationTests.registryRecognizesZaya1VL` now returns a real `Zaya1VL`, and `nativeModelUsesZayaCCACachesForEveryBlock` proves 40 `ZayaCCACache` layers. |
| Real image generation | `/tmp/zaya1vl_jangtq2_vl_batch_24_20260510.log` proves BatchEngine image->text then text follow-up on `ZAYA1-VL-8B-JANGTQ2`, compile off and compile on, with coherent full text. |
| TokenIterator vs BatchEngine parity | `/tmp/zaya1vl_jangtq2_vl_cross_validate_20260510.log` proves 16 generated tokens are byte-identical across TokenIterator and BatchEngine at temp=0. |
| Image -> text-only multi-turn cache | `/tmp/zaya1vl_jangtq2_vl_cache_hit_disk_20260510.log`, `/tmp/zaya1vl_jangtq4_vl_cache_hit_disk_20260510.log`, and `/tmp/zaya1vl_mxfp4_vl_cache_hit_disk_20260510.log` prove same-media disk-backed cache hits for JANGTQ2/JANGTQ4/MXFP4. |
| Same text + different image isolation | `/tmp/zaya1vl_jangtq2_vl_media_salt_20260510.log` proves same text with different images has equal tokens, different media salts, and a coordinator MISS for the different image. |
| Structured chat cache matrix | `/tmp/zaya1vl_jangtq2_vl_chat_cache_20260510.log` proves cold image answer, same-media disk HIT replay, different-media MISS, and coherent follow-up output. |
| Real bundle metadata/indexes | `Zaya1VLRegistrationTests.realBundleMetadataDecodes` and `realBundleStructureMatchesNativeAdapterRequirements` cover MXFP4/JANGTQ2/JANGTQ4 config, preprocessor, quant metadata, vision tower, 40 text layers, LoRA surfaces, and sidecars. |
| Non-thinking policy | `realBundleCapabilityStampsAreNonThinking`, `realBundleTemplatesMatchNonThinkingPolicy`, and `ZayaParserDispatchTests` prevent stale `supports_thinking` / `think_in_template` drift. |
| Weight sanitizer prerequisite | `Zaya1VLWeightSanitizerTests` covers local MXFP4/JANGTQ2/JANGTQ4 index rewrites without coercing into text-only 80-layer ZAYA. |
| MLX test harness | The 2026-05-10 expanded default-parallel verifier passed 132 Swift Testing tests in 20 suites plus 23 XCTest cases after `mlx.metallib` aliasing and unified MLX test serialization. |

## Not Proven / Must Not Be Claimed

| Missing surface | Why it matters | Required proof before claiming production |
|---|---|---|
| B>1 media isolation | Continuous batching must not cross-contaminate image masks, CCA state, or LoRA-gated tokens. | B=2/B=4 real rows with simultaneous image/text slots, active-slot overlap proof, distinct outputs, cancellation isolation, and cache-on overlap. |
| Longer output / broader visual semantics | Current rows use short synthetic images and short answers. | Longer image prompts with enough `maxTokens`, full output saved, no loop/leak/empty-content, speed recorded. |
| Real reasoning-on image row | Current ZAYA1-VL templates have image markers but no thinking branch. | Only claim image reasoning if a bundle template actually exposes it and real `enable_thinking=true/false` rows prove clean parser separation. |
| TurboQuant KV | Not applicable as a ZAYA1-VL cache mode. JANGTQ2/4 are weight formats; ZAYA1-VL cache topology is CCA/path-dependent. | Do not add a `Zaya1VLTurboQuantKVCache`. Generic TurboQuant KV tests only cover compatible KV families. |
| Per-quant speed/coherence beyond cache replay | MXFP4/JANGTQ2/JANGTQ4 have different weight paths. | JANGTQ2 has coherent image/chat/cross-engine rows; JANGTQ4 and MXFP4 have disk-cache restore rows. Add longer semantic generation rows before advertising quality parity. |

## Scheduler / Cache Policy To Implement

ZAYA1-VL should use the same conservative policy as path-dependent ZAYA CCA
until live rows prove otherwise:

1. `cacheRequiresDiskBackedCoordinatorRestore(_:)` must classify ZAYA1-VL's
   cache list as path-dependent because it contains `ZayaCCACache`.
2. Generic paged-prefix restore stays disabled for ZAYA1-VL. Current proof uses
   disk-backed restore, not paged-only prefix reuse.
3. L2 disk restore is enabled only through TQDiskSerializer v2, which carries
   KV plus CCA state (`conv_state`, `prev_hs`) directly in the disk arrays and
   folds `mediaSalt` plus request `cacheScopeSalt` into the key.
4. Continuous batching must gather/scatter per-slot CCA state and per-token
   image masks. It cannot share image masks across slots and cannot treat
   vision-token LoRA as a prompt-only side effect.
5. Async rederive is not a substitute for media/CCA restore. For disk-backed
   path-dependent caches, the runtime must not overwrite serialized CCA state
   with text-only rederived state.

## Implementation Order

1. Add B=2/B=4 media-isolation rows with active-slot overlap and cancellation.
2. Add longer visual-semantic rows for JANGTQ2/JANGTQ4/MXFP4.
3. Add any reasoning-on/off rows only if the active ZAYA1-VL bundle template
   supports a thinking branch.
4. Keep paged-prefix disabled and continue using disk-backed v2 restore for
   ZAYA CCA state.
5. Record Osaurus wiring guidance: route `zaya1_vl` through the VLM factory,
   pass structured Jinja context, and do not map JANGTQ weight format to a fake
   TurboQuant KV cache mode.

Any step that cannot produce a failing test first should be documented as a
design note, not hidden behind a production switch.
