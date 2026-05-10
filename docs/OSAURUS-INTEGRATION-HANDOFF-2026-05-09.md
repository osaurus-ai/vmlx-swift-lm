# Osaurus Integration Handoff — vmlx-swift-lm Changes 2026-05-02 → 2026-05-09

**Date:** 2026-05-09
**Audience:** osaurus integrator(s) updating the vmlx-swift-lm pin
**Source-of-truth:** `docs/PRODUCTION-READINESS-MATRIX-2026-05-09.md` (CODEX) for what's currently proven; `docs/RUNTIME-OPEN-GAPS-2026-05-09.md` for what's still open.
**Companion docs:** `docs/PER-ARCH-REAL-TEST-PLAN-2026-05-09.md` (per-family scenarios); `docs/CLAUDE-DELIVERABLES-CROSSCHECK-2026-05-09.md` (proven-vs-unproven list).
**Prior handoff this supersedes:** `Libraries/MLXLMCommon/BatchEngine/OSAURUS-RUNTIME-HANDOFF-2026-05-06.md` (parts; cherry-pick specifics noted below).

> **Reading order:** §0 What changed at a glance → §1 Source-changing fixes → §2 No-action-needed (auto-engages) → §3 Caller-wiring requirements → §4 Env flags → §5 Behavior changes → §6 Test surface to add osaurus-side → §7 Open items → §8 Pinning checklist

---

## §0 What changed at a glance

### ⚠️ CRITICAL boundary: solo `BatchEngine.generate` ≠ real continuous batching

Per CODEX's `docs/PRODUCTION-READINESS-MATRIX-2026-05-09.md` §"Continuous Batching Acceptance Matrix" (added 2026-05-09 16:48 PDT): **the 46-49 tok/s MiniMax proof is solo `BatchEngine.generate` only.** It does NOT prove real continuous batching (B>1). osaurus must NOT advertise B>1 throughput, cancel-mid-stream isolation, cache-on overlap, or true-batch fairness based on the solo numbers.

What's proven (solo, B=1):
- ✅ MiniMax M2.7 JANGTQ family decode at 46-49 tok/s on solo `BatchEngine.generate`.
- ✅ Stage 1B.3 single-slot compile path engagement.
- ✅ Coherent output, no leak, no loop on the solo path.

What is NOT proven on current tree (B>1):
- ❌ B=2/B=4 active-slot overlap (two requests actively decoding concurrently).
- ❌ Cancel one slot mid-stream while another continues to EOS without KV/cache poisoning.
- ❌ Multi-turn warm-session overlap (two conversations sharing the live engine).
- ❌ L2 disk restore overlap across concurrent requests.
- ❌ Paged-prefix overlap on standard-KV topologies.
- ❌ TurboQuant KV overlap (`kvMode=.turboQuant` with B>1).
- ❌ Reasoning on/off cross-slot isolation (one slot thinking, one not).
- ❌ Media salt cross-slot isolation (one slot with image, one text-only).
- ❌ Long-output soak (256+ tokens) on the B>1 path.

**osaurus integration tests** that exercise B>1 must reproduce these on the new pin BEFORE claiming continuous-batching production status. See §6.b below for the row-shape recipe.

### Speed/perf wins osaurus inherits automatically (no API change required)

| Fix | Where | Net effect | Status |
|---|---|---|---|
| Hadamard scratch `newv[64]→newv[8]` | `Libraries/MLXLMCommon/JANGTQKernels.swift` (`kHadamardMultiblockSource`) | MiniMax M2.7 JANGTQ family went 30 → 46-49 tok/s on solo `BatchEngine.generate`. Likely benefits Laguna/Ling/DSV4/Mistral 3.5 too at the kernel level (smoke verified, head-to-head delta unproven). | ✅ Source landed, ✅ proven on MiniMax solo |
| `cachedJANGTQMeta` per-dispatch meta-array cache | `JANGTQKernels.swift:31-49` + every `JANGTQKernelLibrary.*` launcher | Eliminates per-step MLXArray allocation for kernel meta tensors. Compounding effect with the Hadamard fix; together responsible for the 30 → 46-49 jump. | ✅ Source landed, ✅ proven |
| Hadamard fast-path fp32 pre-cast removal | `JANGTQKernels.swift:496-499` | Saved ~10% graph nodes on MiniMax decode (4045 → 3673). Marginal speed contribution; correctness-equivalent. | ✅ Source landed |
| MiniMax router env alias | `Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift:30-49` | Canonical `VMLX_MINIMAX_ROUTER_COMPILE` now works (was previously inert; only the typo `VMLINUX_*` was read). Legacy typo still works for backwards compat. | ✅ Source landed |
| Bench-side stop-token fix ("stopfix2") | `RunBench/Bench.swift` (verifier path; NOT model code) | Bench harness no longer drops `LMInput.cacheScopeSalt` on raw-token replay. Production is unaffected — this was a verifier bug. | ✅ Local-only; not osaurus-side |

### Correctness fixes osaurus inherits automatically

| Fix | Where | Net effect | Status |
|---|---|---|---|
| `LMInput.cacheScopeSalt` + `computeCacheSalt(for:)` | `Libraries/MLXLMCommon/LanguageModel.swift` (struct property + initializer params) and `Libraries/MLXLMCommon/Cache/MediaSalt.swift` (helpers); threaded through LLM and all 15 VLM processors that render `additionalContext` | Reasoning on/off + reasoning_effort variations + media salt now produce distinct cache keys at request time. Defense-in-depth: doesn't rely on prompt-token differences alone. | ✅ Source landed, ✅ unit-tested (`CacheCoordinatorModeKeyIsolationTests`, `LLMCacheScopeSourceCoverageTests`, `VLMCacheScopeSourceCoverageTests`); ❌ real-model multi-turn proof OPEN |
| `planLiveCacheReuse(...)` warm-session restore-skip planner | `Libraries/MLXLMCommon/Cache/...` + `Evaluate.swift` + `BatchEngine.swift` | When live `[KVCache]` already covers `matchedTokens` from coordinator hit, skip the lossy restore step. Saves ~50ms re-prefill on warm-session multi-turn. | ❌ NOT IMPLEMENTED — symbol absent from code (`grep planLiveCacheReuse Libraries/ -r` → 0 hits 2026-05-10). Earlier "Source landed, unit-tested" claim was incorrect. **Osaurus integrator: do not expect this win on pin update.** |
| TQDiskSerializer `LayerKind` `CaseIterable` | `Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift:80` | Future persistable layer kinds fail loudly until disk round-trip coverage added. Build-time guard. | ✅ Source landed, ✅ tested |
| MiniMax JANGTQ_K per-projection bits + Hadamard kernel shmem fix (Mistral 3.5) | `MiniMaxJANGTQ.swift`, `TurboQuantSwitchLinear.swift`, `JANGTQKernels.swift` | Pre-existing fixes from earlier May iterations. Production path verified coherent via `RUNTIME-AUDIT-2026-05-05.md`. | ✅ Already in tree |

### New caller-facing surface osaurus may want to consume

| Surface | Where | Use |
|---|---|---|
| `LMInput.cacheScopeSalt` (request-level salt) | `Libraries/MLXLMCommon/LanguageModel.swift` (property + init params); helpers in `Libraries/MLXLMCommon/Cache/MediaSalt.swift` | Caller sets when reasoning mode / media is part of cache identity. **Optional**: if unset, falls back to model-key-only behavior; cache is still safe at the token-hash level when prompt rendering already differs. |
| `ToolCallFormat.fromCapabilityName(...)` | `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:339-348` | Maps JANG capability names to tool-call parsers. Now covers Qwen, MiniMax, GLM4/Deepseek, Nemotron, Mistral, LFM2, Kimi-K2, DSV4, Zaya. Osaurus-side dispatch should consume this rather than maintaining its own family table. |
| `JangCapabilities.reasoningParser` fallback chain | `Libraries/MLXLMCommon/JangLoader.swift:265-292` | Ordered: (a) `jang_config.capabilities.reasoning_parser` if present; (b) `reasoningStampFromModelType(modelType)` model-type prefix match; (c) `"none"`. **Important for non-JANG bundles or bundles with incomplete capabilities** — see PER-ARCH-REAL-TEST-PLAN §X for Ling-class detection-fallback details. |
| `WiredMemoryTicket` / `WiredSumPolicy` | `Libraries/MLXLMCommon/WiredMemoryPolicies.swift`, `WiredMemoryUtils.swift` | Caller-controlled wired memory budget. Currently NOT used by RunBench/osaurus; opt-in. Python `jang_tools` auto-sets to 96 GB; Swift defers to caller. Not a blocker for current MiniMax 46-49 tok/s evidence. |
| `ModelContainer.defaultGenerateParameters(fallback:)` | `Libraries/MLXLMCommon/ModelContainer.swift` (returns `GenerateParameters(generationConfig: configuration.generationDefaults, fallback:)`) | Opt-in convenience for applying a bundle's stamped `generation_config.json` defaults (max_new_tokens, temperature, top_p, top_k, min_p, repetition_penalty, do_sample) to a `GenerateParameters` instance. Pass any per-request runtime overrides via `fallback`. `generate()`/`streamGenerate()` do NOT consume this automatically — pass the result into the `GenerateParameters` argument when you want bundle defaults to win over your fallback values. |

---

## §1 Source-changing fixes (commits to inherit)

These are the actual edits in the worktree as of 2026-05-09 16:35 PDT (uncommitted, will be in the next vmlx-swift-lm release commit). When osaurus updates the pin, these come for free:

### §1.1 `Libraries/MLXLMCommon/JANGTQKernels.swift`
- Added `JANGTQMetaCacheKey` + `cachedJANGTQMeta(...)` private helper at top of file.
- All four `JANGTQKernelLibrary.*` launchers (`hadamardMultiblock`, `fusedGateUpSwiGLU`, `gatherTQ`, `gatherTQTopK`) now read meta tensors through `cachedJANGTQMeta(kind:values:)` instead of `MLXArray(...)` per call.
- `kHadamardMultiblockSource` `newv[64]` → `newv[8]` (line ~104).
- `hadamardRotate(_:signs:dim:)` removes pre-cast `xFlat.asType(.float32)` from fast path (line 496-499). Slow path keeps the cast (Mistral 3.5 only path).

### §1.2 `Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift`
- New `miniMaxJANGTQRouterCompileEnabled(environment:)` helper at line 30. Reads canonical `VMLX_MINIMAX_ROUTER_COMPILE` first, then legacy typo `VMLINUX_MINIMAX_ROUTER_COMPILE`.
- Router cache build at line 61 reads through new helper.

### §1.3 `Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift`
- `LayerKind: Int32, CaseIterable, Sendable` (line 80). Previously not `CaseIterable`.

### §1.4 `Libraries/MLXLMCommon/JANGTQStreamingExperts.swift`
- JANGTQ streaming-expert changes are included in the vmlx release surface. Osaurus integrators should still run the package build and focused cache/batching tests after each pin update.

### §1.5 `Tests/MLXLMTests/...`
- New: `CacheCoordinatorModeKeyIsolationTests`, `SSMReDeriveParityTests`, `LocalChatTemplateFamilySnapshotTests`, `ChunkedPrefillVLMTests`, `TQDiskSerializerTests`, `JANGTQStreamingExpertsTests`.
- Extended: `CacheHelpersTests`, `JANGTQKernelsTests`, `MiniMaxJANGTQConfigTests`.

---

## §2 No-action-needed (auto-engages on pin update)

osaurus does NOTHING and gets:
- MiniMax M2.7 JANGTQ family at 46-49 tok/s on solo `BatchEngine.generate` (verified bundles: `MiniMax-M2.7-JANGTQ_K`, `-Small-JANGTQ`, `-JANGTQ`, `-JANGTQ_K-CRACK`).
- Reduced AsType count + dispatch count across all JANGTQ models on the Hadamard kernel paths.
- Cache identity correctness for reasoning + media when `LMInput.cacheScopeSalt` is set by callers (see §3.1).
- ~~Warm-session live-cache restore-skip optimization on coordinator hits~~ **NOT IN THIS PIN** — `planLiveCacheReuse` was previously claimed landed; verified absent on 2026-05-10. The ~50ms warm-session savings are still on the roadmap, not in the inheritable surface.

osaurus's existing `BatchEnginePlan.openBlockers` should remain `[]` — both `kvQuantization` and `compileSupport` were already closed in the prior handoff and are not affected by these changes.

---

## §3 Caller-wiring requirements (action items for osaurus)

### §3.1 Set `LMInput.cacheScopeSalt` for reasoning and media variations (RECOMMENDED)

**Why:** Defense-in-depth. The token-hash naturally separates cache entries when prompt rendering differs (typical case). But if your caller path passes `enable_thinking=true/false` as a request flag without it materializing into different prompt tokens, two semantically different requests could hash to the same cache key.

**How (osaurus side):**

```swift
// Build additionalContext that the chat-template + cache-key pipeline understands.
let additionalContext: [String: any Sendable] = [
    "enable_thinking": req.reasoningOn,           // recognized scope key
    // future scope flags can be added here without changing this call site
]

// Helper at `Libraries/MLXLMCommon/Cache/MediaSalt.swift:cacheScopeSalt(from:)`
// returns "reasoning=on" / "reasoning=off" / nil per the recognized-key contract.
let salt: String? = cacheScopeSalt(from: additionalContext)

// `text` here is `LMInput.Text` (struct wrapping tokens + optional mask),
// NOT a Swift String. Use `LMInput.Text(tokens: mlxArray, mask: nil)` if
// you have raw tokens, or use the convenience `LMInput(tokens:mask:cacheScopeSalt:)`
// init at LanguageModel.swift:150 which wraps tokens for you and is the
// shorter path when you have no media.
let lmInput = LMInput(
    text: text,
    image: processedImage,
    video: processedVideo,
    audio: processedAudio,
    cacheScopeSalt: salt)         // <— wire this through

// The TokenIterator + BatchScheduler internally call
// `computeCacheSalt(for: lmInput)` (which combines `cacheScopeSalt` with
// the SHA256 over media bytes) and pass the combined salt through to the
// cache coordinator. Callers do NOT need to call computeCacheSalt directly.
```

**API signatures (`Libraries/MLXLMCommon/Cache/MediaSalt.swift`):**

```swift
public func cacheScopeSalt(from additionalContext: [String: any Sendable]?) -> String?
public func computeCacheSalt(for input: LMInput) -> String?
public func computeMediaSalt(for input: LMInput) -> String?
```

`cacheScopeSalt(from:)` decides what additionalContext keys are recognized (currently just `enable_thinking`); unrelated keys deliberately return `nil` so they do NOT fragment cache keys. `computeCacheSalt(for:)` is internal call-site machinery — most callers only need `cacheScopeSalt(from:)` + the `LMInput.cacheScopeSalt:` initializer parameter.

**Real-model proof status:** UNIT-PROVEN (mode + media salt key isolation in `CacheCoordinatorModeKeyIsolationTests`). REAL-MODEL multi-turn proof is OPEN per CODEX matrix. osaurus may want to add its own e2e proof on the request flow before flipping reasoning toggles in production.

### §3.1.b Reasoning policy by family (Swift Jinja + cache salt)

Osaurus should treat model capability stamps as the source of truth and pass
the user/request reasoning choice through `additionalContext["enable_thinking"]`
so Swift Jinja and the cache-key salt see the same mode.

| Family | Runtime policy | Osaurus wiring |
|---|---|---|
| ZAYA text | Reasoning-capable. Correct bundle stamp is `supports_thinking=true`, `think_in_template=false`, `reasoning_parser=qwen3`. Runtime does not auto-repair stale ZAYA stamps. | Expose reasoning toggle once real on/off rows pass. Pass `enable_thinking` from the request; include `cacheScopeSalt(from:)` in `LMInput`. |
| ZAYA1-VL | Native image/text generation is wired. Current proof covers JANGTQ2 image->text + text follow-up, same-media disk HIT, different-image MISS, TokenIterator/BatchEngine byte identity, plus JANGTQ4/MXFP4 disk-backed cache restore. | Route through VLM factory, pass images as structured `UserInput`, and keep conservative CCA cache policy: disk-backed v2 restore only, no paged-prefix claim, no TurboQuant-KV mapping. B>1 media isolation, cancellation, longer semantic rows, video, and any reasoning-on image rows remain open. |
| Ling/Bailing | Production default is non-thinking. Stamps should be `supports_thinking=false`; Swift seeds `enable_thinking=false` for these capability stamps. | Do not expose a normal reasoning toggle unless a future model-specific diagnostic path is intentionally added and tested. Still pass the default through `additionalContext` so salts stay explicit. |
| Qwen/Nemotron/DSV/Kimi/MiniMax reasoning families | Capability/model-type parser selects the reasoning parser. | Wire toggle per model support and run mixed-turn cache rows before product exposure. |

Do not add Osaurus-side family-name rewrites for ZAYA. If a local user model
has stale ZAYA `supports_thinking=false`, surface it as a bundle validation
problem or restamp the model locally with the converter rules. The only runtime
default-off behavior that should happen automatically is capability-driven
`supports_thinking=false` handling for non-thinking families such as
Ling/Bailing.

### §3.2 `enableCachingAsync()` is still model-key-only at config-build time (LATENT GAP)

**Why this still matters:** `Libraries/MLXLMCommon/ModelContainer.swift:64-99` `enableCachingAsync()` sets `config.modelKey = modelConfig.name` — does NOT include reasoning mode in the model key. The request-scope `cacheScopeSalt` (§3.1) compensates for this in well-wired callers, but if osaurus has any code path that uses `enableCachingAsync()` and bypasses `cacheScopeSalt`, two different reasoning modes could share a model-key namespace.

**osaurus action options:**
- (preferred) Always set `cacheScopeSalt` on every `LMInput` (§3.1).
- (alternative) When constructing the `CacheCoordinatorConfig`, manually add reasoning mode to the model key: `config.modelKey = "\(bundleName)|reasoning=\(modeKey)"`.

**Status:** Request-scope `cacheScopeSalt` is the production path. A future engine-side improvement can also fold reasoning mode into `enableCachingAsync()` model-key assembly, but Osaurus should not depend on that until it appears in source and tests.

### §3.3 Tool-call format dispatch — consume `ToolCallFormat.fromCapabilityName(...)` and `fromModelType(...)`

If osaurus has its own family-to-parser dispatch table, prefer the vmlx-side dispatch:

```swift
// JANG-stamped bundles (preferred — explicit capability)
if let cap = jangConfig.capabilities?.toolCallFormat {
    parser = ToolCallFormat.fromCapabilityName(cap)?.makeParser()
}
// Non-JANG fallback
if parser == nil, let mt = config.modelType {
    parser = ToolCallFormat.infer(from: mt)?.makeParser()
}
```

Coverage as of 2026-05-10: Qwen3.6, MiniMax M2, GLM4-MoE, Deepseek, Nemotron-H, Mistral, LFM2, Kimi-K2, Zaya, DSV4, Gemma-4, and Hy3/Tencent Hunyuan. JANG capability stamps additionally remap to the underlying parser (e.g. JANG `qwen` capability → xml_function parser; Hy3 `tool_parser=hunyuan` → Hunyuan parser).

Hy3/Hunyuan note: its parser can return multiple calls from one
`<tool_calls>...</tool_calls>` block. The Swift `ToolCallProcessor` must run
the parser's `parseEOS` path when the wrapper closes; otherwise only the first
call in a completed block is emitted. This is covered by
`Hy3ToolCallParserRoutingTests.processorExtractsEveryClosedWrapperCall`.

### §3.4 ChatTemplate override env

Two opt-in env vars stay live and unchanged from prior handoff:
- `VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja` — for Gemma-4 swift-jinja interaction (`Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja` and `Gemma4WithTools.jinja` ship with the package).
- `VMLX_TOKENIZER_CLASS_OVERRIDE=Qwen2Tokenizer` — for `mlx-community/Qwen3.5-VL-9B-8bit` class.

No changes; both default-off.

### §3.5 Swift Jinja / mlx-swift integration notes

- Swift Jinja receives `additionalContext` during prompt rendering. Osaurus
  should not pre-render ZAYA/Ling reasoning prompts itself; pass structured
  flags (`enable_thinking`, future effort keys) and let the tokenizer/template
  path render them.
- `cacheScopeSalt(from:)` intentionally recognizes only semantic cache-scope
  keys. Random UI state must not enter the salt or L2/paged cache reuse will
  fragment.
- ZAYA/ZAYA1-VL use Zaya CCA/path-dependent cache state. Do not route them
  through a TurboQuant-KV-only cache path; JANGTQ2/4 are weight formats, not KV
  cache formats. For ZAYA1-VL the proven cache path is disk-backed
  TQDiskSerializer v2 restore keyed by media/request salt, carrying KV plus
  `conv_state` and `prev_hs`; paged-prefix restore stays disabled.
- Ling/Bailing hybrid recurrent state must be treated like other
  path-dependent families: default no-thinking, disk-backed coordinator
  restore where enabled, and no paged-prefix claim unless the topology reports
  it safe.
- Hy3/Hunyuan v3 is text-only standard KV from the Osaurus API perspective
  today. Parser/config recognition is wired (`reasoning_parser=qwen3`,
  `tool_parser=hunyuan`, `cache_type=kv`), and the Swift no-load native runtime
  now includes `Hy3Attention`, `Hy3MoE`, layer-0 dense FFN, JANGTQ routed
  experts, MTP-drop sanitize, and fp32 dense/quantized `lm_head` projection.
  As of 2026-05-10, the local `Hy3-preview-JANGTQ` bundle has passed real Swift rows for
  single-stream coherent decode, B=2 active-slot overlap, 3-turn chat,
  paged-prefix cache hit, and L2 disk restore. Keep it pre-production for
  speed until the dispatch/cast surface is reduced (`14.9 tok/s`,
  `decodeNodes=6684`, `AsType=1258` in the current QKV-fused proof row).
- mlx-swift / Metal test harness fixes (`MLXMetalTestLock`, metallib aliasing)
  are test-only. Osaurus runtime should not copy those locks into request
  scheduling; production batching correctness is proven by real B>1 overlap
  rows, not by serializing tests.
- Test runner caveat: large mixed XCTest + Swift Testing filters that include
  MLX-backed image/cache assertions can still collide across SwiftPM helper
  processes under default parallelism. Use `swift test --no-parallel` for broad
  integration guard sweeps. This is a verifier constraint, not an Osaurus
  runtime scheduling rule.

---

## §4 Env flags relevant to perf/behavior

These can be set process-wide on the osaurus side. None are required for the 46-49 tok/s MiniMax solo result; flags are for fine-tuning or debugging.

| Env | Effect | Default | Notes |
|---|---|---|---|
| `VMLX_MINIMAX_ROUTER_COMPILE=1` | Compile MiniMax MoE topK router as a single MLX trace | off | Now actually wired (was previously inert under canonical name). Marginal gain (~1 tok/s on MiniMax). |
| `VMLINUX_MINIMAX_ROUTER_COMPILE=1` | Legacy typo alias for above | off | Kept for backwards compat. |
| `VMLX_TQ_SWITCH_GLU_COMPILE=1` | Compile the full SwitchGLU 4-kernel chain via `mx.compile(shapeless: false, body)` | off | Memory + repeat tests showed this DOES NOT recover speed on MiniMax M2.7 JANGTQ_K post-`newv[8]`. Comment in `TurboQuantSwitchLinear.swift:190` notes it can regress on some bundles. Keep off unless you have a specific bundle that benefits. |
| `MLXPRESS_STREAMING_EXPERTS=1` | Alternative streaming MoE expert path | off | Untested at scale on current tree post-`newv[8]`. Use only with explicit testing. |
| `VMLX_CHAT_TEMPLATE_OVERRIDE=path` | Override tokenizer's chat template | unset | See §3.4. |
| `VMLX_TOKENIZER_CLASS_OVERRIDE=name` | Override tokenizer class | unset | See §3.4. |
| `DSV4_KV_MODE=sliding|full|tq` | DSV4 cache topology selection | `sliding` | Per-layer DSV4Cache vs full KVCacheSimple vs TurboQuant. |
| `VMLX_MISTRAL3_PROJ_PROBE=1` | Per-projection L2 probe for Mistral 3.5 | off | Diagnostic for the 12288 hidden-dim Hadamard recursion path. |
| `VMLX_MOE_TOPK_OVERRIDE=4` | Opt-in runtime routed-MoE top-k lowering for compatible families | unset | Implemented source-side, not a default. Only lowers `currentTopK > requestedTopK`; never raises ZAYA/ZAYA1-VL top-1 to 4; never touches sampler `top_k`, DSV4 NSA `index_topk`, group-routing fields, or speculative decode knobs. Cache model keys are scoped with `|moeTopK=<K>` when valid. Keep disabled unless a specific model has full text/cache/B>1 rows. |
| `VMLINUX_MOE_TOPK_OVERRIDE=4` | Legacy typo alias for above | unset | Accepted for backwards compatibility only; new Osaurus wiring should use `VMLX_MOE_TOPK_OVERRIDE`. |

---

## §5 Behavior changes osaurus integration tests should verify

After updating the pin, osaurus integration suite should assert these still hold:

1. **MiniMax M2.7 JANGTQ family decode at ≥45 tok/s** — pin the bench cell on at least one bundle (e.g. `MiniMax-M2.7-Small-JANGTQ`) and assert solo `BatchEngine.generate` decode tok/s ≥ 45. Replaces any prior assertion that this was at 30 tok/s.
2. **No regression on Qwen3.6, Gemma4, Nemotron speeds** — sample row per family.
3. **Cache identity isolation across reasoning mode** — request with `reasoning=on` then `reasoning=off` on identical text → distinct cache keys / no cross-contamination.
4. **Cache identity isolation across media** — request with `image=A`, then same text with `image=B` → distinct cache keys.
5. **Tool-call dispatch covers all `~/models` families** — JANG capability stamp resolves to correct parser.
6. **EOS / stop-token handling unchanged** across families.
7. **Stage 1B.3 single-slot compile path engaged** for `.simple`, `.turboQuant`, `.rotating` cache families when `enableCompiledBatchDecode=true` AND `maxBatchSize=1`.
8. **Any future routed-MoE top-k override remains explicit and topology-safe** — if osaurus adopts a Swift top-k override later, assert that ZAYA/ZAYA1-VL top-1 is unchanged, MiniMax/Hy3/Qwen/Ling/Nemotron/Gemma rows only lower when their config top-k is >4, and decoded full text remains coherent across default vs top-4 rows.

---

## §6 Test surface to add osaurus-side

### §6.a Solo path coverage (low-risk, easy)

- **e2e reasoning-toggle multi-turn** — turn 1 reasoning_on / turn 2 reasoning_off / turn 3 reasoning_on on the same prompt prefix; assert no leak, no stale cache state. (Real-model proof of cacheScopeSalt at the osaurus request layer.)
- **e2e VLM media-then-text** — turn 1 with image, turn 2 text-only follow-up, turn 3 same text with different image. Assert cache miss/bypass on text-only and on image-change.
- **e2e SSM hybrid warm-session** — Qwen3.5/3.6/Nemotron-H/Bailing — multi-turn with reasoning toggles; assert SSM state restored from coordinator hit and no contamination from prior turn's reasoning tokens. **Note:** the SSM re-derive path uses inline seed at prefill-end as of 2026-04-15; the historical `SSMReDeriver` pattern was reverted (Metal race). osaurus tests should observe `SSMStateCache` hit/miss counts via `cacheCoordinator.ssmStateCache.reDerives` and `.fetchHits` to detect regressions.
- **Pinning bench rows** — record a baseline tok/s per family and assert ≥ 95% of baseline on the new pin.

### §6.b Real continuous batching (B>1) row-shape recipe

Per CODEX's `docs/PRODUCTION-READINESS-MATRIX-2026-05-09.md` §"Continuous Batching Acceptance Matrix": each B>1 row must record the command, log path, model path, git SHA, generation parameters, cache mode, `maxBatchSize`, and enough output text to inspect coherence. A passing row must also show that the request did not silently fall back to serial single-request generation.

**Minimum evidence per row:**
1. Two or more requests submitted to the SAME live `BatchEngine` while both can be active at once.
2. Active-slot overlap or scheduler-side marker proving B>1 decode actually happened.
3. Distinct prompt content + distinct expected facts, so cross-slot prompt/cache contamination is visible.
4. Coherence inspection: no loop, no stop-token leak, no reasoning-tag leak, no accidental tool-call/media carryover.
5. Cache diagnostics that distinguish no-cache, live-session reuse, paged prefix, L2 disk restore, TurboQuant KV, and topology-specific companion state.

**Required common rows** (each architecture class needs all of these before being declared real-continuous-batching production-covered):

| Row | What to assert | Common false-positive to reject |
|---|---|---|
| `B=2 no-cache overlap` | Two concurrent text prompts complete coherently with isolated outputs. | If requests ran sequentially / only solo path engaged → reject. |
| `B=4 queue/admission` | Four prompts mixed lengths complete without slot starvation or output mixing. | If only one slot decodes at a time for the whole row → reject. |
| `Cancel isolation` | Cancel one slot mid-stream; other slot continues to EOS/length, output unaffected. | Verify cancelled slot doesn't poison remaining KV/cache state. |
| `Multi-turn overlap` | Two conversations each turn 1 + turn 2; turn 2 recalls only its own facts. | Reject if turn 2 reconstructed from raw tokens dropped `cacheScopeSalt`. |
| `L2 disk restore overlap` | Turn 2 in each conversation hits disk on topologies where L2 is supported. | Fresh coordinator probe must reuse the SAME salt as the prepared `LMInput`. |
| `Paged prefix overlap` | Standard-KV topologies hit paged tier across both slots. | Hybrid/CCA/rotating may intentionally report paged-incompatible — pass only if documented (e.g. Zaya). |
| `TurboQuant KV overlap` | `kvMode=.turboQuant` with B>1, output isolation, no speed cliff beyond expected compression overhead. | Synthetic-only proof rejected; need a real model where TQ KV is allowed. |
| `Reasoning on/off overlap` | Slot A thinking-enabled + slot B thinking-disabled, then swap modes on turn 2. | Cache keys must split by mode; no `<think>` leakage into non-thinking output. |
| `Media salt overlap` (VLM) | Image request + text-only request near each other, then same text with different image. | Same text with different media must miss/bypass any contaminated cache state. |
| `Long-output soak` | At least one 256+ token row on a small safe model. | Inspect actual text; tok/s alone is insufficient. |

**Topology-specific rows** (in addition to the common rows above):

| Architecture class | Required cache/state checks (post B>1 baseline) |
|---|---|
| Standard KV MoE / dense | B=2/B=4 overlap + L2 disk + paged + TurboQuant if enabled. |
| Sliding-window + full | Rotating/full layer isolation, SWA boundary on long prompt, L2 restore preserves rotating offset. |
| Hybrid SSM/Mamba/GLA | SSM companion store/fetch under B>1, warm-pass vs async-rederive deviation, reasoning-mode salt split. |
| Zaya CCA | CCA state arrays round-trip, paged intentionally incompatible (verify graceful skip), L2 disk restore with CCA state. |
| DSv4 hybrid pool | Compressor/indexer pool restore, graph/AsType count, long-output coherence, B=2 ONLY after single-stream L2 proven. |
| VLM | Media salt, image-then-text turn, same text + different media miss, no media-contaminated KV reuse. |

osaurus integration suite should treat each B=1 family pass + the corresponding B>1 row set as separate gating signals. **Do NOT collapse them in osaurus's user-facing release notes.**

---

## §7 Open items on the vmlx-swift-lm side

These are NOT done as of 2026-05-09 16:35 PDT. osaurus should NOT assume they're proven on the next pin update:

- **MiniMax post-fix B=2 / cancel mid-stream / `submit` raw / long soak** — older test rows predate the speed fix; current re-verification not yet logged.
- **Reasoning mixed-turn real-model rows** for Ling, Qwen3.5/3.6, Bailing, Nemotron, Zaya.
- **Media salt real image-then-text rows** for Qwen-VL, Gemma-4 VLM, Nemotron-Omni, Mistral/Pixtral.
- **DSV4 Python A/B + L2 disk restore + long-output reasoning** — current speed is 14.1 tok/s and explanation hypothesis-only.
- **Gemma E2B/E4B/26B current-tree speed rows** — older E2B matrix exists; current refresh open.
- **Mistral-Small-4-119B-JANG_2L** — explicitly deprioritized; do not run without cleared window.
- **Qwen3.6-27B-JANG_4M / MXFP4** — no current row; loader/topology guard, speed/coherence row, cache row open.
- **Kimi K2.6** — only chat-template snapshot proven; speed/cache row open.
- **TokenIterator-loops-at-50.8** — pre-existing in the compiled iterator path; root cause open. **Not a blocker for BatchEngine.generate path.** osaurus production callers should prefer `BatchEngine.generate` over raw TokenIterator until iter loop is root-caused.

---

## §8 Pinning checklist for osaurus integrator

When opening the pin-update PR:

- [ ] Confirm `vmlx-swift-lm` head SHA on the pin matches the commit that includes JANGTQKernels + MiniMaxJANGTQ + TQDiskSerializer changes per §1.
- [ ] Run osaurus's existing model coverage tests on the new pin; expect no failures.
- [ ] Run a bench cell on `MiniMax-M2.7-Small-JANGTQ` and assert ≥ 45 tok/s on solo `BatchEngine.generate`. Replace any prior 30 tok/s baseline assertion.
- [ ] Audit any internal osaurus `LMInput` construction sites; pass `cacheScopeSalt` per §3.1 for reasoning + media identity.
- [ ] Audit any `enableCachingAsync()` call sites; if osaurus wants reasoning-mode-keyed cache without relying on `cacheScopeSalt`, add `|reasoning=...` suffix to `config.modelKey` per §3.2.
- [ ] Audit tool-call parser dispatch table; switch to vmlx-side `ToolCallFormat.fromCapabilityName`/`fromModelType` per §3.3 if not already.
- [ ] Recommend keeping `mlxBatchEngineMaxBatchSize` default at 1 (compile-engagement) per the Osaurus PR #1037 prior fix (`fa694e9e`).
- [ ] Run e2e reasoning-toggle + media-salt isolation tests per §6 and confirm green.
- [ ] Update osaurus's user-facing docs / release notes to reflect the new MiniMax baseline.
- [ ] Do NOT advertise speed numbers for paths that aren't proven (raw `submit`, B>1 post-fix, TokenIterator).

---

**End of integration handoff.** Local-only doc (`docs/` is gitignored). Companion `docs/PER-ARCH-REAL-TEST-PLAN-2026-05-09.md` covers the per-family scenarios osaurus may want to mirror in its own integration tests.
