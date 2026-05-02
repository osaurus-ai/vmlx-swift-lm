# Osaurus Production Reference — 2026-05-01

End-to-end integration guide covering every runtime axis osaurus needs
to be aware of when shipping vmlx-swift-lm. Sister doc to
`OSAURUS-INTEGRATION-2026-05-01.md` (which is the chronological diagnostic
log of this iteration); this doc is the **flat reference** organized by
runtime component, refreshed with the iter-12 fix sweep state.

If you're integrating the SDK at osaurus, read each section below and
verify your call sites against the contracts.

---

## Table of contents

1. [Reasoning ON / OFF](#1-reasoning-on--off)
2. [Hybrid SSM (Mamba + attention)](#2-hybrid-ssm-mamba--attention)
3. [SSM state cache + async re-derive](#3-ssm-state-cache--async-re-derive)
4. [Pixtral VL cache (Mistral 3 / 3.5)](#4-pixtral-vl-cache-mistral-3--35)
5. [Sliding-Window Attention (SWA)](#5-sliding-window-attention-swa)
6. [Default TurboQuant cache](#6-default-turboquant-cache)
7. [Layer-kind map (cache topology)](#7-layer-kind-map-cache-topology)
8. [Compile path (Stages 1B–5)](#8-compile-path-stages-1b5)
9. [Tool calling formats](#9-tool-calling-formats)
10. [Cache disk serializer (L2)](#10-cache-disk-serializer-l2)
11. [Mistral 3.5 family — special notes](#11-mistral-35-family--special-notes)
12. [Per-model status matrix](#12-per-model-status-matrix)
13. [Known limitations + open issues](#13-known-limitations--open-issues)
14. [Bench harness reference](#14-bench-harness-reference)

---

## 1. Reasoning ON / OFF

### Capability stamps

`ReasoningParser` is selected per-bundle from one of:

| Source                                              | Used when …                          |
|-----------------------------------------------------|--------------------------------------|
| `jang_config.json::capabilities.reasoning_parser`   | bundle ships a JANG capability stamp |
| `model_type` heuristic (`ReasoningParser.infer`)    | no explicit stamp                    |

`ModelContext.configuration.reasoningParserName` reflects the resolved
parser name. Surface this in osaurus logs at load time so the support
team can verify a regression is in the parser-stamping vs decode path.

### Parser families currently shipped

| Parser         | Models                                                  |
|----------------|----------------------------------------------------------|
| `think_xml`    | Qwen 3.5 / 3.6 / Qwen-3-Next, Laguna, DSV4, Holo3       |
| `harmony`      | Gemma 4 (`<\|channel>thought…<channel\|>` block)       |
| `deepseek_r1`  | NemotronH-Omni JANGTQ2, DSV4 some variants              |
| `none`         | Mistral 3.5 (no trained reasoning), Gemma 2-line        |

### `enable_thinking` toggle contract

When the bundle's chat template supports it (Qwen-style):

```swift
ui.additionalContext = ["enable_thinking": false]
```

`ToolCallProcessor`+`ReasoningParser` cooperate so:
- Reasoning content goes to `.reasoning` events (not `.chunk`)
- Toggle ON→OFF→ON across multi-turn is supported
- BENCH_OMNI rows 7 (`reasoning OFF`) + 8 (`ON→OFF→ON toggle`) gate this

### Stream events — chunk vs reasoning

`BatchEngine.generate(input:parameters:)` emits `.chunk(String)` for
user-facing tokens and `.reasoning(String)` for reasoning-channel
tokens. **Osaurus must wire BOTH** to its UI (chunk → message body,
reasoning → fold-out reasoning panel). Harnesses that count only
`.chunk` will miss output from reasoning-heavy models like Qwen-3.6
hybrid SSM and any `enable_thinking=true` Qwen call.

### Marker leak invariants

Tested in `BENCH_HARMONY_CHECK` (Gemma 4) and `BENCH_QWEN_THINKING_CHECK`
(Qwen 3.x). Asserts:
- No `<\|channel>` / `<channel\|>` / `<\|channel\|>` in `.chunk`
- No `<think>` / `</think>` in `.chunk`

If these fire in production, the parser stamp is misrouted — log the
prompt + chunk text and ping the fork team.

---

## 2. Hybrid SSM (Mamba + attention)

### Models in scope

| Model              | Hidden | Mamba layers       | Attn layers        |
|--------------------|--------|--------------------|--------------------|
| Qwen-3.5 35B-A3B   | 4096   | 1:6 mamba-attn     | repeated           |
| Qwen-3.6 35B       | 4096   | hybrid             | hybrid             |
| Qwen-3-Next        | 5120   | hybrid             | hybrid             |
| MiniMax M2.7-Small | 3072   | hybrid             | hybrid             |
| MiniMax M2.7       | 6144   | hybrid             | hybrid             |
| NemotronH-Omni     | 2688   | hybrid             | hybrid             |

### Cache topology per layer

`model.newCache(parameters:)` returns one entry per layer:
- Attention layer → `KVCacheSimple` (or `TurboQuantKVCache` after promotion)
- Mamba layer → `MambaCache` (`ArraysCache(size:2)` legacy variant)

The coordinator MUST be told the cache is hybrid:

```swift
coordinator.setHybrid(true)
```

`BatchEngine` does this automatically when admitting a slot whose cache
contains any `MambaCache`/`ArraysCache` layer (`BatchEngine.swift:652`).
Osaurus consumers don't need to call it directly.

### Hybrid scheduling guarantees

- Per-layer `cache[i]` is always passed to the matching layer (no
  cross-layer mixing). Validated by `BENCH_OMNI` row 11 (hybrid SSM
  warm-pass parity).
- Disk round-trip preserves both KV blocks AND SSM states. Validated
  by `BENCH_STABILITY` S10 (hybrid SSM disk round-trip).

---

## 3. SSM state cache + async re-derive

### Current implementation: inline seed at prefill end

(commit `fde3bb9`, no async path)

After prefill completes, BatchEngine:
1. `extractSSMStates(from: slot.cache)` — snapshots Mamba state per
   layer keyed by prompt-token-length.
2. `coordinator.put(...ssmStates: ..., diskArrays: ...)` — stores
   alongside the KV-block disk fingerprint.

On a future request whose paged-tier hit covers the SAME prompt prefix:
1. `coordinator.fetch(tokens:mediaSalt:) -> .hit(_, _, _, blocks,
   ssmStates, diskArrays)`
2. `restoreSSMStates(ssm, into: slot.cache)` — re-installs Mamba state.

### Why async re-derive was reverted

The earlier `SSMReDeriver` ran the recurrence on a separate Metal
command queue concurrent with the next prefill. **It raced with the
prefill's command buffer** under cache pressure → Metal command-buffer
abort with `notifyExternalReferencesNonZeroOnDealloc`. Reverted —
the inline-at-prefill-end path has no race.

Search terms in code: `SSMStateCache`, `extractSSMStates`,
`restoreSSMStates`. **Don't grep for `SSMReDeriver` — it's gone.**

### Hybrid-SSM full-disk-hit fix (commit `227332f`)

Trim KV by 1 + re-feed last token is wrong for hybrid SSM (Mamba state
already includes that token's recurrence). `BatchEngine` now rolls back
to full prefill on:
- `unsafePartial = remaining.nonEmpty && (hasVisualContent || hasSSMLayer)`
- `unsafeFullHit = remaining.isEmpty && hasSSMLayer`

Trade-off: hybrid SSM 2nd-request pays full-prefill cost rather than
silently emitting 0 tokens. Pure-attention caches keep the trim+re-feed
fast path.

---

## 4. Pixtral VL cache (Mistral 3 / 3.5)

### Image preprocessing

`PixtralImageProcessor` (in vmlx-swift-lm) produces:
- CHW float32 tensor (mean/std normalized)
- Per-patch placeholder tokens spliced into the text token stream at
  `[IMG]` markers

### `patch_conv` weight layout fix (commit `890e3ed`)

HuggingFace ships `patch_conv.weight` as `(out, in, kh, kw)` (PyTorch).
MLX `Conv2d` needs `(out, kh, kw, in)`. The fix transposes inside both
`Mistral3VLM.sanitize` and `Mistral3VLMJANGTQ.sanitize` (idempotent
via `PixtralVision.checkArrayShape`).

Without this fix, every image input crashed with:

```
[conv] Expect input channels in input and weight to match
       input: (1,224,224,3)  weight: (1664,3,14,14)
```

### Cache key includes image fingerprint

`mediaSalt` is mixed into the cache key when `slot.originalInput.image
!= nil` or `.video != nil`. Prevents cross-image cache leakage in:
- Multi-turn conversations with different images
- Concurrent batched requests on the same coordinator

Validated by `BENCH_VL` Turn 2 cache reuse and `BENCH_OMNI` row 10
(media-salt isolation).

### Vision-token region MUST NOT cross cache boundaries

If a prefix-paged hit lands inside the vision-token region,
`mergeInputIdsWithImageFeatures` traps `SmallVector out of range`.
`BatchEngine` rolls back to full prefill on partial VL hits:

```swift
let hasVisualContent = slot.originalInput.image != nil
                    || slot.originalInput.video != nil
```

This is the same `unsafePartial` guard as hybrid SSM (see §3).

---

## 5. Sliding-Window Attention (SWA)

### Models in scope

| Model              | Sliding pattern         | Window    |
|--------------------|-------------------------|-----------|
| Gemma 4            | 1:5 sliding:full mix    | 1024      |
| Mistral 4          | configurable            | 1024 def. |
| Laguna XS.2        | full at layer 0+, sliding pattern thereafter | 256 |
| Mistral 3.5        | **NONE** (`sliding_window: null`) | n/a |

### Per-layer cache type

```swift
return layerTypes.map { layerType in
    if layerType == "sliding_attention", let slidingWindow {
        return RotatingKVCache(maxSize: slidingWindow)
    } else if let maxKVSize = parameters?.maxKVSize {
        return RotatingKVCache(maxSize: maxKVSize, keep: 4)
    } else {
        return KVCacheSimple()
    }
}
```

→ Heterogeneous cache (full + sliding) when `layer_types` is mixed.

### `swaSeam` mask

When prefill `t > 1` and there's at least one sliding layer:

```swift
let swaOffset = min(slidingWindow, cache[swaIndex].offset)
swaMask = .array(createCausalMask(n: t, offset: swaOffset, windowSize: slidingWindow))
```

`Mistral3.swift` and `Mistral3VLMJANGTQ.swift` now gate this on BOTH
`swaIndex != nil` AND `slidingWindow != nil` AND `t > 1`. Mistral 3.5
production bundles have `sliding_window: null` so this branch is
skipped — `cache[swaIndex].offset` (which would `.item()` on
CompilableKVCache and crash compile) is never read.

### `LayerKind.rotating = 6` (TQDiskSerializer v2, commit `bf942a8`)

The disk-serializer carries `RotatingKVCache` as kind=6 (post-iter SLIDING-1).
`hasRotatingCache` guards were removed because the canonical SDPA path
correctly handles `.array(mask)` for any window size.

---

## 6. Default TurboQuant cache

### Coordinator-owned KV sizing (commit `35820ba`)

```swift
var cfg = CacheCoordinatorConfig()
cfg.usePagedCache = true
cfg.maxCacheBlocks = 2000
cfg.enableDiskCache = true
cfg.diskCacheDir = URL(fileURLWithPath: "/tmp/osaurus_disk_cache")
cfg.defaultKVMode = .turboQuant(3, 3)            // ← TQ default
cfg.defaultMaxKVSize = 65536                     // ← admission cap
```

**Per-request explicit `parameters.kvMode` / `parameters.maxKVSize`
always WIN over coordinator defaults**. Coordinator only fills `nil`
slots at admission time (`BatchEngine.swift:610-630`).

### `BatchQuantize.maybeCompress` — phase swap

A slot starts in `.fill` phase with `KVCacheSimple` layers. When the
offset crosses `min_tokens_for_compression` (per layer), `maybeCompress`
swaps that layer to `TurboQuantKVCache` in `.compressed` phase. Decode
continues unchanged from the slot's POV.

### Disk round-trip

`TurboQuantKVCache.restoreFromDecodedKV(keys:values:sourceOffset:)`
seats decoded floats as the compressed-phase prefix without re-encoding.
Validated by `BENCH_STABILITY` S8 (TQ KV mode + disk round-trip).

### Compile-ON path on TQ (Stage 2)

`CompilableTurboQuantKVCache(from: layer)` promotes once all slots are
in compressed phase. The compile trace specializes on shape and offset
counter (now MLXArray-backed, no `.item()` calls). Stage-2 drift
verified at FP precision (~5e-7) on iter 21.

---

## 7. Layer-kind map (cache topology)

`LayerKind` is the discriminator the disk-serializer + cache coordinator
use to route per-layer state correctly across processes.

| Kind | Enum case          | Cache class                              | Notes                          |
|------|--------------------|------------------------------------------|--------------------------------|
| 0    | `.kvSimple`        | `KVCacheSimple` / `CompilableKVCache`    | Standard attention             |
| 1    | `.tqCompressed`    | `TurboQuantKVCache.compressed`           | Post-promotion                 |
| 2    | `.qkv`             | legacy                                   | Pre-coordinator cache          |
| 3    | `.mamba`           | `MambaCache`                             | Hybrid SSM                     |
| 4    | `.kv`              | legacy                                   | Pre-coordinator                |
| 5    | `.skip`            | n/a                                      | Layer not in decode            |
| 6    | `.rotating`        | `RotatingKVCache` / `CompilableRotating` | SWA                            |

`BatchEngine` writes per-layer kind into the disk metadata; restore
reads it back to construct the right cache class. Mismatch is caught
at `restoreLayerData` and returns 0 → coordinator marks miss.

---

## 8. Compile path (Stages 1B–5)

### Promotion gates (`maybePromoteToCompiledDecode`)

All must hold:
- `params.enableCompiledBatchDecode == true`
- `maxBatchSize == 1` (Stage 1B.3 scope; Stage 1B.4 lifts via shared
  `[B, H, maxLen, D]` buffers — pending)
- `HardwareInfo.isCompiledDecodeSupported` (dodges MLX#3329 on macOS
  Tahoe Metal driver)
- `CacheFamily.classify(slot.cache)` matches a Stage-supported family

### Per-family promotion

| Family        | Promotion target              | Stage  | Status                   |
|---------------|-------------------------------|--------|--------------------------|
| `.simple`     | `CompilableKVCache`           | 1B.3   | Shipped                  |
| `.turboQuant` | `CompilableTurboQuantKVCache` | 2      | Shipped (iter 21)        |
| `.rotating`   | `CompilableRotatingKVCache`   | 3      | Shipped (iter 13)        |
| `.cacheList`  | `CompilableCacheList`         | 5      | Shipped (iter 22)        |
| `.mamba`      | n/a                           | 4      | Pending (hybrid trace)   |
| `.heterogeneous` | n/a                        | 4      | Pending (hybrid trace)   |

### Mistral 3.5 compile-ON unblock (commit `7389453`)

`CompilableKVCache.offset` getter calls `.item()` which crashes inside
MLX compile. Mistral 3 family was the only consumer that read
`cache.first?.offset` per layer (for `getLlama4AttentionScale`). Fix:
when `llama_4_scaling_beta == 0` (Mistral 3.5 production), skip the
offset read entirely and use a constant `MLXArray(1.0)` as the scale.

Verified: `BENCH_VL_BATCH_CHAT` compile-ON now produces text
byte-identical to compile-OFF, with 9× TTFT speedup (24.8s → 2.7s).

---

## 9. Tool calling formats

`ToolCallFormat` resolved from `jang_config.capabilities.tool_parser`
or `model_type` prefix:

| Prefix              | Format          | Models                                  |
|---------------------|-----------------|------------------------------------------|
| `qwen3_5*` / `qwen3_6*` / `qwen3_next` | `.xmlFunction`  | Qwen 3.x family       |
| `mistral3*` / `ministral3*`            | `.mistral`      | Mistral 3 / 3.5       |
| `mistral4*`                            | `.mistral`      | Mistral 4             |
| `laguna*`                              | `.glm4`         | Poolside Laguna       |
| `gemma4*`                              | `.gemma4`       | Gemma 4               |
| `kimi_k2*` / `kimi_k25*`               | `.kimiK2`       | Kimi K2 / K2.6        |
| `minimax_m2*`                          | `.minimaxM2`    | MiniMax M2 / M2.7     |
| `deepseek_v3*` / `deepseek_v4*`        | `.glm4`         | DeepSeek V3 / V4      |
| `nemotron*`                            | `.dsml`         | NemotronH-Omni        |
| `gpt_oss*`                             | `.harmony`      | OpenAI OSS GPT        |

Cap stamp wins over heuristic when both present.

### Pipeline order

```
model_forward
  → logits → sample → tokenIDs
  → tokenizer.decode (+ stop tokens)
  → ReasoningParser (split text into chunk/reasoning streams)
  → ToolCallProcessor (extract tool calls if format matches)
  → AsyncStream<GenerateEvent>
```

`BatchEngine.generate` and `Evaluate.swift` (TokenIterator-based) both
run this pipeline. Validated by `BENCH_QWEN_MULTITURN_TOOL` (3-turn
chat + tool call dance).

---

## 10. Cache disk serializer (L2)

### Format (TQDiskSerializer v2, commit `bf942a8`)

```
[file_header]
[layer_count : uint32]
for each layer:
    [layer_index : uint32]
    [layer_kind : uint8]    # 0..6 per LayerKind table
    payload depends on kind:
        kvSimple        → keys + values (fp16)
        tqCompressed    → packed + scales + biases + offset
        rotating        → keys + values (fp16) + idx + offsetArray
        mamba           → arraysCount + arrays...
        skip            → nothing
[trailer]
```

### Atomic write

`PagedCacheManager.store` writes to `.tmp` then `rename` — interrupted
writes don't poison the cache.

### Disk path

Default `CacheCoordinatorConfig.diskCacheDir = nil` (disk disabled).
Set explicitly:

```swift
cfg.diskCacheDir = URL(fileURLWithPath: "/tmp/osaurus_disk_cache")
cfg.enableDiskCache = true
```

Osaurus production should pick a persistent location (NOT `/tmp`).

### Invalidation triggers

- `clearCache()` on coordinator → wipes both L1 paged + L2 disk
- Cache key includes:
  - Tokens (full prefix)
  - `mediaSalt` (image/video/audio fingerprint)
  - Model path / load fingerprint
- Mismatch on any → coordinator returns `.miss`

---

## 11. Mistral 3.5 family — special notes

### What works (mxfp4 — production-ready)

| Path                                       | Status |
|--------------------------------------------|--------|
| Text single-turn (`BENCH_HARMONY_CHECK`)   | ✅ "What is 2+2?" → "4" |
| Image describe (`BENCH_VL` Turn 1)         | ✅ "The image displays a gradient transitioning…" |
| Image follow-up (`BENCH_VL` Turn 2 reuse)  | ✅ "The color that dominates the top edge is red." |
| BatchEngine VL compile-OFF + ON            | ✅ byte-identical text, **9× TTFT speedup ON** |
| `BENCH_STABILITY` 12 rows                  | ✅ 12/12 PASS |
| Long-context narrative continuation        | ✅ coherent (80-token doc summary, 60-token narrative) |

### What works structurally (JANGTQ — kernel correct, text degenerate)

| Path                                       | Status |
|--------------------------------------------|--------|
| Bundle load                                | ✅ |
| `patch_conv` (image input doesn't crash)   | ✅ |
| `BENCH_VL` 2-turn (no crash, cache reuse)  | ✅ |
| `BENCH_VL_BATCH_CHAT` compile-OFF + ON     | ✅ |
| `BENCH_STABILITY` rows S1–S10              | ✅ 10/10 |
| Hadamard L2 preservation + determinism     | ✅ verified at d ∈ {4096, 8192, 12288, 28672} |
| Coherent text                              | ❌ outputs `"1, a a a a…"` low-ID attractor |

### Outstanding for JANGTQ Mistral 3.5

The Hadamard kernel is provably correct (3 fixes landed: shmem 4096→
8192, per-block isolation, H_2n recursion for blocks > 8192). Residual
stream now saturates like mxfp4 (~400 vs 555 — 28% magnitude gap).
Without a Python or upstream-vmlx reference output to byte-compare
against (none exists — both refs filter `.scales` and don't run JANGTQ
Mistral 3.5), we cannot localize whether the remaining gap is:

1. A subtle remaining kernel/wrapper bug
2. 2-bit codebook precision compounded over 88×12288×28672 dense layers

**Action item for jang-tools**: emit a JANGTQ4 (4-bit, 16-entry codebook)
Mistral 3.5 bundle. Existing JANGTQ2 bundles for OTHER models (NemotronH-
Omni 2-bit, MiniMax M2.7 2-bit) are coherent at smaller scale. The JANGTQ4
would close the precision gap while the runtime stays unchanged.

### Mistral 3.5 config quirks osaurus must know

- `tie_word_embeddings: false` — `lm_head.weight` is fp16 passthrough
- `sliding_window: null` — no SWA layers; all 88 layers full-attention
- `llama_4_scaling_beta: 0` — `getLlama4AttentionScale` returns
  identically 1; the runtime short-circuits the position read for
  compile compatibility
- `rope_type: yarn`, `factor: 64`, `original_max_position_embeddings:
  4096`, `rope_theta: 1e6` — full YaRN scaling applied via `YarnRoPE`
- 88 layers × 12288 hidden × 28672 intermediate → largest dense
  decoder in the Mistral family

---

## 12. Per-model status matrix

(as of 2026-05-01 with all iter-12 fixes landed)

| Model                                   | Format     | Coherent?                  | BENCH_STABILITY | BENCH_OMNI |
|------------------------------------------|------------|----------------------------|-----------------|------------|
| Mistral-Medium-3.5-128B                 | mxfp4      | ✅ multi-turn + image      | 12/12           | n/a        |
| Mistral-Medium-3.5-128B                 | JANGTQ (2-bit) | ❌ degenerate text     | 10/10 partial   | n/a        |
| NemotronH-Nano-Omni-30B                 | MXFP4      | ✅                         | 12/12           | 13/13      |
| NemotronH-Nano-Omni-30B                 | JANGTQ4    | ✅                         | 12/12           | 13/13      |
| NemotronH-Nano-Omni-30B                 | JANGTQ2    | ✅                         | passes          | 13/13      |
| Qwen-3.6-35B-A3B                        | JANGTQ4    | ✅                         | 12/12           | n/a        |
| MiniMax-M2.7                            | JANGTQ4    | ✅                         | passes (S2)     | n/a        |
| MiniMax-M2.7-Small                      | JANGTQ     | ✅                         | passes          | n/a        |
| Holo3-35B-A3B (qwen3_5_moe)             | mxfp4      | ✅ via reasoning stream    | 12/12           | n/a        |
| Laguna-XS.2                             | JANGTQ     | ✅ "2+2 equals 4."         | passes          | n/a        |
| Laguna-XS.2                             | mxfp4      | ✅ "2+2 equals 4." (4699d3a) | 12/12         | n/a        |
| DeepSeek-V4-Flash                       | JANGTQ     | ✅ "2+2 = 4..."           | passes          | n/a        |
| Kimi-K2.6-Small / Med                   | JANGTQ     | bundle missing tokenizer   | blocked         | n/a        |

---

## 13. Known limitations + open issues

| # | Item                                  | Owner       | Action                          |
|---|---------------------------------------|-------------|----------------------------------|
| 1 | JANGTQ Mistral 3.5 text degenerate    | jang-tools  | Emit JANGTQ4 bundle              |
| 2 | ~~Laguna mxfp4 expert format~~        | ✅ done     | Polymorphic MoE landed in `4699d3a` |
| 3 | Kimi-K2.6 bundle missing tokenizer    | jangq-ai    | Re-publish bundle with `tokenizer.json` |
| 4 | BatchEngine LM-only double-load OOM   | bench harness | Restructure `BENCH_BATCH_CHAT` to share one loaded model |
| 5 | Stage 4 hybrid-trace compile pending  | vmlx-swift  | Mamba+attn unified compile trace |
| 6 | `maxBatchSize > 1` compile (Stage 1B.4) | vmlx-swift | Per-bucket shared `[B, H, maxLen, D]` |

---

## 14. Bench harness reference

| Bench knob                       | Validates                                              |
|----------------------------------|--------------------------------------------------------|
| `BENCH_HARMONY_CHECK=1`          | Reasoning marker leak invariants (Gemma 4 harmony)     |
| `BENCH_QWEN_THINKING_CHECK=1`    | Qwen 3.x prefilled-think (`<think>\n` channel)        |
| `BENCH_QWEN_MULTITURN_TOOL=1`    | Qwen 3-turn + tool calling                             |
| `BENCH_VL=1`                     | TokenIterator VL 2-turn (image describe + follow-up)   |
| `BENCH_VL_BATCH_CHAT=1`          | BatchEngine VL 2-turn, both compile OFF and ON         |
| `BENCH_VL_BATCH_VIDEO=1`         | Video ingest end-to-end                                |
| `BENCH_VL_MIXED=1`               | Single coordinator + 4 turns flipping mod/think        |
| `BENCH_OMNI=1`                   | NemotronH-Omni full multi-modal matrix (13 rows)       |
| `BENCH_STABILITY=1`              | 12 stability rows (text/disk/prefix/8-turn/16k/concurrent/cancel/TQ-KV/clearCache/SSM-disk/60k-token/mid-reasoning) |
| `BENCH_BATCH_CHAT=1`             | BatchEngine 3-turn with HF tokenizer + chat template   |
| `BENCH_CROSS_VALIDATE=1`         | TokenIterator vs BatchEngine byte-identity (temp=0)    |
| `BENCH_SIMPLE=1`                 | Single-load + N-token decode smoke                     |

### Diagnostic env-vars (production: leave unset)

- `VMLX_MISTRAL3_LAYER_PROBE=1` — per-layer residual L2 (mxfp4 + JANGTQ)
- `VMLX_MISTRAL3_PROJ_PROBE=1`  — layer 0..2 q/k/v/o L2 + tq_norms
- `DSV4_LONG_CTX=0` — opt-out of full-context KV (default ON since 2026-05-01)
- `DSV4_KV_MODE=full|tq|sliding` — KV mode override for DSV4
- `DSV4_FORCE_JANGTQ=1` — force JANGTQ dispatch when bundle weight_format mislabeled
- `OSAURUS_MLX_CLEAR_LIBRARY_TRACE=1` — Metal library-eviction trace
- `MLX_CLEAR_LIBRARY_RELEASE=1` — restore eager-release (testing only)

---

## 15. Component invariants (osaurus must respect these)

These are the runtime contracts that the SDK guarantees, and that osaurus
should NOT try to bypass. Violating any of these will silently corrupt
state — there's no defensive runtime check.

### KV cache

| Invariant                                                          | Why                                                                 |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| `coordinator.setHybrid(true)` MUST be called before admitting any slot whose cache contains `MambaCache`/`ArraysCache` | SSM-state extract/restore is gated on this flag — silent skip otherwise |
| `parameters.maxKVSize` overrides `coordinator.config.defaultMaxKVSize` | Coordinator only fills `nil` slots at admission                      |
| `parameters.kvMode` overrides `coordinator.config.defaultKVMode`    | Same reason                                                          |
| `slot.cache` layer count must equal `model.kvHeads.count`           | Per-layer routing assumes 1:1                                        |
| `mediaSalt` MUST be computed from image/video/audio bytes (not from path) | Path-based salt would alias different content with same path        |

### Reasoning + tool calling

| Invariant                                                          | Why                                                                 |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| Wire BOTH `.chunk` and `.reasoning` events through to UI            | Reasoning-heavy models (think_xml, harmony, deepseek_r1) emit during reasoning phase ONLY |
| `enable_thinking: false` in `additionalContext` toggles off reasoning prefill where supported | Otherwise reasoning runs even when caller doesn't want it          |
| `ToolCallProcessor` runs AFTER `ReasoningParser` in the pipeline    | Tool calls embedded in reasoning channel get extracted correctly    |
| Don't strip harmony/think markers before `BatchEngine.generate` | The pipeline does it — pre-stripping breaks parser state machines   |

### Hybrid SSM

| Invariant                                                          | Why                                                                 |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| Mamba state is restored DURING prefill, NOT post-decode             | `SSMReDeriver` async path was reverted (Metal race); inline-at-prefill-end is the only correct path |
| Full disk hit on hybrid SSM rolls back to full prefill              | "Trim KV by 1 + re-feed last token" double-counts SSM recurrence    |
| Partial disk/paged hit on hybrid SSM rolls back to full prefill     | SSM recurrence is path-dependent on FULL prefix                     |

### Compile path

| Invariant                                                          | Why                                                                 |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| Compile only engages with `maxBatchSize == 1` (Stage 1B.3)         | Stage 1B.4 (per-bucket shared buffers) pending                       |
| Heterogeneous cache (`KVCacheSimple` + `RotatingKVCache` mix) skips compile | Stage 4 trace grouping pending                                       |
| `cache.first?.offset` Int read MUST NOT happen inside a compile trace | `CompilableKVCache.offset` getter calls `.item()` which crashes inside `MLX.compile` |
| Mistral 3.5 family must short-circuit `getLlama4AttentionScale` when `beta == 0` | Same as above — it reads cache offset                              |

### Hadamard kernel + JANGTQ codebook

| Invariant                                                          | Why                                                                 |
|--------------------------------------------------------------------|----------------------------------------------------------------------|
| Sidecar (`jangtq_runtime.safetensors`) MUST be loaded before any `JANGTQDenseLinear`/`TurboQuantSwitchGLU` forward | The kernels fatalError otherwise                                      |
| Hadamard kernel allocates `shmem[8192]` (32 KB Apple Silicon limit) | Larger blocks must use the H_2n Swift recursion in `hadamardRotate`  |
| `signs` vector deterministic by `(in_features, seed)`               | Converter uses `numpy.default_rng(seed).choice([-1, 1], size=in_features)` — sidecar carries the materialized result |
| `codebook` symmetric Lloyd-Max for Beta((d-1)/2, (d-1)/2)           | Compute-once-per-(in_features, bits)                                 |
| Per-row `tq_norms` always positive (0 → bias to 1e-10 in converter) | Avoids divide-by-zero at inference                                  |

### Bundle-format dispatch (factory)

| `weight_format` field in config.json                  | Routes to                                              |
|-------------------------------------------------------|---------------------------------------------------------|
| `"mxtq"` (or any `"jangtq*"` alias)                   | JANGTQ codebook path (`TurboQuantSwitchGLU`/`JANGTQDenseLinear`) |
| `"mxfp4"` (or absent + `quantization.bits == 4`)      | mxfp4 affine path (`SwitchGLU` + `QuantizedLinear`)     |
| `"jang"` / `"jang_2l"` / `"jang_4m"`                  | JANG (mixed-quant) — affine via per-layer override     |
| absent + bf16 weights                                 | Plain `Linear` / unquantized                            |

For Laguna specifically (commit `4699d3a`): factory checks `weight_format == "mxtq"`
**OR** `mxtq_bits` top-level key presence to decide MoE primitive. Both
formats fuse `gate_up_proj` → sanitize splits unconditionally.

---

## Appendix: iter-12 commit log (2026-05-01)

| Hash      | Description                                                      |
|-----------|------------------------------------------------------------------|
| `a1bfe65` | Hadamard kernel `shmem[4096]→[8192]`, `newv[4]→[64]`            |
| `38086ca` | Per-block kernel rewrite (each block uses its own shmem)         |
| `890e3ed` | Mistral 3.5 VLM `patch_conv` weight transpose                    |
| `227332f` | Hybrid-SSM full-disk-hit `unsafeFullHit` guard                   |
| `53f7671` | Kimi-K2.6 (`kimi_k25`) `text_config` unwrap + `language_model.` strip |
| `7389453` | Mistral 3.5 compile-ON `cache.offset` skip when beta=0           |
| `9703b49` | Mistral3Text sibling fix                                         |
| `6096875` | Hadamard `H_2n` Swift-recursion for blocks > 8192               |
| `a8ac486` | L2-preservation regression test (4096/8192/12288/28672)          |
| `1125e20` | Hadamard determinism guard at dim=28672                          |
| `e9b7e7b` | OSAURUS doc Python/vmlx-upstream parity audit                   |
| 3 docs    | OSAURUS/memory updates                                           |
