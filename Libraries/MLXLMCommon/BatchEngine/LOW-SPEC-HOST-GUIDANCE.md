# Lower-spec Apple Silicon host operating guidance

For osaurus / vmlx-swift-lm operators on Apple Silicon hosts with less
than 64 GB unified memory. Pinned to vmlx revision `de521c7`.

This document is the operating envelope: what works on smaller hosts,
what to avoid, and why. Companion to `KV-SIZING-CONTRACT.md` and
`MEDIA-MODEL-MATRIX.md`.

---

## 1. Memory budget per host class

Empirical from RunBench OmniBench + StabilityBench on real
Nemotron-3-Nano-Omni-30B-A3B-MXFP4 weights at vmlx pin `de521c7`:

| Host class | Unified | Recommended ceiling | Practical headroom |
|---|---|---|---|
| M1 / M1 Pro 8 GB | 8 GB | dense LLMs ≤ 1B params | inference-only; weights spill to swap |
| M1 / M2 16 GB | 16 GB | dense LLMs ≤ 4B params + paged cache | lite VLMs (gemma-3, smolvlm) |
| M1 / M2 / M3 24 GB | 24 GB | dense ≤ 8B + paged + L2 disk cache | image VLMs ≤ 4B; no video / audio for omni 30B |
| M3 / M4 32 GB | 32 GB | hybrid ≤ 16B (Qwen 3.5 9B-VL works); 30B MXFP4 dense LLMs | omni 30B text-only with chunked prefill (Bug 2 fix); 30B audio/video need 64 GB+ |
| M4 Pro 36–48 GB | 36–48 GB | hybrid 30B; omni 30B text + 1 image | omni 30B + audio OR video, not both |
| M4 Max 64 GB | 64 GB | omni 30B text + image + 30 s audio + 8-frame video, simultaneously | comfortable headroom for batched B=2 |
| M5 Max 128 GB | 128 GB | full multimodal at long context | Bug 2 pre-fix only fit here via swap |

The "before Bug 2 fix, omni hybrid at L=16k tokens needed > 100 GB
peak" — this is documented in the Bug 2 commit. After the fix the
same workload fits in O(chunk_size²) per Mamba layer instead of
O(prompt_length²). Smaller hosts gained the most.

---

## 2. Wired memory + Metal allocation behavior

Apple Silicon's unified memory model means VRAM and system RAM are the
same physical pool. The Metal allocator's `wired_limit_` controls how
much can be page-locked — on default-tuned systems this is roughly
70–90% of physical memory. Allocations exceeding `wired_limit_` will
spill to swap, which is fatal for inference (decode tok/s drops orders
of magnitude).

**Knobs the host can use to stay under the wired limit:**

| Knob | Default | Recommended on tight hosts |
|---|---|---|
| `CacheCoordinatorConfig.defaultMaxKVSize` | unset | 8192 — caps rolling KV window for prompts > 16384 tokens (`longPromptMultiplier × defaultMaxKVSize`) |
| `CacheCoordinatorConfig.defaultKVMode` | `.none` | `.turboQuant(keyBits: 3, valueBits: 3)` — ~5× KV memory savings |
| `CacheCoordinatorConfig.usePagedCache` | true | true — pages KV in fixed-size blocks instead of slab-allocating |
| `CacheCoordinatorConfig.enableDiskCache` | false | true — L2 spillover frees in-memory blocks under pressure |
| `CacheCoordinatorConfig.diskCacheMaxGB` | 4 | 8–16 on hosts with > 200 GB free disk |
| `MLX_CLEAR_LIBRARY_RELEASE` env | unset | leave unset — default is the leak-stale-pipelines safe path (Bug 1 fix) |
| `OSAURUS_MLX_MALLOC_TRACE` env | unset | enable when investigating OOMs to capture allocation backtraces |
| `DSV4_KV_MODE` env (DSV4 only) | unset (rotating 128) | `full` — switches new caches to KVCacheSimple for full prompt visibility |

**Bug 2 fix interaction**: chunked prefill in `NemotronHOmni.prepare`
bounds peak to `O(prefillStepSize²)` per Mamba layer instead of
`O(L²)`. The default `prefillStepSize = 512` is appropriate for hosts
≥ 32 GB. On 16 GB hosts: lower to 256 explicitly via
`GenerateParameters.prefillStepSize = 256` to halve the per-chunk peak
at modest TTFT cost.

---

## 3. Bug 1 — `notifyExternalReferencesNonZeroOnDealloc` on warm disk cache

### Original symptom

Two-request sequence to the same model with disk KV cache enabled:
turn 1 completes and writes the cached entry; turn 2 with identical
prompt restores the prefix from disk; first Metal dispatch on turn 2
hits a Metal-validation assertion inside `Device::clear_library` —
specifically a `MTLComputePipelineState` released while still
referenced by an in-flight command buffer.

The dangling reference originates in `mlx::core::fast::CustomKernel::eval_gpu`
on the same call stack as the `clear_library` call. A custom kernel's
compiled pipeline gets dropped on a stack frame that's still building
the command buffer using it.

### Fix

mlx-swift submodule advanced to `osaurus-ai/mlx@f58e52da` carrying:

- `MetalAllocator::clear_cache` no longer drops kernel pipelines
  immediately. Stale pipelines are leaked (semantically correct on
  macOS 14+ where command buffers retain their own refs to
  pipelines they're still encoding) until the next Metal-driven
  GC sweep clears them safely.
- Reverter knob: `MLX_CLEAR_LIBRARY_RELEASE=1` runtime env switches
  back to eager release for A/B testing. Off by default.

### What's been verified

- StabilityBench S2 (warm L2 disk hit, identical 2nd request) — log-confirmed `Cache disk hit` line firing without the assertion.
- StabilityBench S8 (TQ KV mode + disk round-trip across two BatchEngine instances).
- StabilityBench S10 (hybrid SSM disk round-trip).
- StabilityBench S12 (warm disk cache + mid-reasoning state — second turn restores prefix while turn-1 cached state contains buffered `<think>` tokens; the highest-fidelity reproduction of the original report's mechanism).
- OmniBench S2 + S11 multi-turn cache reuse on real Nemotron-3 weights.

### What hasn't been directly reproduced

The original Metal-validation assertion was reported on M4 Pro Debug
builds (AGXG16 family). M5 Max (AGXG17 family) does not surface the
assertion even on the unfixed mlx submodule — the validation layer
fires the assertion based on Metal driver internals that vary by GPU
family. The fix is path-confirmed on M5 (the same disk-restore code
flow runs without crashing); direct Metal-validation assertion repro
on M4 Pro Debug is the responsibility of operators who hit it in the
original report.

If the assertion resurfaces on any host, the runtime reverter
(`MLX_CLEAR_LIBRARY_RELEASE=1`) restores the pre-fix behavior for A/B
isolation without rebuilding.

---

## 4. Reasoning toggle + parser edges

`enable_thinking` Jinja kwarg flows through `UserInput.additionalContext`
to swift-jinja, which renders the model's `chat_template.jinja` with
the kwarg in scope. The template controls whether `<think>` opener +
closer are prefilled into the prompt tail.

**Known edges, all locked by tests:**

| Edge | Test | Location |
|---|---|---|
| `enable_thinking=false` with think_xml stamp (template prefills `<think></think>`) | `B1: enable_thinking=false with think_xml stamp` | ReasoningParserTests |
| max_tokens cap MID-reasoning (turn 1 saves mid-think state, turn 2 resumes) | StabilityBench S12 | RunBench |
| ON → OFF → ON across 3 turns (toggle without state contamination) | OmniBench row 8 | RunBench |
| Empty think block (`<think></think>` with no content between) | `unclosed pre-</think> on EOS flushes to reasoning` | ReasoningParserTests |
| Stray `</think>` in content mode (interleaved-thinking leak) | `Stray </think> in content mode is stripped` | ReasoningParserTests |
| char-by-char streaming with no token boundary alignment | `char-by-char streaming of pre-</think> reasoning` | ReasoningParserTests |
| Harmony channel envelope (Gemma-4 style `<\|channel>thought\n…`) | 5 streaming tests | ReasoningParserTests |

The `disableThinking` host model option flips through to
`additionalContext["enable_thinking"]` at `MLXBatchAdapter.swift:273-278`.
Default behavior on osaurus: when `disableThinking` is unset, send
`enable_thinking: true` so models that respect the kwarg activate CoT;
templates that don't reference the kwarg silently ignore it (proven
safe across the entire `LLMModelFactory` registry).

---

## 5. Tool-call parser coverage

`ToolCallFormat.fromCapabilityName(_:)` resolves a capability name to a
parser. Resolution precedence:

1. Explicit `jangConfig.tool_parser` field if present
2. Outer `model_type` heuristic via `hasPrefix` on the family list
3. Inner `text_config.model_type` for VLM wrappers
4. `none` default — content streams as plain `.chunk` events

**Per-family verification (live, Nemotron-3 hot path; others
source-confirmed via per-prefix tests):**

| Family | Capability | Test |
|---|---|---|
| Nemotron / NemotronH | `nemotron`, `nemotron_h` → `xmlFunction` (NeMo-style) | `nemotronAlias` |
| Qwen 3 / 3.5 / 3.6 / coder | `qwen3*`, `qwen3_5`, `qwen3_6` → `xmlFunction` | `qwenAliases` |
| Mistral 3 / Mistral 4 / Pixtral | `mistral*` prefix → mistral parser | `mistral3` hasPrefix |
| Minimax M2 / M2.7 | `minimax`, `minimax_m2` → `minimaxM2` | `minimaxAlias` |
| GLM4-MoE / GLM5 / DeepSeek | `glm*`, `deepseek*` → `glm4` family | `glmAndDeepseekAliases` |
| Kimi K2 / K2.5 | `kimi`, `kimi_k2` → `kimiK2` | `directRawValueWins` |
| LFM2 / DSV4 / Apertus / dense LLMs | fall through → `none` | `unknownReturnsNil` |

Tool-call edge cases (25 streaming tests in `Tool-Call Edge Cases (iter 65+)`)
all green at pin `de521c7`.

---

## 6. Context injection (additionalContext kwargs)

vmlx threads `[String: any Sendable]` from `UserInput.additionalContext`
into `applyChatTemplate(messages:tools:additionalContext:)`. swift-jinja
honors arbitrary kwargs in the template via its `Filters` infrastructure.

**Verified kwargs in active use by templates:**

| Kwarg | Templates that respect it | Default applied by host |
|---|---|---|
| `enable_thinking` | Qwen 3.5/3.6, NemotronH/Cascade, Holo3, Laguna, MiniMax M2, Kimi, DeepSeek V3/V4, GLM5 | `disableThinking` model option, default ON |
| `reasoning_effort` | DSV4 (`'max'` triggers max-effort preface) | unset by default |
| `auto_truncate_thinking` | per-template, rare | unset |

Templates that don't reference a given kwarg silently ignore it. The
osaurus host can safely send any of these unconditionally without
breaking models that don't consume them.

---

## 7. JANG config preflight

`jang_config.json` v2 schema:

```json
{
  "weight_format": "mxtq" | "mxfp4" | absent,
  "source_model": { "architecture": "..." },
  "has_vision": bool,
  "has_audio": bool,
  "vision_arch": "pixtral" | "siglip" | "radio" | ...,
  "quantization": {
    "bit_widths_used": [int, ...],
    "profile": "JANGTQ4" | "JANGTQ2" | "JANG_4M" | ...,
    "mxtq_seed": int
  },
  "mxtq_bits": {
    "attention": int,
    "shared_expert": int,
    "routed_expert": int,
    "embed_lm_head": int,
    "vision_tower": "passthrough_fp16" | int,
    "multi_modal_projector": "passthrough_fp16" | int,
    "lm_head": "passthrough_fp16" | int
  }
}
```

Host preflight (`ModelRuntime.validateJANGTQSidecarIfRequired`) catches
the most common JANGTQ deployment failure: `weight_format == "mxtq"`
declared but `jangtq_runtime.safetensors` sidecar missing. Without the
preflight, vmlx hits `abort()` inside `TurboQuantSwitchLinear` on the
first forward pass, killing the whole process. With it, the user gets
a clear error and the server stays up.

`VLMModelFactory.swift:362-403` merges `jang_config.json` fields into
`config.json` at decode time, so JANGTQ tier dispatch (4 bits / 2 bits /
mixed) flows from the bundle metadata uniformly across LLM and VLM
factories.

---

## 8. Async cancellation + recovery

`BatchEngine.generate` returns `AsyncThrowingStream<GenerationEvent, Error>`.
Cancellation flows via Swift's structured concurrency: dropping the
stream's `Task` cancels prefill mid-encode. State after cancellation:

- The cancelled slot is removed from `activeSlots` cleanly
- The cache associated with the slot is NOT persisted to disk (the
  partial-prefill state is incomplete and would corrupt prefix-hash
  lookup if stored)
- The next request on the same `BatchEngine` admits cleanly without
  any "stuck slot" residue

Verified: StabilityBench S7 (cancel mid-decode + next request)
explicitly drops the stream after 3 events and asserts the next
request completes normally. Currently passes 11/12 on M5 Max.

---

## 9. Recommended host config for tight memory budgets

```swift
let config = CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    diskCacheMaxGB: 8,
    diskCacheDir: pathToDiskCache,
    modelKey: modelId,
    defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),  // ~5× savings
    defaultMaxKVSize: 8192,                                 // 8K cap
    longPromptMultiplier: 2.0                                // > 16K → cap
)

// Per-request override for hosts with < 24 GB unified memory:
var params = GenerateParameters(...)
params.prefillStepSize = 256  // halve per-chunk peak (Bug 2 fix)
```

This config is the recommended baseline for memory-bounded inference
across arbitrary prompt lengths on hosts with 32 GB or less.
