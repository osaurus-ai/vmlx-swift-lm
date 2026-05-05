# Osaurus Integration: DeepSeek-V4-Flash JANGTQ

**Status:** production-ready as of vmlx-swift-lm `main` 2026-05-04.
**Audience:** osaurus host-app engineers integrating DSV4-Flash JANGTQ.
**Companion docs:**
- `docs/OSAURUS-CACHE-CONTRACT.md` — multi-turn / paged / prefix cache rules.
- `docs/OSAURUS-JANGPRESS.md` — production weights cache.
- `docs/superpowers/specs/2026-05-04-dsv4-flash-jangpress-prod-design.md` — design rationale.

## What is DSV4-Flash?

DeepSeek-V4-Flash is a 284B parameter MoE model. Its runtime in
vmlx-swift-lm covers:

- **MLA attention** with `head_dim = 512`, `num_kv_heads = 1` (single
  latent KV head broadcast to all 64 Q heads via GQA), partial RoPE on
  the last 64 dims.
- **mHC residual stream** (`hc_mult = 4` parallel copies, collapse +
  expand per block via Sinkhorn-normalized comb matrix).
- **Hybrid attention** combining three shapes per layer:
  - **SWA** (Sliding-Window Attention) — every layer keeps a
    `RotatingKVCache(maxSize = 128)` for local context.
  - **CSA** (Compressed-Summary Attention) — layers with
    `compress_ratio > 0` (~41 of 43 layers) augment local KV with
    pooled summaries of older tokens.
  - **HSA** (Hash-Selected Attention) — layers with `compress_ratio = 4`
    additionally use an Indexer that picks the top
    `index_topk = 512` pool rows per query position.
- **Learned per-head `attn_sink`** prepended to the softmax denominator.
- **Inverse RoPE on attention output** (strips positional info before
  the mHC residual add-back).
- **Grouped low-rank O projection** (`o_groups = 8`,
  `o_lora_rank = 1024`).
- **MoE routing** via `sqrtsoftplus(logits)` (not softmax).
- **Hash routing** for the first 3 layers (`tid2eid` lookup table —
  deterministic per token id; weights are still the
  per-token sqrtsoftplus scores at the selected expert ids).
- **Limited SwiGLU** activation:
  `silu(min(gate, 10)) * clip(up, -10, 10)`. Both gate AND up are
  clamped — without this, deep MoE stacks diverge numerically.
- **lm_head matmul in fp32** — bf16 over a 4096-dim contraction drops
  ~0.5 ULP per logit, enough to flip arithmetic answers.

## Bundle requirements

A DSV4-Flash JANGTQ bundle must ship:

| File | Required | Notes |
|---|---|---|
| `config.json` | yes | `model_type = "deepseek_v4"` |
| `jang_config.json` | yes | `weight_format`, `profile`, `mxtq_seed`, `mxtq_bits` (or `routed_expert_bits`) |
| `tokenizer.json` / `tokenizer_config.json` / `chat_template` | yes | the chat template MUST be present; missing template breaks chat at runtime |
| `model-*.safetensors` shards | yes | per-expert weights OR prestacked routed-expert tensors directly at `switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}` (`routed_expert_layout: prestacked`) |
| `jangtq_runtime.safetensors` | yes (JANGTQ) | sidecar with codebook + Hadamard signs |
| `jangtq_stacked.safetensors` | **deprecated, optional** | pre-stacked routed-expert overlay. The loader recognizes both `tq_packed`/`tq_norms` and the un-prefixed `packed`/`norms` aliases. Bundles built with `routed_expert_layout: prestacked` ship the same tensors directly in the main shards — the sidecar is redundant. See **§ Sidecar deprecation** below. |
| `model.safetensors.index.json` | optional | speeds up weight discovery |

### Sidecar deprecation

`jangtq_stacked.safetensors` was added when DSV4 bundles still stored
routed-expert weights per-expert (`mlp.experts.{E}.gate_proj.…`) and
the loader had to call `MLX.stacked(...)` at load time to materialize
the `(n_experts, …)` tensors the runtime expects. The sidecar
pre-computed those stacked tensors so loading was fast AND mmap-friendly.

Newer bundles ship with `routed_expert_layout: prestacked` —
the routed-expert weights live directly at
`model.layers.{L}.mlp.switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}`
inside the main `model-*.safetensors` shards. Same MXTQ codec, same
seed, same Hadamard rotation — just relayouted on disk so the loader
doesn't have to restack. **In this layout the sidecar is fully
redundant** and shipping it doubles bundle disk size (~65 GB on
JANGTQ_K).

The Swift loader supports BOTH layouts today (commit `2534e88`'s
`packed` → `tq_packed` alias rewriter handles whichever file the
keys come from). Migration plan:

| Phase | Bundle action | Loader behavior |
|---|---|---|
| **A (current)** | Sidecar shipped + prestacked-in-shards | Both files loaded; main shard wins on key collision |
| **B (validation)** | Sidecar shipped + prestacked-in-shards | Set `VMLX_DSV4_SKIP_SIDECAR=1` to deliberately skip the sidecar at load. Validates the sidecar-free path works without rebuilding the bundle. |
| **C (deprecation)** | Sidecar removed from HF release | Old Swift loaders (pre-`2534e88`) fail; new loaders work. Bundle disk usage drops ~65 GB. |
| **D (cleanup)** | Sidecar gone everywhere | Drop the alias rewriter at `DeepseekV4.swift:1086-1096` and the per-expert stacking path at `1098-1149`; bundles use canonical `tq_packed`/`tq_norms` natively. |

Hosts that want to validate Phase C against the current bundle
(without rebuilding) can run any DSV4 bench with
`VMLX_DSV4_SKIP_SIDECAR=1` set. The skip is logged to stderr as
`[loadWeights] VMLX_DSV4_SKIP_SIDECAR=1 — skipping jangtq_stacked.safetensors`.

Audit a bundle from Python with the codex kit:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  /Users/eric/jang/codex_dsv4_fixkit/scripts/audit_dsv4_bundle.py \
  /path/to/DSV4-Flash-JANGTQ2
```

The audit refuses bundles with critical F32 control tensors stored as
F16 (mHC `hc_*`, `attn_sink`, `ffn.gate.bias`).

## Minimum SDK call sequence

```swift
import MLXLLM
import MLXLMCommon

// 1. Load configuration — selects JangPress profile, mmap policy, etc.
let loadConfig = LoadConfiguration.default

// 2. Resolve model — the factory dispatches to DeepseekV4JANGTQModel
//    based on `model_type = "deepseek_v4"` and `weight_format = "mxtq"`.
let context = try await ModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(directory: bundleURL),
    progressHandler: nil,
    loadConfiguration: loadConfig)

// 3. Chat: ChatSession handles chat template + reasoning parser + tools.
let session = ChatSession(
    context,
    instructions: nil,            // DSV4 always thinks; system prompt is optional
    generateParameters: GenerateParameters(maxTokens: 512, temperature: 0))

// 4. Per turn:
for try await event in session.streamDetails(
    to: "What is 17 + 28?", images: [], videos: [])
{
    if let chunk = event.chunk {
        // user-visible text (after reasoning parser strips <think>...</think>)
        print(chunk, terminator: "")
    } else if case .reasoning(let r) = event {
        // present in a separate UI region — DSV4 always emits a thinking trace
        ui.appendReasoning(r)
    }
}
```

## Cache mode env vars (`DSV4_KV_MODE`)

DSV4-Flash IS a hybrid SWA+CSA+HSA architecture by definition. As of
2026-05-04, the per-layer cache shape is fixed:

- `compress_ratio == 0` (layer 0 and the last layer) →
  `RotatingKVCache(maxSize = sliding_window = 128)` (pure SWA).
- `compress_ratio > 0` (every other layer) →
  `DeepseekV4Cache(slidingWindow = 128, compressRatio = cr)` — the
  rotating window plus the compressor + indexer pools.

The host can override the SWA-component sizing via `DSV4_KV_MODE`:

| Value | Behavior | When to use |
|---|---|---|
| unset / `sliding` (default) | Hybrid cache. SWA window = 128; CSA + HSA pools persist across decode steps for `cr > 0` layers. | Chat, reasoning, normal traffic. |
| `full` | Plain `KVCacheSimple` on every layer. No rotation, no pool. Memory grows linearly with sequence length. | Long-reasoning runs that fit in memory and don't benefit from the pool. |
| `tq` | `KVCacheSimple` at construction; `BatchEngine.maybeCompress` swaps in `TurboQuantKVCache` once offset crosses the min-tokens threshold. Caller MUST also set `GenerateParameters.kvMode = .turboQuant(...)`. | Long reasoning where memory is tight. |

The legacy `DSV4_LONG_CTX={0,1}` toggle has been **removed**. There is
no "short-context" path anymore — DSV4 always uses the hybrid cache for
`cr > 0` layers.

## Reasoning parser

`reasoningStampFromModelType("deepseek_v4")` returns `"think_xml"`.
ChatSession's stream auto-strips `<think>...</think>` from the chunk
stream before tool-call processing fires. Use `streamDetails(...)` to
receive the raw `.reasoning(String)` events alongside the regular
`.chunk(String)` events.

DSV4 ALWAYS emits a `<think>...</think>` block before its final answer.
A `.chunk` stream that arrives empty after a `<think>` block usually
means `maxTokens` ran out before the model closed `</think>`. Bumping
`max_tokens` to 1024+ is normal for DSV4 chat.

## Multi-turn caveats

- Pool buffers (compressor + indexer pools, per-branch
  incomplete-window buffer state) DO survive disk round-trip and prefix
  cache restoration as of 2026-05-04 (`LayerKind.deepseekV4 = 7`). The
  host does NOT need to do anything special to enable this; passing the
  same `cache: [KVCache]` array across turns just works.
- Pre-2026-05-04 builds had a known bug where `DeepseekV4Cache.trim(_:)`
  only delegated to `local.trim(n)`, leaving stale pool rows from
  output tokens that survived multi-turn prefix-cache reuse. Symptom:
  "polite-assistant attractor loops" on long `/v1/chat/completions`
  traces. **Upgrade to 2026-05-04 or later if you see that pattern.**
- DSV4 with `compress_ratio > 0` layers is NOT compatible with the
  paged KV cache (`PagedCacheManager`). The paged block ring assumes
  uniform per-token KV; DSV4's pool rows summarize variable spans of
  raw KV. **The runtime detects this automatically as of 2026-05-04**:
  at first slot admission the `CacheCoordinator` flips to
  `isPagedIncompatible = true` and skips the paged tier for fetch +
  store; the disk tier (which understands `LayerKind.deepseekV4`)
  becomes the exclusive prefix-reuse path. Hosts do NOT need to set
  `CacheCoordinatorConfig.usePagedCache = false` — the default value
  is fine; the runtime guard activates on its own. Without the guard,
  the paged tier would silently report a token-id-hash hit on empty
  blocks for DSV4 prompts, suppressing the disk-tier lookup that
  would actually hit.

## Known good bundle smoke test

```bash
# kill any other runtimes first (per CLAUDE.md memory)
pkill -9 -f "ollama|lms|mlx_lm|RunBench|xctest"

# 3-turn coherence smoke against the production JANGTQ2 bundle
BENCH_COHERENT=1 BENCH_MAX_TOKENS=400 \
  BENCH_MODEL=/Volumes/EricsLLMDrive/jangq-ai/DeepSeek-V4-Flash-JANGTQ2 \
  .build/release/RunBench
```

Expected output (verified 2026-05-04 on M5 Max):

- Turn 1 ("My favorite color is blue.") → reasoning content acknowledges
  the color.
- Turn 2 ("What is my favorite color?") → reasoning recalls "blue"
  correctly (multi-turn cache reuse working).
- Turn 3 ("Is that a warm or cool color?") → reasoning correctly
  categorizes blue as cool.

Decode latency on M5 Max for the JANGTQ2 (74 GB) bundle:
- Cold load: ~25 s (mmap warm-up).
- Per-turn TTFT after warm load: ~6-30 s depending on prompt length.
- Decode: limited by the 43-layer × 6-active-experts MoE. Acceptable
  for chat workloads; not suited to high-tps API service.

## Architecture summary table

| Layer | `compress_ratio` | Cache type | Attention shape |
|---|---:|---|---|
| 0 | 0 | `RotatingKVCache(128)` | SWA only |
| 1 | 128 | `DeepseekV4Cache(128, 128)` | SWA + CSA |
| 2 | 4 | `DeepseekV4Cache(128, 4)` | SWA + CSA + HSA |
| 3 | 128 | `DeepseekV4Cache(128, 128)` | SWA + CSA |
| ... | alternates | ... | ... |
| 41 | 4 | `DeepseekV4Cache(128, 4)` | SWA + CSA + HSA |
| 42 | 0 | `RotatingKVCache(128)` | SWA only |

The `DSV4_KV_MODE = full` / `tq` overrides replace the per-layer cache
class with `KVCacheSimple` for every layer, dropping the pooled global
context entirely.

## P0 correctness contracts (must not regress)

1. JANGTQ fused gate/up Metal kernel applies the `swiglu_limit = 10`
   clamp — `silu(min(gate, 10)) * clip(up, -10, 10)`.
2. Affine MoE path applies the same clamp via the `glue:` 2-arg
   closure on `SwitchGLU`.
3. Hash-routing weights mirror Python `mx.take_along_axis(scores,
   inds, axis=-1)` — per-token sqrtsoftplus scores at the selected
   expert ids, optionally renormalized.
4. `lm_head` matmul runs in fp32.
5. Per-head Q RMSNorm uses the cached unit-weight ones tensor.
6. Attention prefill uses
   `_dsv4_window_visibility ∥ _dsv4_compressed_visibility ∧
   indexer_selection_mask` for the mask. Decode (L=1) gathers
   `(B, 1, k, D)` only.
7. `DeepseekV4Cache.trim(n)` does proportional pool-row truncation
   and unconditionally clears partial-window buffers.
8. Disk round-trip persists rotating window + pool + buffer state
   under `LayerKind.deepseekV4 = 7`.
