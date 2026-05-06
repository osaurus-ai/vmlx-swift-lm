# Osaurus Integration: DeepSeek-V4-Flash JANGTQ

**Status:** production-ready as of vmlx-swift-lm `main` 2026-05-04.
**Audience:** osaurus host-app engineers integrating DSV4-Flash JANGTQ.
**Companion docs:**
- `docs/OSAURUS-CACHE-CONTRACT.md` — multi-turn / paged / prefix cache rules.
- `docs/OSAURUS-JANGPRESS.md` — production weights cache.
- `docs/superpowers/specs/2026-05-04-dsv4-flash-jangpress-prod-design.md` — design rationale.
- `~/mlx/vllm-mlx/docs/DSV4_RUNTIME_REGRESSION_TRACE.md` — Python
  runtime regression trace. The originally requested
  `DSV4_FIX_NUANCES.md` filename was not present locally on 2026-05-06.

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
| `tokenizer.json` / `tokenizer_config.json` | yes | The local Flash bundle has `chat_template = null`; Swift uses the tracked `DSV4Minimal` fallback. Bundles may ship a compatible template, but absence is no longer a runtime blocker. |
| `model-*.safetensors` shards | yes | per-expert weights OR prestacked routed-expert tensors directly at `switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}` (`routed_expert_layout: prestacked`) |
| `jangtq_runtime.safetensors` | yes (JANGTQ) | Small required runtime sidecar with codebook + Hadamard signs. The 2026-05-06 Flash bundle needed `signs.4096.42`, `codebook.4096.2`, `signs.2048.42`, and `codebook.2048.2`. |
| `jangtq_stacked.safetensors` | deprecated, optional | Old pre-stacked routed-expert overlay. Do not confuse this large overlay with the small required `jangtq_runtime.safetensors` sidecar. |
| `model.safetensors.index.json` | optional | speeds up weight discovery |

### Stacked overlay deprecation

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
doesn't have to restack. In this layout the old stacked overlay is
redundant and shipping it doubles bundle disk size on large variants.

This does **not** deprecate `jangtq_runtime.safetensors`. The runtime
sidecar remains required for JANGTQ bundles because the TurboQuant kernels
need deterministic codebook and Hadamard-sign tensors. The local DSV4 Flash
JANGTQ bundle was prestacked in the main shards but still needed the small
runtime sidecar rebuilt on 2026-05-06 before live loading was valid.

The Swift loader supports BOTH layouts today (commit `2534e88`'s
`packed` → `tq_packed` alias rewriter handles whichever file the
keys come from). Migration plan:

| Phase | Bundle action | Loader behavior |
|---|---|---|
| **A (current)** | `jangtq_runtime.safetensors` shipped + prestacked-in-shards | Required runtime sidecar loads codebooks/signs; main shards provide prestacked routed tensors. |
| **B (validation)** | No `jangtq_stacked.safetensors`; runtime sidecar present | Validates the overlay-free prestacked path without removing codebooks/signs. |
| **C (deprecation)** | Old stacked overlay removed from releases | Loader still works because routed tensors are in the main shards. Bundle disk usage drops when old overlays are absent. |
| **D (cleanup)** | Overlay path gone everywhere | Drop only the old overlay/per-expert restacking compatibility code after all bundles are prestacked. Keep runtime sidecar handling. |

Hosts that want to validate Phase C against the current bundle
(without rebuilding) should verify that `jangtq_stacked.safetensors` is absent
or unused while `jangtq_runtime.safetensors` remains present and non-empty.

Audit a bundle from Python with the codex kit:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  ~/jang/codex_dsv4_fixkit/scripts/audit_dsv4_bundle.py \
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

// 3. Production chat: use UserInput(chat:) so template kwargs flow into
//    the DSV4 fallback/bundle template.
let coordinator = CacheCoordinator(config: .init(
    usePagedCache: true,
    enableDiskCache: true,
    diskCacheDir: osaurusDiskCacheDir,
    modelKey: bundleURL.lastPathComponent))
let engine = BatchEngine(context: context, maxBatchSize: 4, cacheCoordinator: coordinator)

let input = UserInput(
    chat: [.system("You are concise."), .user("What is 17 + 28?")],
    additionalContext: [
        "enable_thinking": false,
        "reasoning_effort": Optional<String>.none as Any
    ])

let params = GenerateParameters(maxTokens: 512, temperature: 0, topP: 0.95)
let prepared = try await context.processor.prepare(input: input)
let stream = await engine.generate(input: prepared, parameters: params)

for await event in stream {
    switch event {
    case .chunk(let text): ui.appendVisibleText(text)
    case .reasoning(let text): ui.appendReasoning(text)
    case .toolCall(let call): await dispatchTool(call)
    case .info(let info): logCompletion(info)
    }
}

// Hard reasoning / max-effort requests should enable thinking and allocate a
// larger budget. The strict live RunBench row passed with 384 max tokens and a
// max-only repetition penalty; product requests may use larger budgets.
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
| `full` | Plain `KVCacheSimple` on every layer. No rotation, no pool. Memory grows linearly with sequence length. | Diagnostic comparison only; not the osaurus serving path. |
| `tq` | `KVCacheSimple` at construction; `BatchEngine.maybeCompress` swaps in `TurboQuantKVCache` once offset crosses the min-tokens threshold. Caller MUST also set `GenerateParameters.kvMode = .turboQuant(...)`. | Diagnostic memory experiments only; this disables the DSV4 CSA/HSA pool. |

The default hybrid path is the production compression path. A global
`GenerateParameters.kvMode = .turboQuant(...)` or coordinator
`defaultKVMode = .turboQuant(...)` does **not** replace DSV4's hybrid
cache; `DeepseekV4Model.newCache(parameters:)` still returns
`RotatingKVCache` plus `DeepseekV4Cache`, and BatchEngine's TQ hook has no
`KVCacheSimple` layer to promote. TurboQuant is available only through the
explicit diagnostic override `DSV4_KV_MODE=tq`.

For paged/prefix/L2 cache integration, do not put DSV4 on the generic
TurboQuant block-cache path. The coordinator marks DSV4's hybrid pool as
paged-incompatible and uses the disk tier (`TQDiskSerializer` with
`LayerKind.deepseekV4`) to persist the rotating SWA window, CSA/HSA pools,
and incomplete compressor/indexer buffers together.

At long context, the default cache is already roughly the 90% KV-memory cut
the model was designed around. The exact ratio depends on how many layer
positions are in `compress_ratio = 4` versus `128`, but the shape is:
`128` local SWA tokens plus `T / compress_ratio` pooled rows instead of `T`
ordinary KV rows. For DSV4-Flash, layers 0 and 42 are SWA-only; the middle
layers alternate CSA (`cr=128`) and CSA+HSA (`cr=4`).

The legacy `DSV4_LONG_CTX={0,1}` toggle has been **removed**. There is
no "short-context" path anymore — DSV4 always uses the hybrid cache for
`cr > 0` layers.

## Prefix / L2 Cache Contract

DSV4 cache keys are prompt-token keys. The stored cache payload must be a
prompt-boundary snapshot, not the live cache after generated tokens have
been decoded. This is stricter than ordinary KV models because DSV4's
`DeepseekV4Cache` carries:

- local SWA ring-buffer state,
- CSA compressor pool rows,
- HSA indexer pool rows,
- incomplete-window buffers for both compressor and indexer branches.

As of 2026-05-06, both Swift generation paths follow that contract:

- `TokenIterator` captures a prompt-boundary cache snapshot before the
  async generation loop starts and stores from that snapshot on stream
  completion.
- `BatchEngine` captures `BatchSlot.promptCacheSnapshot` immediately
  after prefill and before the first generated token is fed back into the
  model.
- The disk tier receives a copy of that snapshot. If KV TurboQuant is
  requested and the prompt length crosses the TQ threshold, the snapshot
  is compressed before disk write, preserving TQ encoded-block storage
  without using post-decode cache state.

The disk serializer stores DSV4 layers as `LayerKind.deepseekV4` with
rotating SWA state plus CSA/HSA pool and buffer payloads. Nil pool/buffer
slots are marked by `__dsv4_{layer}_nilmask__`; the tensor payload uses a
small non-empty sentinel because safetensors rejects empty arrays.

Validated live on `~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ`:

```bash
BENCH_BATCH_DISK_RESTORE=1 BENCH_MAX_TOKENS=4 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

Expected result: session 1 writes a disk entry, a fresh coordinator
probes a disk hit for 132/132 prompt tokens, and session 2 restores from
disk without re-prefilling the whole prompt.

## Reasoning parser

`reasoningStampFromModelType("deepseek_v4")` returns `"think_xml"`.
ChatSession's stream auto-strips `<think>...</think>` from the chunk
stream before tool-call processing fires. Use `streamDetails(...)` to
receive the raw `.reasoning(String)` events alongside the regular
`.chunk(String)` events.

`enable_thinking=false` is plain-answer mode. The current Swift fallback
renders `<｜Assistant｜></think>`, so `ReasoningParser.forPrompt` starts in
visible-answer mode and output flows through `.chunk`. The 2026-05-06 strict
full-weight chat row recalled `sapphire-42` across three turns with
`unclosedReasoning=false`.

`enable_thinking=true` opens `<think>` and routes thought text through
`.reasoning`. A `.chunk` stream that arrives empty at a tiny budget can still
be a valid reasoning-only or length-finished state; production should surface
that telemetry rather than treating it as marker leakage.

## Multi-turn caveats

- Pool buffers (compressor + indexer pools, per-branch
  incomplete-window buffer state) DO survive disk round-trip and prefix
  cache restoration as of 2026-05-06 (`LayerKind.deepseekV4 = 7` plus
  `__dsv4_{layer}_nilmask__`). The
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

# Full DSV4 production gate on the local Flash JANGTQ bundle
BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=all \
  BENCH_MAX_TOKENS=128 BENCH_DSV4_REASONING_MAX_TOKENS=384 \
  BENCH_DSV4_LONG_REPEAT=220 BENCH_DSV4_LONG_MAX_TOKENS=96 \
  BENCH_MODEL=~/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

Expected output (verified 2026-05-06 on M5 Max):

- Turn 1 stores `sapphire-42`; turn 2 recalls it; turn 3 answers the
  sapphire/blue follow-up.
- Reasoning off/on/max all answer `12` with no raw `<think>` leakage.
- The 5,568-token long row recalls `CERULEAN RIVER / OSLO`.
- Finish state is `stop=stop`, `unclosedReasoning=false`, clean process exit.

Latest strict full-row sample on M5 Max:
- Load: about 7 s from warm local storage.
- Full chat + reasoning + 5,568-token long row: about 132 s wall time.
- Max RSS: about 69.1 GB; peak memory footprint: about 111.9 GB.

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
