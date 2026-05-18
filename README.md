# mlx-swift-lm (Osaurus fork)

**Maintained by [Osaurus](https://osaurus.ai)** · Fork of [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

A Swift package for running LLMs and VLMs on Apple Silicon, powered by [MLX Swift](https://github.com/ml-explore/mlx-swift). **This fork is the inference engine for [Osaurus](https://github.com/osaurus-ai/osaurus)** — primary consumer, primary integration target. Other Swift apps can use it too, but Osaurus's needs drive what's prioritized.

It tracks upstream's full model surface and adds:

- **Continuous batching** (`BatchEngine`) with per-slot KV isolation, image-mask isolation, and SSM-state merge for hybrid models
- **Multi-tier KV cache** — paged in-memory (L1) + SQLite-indexed disk (L2) + SSM-companion tier; survives process restart
- **TurboQuant KV compression** — ~5× cache memory savings ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
- **Speculative decoding** — classic AR drafter, DFlash, DDTree
- **JANG mixed-precision** — per-layer attention/MLP/expert bit widths from `jang_config.json`
- **MoE / hybrid SSM dispatch reduction** — 2-4× decode speedup vs upstream on routed-MoE and recurrent-attention models
- **Models added on top of upstream**: Gemma 4, Mistral Small 4, Qwen 3.5 / 3.6 (text + VL), DeepSeek-V4, NemotronH, Hunyuan v3 (Hy3), ZAYA / ZAYA1-VL, and the JANGTQ variants of all of the above

Existing upstream consumers don't need to change anything — every upstream API is preserved.

---

## Performance

Speed + model attention-architecture research by **[eric@osaurus.ai](mailto:eric@osaurus.ai)** and the **Osaurus team**. The Swift-runtime-for-pure-speed thesis that motivated this fork is **[t@osaurus.ai](mailto:t@osaurus.ai)** (tpae)'s. Numbers from M4 Max 128 GB with all other inference processes killed before each run.

### Single-stream decode (sustained tok/s)

| Model | Architecture | Upstream Swift | This fork | Python `mlx_lm` | Gain vs upstream |
|---|---|---:|---:|---:|---:|
| Qwen 3.5-35B-A3B | hybrid SSM + MoE | 41 | **103** | 94 | +151% |
| Gemma 4 26B-A4B | dense MoE | 27 | **87** | — | +222% |
| Gemma 4 E2B | dense | 120 | **121** | 128 | — |
| Mistral Small 4 119B | MLA + MoE | 16 | **70** | 45-50 | +338% |
| NemotronH 30B-A3B | hybrid SSM + MoE | 45 | **110** | 15.5 | +144% |
| MiniMax M2.5 172B | MoE (256 expert) | 14 | **46** | 51 | +229% |

Python baselines were measured on M3 Ultra 256 GB (~1.5× more memory bandwidth), so Swift matching Python on M4 Max means Swift is faster per-bandwidth.

The big wins on MoE / MLA / hybrid SSM come from cutting Metal kernel-dispatch overhead: this fork reduces graph-level `AsType` ops by 71-95% across those families. Dense models like Gemma 4 E2B were already near-optimal upstream. Full breakdown in [`docs/SPEED-FIXES.md`](docs/SPEED-FIXES.md) and [`docs/STRESS-TEST-RESULTS.md`](docs/STRESS-TEST-RESULTS.md).

### Multi-turn (5 turns at 256 tok/turn, growing context)

**Qwen 3.5-35B-A3B** — long context, 21 → 10,932 tokens. Full table in [`BENCHMARK-QWEN3.5-35B.md`](BENCHMARK-QWEN3.5-35B.md):

| Backend | Decode T1 | Decode T5 | Prefill avg | TTFT T1 | Overall T2-5 |
|---|---:|---:|---:|---:|---:|
| Python `mlx_lm` 0.31.2 | 122.1 | 106.1 | **1520** tok/s | 281 ms | **62.4** tok/s |
| **vmlx-swift-lm** | 106 (peak 111) | **96.2** | 1335 tok/s | **53 ms** | ~58 tok/s |
| omlx 0.3.2 | ~83 | ~57 | (broken streaming TTFT) | (broken) | 47.0 tok/s |
| LM Studio 0.4.x | 107.5 | 25.5 (no prefix cache) | n/a | (broken) | 42.4 tok/s |

vmlx-swift-lm has the fastest cold-start TTFT (5× Python), is the fastest Swift-binding runtime (+37% over LM Studio, +23% over omlx), and trails Python by ~10% on long-context decode.

**Gemma 4 26B-A4B** — short context, 25 → 5,499 tokens ([`BENCHMARK-GEMMA-4-26B.md`](BENCHMARK-GEMMA-4-26B.md)):

| Backend | Decode T1 | Decode T5 | Avg decode T2-5 |
|---|---:|---:|---:|
| **vmlx-swift-lm** | **98.2** | **86.5** | **88.4** |
| Python `mlx_lm` 0.31.2 | 71.6 | 78.1 | 77.4 |
| omlx 0.3.2 | 77.7 | 68.6 | 71.0 |

**Llama 3.2 1B 4-bit** (pure dense baseline, 47 → 11,022 tokens) — confirms that on dense models without MoE/SSM all four runtimes land within 8% of each other; the dispatch-tax wins are MoE/MLA/hybrid-specific. Full table in [`BENCHMARK-LLAMA-3.2-1B.md`](BENCHMARK-LLAMA-3.2-1B.md).

### Recent single-stream additions (B=1, 96-128 token decode budget)

Numbers from the 2026-05-09 evidence pass at commit `b9da180` against real local bundles, M4 Max 128 GB. Three runs each, median of three reported. Full per-run logs in [`build/evidence-20260509/`](build/evidence-20260509/).

| Model | Architecture | Format | TTFT | Decode (median) | Provenance |
|---|---|---|---:|---:|---|
| ZAYA1-8B | top-1 CCA + MoE | JANGTQ2 | 64-68 ms | 57.1 tok/s | [evidence log](build/evidence-20260509/zaya_ZAYA1-8B-JANGTQ2_perf.log) |
| ZAYA1-8B | top-1 CCA + MoE | JANGTQ4 | 63-66 ms | 57.2 tok/s | [evidence log](build/evidence-20260509/zaya_ZAYA1-8B-JANGTQ4_perf.log) |
| ZAYA1-8B | top-1 CCA + MoE | MXFP4 | 73-75 ms | 71.8 tok/s | [evidence log](build/model-speed-targets-20260509/zaya-mxfp4-no-cache-128.log) |
| Ling-2.6-flash | recurrent GLA + MoE | JANGTQ2 | 151-171 ms | 57.5 tok/s | [evidence log](build/evidence-20260509/Ling-2.6-flash-JANGTQ2-CRACK__perf.log) |
| Ling-2.6-flash | recurrent GLA + MoE | MXFP4 | 375-381 ms | 10.8 tok/s | known regression — [evidence log](build/evidence-20260509/Ling-2.6-flash-MXFP4-CRACK__perf.log) |
| Nemotron-Omni-Nano | hybrid SSM + MoE | JANGTQ | 86-92 ms | 117.3 tok/s | [evidence log](build/evidence-20260509/Nemotron-Omni-Nano-JANGTQ-CRACK__perf.log) |
| Nemotron-Omni-Nano | hybrid SSM + MoE | MXFP4 | 77-89 ms | 140.5 tok/s | [evidence log](build/evidence-20260509/Nemotron-Omni-Nano-MXFP4-CRACK__perf.log) |
| Hy3-preview | dense GQA + 192-expert MoE | JANGTQ2 | 663 ms | 14.9 tok/s | functional, speed-open — [evidence log](build/evidence-20260510/vmlx_hy3_jangtq_single_perf_qkv_fused_20260510.log) |

Same-machine multi-turn cache rows (cache-hit TTFT, disk-restore) for these families are in the matching `__batch_cache_hit.log` and `__batch_disk_restore.log` files alongside.

### Model coverage vs other Apple-Silicon runtimes

A few of the families we benchmark above aren't yet natively handled by other on-device MLX runtimes (verified against the `mlx-engine` source tree on `lmstudio-ai/mlx-engine` and the `mlx-lm` model registry on `ml-explore/mlx-lm`, which is what omlx wraps):

| Family | This fork | Python `mlx_lm` (omlx) | LM Studio (`mlx-engine`) |
|---|:---:|:---:|:---:|
| ZAYA / ZAYA1-8B (top-1 CCA + MoE) | ✓ | — | — |
| Ling / Bailing-Hybrid (recurrent GLA + MoE) | ✓ benched | arch in upstream | — |
| Hunyuan v3 (Hy3) | Phase A | — (only v1 / v1-dense) | — |
| NemotronH-Omni / Nemotron-Omni-Nano | ✓ benched | partial | — |
| Mistral Small 4 (MLA + MoE) | ✓ | partial | — (only Mistral3) |
| MiniMax M2 / M2.5 | ✓ | ✓ | — |

That's the practical reason most of the perf rows above don't have an LM Studio or omlx column — those runtimes can't load the bundle to begin with. Where they can (Qwen 3.5-35B, Gemma 4, Llama 3.2 1B) we measure them head-to-head in the multi-turn tables.

### Continuous batching

Throughput scales near-linearly to ~6-8 concurrent slots for routed-MoE bundles before per-batch GPU saturation kicks in. Osaurus's stress harness ran **199/199 mixed concurrent requests passing** through the BatchEngine + cache stack — see [`docs/STRESS-TEST-RESULTS.md`](docs/STRESS-TEST-RESULTS.md).

Hy3 JANGTQ2 now has a current B=2 proof on the Swift engine: active-slot overlap reached 2 and both prompts completed coherently in [`build/evidence-20260510/vmlx_hy3_jangtq_b2_concurrent_after_coalesce_20260510.log`](build/evidence-20260510/vmlx_hy3_jangtq_b2_concurrent_after_coalesce_20260510.log). The same bundle also passed post-QKV-fusion [3-turn chat](build/evidence-20260510/vmlx_hy3_jangtq_batch_chat_qkv_fused_20260510.log), [paged-prefix cache](build/evidence-20260510/vmlx_hy3_jangtq_cache_hit_qkv_fused_20260510.log), and earlier [L2 disk-restore](build/evidence-20260510/vmlx_hy3_jangtq_disk_restore_20260510.log) rows. Speed remains open at 14.9 tok/s, so it is functional rather than performance-ready.

The MiniMax M2.7 B=2 cross-slot row in [`build/evidence-20260509/minimax_m27_tq_b2.log`](build/evidence-20260509/minimax_m27_tq_b2.log) currently shows a TurboQuant cross-slot drift — that family's B>1 production claim is gated on closing that row.

**MiniMax M2.7 Osaurus speed update (2026-05-10):** the app-path 30 tok/s
regression was traced to two engine issues: the pinned source was missing the
documented single-slot `BatchEngine.generate` fast path, and the JANGTQ
Hadamard/meta optimization was not actually in the source despite stale notes
claiming it was. This tree restores both. Fresh Release `RunBench` rows on this
machine show `MiniMax-M2.7-JANGTQ` at 46.6 tok/s through
`BatchEngine.generate`, and 46.4 tok/s with a production-style
`CacheCoordinator` attached. Both rows are coherent and stop cleanly. See
[`docs/MINIMAX-OSAURUS-DECODE-SPEED-DISCREPANCY-2026-05-10.md`](docs/MINIMAX-OSAURUS-DECODE-SPEED-DISCREPANCY-2026-05-10.md).
The 74 GB `JANGTQ_K` / CRACK rows still need a follow-up run in a clear memory
window.

The 2026-04-13 multi-turn benchmark numbers above are still representative for those exact models at commits `cf55f6d` / `21176a4`. Re-running them against `main` is a periodic exercise — open follow-up.

---

## Engine modes

### Single-stream generation

Plain prefill + decode — same surface as upstream `mlx-swift-lm`. JANG, Gemma 4, Mistral Small 4, and all upstream model families load through the same `loadModelContainer(...) → ModelContainer.generate(...)` API.

```swift
let container = try await loadModelContainer(
    from: HubClient.default, using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(container)
print(try await session.respond(to: "What is the capital of France?"))
```

### Continuous batching

`BatchEngine` admits multiple concurrent requests, runs prefill + decode batched with per-slot KV isolation, and streams tokens back to each caller independently:

```swift
let engine = await container.makeBatchEngine(maxBatchSize: 8)
let stream1 = await engine.generate(input: input1, parameters: params)
let stream2 = await engine.generate(input: input2, parameters: params)
for await gen in stream1 { print(gen.text, terminator: "") }
```

Per-slot guarantees: independent `temperature`/`topP`/`maxTokens`, image-mask isolation across slots (no cross-contamination of vision-token routing), SSM-state merge for hybrid Mamba/Arrays caches (Qwen 3.5, NemotronH, Bailing, FalconH1, Jamba, MiMoV2Flash, ZAYA), B>1 reasoning-mode salt isolation, and request cancellation.

### Multi-tier KV cache

```swift
await container.enableCachingAsync()  // auto-detects hybrid models
// — or —
container.enableCaching(config: CacheCoordinatorConfig(
    pagedBlockSize: 64, maxCacheBlocks: 1000,
    diskCacheDir: URL(filePath: "/path/to/cache"), diskCacheMaxGB: 10.0))
```

Three tiers run together transparently:

| Tier | Storage | Granularity | Persistence |
|---|---|---|---|
| **L1 paged** | in-memory block pool | 64-token blocks (configurable), SHA-256 chain hashing | per-process |
| **L2 disk** | SQLite + safetensors | per-prompt-hash | survives process restarts |
| **SSM companion** | in-memory | per-layer Mamba/Arrays/ZayaCCA state | per-process; disk-paged for hybrids when L2 is enabled |

Cache key includes `modelKey` (from model config) plus `cacheScopeSalt` (reasoning-mode + media salt) plus the optional `moeTopK` suffix when `VMLX_MOE_TOPK_OVERRIDE` is set, so concurrent requests at different reasoning modes / topK never alias.

For hybrid SSM models (Mamba, ArraysCache, ZayaCCA), path-dependent state is captured at the prompt boundary via inline-during-prefill or sync re-derive, then restored on multi-turn cache hits — preventing the position-mismatch contamination that would otherwise garble turn 2+.

### TurboQuant KV cache compression

```swift
let params = GenerateParameters(kvMode: .turboQuant(keyBits: 3, valueBits: 3))
```

Compresses the KV cache 4.7-5.0× during inference using randomized Hadamard rotation + Lloyd-Max codebook + (keys only) QJL residual correction. Works with `ChatSession`, the `BatchEngine`, the multi-tier cache, and speculative decoding. Skips non-KV layers (MambaCache for SSM, RotatingKVCache for sliding-window, ZayaCCA) automatically.

### Speculative decoding

Three strategies, one field on `GenerateParameters.draftStrategy`:

```swift
params.draftStrategy = .dflash(drafterPath: …, blockSize: 16)
params.draftStrategy = .ddtree(drafterPath: …, branchingBudget: 32, blockSize: 16)
// or pass a smaller draft container directly for classic autoregressive drafting
```

At `temperature: 0`, output is byte-identical to plain greedy AR. Drafter quality affects speed, not tokens. Drafters load from HuggingFace `z-lab/<model>-DFlash` snapshots; target models must conform to `HiddenStateCaptureModel & TokenEmbedderModel`.

### Routed-MoE top-k runtime override (opt-in)

For supported MoE families (MiniMax, Hy3, NemotronH, Qwen3 / Qwen 3.5 / Qwen 3.6, BailingHybrid, Gemma 4), an opt-in env var **lowers** the routed-expert top-k at load time:

```sh
VMLX_MOE_TOPK_OVERRIDE=4   # also accepts the legacy VMLINUX_ prefix
```

Lower-only (never raises top-1 / top-2 architectures), respected per-load, folded into `CacheCoordinatorConfig.modelKey` so a topK=4 cache is never reused after restart at a different topK. ZAYA / ZAYA1-VL top-1 routing is firewalled; sampler `top_k`, speculative-decode top-k, group-routing selectors, and DSV4 NSA `index_topk` are explicitly out of scope. See [`docs/JANGTQ-TOPK-OVERRIDE-PLAN-2026-05-10.md`](docs/JANGTQ-TOPK-OVERRIDE-PLAN-2026-05-10.md).

---

## Supported models

### LLMs

Llama, Mistral, Phi, Phi-3, Phi-MoE, Gemma 2 / 3 / 3n / **4**, Qwen2 / 3 / 3-MoE / 3.5 / 3.5-MoE / 3.6, DeepSeek-V3 / **V4**, Cohere, OpenELM, InternLM2, Starcoder2, MiniCPM, Granite, Granite-MoE-Hybrid, MiMo, MiMo-V2-Flash, MiniMax (M2 / M2.5), GLM-4 / GLM-4-MoE, Falcon-H1, Bitnet, SmolLM3, ERNIE 4.5, LFM2 / LFM2-MoE, Baichuan-M1, Exaone4, GPT-OSS, Lille-130m, OLMoE / OLMo2 / OLMo3, **Bailing-MoE / Bailing-Hybrid / Ling / Ling-2.6-flash** (recurrent GLA + MoE), NanoChat, **NemotronH** (incl. NemotronH-Omni / Nemotron-Omni-Nano), AF-MoE, Jamba, **Mistral Small 4** (MLA + MoE), Mistral3, Apertus, Laguna, Kimi-K2, **ZAYA / ZAYA1-8B** (top-1 CCA + MoE).

**Functional but speed-open:** **Hunyuan v3 (Hy3)** — text-only dense GQA + 192-expert MoE + MTP-preserved-disabled. Native Swift decode, B=2 overlap, 3-turn chat, paged cache, and L2 disk restore work on the local JANGTQ2 bundle; speed is still low and MTP speculative decode is not enabled.

### VLMs

PaliGemma, Qwen2-VL / 2.5-VL / 3-VL / 3.5 / 3.5-MoE, Gemma 3, **Gemma 4**, SmolVLM2, FastVLM, Pixtral, **Mistral Small 4** (MLA + Pixtral), Mistral3, LFM2-VL, GLM-OCR, Idefics3, NemotronH-Omni, **ZAYA1-VL**.

**ZAYA1-VL status:** native image/text generation and disk-backed CCA cache restore are proven on local JANGTQ2/JANGTQ4/MXFP4 bundles. It intentionally uses `ZayaCCACache` with media/request cache salt, not TurboQuant KV and not paged-prefix restore. B>1 media isolation and longer visual-semantic rows remain open; see [`docs/ZAYA1-VL-PRODUCTION-GAP-LEDGER-2026-05-10.md`](docs/ZAYA1-VL-PRODUCTION-GAP-LEDGER-2026-05-10.md).

### Embedders

Sentence Transformers, BERT, and other popular embedding models.

### JANG / JANGTQ mixed-precision

[JANG](https://jangq.ai) bundles use per-layer mixed-precision — attention at 6-8 bit, MLP / experts at 2-4 bit (MXTQ for routed experts in JANGTQ2 / 4 / K). Loaded automatically when `jang_config.json` is present:

```swift
let container = try await loadModelContainer(
    from: URL(filePath: "/path/to/Gemma-4-26B-A4B-it-JANG_4M"),
    using: TokenizersLoader())
```

JANG capability stamps (`reasoning_parser`, `tool_parser`, `supports_thinking`, `think_in_template`, `cache_type`, `draft_strategy`) are honored at load time so reasoning / tool surfaces wire automatically without per-bundle code.

---

## Quick start

Add the package:

```swift
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", branch: "main"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.1.0"),
```

Target dependencies:

```swift
.target(name: "YourTarget", dependencies: [
    .product(name: "MLXLLM", package: "mlx-swift-lm"),
    .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
    .product(name: "MLXLMHuggingFace", package: "swift-hf-api-mlx"),
])
```

### Chat session

```swift
import MLXLLM
import MLXLMHuggingFace
import MLXLMTokenizers

let container = try await loadModelContainer(
    from: HubClient.default, using: TokenizersLoader(),
    id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(container)
print(try await session.respond(to: "What's the time complexity of binary search?"))
```

### Local model

```swift
let container = try await loadModelContainer(
    from: URL(filePath: "/path/to/model"),
    using: TokenizersLoader())
```

### VLM detection

```swift
if await container.isVLM {
    // safe to pass UserInput with .images
}
```

`VLMTypeRegistry.supportedModelTypes.contains(modelType)` lets you check synchronously before loading.

---

## `VMLXServer` + `vmlx-cli` — headless inference server

In addition to the inference libraries, this package ships an
OpenAI-compatible HTTP server (`VMLXServer` library) and a headless CLI
wrapper (`vmlx-cli` executable). They are **optional products** — the
inference libraries (`MLXLLM`, `MLXVLM`, `MLXLMCommon`, `MLXEmbedders`)
don't pull them in. Only consumers of `.product(name: "VMLXServer", …)`
or `.product(name: "vmlx-cli", …)` see the SwiftNIO / IkigaJSON /
swift-argument-parser dependencies.

`VMLXServer` is the engine that powers the [Osaurus](https://github.com/osaurus-ai/osaurus)
Mac app. It exposes:

- `OsaurusServer` — SwiftNIO HTTP server; `.start(_:)` / `.stop()` are async.
- OpenAI-compatible endpoints: `/v1/chat/completions` (streaming + non-streaming),
  `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions`, plus
  Anthropic Messages and an "Open Responses" wire format.
- `InferenceServices` registry — 17 host-supplied seam protocols
  (`ModelDirectoryProvider`, `ChatEngineProvider`, `EmbeddingProvider`,
  `SpeechProvider`, `BackgroundTaskService`, `APIKeyValidating`, …)
  with No-Op defaults the CLI uses out of the box.
- Wire-format Codable types (`ChatCompletionRequest` / `ChatMessage` /
  `Tool` / `AnthropicMessagesRequest`, …) — useful even when not
  hosting the server (e.g. building a client).

### Quick start — `vmlx-cli`

```sh
swift run vmlx-cli serve --host 127.0.0.1 --port 8080 \
                         --model-dir ~/.vmlx/models
swift run vmlx-cli list       # installed models in --model-dir
swift run vmlx-cli version    # vmlx-cli 0.1.0
```

The CLI scans a `<root>/<org>/<repo>` two-level layout (matching the
Hugging Face cache shape). Point `--model-dir` at the directory that
contains downloaded MLX-format model bundles.

### Embedding `VMLXServer` in another Swift app

```swift
// Package.swift
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", from: "..."),
// ...
.product(name: "VMLXServer", package: "vmlx-swift-lm"),
```

```swift
import VMLXServer

InferenceServices.register(modelLocator: MyModelLocator())
InferenceServices.register(modelDirectory: MyModelDirectoryProvider())
// register the rest of the seams you need; everything else stays no-op.

let server = OsaurusServer()
let config = OsaurusServer.Config(
    host: "127.0.0.1",
    port: 8080,
    serverConfiguration: .default,
    validatorFactory: { NoOpAPIKeyValidator() },
    preHandlerFactory: nil,
    trustLoopback: true
)
try await server.start(config)
```

### Releases

Binary releases of `vmlx-cli` are attached to tags of the form
`vmlx-cli-vX.Y.Z` (Apple Silicon only — MLX is arm64-only). The
release artifact is a tarball containing a stripped, ad-hoc-signed
`vmlx-cli` binary plus a SHA-256 sidecar.

---

## Why this fork is faster on MoE / MLA / hybrid models

The "Swift native runtime as the pure-speed lane for MLX inference" thesis is **[t@osaurus.ai](mailto:t@osaurus.ai)** (tpae)'s — pursuing the Swift library specifically for runtime speed is what justified investing in this fork in the first place. Root-cause investigation + per-architecture fixes by **[eric@osaurus.ai](mailto:eric@osaurus.ai)** and the **Osaurus team**.

Both Python `mlx-lm` and Swift `mlx-swift-lm` use the same C++ / Metal backend (`libmlx`, same shaders, same GPU). The speed gap is entirely about how many kernel dispatches happen per token — and Swift's graph was 2× larger than Python's because Python's `pybind11` bindings infer scalar dtypes from context while Swift defaults `MLXArray(0.5)` to `float32`, which then forces an `AsType` cast when multiplied against bfloat16 tensors.

| Model | Upstream Swift `AsType` ops | This fork |
|---|---:|---:|
| Qwen 3.5-35B | 1,176 | 60 (-95%) |
| Mistral Small 4 119B | 988 | 72 (-93%) |
| MiniMax JANG | 1,245 | 248 (-80%) |
| NemotronH Cascade | 562 | 161 (-71%) |

Each `AsType` is a separate ~20µs Metal dispatch. At 1,100 extra dispatches per decode step, that's ~22 ms of pure overhead per token — exactly the difference between 41 and 103 tok/s on Qwen 3.5-35B-A3B. Six root-cause fixes (precise softmax, scalar-dtype inference, sigmoid cast removal, universal bfloat16 conversion, identity-weight dtype, MoE gate zero-out dtype) plus compiled GeGLU / SwiGLU activations close the gap. Full per-model breakdown in [`docs/SPEED-FIXES.md`](docs/SPEED-FIXES.md).

The contributor rule: **every `MLXArray` scalar created at runtime MUST specify `dtype:`**.

```swift
// BAD — triggers AsType cascade
MLXArray(someFloat) * bfloat16Tensor
softmax(x.asType(.float32), …)

// GOOD — zero unnecessary casts
MLXArray(someFloat, dtype: tensor.dtype) * bfloat16Tensor
softmax(x, axis: -1, precise: true)
```

---

## Migrating from upstream

Change your package URL:

```swift
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", branch: "main"),
```

Everything upstream still works — the fork preserves every public API. You gain JANG support, Gemma 4, Mistral Small 4, BatchEngine, multi-tier cache, TurboQuant, and SpecDec for free.

If you're on upstream `2.x`, also see the version 3 migration notes below.

---

## Known limitations

- **Hy3** — native text decode is functional on the local JANGTQ2 bundle, including B=2 overlap, 3-turn chat, paged-prefix cache, and L2 disk restore. It remains speed-open and MTP speculative decode is preserved-disabled. Tracked in `docs/PRODUCTION-READINESS-MATRIX-2026-05-09.md` and `the production-readiness matrix`.
- **ZAYA1-VL** — native generation is wired and tested on local bundles. Current proof includes image->text, text follow-up, same-media disk-backed cache HIT, different-image MISS, TokenIterator/BatchEngine byte identity, and JANGTQ2/JANGTQ4/MXFP4 cache restore rows. Still open: B>1 media isolation, cancellation, longer semantic rows, and video.
- **Audio** — Gemma 4 supports audio natively, but the audio encoder is not yet implemented; `Gemma4.prepare` throws `VLMError.processing` on `LMInput.audio`.
- **Speculative decoding + RotatingKVCache** — speculative decoding requires trimmable caches and is not compatible after the ring wraps.
- **Raw HuggingFace checkpoints** — JANG and pre-converted mlx-community models are supported. Raw HF `transformers` checkpoints with fused `gate_up_proj` need conversion first.

---

## Migrating to version 3 (from upstream 2.x)

Version 3 decouples tokenizer and downloader implementations.

```swift
// Before (2.x)
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", from: "2.30.0")
import MLXLLM

// After (3.x)
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm/", branch: "main")
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx/", from: "0.1.0")
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx/", from: "0.1.0")
import MLXLLM
import MLXLMHuggingFace  // downloader adapter
import MLXLMTokenizers   // tokenizer adapter
```

API renames: `hub:` → `from:` (any `Downloader` or local `URL`), `HubApi` → `HubClient`, `decode(tokens:)` → `decode(tokenIds:)`.

---

## For Osaurus integrators

If you're consuming this engine from [osaurus-ai/osaurus](https://github.com/osaurus-ai/osaurus), these are the authoritative references:

| Doc | Use it for |
|---|---|
| [`Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md`](Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md) | Per-symbol map — every type/method/property Osaurus consumes, with osaurus file+line callers |
| [`Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md`](Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md) | Migration recipe + `additionalContext` (`enable_thinking`, `reasoning_effort`) plumbing |
| [`Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md`](Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md) | Continuous-batching architecture + real-model verification matrix |
| [`Libraries/MLXLMCommon/BatchEngine/KV-SIZING-CONTRACT.md`](Libraries/MLXLMCommon/BatchEngine/KV-SIZING-CONTRACT.md) | Coordinator-owned KV sizing (`CacheCoordinatorConfig.defaultKVMode` / `.defaultMaxKVSize`) |
| [`Libraries/MLXLMCommon/SpecDec/OSAURUS-SPECDEC.md`](Libraries/MLXLMCommon/SpecDec/OSAURUS-SPECDEC.md) | DraftStrategy (DFlash + DDTree), drafter checkpoint map, byte-parity invariant |

Per-topic skill references live under [`skills/mlx-swift-lm/references/`](skills/mlx-swift-lm/references/) — particularly [`tool-calling.md`](skills/mlx-swift-lm/references/tool-calling.md), [`reasoning-parser.md`](skills/mlx-swift-lm/references/reasoning-parser.md), and [`speculative-decoding.md`](skills/mlx-swift-lm/references/speculative-decoding.md).

`Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift` is byte-identical with [ml-explore/mlx-swift-lm `main`](https://github.com/ml-explore/mlx-swift-lm) — Osaurus can pin to either repo without drift.

---

## Documentation

- Per-symbol Osaurus surface: [`Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md`](Libraries/MLXLMCommon/BatchEngine/OSAURUS-API-SURFACE.md)
- Continuous batching architecture: [`Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md`](Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md)
- KV sizing contract: [`Libraries/MLXLMCommon/BatchEngine/KV-SIZING-CONTRACT.md`](Libraries/MLXLMCommon/BatchEngine/KV-SIZING-CONTRACT.md)
- Speculative decoding: [`Libraries/MLXLMCommon/SpecDec/OSAURUS-SPECDEC.md`](Libraries/MLXLMCommon/SpecDec/OSAURUS-SPECDEC.md)
- ZAYA1-VL production gap ledger: [`docs/ZAYA1-VL-PRODUCTION-GAP-LEDGER-2026-05-10.md`](docs/ZAYA1-VL-PRODUCTION-GAP-LEDGER-2026-05-10.md)

Upstream API docs (still authoritative):

- [MLXLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon)
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm)
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm)
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders)

---

## License

MIT License. See [LICENSE](LICENSE).

Based on [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) by Apple's ML Explore team.

## Acknowledgments

- **[t@osaurus.ai](mailto:t@osaurus.ai)** (tpae) — originated the "Swift library as the pure-speed runtime lane for MLX inference" thesis that justified investing in this fork; the dispatch-count work below is the answer to that question
- **[eric@osaurus.ai](mailto:eric@osaurus.ai)** and the **Osaurus team** — speed + model attention-architecture research, MoE / MLA / hybrid SSM dispatch reduction, JANG mixed-precision integration, BatchEngine / multi-tier cache / TurboQuant / SpecDec, and the per-family native runtime work for Gemma 4, Mistral Small 4, Qwen 3.5 / 3.6, ZAYA, NemotronH, Hy3
- [Apple ML Explore](https://github.com/ml-explore) — MLX and mlx-swift-lm
- [JANG](https://jangq.ai) — mixed-precision quantization format
- [Google DeepMind](https://deepmind.google) — Gemma 4 architecture
- [Tencent](https://github.com/Tencent) — Hunyuan v3 (Hy3) architecture
- [Zyphra](https://www.zyphra.com) — ZAYA / ZAYA1-VL CCA architecture
