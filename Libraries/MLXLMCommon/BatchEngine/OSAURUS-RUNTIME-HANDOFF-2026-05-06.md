# Osaurus Runtime Handoff - 2026-05-06

Audience: osaurus agents wiring `vmlx-swift-lm` into the production runtime.

Scope: text runtime, chat templates, reasoning streams, DSV4, Ling/Bailing,
Gemma/Laguna/Qwen/MiniMax/Nemotron-Omni, BatchEngine cache tiers, SSM companion
state, and TurboQuant KV correctness. Distributed inference and JangPress speed
work are intentionally out of scope for this handoff.

## Current Position

Use `BatchEngine.generate(input:parameters:)` as the production path for normal
osaurus chat/runtime traffic. Use `UserInput(chat:)` for chat models, not a raw
`"User:\nAssistant:"` transcript, whenever template semantics matter.

The runtime is coherent at the minimum bar across the currently tested local
families. Speed tuning remains open, especially MiniMax M2.7 throughput and
Ling MXFP4 affine decode after removing the oversized fused gate/up cache. Do
not block integration on speed unless product policy requires a token/s
threshold.

## Osaurus PR Agent Notes

This is the handoff document to attach or cite from the first osaurus PR that
wires this library. The PR should be a runtime-coherence integration PR, not a
speed PR and not a JangPress PR.

PR scope:

- Use `vmlx-swift-lm` as the library/runtime source of truth.
- Wire `BatchEngine.generate(input:parameters:)` for production chat.
- Build production requests with `UserInput(chat:)` and explicit
  `additionalContext`.
- Consume `.chunk`, `.reasoning`, `.toolCall`, and `.info` as separate stream
  events.
- Surface reasoning-only length finishes cleanly in osaurus UI/API telemetry.
- Keep cache ownership in vmlx via `CacheCoordinator` and model-specific cache
  topology.

Do not include in the first PR:

- JangPress default-on behavior.
- Distributed inference.
- Speed-target enforcement.
- App-layer guards for DSV4/Ling cache behavior.
- Osaurus-side parsing of raw `<think>`, harmony, DSML, Qwen XML tool, or
  MiniMax tool markers from visible text.

Production-ready minimum as of this handoff:

- Ling JANGTQ/JANGTQ2/MXFP4 load and answer coherently in multi-turn recall.
  JANGTQ2 is the preferred Ling production bundle for memory. MXFP4 is now
  memory-bounded but slow.
- DSV4 Flash loads, routes reasoning modes, and stores/restores cache state.
  Short chat recall and the 3K-token long-context row are revalidated after
  the local sidecar rebuild. The 5K+ stress row remains a memory-hardening
  task, not a minimum coherence blocker.
- Qwen 35B, Qwen 27B, Gemma4 26B, Laguna XS.2, MiniMax M2.7, and Nemotron Omni
  have live coherent-output coverage without marker leakage in the tested rows.
- Paged/prefix cache, L2 disk restore, SSM companion state, and TurboQuant KV
  disk round-trip have targeted coverage.

Hygiene notes for PR agents:

- `RunBench/` and `docs/` are gitignored in this checkout. Local benchmark
  harness edits and raw logs are audit artifacts unless explicitly force-added.
- Keep PR code to library/runtime files plus this handoff. Do not commit stale
  local logs by accident.
- There is no new duplicate runtime path for the first PR. Use the existing
  `BatchEngine`, `CacheCoordinator`, `ReasoningParser`, `ToolCallProcessor`,
  model loaders, and tokenizer processor APIs.
- Treat old JangPress docs that say "production ready" as historical for this
  phase. JangPress is deferred until runtime coherence is wired.
- Treat old speed docs as constraints for future optimization, not acceptance
  blockers for this PR.

## 2026-05-06 Local Revalidation

This pass revalidated the integration surface after pinning Swift-Jinja through
the osaurus-owned dependency chain and fixing RunBench's tokenizer smoke path.

| Area | Result |
|---|---|
| Package dependency graph | PASS. `Package.swift` resolves `osaurus-ai/Jinja` tag `2.3.5` and `osaurus-ai/swift-transformers` revision `b4a094b34b997167549c7f45bde16c80f18ed5a8`; no active `huggingface/swift-jinja` edge remains. |
| Release build | PASS. `swift build -c release --product RunBench`. |
| Focused Swift tests | PASS with Xcode toolchain and `--no-parallel`: paged/prefix cache, cache coordinator, SSM state cache, TQ disk serializer CacheList, interleaved reasoning/tool calls, Gemma4 template probes, and CompilableTurboQuantKVCache parity. |
| Kimi K2.6 Small JANGTQ config/template | PASS after RunBench switched to the production tokenizer substitution path. Remaining warning: tokenizer EOS `163585` is not in effective EOS `[163586]`. |
| DSV4 template kwargs | PASS. `enable_thinking` and `reasoning_effort=max` reached the shipped DSV4 template through Swift-Jinja. |
| DSV4 production chat row | PASS. Full-weight `BENCH_DSV4_COHERENCE BENCH_DSV4_ROW=chat` recalled `sapphire-42` and answered the sapphire/blue follow-up. Wall time was 503 s under memory pressure; turn 2 finished with answer content inside `.reasoning` and empty `.chunk` at 128 tokens. |
| Laguna XS.2 JANGTQ live loop probe | PASS. No loop and no raw marker leak. Thinking-on stayed in `.reasoning` at short budget. Peak memory footprint about 11.9 GB. |
| Qwen3.6 35B JANGTQ live reasoning gate | PASS. No `<think>` leakage in `.chunk`; reasoning deltas populated. Cosmetic VLM factory warning still appears before LLM factory succeeds. |
| Gemma4 26B JANG live harmony gate | PASS. No harmony markers in `.chunk`; the selected prompt did not elicit `.reasoning` deltas. Peak memory footprint about 27.9 GB. |
| Qwen cache stack live probes | PASS. Paged-prefix hit fired with hybrid rollback semantics; disk L2 restore hit with `ssm_companion`; TurboQuant KV B=2 kept plain-slot output byte-identical beside a TQ slot. |
| Ling 2.6 flash JANGTQ/JANGTQ2/MXFP4 live ChatSession | PASS compile off/on. Multi-turn recall stayed coherent through `ArraysCache` reuse. JANGTQ2 peak footprint about 30.4 GB; MXFP4 peak footprint fixed from ~110 GB to ~66.8 GB by disabling oversized fused gate/up cache materialization. |
| MiniMax M2.7 JANGTQ live perf/coherence probe | PASS for coherent visible text, no loop, no raw marker leakage. Throughput remains below target and is deferred to speed work. |

### 2026-05-06 Cache/SSM WIP Recheck

This recheck was run in `/Users/eric/vmlx-swift-lm` after the Ling
`enable_thinking` fix landed at `82ce729`.

| Row | Result |
|---|---|
| `swift test -c release --filter SSMStateCacheTests --no-parallel` | PASS with Xcode toolchain after copying `default.metallib` / `mlx.metallib` beside the release test bundle. 7 tests passed, including SSM companion disk round-trip, media-salt isolation, and over-cap eviction. |
| `swift build -c release --product RunBench` | PASS with existing distributed/VL/model warnings only. |
| Ling JANGTQ `BENCH_COHERENT=1` | PASS. Compile off/on both recalled favorite color, identified blue as cool, and produced no loop or raw marker leak. |
| DSV4 Flash short `BENCH_DSV4_COHERENCE BENCH_DSV4_ROW=chat` | PASS before the local DSV4 bundle was removed for redownload. Three turns recalled `sapphire-42`; visible answer content was routed through `.reasoning`, not `.chunk`, as expected for this bundle. |
| DSV4 Flash long row | PASS after local sidecar rebuild. The 3,068-token row recalled `CERULEAN RIVER / OSLO` with reasoning present, visible final answer, and no loop. The 5,568-token stress row was killed under overlapping DSV4 load and is not a clean verdict. |
| Qwen3.6 35B JANGTQ `BENCH_BATCH_TQ_B2=1` | PASS. Plain KV slot stayed byte-identical beside a TurboQuant KV neighbor; both TQ slots decoded coherent text. |
| MiniMax M2.7 Small JANGTQ `BENCH_BATCH_DISK_RESTORE=1` | PASS. Fresh coordinator hit disk L2; prompt time dropped from 13.608s to 0.179s. |
| MiniMax M2.7 Small JANGTQ `BENCH_BATCH_CACHE_HIT=1` | PASS. Paged prefix probe hit 128/186 tokens and warm/cold prompt ratio was 0.36. |
| MiniMax M2.7 full-size speed row | NOT COMPLETED. The process never entered a real heavy load path and was stopped after several minutes at ~300 MiB RSS. Keep prior MiniMax speed status as open. |

Runtime change made in this pass: `CacheCoordinatorConfig.enableSSMReDerive`
now gates the extra synchronous prompt-boundary SSM companion rederive/store
pass in both `Evaluate` and `BatchEngine`. Direct prompt-end SSM seed handling
remains enabled for correctness. Detached async SSM rederive is still not a
production path.

### 2026-05-06 Live Model Continuation

Follow-up rows were run after the DSV4 redownload completed enough to restore
the model shards. These rows are live model checks unless explicitly marked as
metadata/template only.

| Row | Result |
|---|---|
| Qwen3.6 35B JANGTQ `BENCH_QWEN_THINKING_CHECK=1` | PASS. 63 reasoning deltas, empty `.chunk`, no `<think>` marker leak. Peak footprint about 11.2 GB. |
| Qwen3.6 35B JANGTQ `BENCH_QWEN_MULTITURN_TOOL=1` | PASS. Three prompt/tool-style turns had zero reasoning-envelope marker leakage in `.chunk`. The selected budget stayed inside `.reasoning`, which is acceptable for the leak gate. |
| Gemma4 26B JANG_4M `BENCH_HARMONY_CHECK=1` | PASS. Coherent README-template visible text, no harmony markers in `.chunk`. The prompt did not elicit reasoning deltas. |
| Laguna XS.2 JANGTQ `BENCH_LAGUNA_LOOP=1` | PASS. Thinking off/on both produced coherent folder summaries with `loop=NO`; thinking-on split reasoning and visible text without marker leakage. |
| MiniMax M2.7 JANGTQ `BENCH_COHERENT=1` | PASS. Compile off/on both recalled blue and answered cool color correctly. Peak footprint about 61.2 GB. |
| MiniMax M2.7 JANGTQ_K `BENCH_COHERENT=1` | COHERENT / PERF ISSUE. Compile off/on both recalled blue. The first cold run had 163s TTFT under cold shader/cache conditions; rerun after warmup had about 4.2s first TTFT and warm turns around 350 ms. Peak footprint stayed about 80.0 GB, so memory/speed remain optimization work. |
| Qwen3.6 27B JANG_4M `BENCH_COHERENT=1` | PASS with explicit visible-answer policy. The generic 48-token row stayed entirely in reasoning, but `BENCH_THINK_LOOP_PROBE=1 THINK=0` produced visible content with zero reasoning chars, EOS stop, no loop, and no marker leak. |
| Qwen3.6 27B JANG_4M `BENCH_QWEN_THINKING_CHECK=1` | PASS for reasoning split. 95 reasoning deltas, empty `.chunk`, no `<think>` marker leak. Osaurus should set explicit `enable_thinking=false` for normal visible-answer mode on Qwen-family traffic. |
| Ling 2.6 flash JANGTQ2 `BENCH_COHERENT=1` | PASS. Compile off/on both recalled blue and cool color. Peak footprint about 30.4 GB. |
| Ling 2.6 flash MXFP4 `BENCH_COHERENT=1` | PASS. Compile off/on both recalled blue and cool color. Peak footprint about 66.8 GB, not the earlier ~110 GB failure mode. |
| Nemotron Omni Nano JANGTQ `BENCH_OMNI=1 BENCH_OMNI_BATCH=1` | PASS. 17/17 rows passed: text single/multi-turn, image, video encoder, audio encoder, video/audio LMInput, reasoning ON/OFF toggle, mixed image+audio, media-salt isolation, hybrid SSM warm-pass, BatchEngine B=1/B=2/image/audio. |
| Production bundle `BENCH_CONFIG_SMOKE=1` sweep | PASS for Qwen 35B, Qwen 27B, Gemma4, Laguna, MiniMax JANGTQ/JANGTQ_K, Ling JANGTQ2/MXFP4, Nemotron Omni JANGTQ, and DSV4 Flash. DSV4 reports `sidecar=true` after local sidecar rebuild. |
| Production bundle `BENCH_TEMPLATE_SMOKE=1` sweep | PASS for the same set. DSV4 always opens `<think>` even with `enable_thinking=false`; `reasoning_effort=max` changes the rendered prompt. Qwen/MiniMax/Nemotron close thinking for `thinking_false`. Ling renders the Bailing "detailed thinking off" system hint in all tested toggle rows. |
| DSV4 Flash `BENCH_DSV4_COHERENCE BENCH_DSV4_ROW=reasoning` | PASS. Reasoning-off and reasoning-on both answered `12`; reasoning-on routed thought text through `.reasoning`. Max-effort produced reasoning-only within the 128-token cap, which is stream-correct but means production should allocate enough token budget if visible final text is required after max reasoning. |

Runtime fix made after this matrix: `MiniMaxJANGTQConfiguration` now decodes
the real JANGTQ_K nested bit map directly from `mxtq_bits.routed_expert`, not
only from factory-normalized `mxtq_gate_up_bits` / `mxtq_down_bits` fields.
Focused unit coverage verifies uniform bits, gate/up/down projection bits,
`quantization` fallback, and explicit field precedence. Targeted smoke on the
local MiniMax M2.7 JANGTQ_K bundle passes with
`routedBits=gateUp:2,down:4`; template smoke also passes for thinking on/off,
multi-turn, and tool rows.

DSV4 model-file issue found during this continuation: the copied
`/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ` bundle had all 78 model
shards, `model.safetensors.index.json`, and tokenizer files, but it was missing
`jangtq_runtime.safetensors` while `jang_config.json` declared
`weight_format="mxtq"`. The safetensors index confirms this DSV4 bundle is a
prestacked JANGTQ layout (`format="jangtq"`,
`rebundled_layout="prestacked-switch_mlp"`) with `tq_packed` / `tq_norms` /
`tq_bits` tensors in the model shards. The small runtime sidecar still must
exist because the TurboQuant kernels need deterministic `signs.{dim}.42` and
`codebook.{dim}.2` tensors. Local sidecar rebuilt with keys:
`signs.4096.42`, `codebook.4096.2`, `signs.2048.42`, and `codebook.2048.2`,
then copied back to `/Volumes/eric-1/models/JANGQ/DeepSeek-V4-Flash-JANGTQ`.
Both copies have SHA-256
`f488d42982781d5653f5bbd6e6d6bd6d93416c9759e2dceaabde4a9817ad571c`.
Model publishing should include that sidecar in the DSV4 Flash bundle.

DSV4 long-context retry status: after the sidecar rebuild, DSV4 loaded as
`DeepseekV4JANGTQModel`. A 5,568-token stress prompt was killed after 285s at
about 118 GB peak footprint while another DSV4 validation process overlapped,
so that row is not a clean exclusive-host verdict. The documented 3K-ish row
then passed with `BENCH_DSV4_LONG_REPEAT=120`: prompt 3,068 tokens, 305s wall
time, about 108 GB peak footprint, visible answer
`CERULEAN RIVER / OSLO`, routed reasoning present, and no loop.

Latest speed/coherence sample from the local ignored `RunBench` perf harness
after adding output previews to the audit logs:

| Model | Target | BatchEngine observed | TokenIterator observed | Coherence status |
|---|---:|---:|---:|---|
| Qwen3.6 35B JANGTQ | 80 tok/s | 78.8 tok/s | 94.2 tok/s | Coherent visible text, no loop, no leaks. |
| Qwen3.6 27B JANG_4M | 25 tok/s | 25.5 tok/s | not rerun | Coherent visible text, no loop, no leaks. |
| Gemma4 26B JANG_4M | 80 tok/s | 79.1 tok/s | 94.4 tok/s | Coherent visible text, no loop, no leaks. |
| Laguna XS.2 JANGTQ | 80 tok/s | 29.2 tok/s | 31.3 tok/s | Coherent visible text, no loop, no leaks. |
| MiniMax M2.7 JANGTQ | 45-50 tok/s | 28.1 tok/s | 30.0 tok/s | Coherent visible text, no loop, no leaks. |
| Ling 2.6 flash JANGTQ2 | 80 tok/s | 30.7 tok/s | 30.3 tok/s | Coherent visible text, no loop, no leaks. |
| DSV4 Flash JANGTQ | 20 tok/s | 11.1 tok/s | 13.2 tok/s | Reasoning stream coherent; visible text empty on the perf prompt at short budget. |

Raw local logs are under `docs/benchmarks/speed-2026-05-06/` but that directory
is gitignored. Recreate them with `BENCH_PERF=1` if the PR needs fresh numbers.

Additional HQ continuation after this table:

- Qwen3.6 35B JANGTQ `BENCH_PERF` BatchEngine, 64 tokens:
  **80.4 tok/s**, coherent text, no loop, no leaks, and no factory fallback noise.
- MiniMax M2.7 JANGTQ TokenIterator, 64 tokens: baseline **34.7 tok/s**;
  `VMLINUX_MINIMAX_ROUTER_COMPILE=1` **34.6 tok/s**;
  `VMLX_TQ_SWITCH_GLU_COMPILE=0` **35.1 tok/s**. The current MiniMax gap is
  therefore not a simple router-compile or SwitchGLU-compile default issue.
- Qwen cache rows after a local ignored RunBench engine-shutdown cleanup:
  `BENCH_BATCH_CACHE_HIT`, `BENCH_BATCH_DISK_RESTORE`, and
  `BENCH_BATCH_TQ_B2` all PASS and exit 0. Before this cleanup the cache-hit row
  printed PASS but could return 139 during process teardown. `RunBench/` remains
  gitignored in this checkout, so treat that harness cleanup as local validation
  hygiene unless the benchmark target is intentionally promoted into source.

The requested Python file
`/Users/eric/mlx/vllm-mlx/docs/DSV4_FIX_NUANCES.md` was not present locally.
The equivalent local Python-runtime notes are in
`/Users/eric/mlx/vllm-mlx/docs/DSV4_RUNTIME_REGRESSION_TRACE.md` and
`/Users/eric/mlx/vllm-mlx/docs/DSV4-PYTHON-AUDIT-2026-05-03.md`; their relevant
requirements are reflected here: force DSV4 thinking mode, keep DSV4 prefill
single-shot, preserve SWA+CSA+HSA hybrid cache state, and never route DSV4
through paged KV ownership.

## Speed / Dtype Contract For New Runtime Work

The previous "speed stuck" cluster was real and should be treated as production
history, not old notes:

| Date/commit cluster | What it fixed | Current contract |
|---|---|---|
| 2026-04-11 to 2026-04-14, `2859808`, `06721aa`, `a8a6a6f`, `d0706af` | Float32 scalar/`AsType` cascades, universal bf16 conversion, and the JANGTQ-native bf16 bypass. | New cache/runtime code must not introduce untyped floating `MLXArray(...)` scalars in decode/prefill paths. Non-JANGTQ parameters convert to bf16 at load; JANGTQ-native keeps fp16 TurboQuant norms because the Metal kernels use norm dtype in their signature. |
| 2026-04-13, `cf55f6d`, `21176a4`, `d4e4e45`, `0e36d38` | Compile micro-fusion islands and Qwen/GatedDelta dtype cleanup. | Keep existing compiled helper paths and `asyncEval` decode ordering intact. Do not add `.item()` or synchronous `eval` inside compiled traces or per-token loops except at the sampler/EOS boundary. |
| 2026-04-15, `fb46fbd` | Fused int4 MoE gate/up gather and SwiGLU path. | Routed JANGTQ MoE should use `TurboQuantSwitchGLU` / JANGTQ kernels, including `gateUpBits` and `downBits` for JANGTQ_K. Do not re-expand per-expert tensors in forward paths. |
| 2026-05-02, `102d80c` | MiniMax router-gate fp32 precision parity with Python. | Router precision exceptions must be model-specific and documented. Do not blindly remove every `.asType(.float32)`; keep MiniMax router fp32, DSV4 mHC wide reductions fp32, MLA single-token fp32 SDPA, and Bailing recurrent GLA fp32 state math. |

Audit result for the new cache/JangPress stack on 2026-05-06:

- `CacheCoordinator`, paged cache, SSM companion, and disk L2 paths use `MLX.eval`
  for cache materialization/store/restore boundaries, not as hidden per-token
  decode work.
- `BatchEngine` keeps the April decode perf contract: B=1 bypass, conditional
  `Task.yield()`, and `asyncEval(logits)` / `asyncEval(sampledTokens)` before
  token readback.
- `TQDiskSerializer` readbacks are metadata/header reads for serialization, not
  model-forward dtype promotion.
- `LoadTimeStacking` materializes load-time per-expert JANGTQ stacks so large 3D
  routed tensors do not retain every per-expert source tensor until final eval.
  It must not be used in model forward paths.
- `ModelFactory` fallback attempts are quiet by default. Set
  `VMLINUX_MODEL_FACTORY_TRACE=1` only when diagnosing factory routing; thrown
  load errors still preserve the most informative factory failure.
- JangPress mmap/prestack is a residency policy. It must not cast weights, change
  JANGTQ norm dtypes, or replace `TurboQuantSwitchGLU`.
- JangPress router advice is default-off because it reads router indices back to
  CPU. It is exact and correct, but it is not yet tok/s-neutral; enable only for
  experiments until the readback path is replaced or proven neutral.

When adding new model/cache code, run the speed-contract grep before calling it
production-ready:

```bash
rg -n "MLXArray\\((0\\.0|1\\.0|[0-9]+\\.?[0-9]*|Float\\(|Double\\()" Libraries
rg -n "softmax\\([^\\n]*asType\\(\\.float32\\)|sigmoid\\([^\\n]*asType\\(\\.float32\\)" Libraries
rg -n "\\.item\\(|MLX\\.eval\\(|Memory\\.clearCache" Libraries/MLXLMCommon/BatchEngine Libraries/MLXLMCommon/Cache
```

Allowed hits must be explained by one of the contracts above. Otherwise treat
them as likely speed regressions until proven with a live model run.

## Non-Negotiable Integration Contract

1. Load through vmlx model loaders.

   Use `loadModel(from:using:loadConfiguration:)` or the existing
   `MLXLMCommon.loadModel(from:using:)` wrappers. Do not bypass the loader for
   JANG/JANGTQ bundles. The loader stamps:

   - `ModelConfiguration.toolCallFormat`
   - `ModelConfiguration.reasoningParserName`
   - effective EOS IDs and extra EOS strings
   - tokenizer fallback for weights-only JANG bundles
   - JANG/JANGTQ metadata, bits, and sidecar paths

2. Build chat requests with `UserInput(chat:)`.

   Production chat should flow:

   ```swift
   var input = UserInput(chat: messages)
   input.additionalContext = [
       "enable_thinking": enableThinking,
       "reasoning_effort": reasoningEffort
   ]
   let lmInput = try await context.processor.prepare(input: input)
   let stream = await batchEngine.generate(input: lmInput, parameters: params)
   ```

   Raw text prompts are acceptable for benchmark/perf probes, FIM/code
   completions, and deliberately template-free tests. They are not a substitute
   for production chat-template coverage.

3. Consume library stream events directly.

   `Generation.chunk(String)` is user-visible assistant text.
   `Generation.reasoning(String)` is a separate reasoning delta.
   `Generation.toolCall(ToolCall)` is authoritative tool-call output.
   `Generation.info(GenerateCompletionInfo)` is telemetry.

   Osaurus should not re-parse `<think>`, harmony channels, DSML, Qwen XML
   function calls, MiniMax tool syntax, or Gemma tool envelopes from `.chunk`.
   If raw markers appear in `.chunk`, treat that as a vmlx bug.

4. Keep request-level reasoning policy explicit.

   Always set `additionalContext["enable_thinking"]` for families with
   template-controlled thinking. When a model supports effort levels, set
   `additionalContext["reasoning_effort"]` to `"high"` or `"max"` only for
   requests that actually need it.

5. Keep cache ownership in vmlx.

   Osaurus should configure `CacheCoordinatorConfig`; vmlx should decide the
   per-model cache topology. Do not bolt on app-layer cache guards around
   DSV4/Ling/SSM. If a prefix hit is unsafe for a topology, BatchEngine must
   roll back or route to the correct tier.

## Reasoning And Tool Streaming

The stream split is library-level:

```swift
for await event in stream {
    switch event {
    case .chunk(let text):
        appendVisibleAssistantText(text)
    case .reasoning(let delta):
        appendReasoningDelta(delta)
    case .toolCall(let call):
        enqueueToolCall(call)
    case .info(let info):
        recordTelemetry(info)
    }
}
```

Important behavior:

- `.reasoning` can be non-empty even when the caller requested
  `enable_thinking=false` on families whose bundle/runtime force an open
  thinking tail for coherence. DSV4 currently does this.
- `.chunk` can be empty at `max_tokens` while `.reasoning` contains the right
  answer. This is a length/state policy issue, not marker leakage. The UI should
  surface a reasoning-only or length-finished state instead of treating it as no
  output.
- Thinking-on short prompts may end before the model closes `</think>`. Do not
  assume every thinking turn produces visible answer text within tiny budgets.
- Tool calls belong in `.toolCall`; do not scan visible chunks for raw tool
  syntax.

Relevant files:

- `Libraries/MLXLMCommon/ReasoningParser.swift`
- `Libraries/MLXLMCommon/Tool/ToolCallProcessor.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift`
- `Libraries/MLXLMCommon/BatchEngine/REASONING-STREAM-EVENT.md`
- `Libraries/MLXLMCommon/BatchEngine/STOP-SEQUENCES-CONTRACT.md`

## Chat Templates And Jinja

The tokenizer/processor path owns template rendering. Use `UserInput(chat:)`
plus `additionalContext`; do not manually splice role markers.

Family notes:

| Family | Template/runtime note |
|---|---|
| DSV4 | Uses DSV4 fallback/encoder semantics with `<｜User｜>`, `<｜Assistant｜>`, `<think>`, DSML tools, and `reasoning_effort`. |
| Qwen 3.x / 3.6 | `enable_thinking` affects prompt tail; parser uses decoded prompt tail via `ReasoningParser.forPrompt`. |
| Gemma 4 | Harmony parser is channel based. Use Gemma4 fallback templates when native Swift Jinja compatibility is not enough. |
| MiniMax M2.7 | JANG fallback detects MiniMax tokens by BOS/EOS, not by fragile `convertTokenToId` checks. |
| Laguna | Thinking-off/on is template-controlled; loop probe is the production smoke. |
| Nemotron-Omni | Text/image/video/audio rows go through Omni processor; reasoning parser strips raw markers from validation summaries. |

Swift-Jinja now resolves through the osaurus-owned chain:
`osaurus-ai/Jinja` carries the HuggingFace 2.3.5 code, and
`osaurus-ai/swift-transformers` is pinned to the osaurus fork that depends on
that package. Do not reintroduce a direct `huggingface/swift-jinja` edge.

## DSV4 Production Notes

DSV4 is its own architecture: SWA + CSA/HSA compressor/indexer hybrid attention,
not ordinary KV, and not Mamba/SSM.

Required runtime behavior:

- Default `newCache` uses:
  - `RotatingKVCache(window=128)` for `compress_ratio == 0` layers.
  - `DeepseekV4Cache(window=128, compressRatio=cr)` for compressed layers.
- `BatchEngine` marks DSV4 hybrid-pool caches as paged-incompatible so the
  paged tier does not claim unsafe prefix hits.
- Disk/L2 serialization uses `TQDiskSerializer` with `LayerKind.deepseekV4`,
  including rotating window state plus compressor/indexer buffers and nil masks.
- DSV4 prefill is single-shot for hybrid-pool caches. Do not chunk DSV4 prompt
  prefill at the app layer.
- `DSV4_KV_MODE=full|tq` is diagnostic/operational override only. Production
  should default to the hybrid cache unless the operator deliberately opts into
  the memory/quality tradeoff.

Reasoning modes:

- `enable_thinking=false`: should be treated as "plain answer requested", but
  current DSV4 bundles may still force the open thinking path for quality. Route
  reasoning deltas separately and allow enough max tokens for final answer text
  when product UX requires visible answer text.
- `enable_thinking=true`: normal reasoning stream.
- `reasoning_effort="max"`: template preface is applied and can consume more
  budget before visible answer.

Live DSV4 gate added:

```bash
BENCH_DSV4_COHERENCE=1 \
  BENCH_DSV4_ROW=chat|reasoning|long|all \
  BENCH_MODEL=/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

Observed on 2026-05-06:

| Row | Result |
|---|---|
| `chat` | PASS, clean exit. Multi-turn recalled `sapphire-42`; turn 3 answered sapphire/blue follow-up. Turn 2 answer stayed in `.reasoning` at 128 tokens. |
| `reasoning` | PASS, clean exit. Reasoning off/on/max routed without raw `<think>` in `.chunk`; arithmetic answer was present in routed stream. |
| `long` repeat=120 | PASS, clean exit. 3,068 prompt tokens, recalled buried `CERULEAN RIVER, OSLO`. |
| `long` repeat=220 | Semantically PASS at 5,568 prompt tokens, recalled `CERULEAN RIVER, OSLO`, but process exited abnormally after printing PASS, likely teardown/memory pressure. |
| `long` repeat=650 | OOM during long prefill/decode. Treat as a memory ceiling until DSV4 long-context memory is hardened. |

For production, use DSV4 long-context chat confidently past the 128-token local
window at the validated 3K-token scale. Treat 5K+ prompt tokens as a live
memory-pressure area on a 128 GB M5 Max until further hardening lands.

## Ling / Bailing Runtime Notes

The Ling RAM/stall issue had multiple causes; do not reduce it to "missing 3D
stacked tensors."

The original multi-turn coherence bug was the MLA cache path:

- Bailing MLA manually updates the MLA/KV cache.
- It must then call the no-update MLA SDPA helper.
- Calling the helper that also updates cache appends keys/values twice, grows
  cache offsets incorrectly, stalls, and breaks multi-turn recall.

The 2026-05-06 MXFP4 110 GB peak was a separate decode-fusion residency bug:

- Ling MXFP4 stores routed expert banks as large pre-quantized 3D
  `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` tensors.
- `SwitchGLU.ensureFusedGateUp()` previously concatenated gate+up into a
  persistent fused cache on first forward. For Ling MXFP4 this was about 1 GiB
  per routed layer and effectively kept a second expert bank resident.
- The fix keeps gate/up fusion for normal-sized MoE layers but skips persistent
  fusion when the fused cache would exceed 512 MiB
  (`VMLX_FUSED_GATE_UP_CACHE_LIMIT_MB` / `_BYTES` override available). Ling
  MXFP4 now uses the regular two-projection path for correctness and memory.
- The loader also now builds `QuantizedLinear` / `QuantizedSwitchLinear`
  modules directly from pre-quantized safetensor arrays, drops the staging
  weight dictionary before post-load materialization, chunk-converts bf16
  casts, and clears the allocator cache after load.

Current expected behavior:

- Do not reset `ArraysCache` on each ChatSession turn.
- Do not cap `ChatSession` prefill at the app layer for Ling.
- Recurrent GLA memory control belongs in the Bailing/Ling recurrent path, not
  in a generic chat-session guard.
- Load-time JANGTQ restacking materializes 3D routed expert tensors before
  source tensors are dropped.

Live validation already recorded:

- `Ling-2.6-flash-JANGTQ BENCH_SIMPLE BENCH_PROMPT_LEN=30 BENCH_MAX_TOKENS=64`:
  PASS, around 29.1 GB peak RSS.
- `Ling-2.6-flash-JANGTQ BENCH_COHERENT BENCH_MAX_TOKENS=64`: PASS, recalled
  `blue` and classified it as cool.
- `Ling-2.6-flash-JANGTQ2-CRACK BENCH_COHERENT BENCH_MAX_TOKENS=96`: PASS
  compile off/on, peak footprint ~30.4 GB, max RSS ~29.8 GB.
- `Ling-2.6-flash-MXFP4-CRACK BENCH_COHERENT BENCH_MAX_TOKENS=96`: PASS
  compile off/on, peak footprint ~66.8 GB, max RSS ~49.2 GB. Before the
  fusion-cache fix this same row peaked at ~110 GB footprint.
- Ling JANGTQ2 perf, `BENCH_PERF_PATH=batch`, 160 tokens: median ~23.7 tok/s.
- Ling JANGTQ2 perf, `BENCH_PERF_PATH=iter`, 160 tokens: median ~28.5 tok/s.
- Ling MXFP4 perf, 160 tokens: median ~6.0 tok/s after memory fix. This is the
  expected tradeoff until a memory-safe fused/streamed MXFP4 gate+up path exists.

Osaurus implication: let vmlx own Ling cache lifecycle. If Ling regresses, fix
`BailingHybrid` / cache topology, not an osaurus-side prompt-size guard.

## Cache Stack

Recommended production coordinator shape:

```swift
let config = CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    pagedBlockSize: 256,
    maxCacheBlocks: 1024,
    diskCacheMaxGB: 10,
    diskCacheDir: kvDir,
    ssmMaxEntries: 64,
    enableSSMReDerive: true,
    modelKey: modelKey,
    defaultKVMode: .none,
    defaultMaxKVSize: nil,
    longPromptMultiplier: 2.0
)
let coordinator = CacheCoordinator(config: config)
let engine = BatchEngine(context: context, maxBatchSize: 1, cacheCoordinator: coordinator)
```

Use `defaultKVMode: .turboQuant(keyBits:valueBits:)` only when the deployment
wants KV memory reduction and has validated the family. TurboQuant KV is a
correctness feature first; current live runs did not show decode-speed wins.

Tier behavior:

| Tier | Purpose | Notes |
|---|---|---|
| Paged L1 | Shared-prefix reuse for ordinary KV models | Exact block prefix hits. Unsafe partial hits roll back for VL/SSM. Disabled for DSV4 hybrid-pool caches. |
| Disk L2 | Session replay/restart and long-lived prefix cache | Uses TQDiskSerializer. DSV4 and SSM companion state serialize here. |
| SSM companion | Mamba/Arrays hidden-state sidecar | Stores prompt-boundary and block-boundary states so KV hits do not lose SSM recurrence state. Controlled by `enableSSMReDerive`; detached async rederive is not used. |
| TurboQuant KV | Optional compressed KV cache | Batch path supports `.turboQuant(keyBits:valueBits:)`; disk round-trip covered by stability and batch probes. |

Cache semantics:

- Exact same prompt across coordinators should hit disk.
- Prefix extension on dense KV can hit paged L1 and prefill only suffix.
- Prefix extension on hybrid SSM or VL may report a coordinator hit but roll
  back to full prefill for correctness.
- DSV4 paged prefix hits are considered incompatible; disk/L2 serializer owns
  DSV4 cache restore.
- Cache store must use prompt-boundary snapshots, not post-decode mutated cache
  state, for prompt-only keys.

Relevant files:

- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift`
- `Libraries/MLXLMCommon/Cache/TQDiskSerializer.swift`
- `Libraries/MLXLMCommon/Cache/SSMStateCache.swift`
- `Libraries/MLXLMCommon/Cache/SSMCompanionDiskStore.swift`
- `Libraries/MLXLMCommon/Cache/SSMReDerive.swift`

## TurboQuant / JANGTQ Notes

JANGTQ model weights and TurboQuant KV are separate concepts:

- JANGTQ weights: routed/expert codebook weight format used by model layers.
- TurboQuant KV: optional KV-cache quantization for runtime memory reduction.

Do not conflate the two in osaurus settings. A JANGTQ model can run with plain
KV, and a non-JANGTQ ordinary KV model can request TurboQuant KV if supported.

Current live correctness:

- Qwen batch cache hit: PASS.
- Qwen disk restore with SSM companion: PASS.
- Qwen `BENCH_BATCH_TQ_B2`: PASS, plain slot byte-identical with TQ neighbor.
- MiniMax-small JP regression: PASS, 3 coherent turns and TQ disk round-trip.
- DSV4 disk restore: PASS at short budget with DSV4 nil-mask fix.

Current speed note:

- TurboQuant KV did not reproduce a decode-speed win in the tested Qwen and
  MiniMax-small settings. It increased TTFT and was neutral/slower for tok/s.
  Treat speed as future tuning, not a correctness gate.
- Ling MXFP4 is memory-correct after disabling oversized persistent gate/up
  fusion, but decode speed is much lower than JANGTQ2. Production should prefer
  Ling JANGTQ2 for serving unless MXFP4 is specifically required.

## Family Minimum Coherence Matrix

| Family/model | Minimum current status |
|---|---|
| DSV4 Flash JANGTQ | PASS for production chat row, reasoning row, and 3K-token semantic long-context row. 5K+ memory pressure open. |
| Ling/Bailing JANGTQ/JANGTQ2/MXFP4 | PASS for multi-turn recall after MLA no-double-update fix and live ArraysCache reuse. |
| Qwen3.6 35B JANGTQ | PASS for thinking marker routing, tool-call multi-turn, paged prefix, disk restore, TQ B=2. Latest marker gate clean-exited with no raw `<think>` in `.chunk`. |
| Gemma4 26B JANG | PASS for harmony parser marker stripping and live perf/coherence smoke. Latest harmony gate clean-exited with no raw harmony markers in `.chunk`. |
| Laguna XS.2 JANGTQ | PASS for thinking-off/on loop probe, no marker leak, no loop. Thinking-on stayed in `.reasoning` at short budget. |
| MiniMax M2.7 JANGTQ | PASS for full-size ChatSession multi-turn coherence, compile off/on, clean exit. Speed remains open. MiniMax-small also passed JP regression and TQ disk round-trip. |
| Nemotron-Omni Nano JANGTQ | PASS for Omni matrix covering text, image, video, audio, media salt, hybrid SSM warm-pass, B=1/B=2. |

Mistral 3.5 was requested, but no local model directory existed under
`/Users/eric/models` during this pass.

## Toolchain Notes

Use Xcode's toolchain for Swift tests on this host:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --no-parallel
```

The Command Line Tools toolchain at `/Library/Developer/CommandLineTools` lacks
`xctest` here, so default `swift test` with CLT fails before it reaches package
logic. GPU-backed SwiftPM tests and live smokes also need MLX's default metallib
available beside the SwiftPM binaries. If it is missing:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -downloadComponent MetalToolchain

(cd .build/checkouts/mlx-swift && \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild build -scheme MLX -destination 'platform=macOS,arch=arm64')

cp /Users/eric/Library/Developer/Xcode/DerivedData/mlx-swift-*/Build/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib \
  .build/arm64-apple-macosx/debug/default.metallib
cp .build/arm64-apple-macosx/debug/default.metallib \
  .build/arm64-apple-macosx/debug/mlx.metallib
cp .build/arm64-apple-macosx/debug/default.metallib \
  .build/arm64-apple-macosx/release/default.metallib
cp .build/arm64-apple-macosx/debug/default.metallib \
  .build/arm64-apple-macosx/release/mlx.metallib
```

Run MLX/GPU Swift tests serially with `--no-parallel`. A parallel mixed
Swift-Testing/XCTest run hit a Swift test helper signal 11 after individual
target tests had passed; the serial reruns were clean.

## Commands Agents Should Reuse

Build:

```bash
swift build -c release --product RunBench
```

DSV4 production gates:

```bash
BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=chat \
  BENCH_MAX_TOKENS=128 \
  BENCH_MODEL=/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench

BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=reasoning \
  BENCH_MAX_TOKENS=192 \
  BENCH_MODEL=/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench

BENCH_DSV4_COHERENCE=1 BENCH_DSV4_ROW=long \
  BENCH_MAX_TOKENS=96 BENCH_DSV4_LONG_REPEAT=120 \
  BENCH_DSV4_LONG_MAX_TOKENS=80 \
  BENCH_MODEL=/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ \
  .build/release/RunBench
```

Ling:

```bash
BENCH_COHERENT=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=/Users/eric/models/JANGQ/Ling-2.6-flash-JANGTQ \
  .build/release/RunBench

BENCH_COHERENT=1 BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Ling-2.6-flash-JANGTQ2-CRACK \
  .build/release/RunBench

BENCH_COHERENT=1 BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Ling-2.6-flash-MXFP4-CRACK \
  .build/release/RunBench
```

MiniMax full-size:

```bash
BENCH_COHERENT=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=/Users/eric/models/JANGQ/MiniMax-M2.7-JANGTQ \
  .build/release/RunBench
```

Qwen reasoning marker gate:

```bash
BENCH_QWEN_THINKING_CHECK=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench
```

Gemma harmony gate:

```bash
BENCH_HARMONY_CHECK=1 BENCH_MAX_TOKENS=96 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Gemma-4-26B-A4B-it-JANG_4M-CRACK \
  .build/release/RunBench
```

Qwen cache stack:

```bash
BENCH_BATCH_CACHE_HIT=1 BENCH_MAX_TOKENS=8 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_BATCH_DISK_RESTORE=1 BENCH_MAX_TOKENS=8 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench

BENCH_BATCH_TQ_B2=1 BENCH_MAX_TOKENS=8 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK \
  .build/release/RunBench
```

Laguna:

```bash
BENCH_LAGUNA_LOOP=1 BENCH_MAX_TOKENS=64 \
  BENCH_MODEL=/Users/eric/models/JANGQ/Laguna-XS.2-JANGTQ \
  .build/release/RunBench
```

Omni:

```bash
BENCH_OMNI=1 BENCH_OMNI_BATCH=1 BENCH_MAX_TOKENS=24 \
  BENCH_MODEL=/Users/eric/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK \
  .build/release/RunBench
```

## Open Risks

- DSV4 at roughly 5K prompt tokens semantically recalled the buried fact but
  exited abnormally after PASS on this host. At 650 filler lines it OOMed. Long
  DSV4 memory hardening is still needed for production long-agent loops.
- DSV4 `enable_thinking=false` still often routes useful answer content through
  `.reasoning` before `.chunk`; osaurus must expose/handle reasoning-only
  length finishes.
- MiniMax M2.7 full-size speed is below the expected 45-50 tok/s band in the
  latest live run. This is a speed task, not a minimum coherence blocker.
- Ling MXFP4 speed is now low (~6 tok/s in BatchEngine perf) because the
  previous faster path kept a second fused gate/up expert bank resident and
  pushed footprint to ~110 GB. Reintroduce speed only with a memory-safe fused
  or streamed MXFP4 path.
- Distributed XCTest targets are opt-in with
  `VMLINUX_ENABLE_DISTRIBUTED_TESTS=1 swift test`; default `swift test` covers
  the active local runtime package surface.
- Factory fallback tracing is opt-in via `VMLINUX_MODEL_FACTORY_TRACE=1`.

## License

Repo-level license is MIT. The root `LICENSE` keeps the upstream
`2024 ml-explore` notice and now also includes `2026 Osaurus contributors` for
the local vmlx-swift-lm additions. Do not remove upstream copyright notices when
syncing or pushing.

## What Not To Do

- Do not add osaurus-side prefill guards for Ling.
- Do not reset ArraysCache per Ling chat turn.
- Do not parse reasoning/tool-call markup in osaurus from visible chunks.
- Do not use raw transcript prompts to certify DSV4 chat-template behavior.
- Do not enable DSV4 `DSV4_KV_MODE=full|tq` as a silent default.
- Do not treat JANGTQ weight format as the same setting as TurboQuant KV cache.
- Do not require JangPress for this phase. JangPress can be wired later as a
  memory/residency policy after runtime coherence is stable.
