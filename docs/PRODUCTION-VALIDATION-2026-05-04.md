# Production Validation — vmlx-swift-lm @ 2026-05-04

**Status:** comprehensive M5 Max validation pass after the
`f5acbeb` build-fix push and the `52eb4d4` GPT-OSS Harmony fix.
Audit covered Python-runtime parity, per-family cache-tier
correctness, chat templates / reasoning, and hybrid-SSM warm-pass.

## Real-bundle smokes — VERIFIED on M5 Max

| Bundle | Family | Attention | Result |
|---|---|---|---|
| Laguna-XS.2-JANGTQ (9.4 GB) | Laguna | dense | ✅ 3-turn coherent (compile ON), recalls "blue" |
| MiniMax-SLURPY-JANGTQ | MiniMax-M2 | dense | ✅ MiniMaxM2Minimal auto-engage, thinking probe PASS, TQ disk round-trip PASS |
| Qwen3.6-35B-A3B-JANGTQ-CRACK (11 GB) | Qwen3.6 MoE | dense MoE | ✅ Both compile passes 3/3 turns coherent |
| Qwen3.6-35B-A3B + JangPress force-on | Qwen3.6 MoE | dense MoE | ✅ JangPress mmap tier engaged (10240 tiles, 7.5 GB routed), TQ disk round-trip PASS |
| Nemotron-Omni-Nano-JANGTQ-CRACK (12 GB) | NemotronH | hybrid SSM (Mamba+attn) | ✅ Both compile passes 3/3 turns coherent, multi-turn cache reuse working |
| Gemma-4-26B-A4B-it-JANG_4M-CRACK (15 GB) | Gemma4 | SWA | ✅ Both compile passes 3/3 turns coherent, harmony parser handles output cleanly |

## Real-bundle smokes — DEFERRED (RAM safety)

| Bundle | Reason |
|---|---|
| DSV4-Flash-JANGTQ_K (137 GB) | Loads OK on M5 (24-44s), but produces no decode events with `BENCH_COHERENT=1`. Silent-empty-output bug — see "Open issues" below. RAM cost too high to keep iterating without a smaller repro. |
| Kimi-K2.6-Small-JANGTQ (143 GB) | Too large for safe iteration this session. |

## Audit findings (parallel-agent surveys)

Four research agents surveyed the runtime against the Python
reference (`~/vmlx`, `~/jang`) and the per-family contracts. Sources
cited inline.

### Real production gaps

| # | Gap | Severity | Scope |
|---|---|---|---|
| 1 | **GPT-OSS Harmony marker leak** | High | All `gpt_oss*` model_types | **FIXED in `52eb4d4`** |
| 2 | **`bailing_hybrid` (Ling-2.6-flash, BailingMoeV2_5)** factory entry missing | High | Ling family | Open — needs new model port (MLA + linear attention + MTP, NOT a config-only fix) |
| 3 | **`CacheList` marked `.skip` on disk serialize** (`TQDiskSerializer.swift:243-245`) | High | FalconH1, BaichuanM1 (any model with `CacheList(MambaCache, KVCacheSimple)` per layer) | Open — multi-turn disk-cache reuse falls back to full re-prefill |
| 4 | **DSV4-Flash silent-empty-output** under `BENCH_COHERENT` chat mode | High | DSV4-Flash | Open — investigation in progress; template matches Python encoder, runtime path unclear |
| 5 | **Mistral 4 chat-mode force-flip absent** (per `~/vmlx/docs/AUDIT-RELEASE-READINESS.md:111`) | Medium | Mistral 4 | Open |
| 6 | **MiniMax-Small thinking-on loop @ 1024 tokens** | Medium | MiniMax-Small specifically (not -M2) | Open — suspected vmlx-side, not yet root-caused |

### Lower-priority observations

| # | Observation | Note |
|---|---|---|
| 7 | DSV4 mask construction (`_dsv4_window_visibility`) not cited in code by audit agent | Implementation IS in `DeepseekV4.swift`; agent reported it as "not cited in contract" — clarify in docs |
| 8 | `QuantizedKVCache` is a runtime-swap (via `maybeQuantizeKVCache`), no direct alloc path; multi-turn restore semantics unclear | Out of any current bundle's path; documented in agent report |
| 9 | MLA non-standard `head_dim=512` / `num_kv_heads=1` broadcast: untested in cache paths | DSV4 works in single-turn (paged-tier guard fires correctly); multi-turn still suspect |
| 10 | Hybrid-SSM `unsafeFullHit` guard rolls back to full prefill on full disk hit (correct trade-off; `SSMStateCache` is the proper warm fast-path) | Working as designed |
| 11 | No end-to-end disk-restore tests for hybrid SSM models | Test gap |
| 12 | Reverted `SSMReDeriver` code is gone — only comments remain referencing the failed attempt | No zombie code |
| 13 | Python `cache_type` is whole-prompt (`"kv" / "mamba" / "hybrid"`) while Swift uses per-layer `LayerKind`; Swift schema is **stricter** and added `deepseekV4=7` for compressor/indexer pool that Python L2 drops | Swift is ahead |

## Cache-tier matrix — current state

| Model family | Eager cache | L1 paged | L2 disk | Prefix reuse | TurboQuant | Notes |
|---|---|---|---|---|---|---|
| Llama / Mistral 3 / Qwen / Gemma 2 (dense) | `KVCacheSimple` | ✅ | ✅ | ✅ | ✅ | Standard path |
| Gemma4 (SWA) | `RotatingKVCache` | ✅ | ✅ (`LayerKind.rotating=6`) | ✅ | ✅ | All paths working |
| Mistral 4 (long-ctx) | `RotatingKVCache` (+QKV swap-in) | ✅ | ✅ (`.rotating` / `.qkv=5`) | ✅ | partial | Chat-mode force-flip missing |
| Qwen3.6 / Qwen3.5 MoE | `KVCacheSimple` | ✅ | ✅ | ✅ | ✅ | Verified end-to-end |
| NemotronH / Jamba / GraniteMoeHybrid (SSM hybrid) | `MambaCache` + `KVCacheSimple` | ✅ | ✅ (`.mamba=3` + `.kv=2` per layer) | ✅ via `SSMStateCache` inline seed | n/a (SSM state can't be quantized) | `unsafeFullHit` guard rolls back to full prefill on full disk hit; `SSMStateCache` is the warm fast-path |
| Qwen3Next | `KVCacheSimple` | ✅ | ✅ | ✅ | ✅ | No SSM despite name |
| BaichuanM1 | `CacheList(RotatingKVCache, MambaCache)` | ✅ | ❌ `.skip` (real bug) | partial | n/a | Multi-turn disk hit re-prefills |
| FalconH1 | `CacheList(MambaCache, KVCacheSimple)` | ✅ | ❌ `.skip` (real bug) | partial | n/a | Multi-turn disk hit re-prefills |
| DSV4-Flash (deepseek_v4) | `RotatingKVCache(128)` (cr=0) + `DeepseekV4Cache` (cr>0) | ❌ paged-incompatible (guard at first slot, commit `8db8ee2`) | ✅ (`.deepseekV4=7` covers rotating + pool + buffer) | ✅ disk only | ✅ TQ swap-in via env `DSV4_KV_MODE=tq` | Decode silent-empty-output observed under chat-mode `BENCH_COHERENT` — needs investigation |
| MiMoV2 Flash | `RotatingKVCache` | ✅ | ✅ | ✅ | ✅ | Working |
| MiniMax-M2 (M2.7 SLURPY) | `KVCacheSimple` | ✅ | ✅ | ✅ | ✅ | Auto-engage MiniMaxM2Minimal template confirmed working |
| GPT-OSS | `KVCacheSimple` | ✅ | ✅ | ✅ | ✅ | Reasoning stamp now `harmony` (`52eb4d4`) |
| Ling-2.6-flash (BailingMoeV2_5) | — | — | — | — | — | **Model class missing** |

## Reasoning-parser stamps — verified mapping

`reasoningStampFromModelType` in `Libraries/MLXLMCommon/ReasoningParser.swift`:

| Family prefix | Stamp | Notes |
|---|---|---|
| `gemma4*` | `harmony` | |
| `gpt_oss*` | `harmony` | NEW in `52eb4d4` |
| `qwen3*` | `think_xml` | qwen3, qwen3_5, qwen3_6, qwen3_moe, qwen3_next |
| `deepseek*` | `think_xml` | deepseek_v3, deepseek_v4, deepseek_r1 |
| `glm4_moe*`, `glm5*` | `think_xml` | |
| `minimax*` | `think_xml` | minimax, minimax_m2, minimax_m3 |
| `kimi*` | `think_xml` | kimi_k2, kimi_k25, kimi_k26 |
| `nemotron_h*` | `think_xml` | |
| `holo*`, `laguna*` | `think_xml` | |
| anything else | `none` | LFM2, LLaMA, Phi, StarCoder2, Cohere, OpenELM, InternLM2, Mistral 3/4, Gemma 2/3/3n, etc. |

JANG bundles override via `jang_config.json:capabilities.reasoningParser`.

## Chat-template fallback chain — verified

`Libraries/MLXHuggingFaceMacros/HuggingFaceIntegrationMacros.swift`:

1. **Custom override** (`VMLX_CHAT_TEMPLATE_OVERRIDE`) — passes `tools` + `additionalContext` (fixed in `2d0d63d`)
2. **MiniMax-M2 thinking-off auto-engage** — fires when `enable_thinking==false` AND tokenizer carries `]~b]` / `[e~[`. Falls back to `MiniMaxM2Minimal.jinja`. Verified firing on real bundle.
3. **DSV4-Flash missing-template fallback** — emits `DSV4Minimal.jinja`. Template matches Python encoder (`encoding_dsv4.py`) byte-for-byte. Verified rendering path engages on real bundle.
4. **Native tokenizer template** — fires when bundle ships its own `chat_template`.
5. **Family-specific minimals** (Gemma4, Nemotron, etc.) — all 7-arg signatures.

## Build state — VERIFIED clean

| Command | Result |
|---|---|
| `swift build -c release` (full package, all targets, fresh `rm -rf .build`) | ✅ 1119 build steps, 193s |
| `swift test --filter "ReasoningStampFromModelTypeTests"` | ✅ 14/14 pass (incl. new GPT-OSS test) |
| `swift test --filter "LoadConfigurationTests\|ShardingPlanTests"` | ✅ 28/28 pass |

## Open follow-ups (for the next session)

1. **DSV4-Flash silent-empty-output under chat mode** — instrument `streamDetails` to confirm whether the model is emitting 0 tokens (EOS at logit 0) or whether the generator path is short-circuiting. Reproduce with the bundled `encoding/test_encoding_dsv4.py` to determine if Python encoder also produces empty output for the same prompt structure.

2. **Ling-2.6-flash (`bailing_hybrid` / BailingMoeV2_5)** — port the model class. Architecture: MLA (`kv_lora_rank=512`, `q_lora_rank=1536`, `qk_head_dim=192`), partial RoPE (`partial_rotary_factor=0.5`), linear attention components (`num_kv_heads_for_linear_attn=32`, `linear_silu=false`), MoE with sigmoid routing (`score_function: sigmoid`, `topk_method: noaux_tc`), MTP head (`num_nextn_predict_layers=1`).

3. **CacheList disk-serialize** — extend `TQDiskSerializer.swift:243` to introspect sub-caches and serialize each with the right `LayerKind`. Add `.cacheList` LayerKind variant or use sub-layer indexing (`{i}.{0}`, `{i}.{1}`). Add E2E disk-restore round-trip test using FalconH1 fixture.

4. **Mistral 4 chat-mode force-flip** — port from Python (`~/vmlx/docs/AUDIT-RELEASE-READINESS.md:111`).

5. **MiniMax-Small thinking-on loop at 1024** — root-cause investigation. May be sampling/RoPE issue or tokenizer mismatch.

6. **Hybrid-SSM E2E disk-restore tests** — add a NemotronH or GraniteMoeHybrid fixture test that exercises trim+re-feed-last-token recipe and `SSMStateCache` warm path through a multi-turn disk hit.

## Pushed commits in this validation pass

```
52eb4d4 fix(reasoning): GPT-OSS gets harmony stamp (was none)
f5acbeb docs(osaurus): record build-fix details + new known-flake list in handoff
f454cf8 fix(build): unblock package build — revert mlx-swift pin, loosen crypto, vendor C ABI
78c91aa docs(osaurus): top-level production handoff for 2026-05-04 push
2d0d63d fix(chat): MiniMax-M2 enable_thinking honored + macro additionalContext plumbed
85c0893 fix(loadconfig): default JangPress to .disabled (opt-in)
```
