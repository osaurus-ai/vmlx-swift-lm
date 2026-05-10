# JANGTQ / MoE Runtime Top-K Override Plan — 2026-05-10

## Status

Implemented as an opt-in runtime helper and config-decode hook. Do not claim
this as a shipped default or a production recommendation until real model rows
pass per family.

The user asked to explore making compatible routed-MoE models use top-4 routing,
then prove multi-turn coherence on local models such as Qwen, MiniMax, Nemotron,
ZAYA, Ling, and Gemma. Treat this as a runtime quality/speed experiment first,
not a bundle conversion or metadata rewrite.

## Core Rule

Only lower a model's existing routed-expert top-k when it is greater than the
requested value.

Examples:

| Model class | Existing routing | Top-4 override behavior |
|---|---:|---|
| MiniMax M2.7 JANGTQ | top-8 | May lower to top-4 under explicit runtime flag |
| Hy3 / Hunyuan v3 | top-8 | May lower to top-4 under explicit runtime flag |
| DSV4 / Kimi-style MoE | often top-6 or top-8 depending bundle | May lower to top-4 if current top-k > 4 |
| Ling / Bailing | config-driven, commonly top-k > 4 in local bundles | May lower only if current top-k > 4 and recurrent-state tests pass |
| Nemotron-H / Omni | config-driven routed MoE | May lower only if current top-k > 4 and group-routing constraints still hold |
| Qwen3 / Qwen3.5 / Qwen3.6 MoE | config-driven routed MoE | May lower only if current top-k > 4 |
| ZAYA text / ZAYA1-VL | top-1 CCA/MoE routing | Do not change. Never raise to 4. |
| Gemma4 MoE | `top_k_experts` from config | Do not change when already <= 4. Lower only if > 4 and VLM rows pass. |

This avoids a fake "top-4" patch that makes top-1/top-2 architectures do more
work or changes their training-time routing semantics in the wrong direction.

## Runtime Surface

Preferred opt-in shape:

```sh
VMLX_MOE_TOPK_OVERRIDE=4
```

`VMLINUX_MOE_TOPK_OVERRIDE` is accepted only as a legacy typo alias. New
integrations should use `VMLX_MOE_TOPK_OVERRIDE`.

Rules:

- Apply after model/config decode and before layer/module construction when
  possible, so every layer sees a single consistent value.
- If a model has multiple routing controls, lower only the routed-expert token
  top-k. Do not alter sampler `top_k`, speculative-decoding top-k, group-count
  selectors, or unrelated tree-search parameters.
- Log a one-line diagnostic when the override fires:
  `MoE top-k override: <modelType> <field> <old> -> <new>`.
- If current top-k is missing, zero, or <= requested top-k, do nothing.
- If requested top-k is <= 0, ignore it.
- If requested top-k is greater than current top-k, do nothing.
- This must not mutate model files or Hugging Face bundles.
- Cache model keys include `|moeTopK=<K>` whenever a valid override is present,
  so L2/paged cache entries created under an override cannot be reused after a
  restart without that override.

## Candidate Swift Fields

The current repo exposes several config-level fields that are candidates for a
shared helper.

| Swift area | Field | Notes |
|---|---|---|
| `MiniMaxConfiguration` / `MiniMaxJANGTQConfiguration` | `numExpertsPerTok` | MiniMax M2.7 target; local Python evidence showed K=4 can improve short-prompt quality and speed. |
| `Qwen3MoEConfiguration` | `numExpertsPerToken` | Standard Qwen3 MoE. |
| `Qwen35TextConfiguration` / `Qwen35JANGTQTextConfiguration` | `numExpertsPerTok` | Qwen3.5/3.6 text and VL text_config families. |
| `BailingHybridConfiguration` | `numExpertsPerTok` | Ling/Bailing; must also prove recurrent-state/cache rows. |
| `NemotronHConfiguration` | `numExpertsPerTok` | Keep `topkGroup` unchanged unless separately proven. |
| `Hy3Configuration` | `numExpertsPerTok` | Recognition/parser only today; runtime implementation still pending. |
| `Gemma4Configuration` | `topKExperts` | VLM/text MoE; only lower if > 4. |
| `ZayaTextConfiguration` | `moeRouterTopk` | Current ZAYA top-1; do not override to 4. |

## Test Matrix Required Before Defaulting

For each compatible model family, collect rows with default top-k and top-4.
Each row must record model path, git SHA, command, env vars, prompt, decoded
full text, token/s, cache mode, and whether top-k actually changed.

Minimum prompts:

1. Short factual: "What is the capital of France?"
2. Short arithmetic: "What is 17 * 23?"
3. One-code answer: "Reverse a string in one line of Python."
4. Long paragraph: "Write one long paragraph describing ocean waves."
5. Multi-turn memory:
   - Turn 1: "Remember the key is blue-17."
   - Turn 2: "What key did I ask you to remember?"
   - Turn 3: same prefix with a different key to detect cache bleed.

Required cache rows:

- no-cache single stream
- cache coordinator on, fresh first turn
- live-session second turn
- L2 disk restore where topology supports it
- B=2 overlap where current BatchEngine supports it
- reasoning on/off overlap for reasoning-capable families
- explicit no-thinking row for Ling/Bailing
- media salt rows for VLM families after native VLM generation exists

Pass criteria:

- top-4 row is coherent, no replacement-character/junk bytes, no looping, no
  stop-token leak, no unwanted `<think>` leak into non-thinking output.
- multi-turn recall is correct and isolated.
- cache hit/miss behavior matches topology.
- speed is not worse by more than 5% unless quality improves and the family is
  explicitly marked quality-first.
- quality-sensitive evals remain open until MMLU/HumanEval or family-specific
  evals are run. Do not treat four prompts as full quality proof.

## Known Boundaries

- This is not a sampler `top_k` change.
- This is not a JANGTQ bundle change.
- This is not a TurboQuant KV-cache change.
- This is not valid for ZAYA top-1 routing.
- This is not DSV4 NSA `index_topk` / `DeepseekV4Compressor.indexTopk`
  compression selection. That module field is named `topK`, but it is not
  routed-expert token routing.
- This is not safe to default globally until per-family live rows pass.

## Implemented Swift Shape

1. `Libraries/MLXLMCommon/RuntimeMoETopKOverride.swift` owns parsing,
   lower-only decisions, diagnostics, and cache-key suffixing.
2. Compatible config decoders call the helper for MiniMax, MiniMax JANGTQ, Hy3,
   Bailing/Ling, Nemotron-H, Qwen3 MoE, Qwen3.5/3.6 text/JANGTQ, and Gemma4
   text/VLM `top_k_experts`.
3. ZAYA and ZAYA1-VL top-1 `moe_router_topk` are intentionally not wired.
4. `ModelContainer.enableCaching(config:)` and `enableCachingAsync()` scope
   `CacheCoordinatorConfig.modelKey` with the override value when valid.
5. `MiniMaxJANGTQ` router compile now reads canonical
   `VMLX_MINIMAX_ROUTER_COMPILE` plus legacy `VMLINUX_MINIMAX_ROUTER_COMPILE`.

## Remaining Implementation Shape

1. Add RunBench rows that print the effective routed top-k and full decoded
   output for default vs top-4.
2. Run no-cache/cache/B>1/reasoning/VLM rows listed above before defaulting.
3. Only after live rows pass, document recommended per-family flags for Osaurus.

No-model unit tests currently prove:
   - MiniMax top-8 lowers to top-4.
   - ZAYA top-1 stays top-1.
   - Gemma top-4 stays top-4.
   - Invalid env values do nothing.
   - Compatible config decoders are wired.
   - ZAYA/ZAYA1-VL are not wired.
   - Cache model keys are scoped by valid overrides.
   - MiniMax router compile reads canonical + legacy env names.
