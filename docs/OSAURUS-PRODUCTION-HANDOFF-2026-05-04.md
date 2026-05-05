# Osaurus Production Handoff — vmlx-swift-lm @ 2026-05-04

**Status:** main is production-ready for the osaurus integration as of
commit `2d0d63d`. SSH push verified.
**Audience:** osaurus host-app engineers picking up vmlx-swift-lm
`main` for the next osaurus release.
**This doc is the entry point.** The companion docs below cover
narrow surfaces; read this one first to know which to load next.

## Companion docs (read on demand)

| Doc | When to read |
|---|---|
| `OSAURUS-INTEGRATION-GUIDE.md` | Wiring `BatchEngine` behind the `mlxBatchEngine` flag — the historical iter-63 handoff. Still authoritative on flags + `BatchEnginePlan` blockers. |
| `OSAURUS-DSV4-INTEGRATION.md` | Loading + chatting with DSV4-Flash JANGTQ. Covers `DSV4_KV_MODE`, the always-thinking `<think>` parser, multi-turn cache semantics. |
| `OSAURUS-JANGPRESS.md` | JangPress cold-tier as a host-controllable cache. Details the `LoadConfiguration` knobs, mmap loader, cache directory layout. |
| `OSAURUS-JANGPRESS-PANEL-SPEC.md` | UI panel spec: status fields, knobs, telemetry surfaces. |
| `OSAURUS-CACHE-CONTRACT.md` | Per-layer cache type table, multi-turn rules, paged-cache compatibility. |
| `OSAURUS-SPEC-COMPLIANCE.md` | OpenAI-compat spec coverage matrix. |

## What changed since the last osaurus rev

### Safety: `LoadConfiguration.default` JangPress is now `.disabled` (commit `85c0893`)

**Action required for hosts that want JangPress on production bundles.**
The default constructor previously engaged `.auto(envFallback: true)`,
which silently turned on the routed-MoE cold-tier on any host whose
bundle was routed AND larger than 50% of physical RAM. JangPress is
production-validated for a narrow family list (DSV4-Flash, NemotronH,
Qwen3.5-35B-A3B) but not the long tail; keeping it default-on was
unsafe for a general host integrator.

The 70% memory caps and the mmap-backed safetensors loader (the
spike-survival parts) **remain on by default**. Only the cold-tier
itself is opt-in.

```swift
// Default — safe for any host, any bundle. JangPress OFF.
let cfg = LoadConfiguration.default

// Opt-in to auto JangPress when the bundle qualifies. Use this when
// you've validated JangPress for the bundle family you're loading.
let cfg = LoadConfiguration.experimentalJangPressAuto

// Force JangPress on with a specific cold fraction.
let cfg = LoadConfiguration(jangPress: .enabled(coldFraction: 0.70))

// Strict pre-iter-23 byte-compat. No JangPress, no caps, no mmap.
let cfg = LoadConfiguration.off
```

The legacy `JANGPRESS=N` / `JANGPRESS=off` env override **only** binds
when the policy is `.auto(envFallback: true)`. With the new
`.disabled` default, env vars are silently ignored unless the host
opts into `experimentalJangPressAuto` or constructs `.auto` directly.
This is intentional — it eliminates a class of "why is JangPress on
in CI" surprises.

### Chat-template: MiniMax-M2 thinking-off honored (commit `2d0d63d`)

Two co-located fixes:

1. **MiniMax-M2 native template ignores `enable_thinking`.** The
   bundle-shipped `chat_template.jinja` unconditionally prefills
   `<think>\n` at the assistant tail. In thinking-off chat workloads
   every output token then routes to `Generation.reasoning`, the
   user-visible chunk stream stays empty, and loop detection trips
   because `</think>` never arrives. We now ship a corrected
   `MiniMaxM2Minimal.jinja` that gates the trailing prefill on
   `enable_thinking` and emits a closed empty `<think>\n</think>\n\n`
   block when thinking-off. The macro tokenizer adaptor auto-engages
   the corrected template when `additionalContext.enable_thinking ==
   false` AND the tokenizer carries the MiniMax-specific `]~b]` /
   `[e~[` tokens. Auto-engage is one-way: thinking-on requests fall
   through to the native template untouched.

2. **`additionalContext` was being silently dropped on all 3 macro
   fallback paths.** Previously the DSV4 fallback, the
   missing-template fallback, and the custom-override fallback all
   used the 2-arg `applyChatTemplate(messages:chatTemplate:)`
   signature. That dropped `tools` and `additionalContext` —
   `enable_thinking`, the tool list, and any other JinjaContext keys
   never reached the renderer. All three paths now route through the
   7-arg signature. Hosts that pass `tools=[...]` or
   `additionalContext=["enable_thinking": false]` now have their
   intent honored regardless of which template path fires.

Set `VMLX_CHAT_TEMPLATE_FALLBACK_LOG=1` to log auto-engage events for
ops visibility.

### DSV4-Flash: paged-tier guard, JANGTQ key alias, hybrid round-trip

These landed earlier in the same branch. See `OSAURUS-DSV4-INTEGRATION.md`
for the full story. Quick recap:

- **`8db8ee2`** — `CacheCoordinator` flips `isPagedIncompatible = true`
  on first DSV4 slot admission. The paged tier would otherwise
  silently report a token-id-hash hit on empty blocks for DSV4
  prompts, suppressing the disk-tier lookup that would actually hit.
  Hosts do NOT need to set `usePagedCache = false` — the runtime
  guard activates on its own.
- **`2534e88`** — JANGTQ overlay loader recognizes both
  `tq_packed`/`tq_norms` and the un-prefixed `packed`/`norms`
  aliases. config.json decoder fix.
- **`9089e8f`** — pure SWA + CSA + HSA correctness pass. Hybrid
  cache survives disk round-trip + prefix-cache reuse.

## Recommended `LoadConfiguration` for osaurus

```swift
import MLXLMCommon

// Canonical production defaults. Use this for every load unless the
// caller has already inspected the bundle and made a different call.
let loadConfig = LoadConfiguration.default

let context = try await ModelFactory.shared.loadContainer(
    configuration: ModelConfiguration(directory: bundleURL),
    progressHandler: nil,
    loadConfiguration: loadConfig)
```

For the validated JangPress bundle family (DSV4-Flash, NemotronH,
Qwen3.5-35B-A3B):

```swift
let loadConfig = LoadConfiguration.experimentalJangPressAuto
```

The auto-detection in `experimentalJangPressAuto` only enables
JangPress when **both** conditions hold:

- `config.json` declares one of `num_local_experts`, `num_experts`,
  `moe_intermediate_size`, `n_routed_experts` (or any of these nested
  under `text_config` / `language_config`) > 1, **and**
- total `*.safetensors` byte size in the bundle > 50% of physical RAM.

A host can override per call via env: `JANGPRESS=70` to force it on at
70% cold fraction, `JANGPRESS=off` to force it off. The env override
**only** binds against `.auto(envFallback: true)` policies.

## Cache mode selection cheat sheet

| Workload | Cache class | LoadConfiguration | Per-call params |
|---|---|---|---|
| Standard chat | model default (per-layer) | `.default` | `GenerateParameters(maxTokens: 512, temperature: 0)` |
| Long reasoning, RAM tight | `TurboQuantKVCache` swap-in | `.default` | `kvMode: .turboQuant(keyBits: 8, valueBits: 8)` and bump `maxTokens` |
| Validated DSV4-Flash bundle | `DeepseekV4Cache` (built-in) | `.experimentalJangPressAuto` | env `DSV4_KV_MODE` (see DSV4 doc) |
| Strict pre-iter-23 parity | `KVCacheSimple` only | `.off` | none |

For full multi-turn rules see `OSAURUS-CACHE-CONTRACT.md`.

## Verified state (M5 Max, 2026-05-04)

| Surface | Status |
|---|---|
| `swift build -c release --target MLXLMCommon` | clean (35.6s) |
| `swift build -c release --target MLXHuggingFaceMacros` | clean (123s) |
| `swift build -c release --target MLXLLM` | clean (18.5s) |
| `swift test --filter LoadConfigurationTests` | 25/25 (one transient setenv-race flake on rerun, pre-existing — not a regression) |
| SSH push to `osaurus-ai/vmlx-swift-lm` main | verified `Hi jjang-ai!` + `1bc6e6b..2d0d63d  main -> main` |

For the broader bundle sweep see `JANGPRESS-VALIDATION-MATRIX-2026-05-04.md`.

## Known issues NOT covered by this rev

These are tracked but were not part of the 2026-05-04 push.

### GPT-OSS Harmony marker leak

`reasoningStampFromModelType("gpt_oss")` currently returns `none`
instead of a Harmony parser variant. The Harmony control markers
(`<|start|>`, `<|end|>`, channel tags) leak into the user-visible
chunk stream for `gpt_oss` model_types. Hosts can work around by
registering a Harmony parser stamp at the call site or stripping the
markers post-stream.

### MiniMax-Small thinking-on loop at 1024 token budget

Distinct from the thinking-off fix in `2d0d63d`. With the corrected
template engaged AND `enable_thinking: true`, MiniMax-Small still
loops past 1024 tokens. Suspected vmlx-side; not yet root-caused.
Workaround: bump `maxTokens` to 2048+ and rely on the native
`<|reasoning_end|>` marker.

### Distributed inference (TP) — blocked on upstream MLX ring

Phase 5 TP scaffolding (sharding plan, Linear-subclass refactor,
`TPRankWorker`, 2-host launcher) all landed in `ba65524`. Vertical
slice is wired end to end; runtime hangs on `mlx_distributed_all_sum`
inside MLX's lazy-graph materialization. See
`docs/DISTRIBUTED-TP-HANDOFF-2026-05-03.md` for the full diagnostic
state and four candidate next-step paths (jaccl loopback, mpi
backend, hand-rolled all_sum, switch to PP-over-NIO).

### INTELLECT-3.1 bundle (#10 from M4 sweep)

Never tested in the 2026-05-04 sweep. Not blocking; flag if it lands
in production traffic.

## Migration checklist for the next osaurus release

1. **Pin to `osaurus-ai/vmlx-swift-lm` main @ `2d0d63d` or later.**
2. **Stop relying on env-only JangPress activation.** If you want
   JangPress in production, switch the call site from
   `LoadConfiguration.default` to `LoadConfiguration.experimentalJangPressAuto`
   (or pass an explicit `.enabled(coldFraction:)`). The env override
   alone no longer activates JangPress against `.default`.
3. **Audit calls into `applyChatTemplate` / chat session.** Anywhere
   you pass `additionalContext: ["enable_thinking": false]`, the
   request now actually reaches the renderer in fallback paths.
   Behaviour for thinking-on requests is unchanged.
4. **Optional**: enable `VMLX_CHAT_TEMPLATE_FALLBACK_LOG=1` in
   staging for one release cycle to surface any unexpected
   auto-engagements.
5. **No breaking API changes.** All public surfaces from the iter-63
   handoff remain valid. New surfaces are purely additive
   (`experimentalJangPressAuto`, `MiniMaxM2Minimal`).

## Contact / next-step coordination

- Open issues against `osaurus-ai/vmlx-swift-lm`.
- For Phase 5 distributed inference unblock, see
  `docs/DISTRIBUTED-TP-HANDOFF-2026-05-03.md` § "What to try next".
- For new JangPress family validation requests, follow the matrix in
  `docs/JANGPRESS-VALIDATION-MATRIX-2026-05-04.md`.
