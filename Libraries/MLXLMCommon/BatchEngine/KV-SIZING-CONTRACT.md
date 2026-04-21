# Coordinator-Owned KV Sizing Contract

**Landed:** 2026-04-21
**Scope:** BatchEngine admission path + CacheCoordinatorConfig
**Motivation:** ferebee's Osaurus 0.17.0 report (55K-token translation crash,
"Local KV cache size setting disappeared")

## Problem

Osaurus 0.17.0 removed its per-request `maxKVSize` UI knob with the code
comment: *"KV cache sizing is owned end-to-end by vmlx-swift-lm's
CacheCoordinator"*. But vmlx-swift-lm had no such logic — it simply
consumed `GenerateParameters.maxKVSize` and `.kvMode` as the caller
supplied them. With the UI gone, every request arrived as
`maxKVSize: nil, kvMode: .none`, meaning:

- `KVCacheSimple` allocated unbounded for every request.
- No quantization applied even when memory pressure was high.
- A 55K-token translation on a 35B MoE piled ~32 GB of float KV into
  GPU memory, plus activations, plus weights — reliable OOM on
  anything below a 96 GB Mac.

## Design

Two knobs on `CacheCoordinatorConfig` drive the policy:

| Field | Type | Applies when | Effect |
|---|---|---|---|
| `defaultKVMode` | `KVQuantizationMode` | request's `kvMode == .none` | sets `parameters.kvMode` for admitted slot |
| `defaultMaxKVSize` | `Int?` | request's `maxKVSize == nil` AND prompt length > `longPromptMultiplier × defaultMaxKVSize` | sets `parameters.maxKVSize` for admitted slot |

`longPromptMultiplier` defaults to `2.0` — short chat turns never pick
up a rotating-window cap they didn't ask for.

## Resolution rules

Codified in `CacheCoordinatorConfig.resolveKVPolicy(kvMode:maxKVSize:promptTokenCount:)`:

1. **Explicit caller values always win.** A request that sets
   `kvMode: .turboQuant(...)` or `maxKVSize: 4096` explicitly is not
   touched. The coordinator fills gaps only.
2. **kvMode gap → defaultKVMode.** When the request's `kvMode` is
   `.none` and the coordinator's `defaultKVMode` is not `.none`, the
   slot runs under the coordinator default.
3. **maxKVSize gap → defaultMaxKVSize only for long prompts.** The
   gate is strict: `promptTokenCount > longPromptMultiplier ×
   defaultMaxKVSize`. Prompts at or below the gate pass through with
   `maxKVSize: nil` (full-attention `KVCacheSimple`).

## Call site

`BatchEngine.admitPendingRequests`, before `BatchQuantize.wrapNewCacheIfNeeded`
and before `context.model.newCache(...)`:

```swift
if let coordinator = cacheCoordinator {
    let promptCount = request.input.text.tokens.size
    let (effMode, effMax) = coordinator.config.resolveKVPolicy(
        kvMode: request.parameters.kvMode,
        maxKVSize: request.parameters.maxKVSize,
        promptTokenCount: promptCount
    )
    request.parameters.kvMode = effMode
    request.parameters.maxKVSize = effMax
}
```

`request.parameters` was changed from `let` to `var` in `BatchPendingRequest`
to allow this gap-fill. Per-caller behavior is unchanged: the struct is
still a value, the caller still sees its own copy.

## Osaurus wiring

The contract gives Osaurus a clean way to configure global defaults
without touching every request site:

```swift
// In Osaurus's model-loading path:
let config = CacheCoordinatorConfig(
    usePagedCache: true,
    enableDiskCache: true,
    modelKey: modelID,
    defaultKVMode: .turboQuant(keyBits: 3, valueBits: 3),  // ~5× KV savings
    defaultMaxKVSize: 8192,                                // 8K sliding window
    longPromptMultiplier: 2.0                              // >16K → apply cap
)
let coordinator = CacheCoordinator(config: config)
let engine = BatchEngine(
    context: modelContext,
    maxBatchSize: 8,
    cacheCoordinator: coordinator
)
```

With those settings, ferebee's 55K-token translation runs with:
- `kvMode: .turboQuant(3, 3)` — ~5× smaller KV than float16
- `maxKVSize: 8192` — ring buffer caps absolute KV growth
- Effective KV memory: bounded regardless of prompt length

Callers that still set per-request `kvMode` / `maxKVSize` explicitly
continue to override these defaults.

## What this fix does NOT change

- `GenerateParameters.kvMode` and `.maxKVSize` semantics are unchanged.
  Existing callers see no behavior difference.
- `ToolCallProcessor`, `ReasoningParser`, Harmony (Gemma-4) parser,
  Qwen3.6 `startInReasoning` prefill handling — all untouched. 75
  reasoning/tool/stop-string tests remain green.
- Model families with hardcoded cache allocation (Gemma-4 SWA,
  Mistral-4 with explicit `maxKVSize`, MiMoV2Flash, BaichuanM1,
  Qwen3.5-VL, Mamba/hybrid-SSM) already build the right cache
  regardless of these knobs. The policy only affects families that
  default to `KVCacheSimple`.

## Verification

- `Tests/MLXLMTests/CacheCoordinatorKVPolicyTests.swift`: 10 unit tests
  covering every resolution path (explicit-wins, gap-fill, short/long
  prompt gate, ferebee scenario, custom multiplier).
- `BatchEngineIntegrationTests`: 7/7 pass with the admission-path
  change. `BatchEngineTurboQuantIntegrationTests`: 6/6 pass.
  `BatchEngineMultiTurnTests`: 6/6 pass. Harmony (Gemma-4) + Qwen3.6
  reasoning parser suites: 75/75 pass.

## Out of scope

Items that remain from the broader long-context triage:

1. **Wired-memory sticky state across reboot.** `mlx-swift`'s
   `WiredMemoryManager` calls `mlx_set_wired_limit` (kernel sysctl
   `iogpu.wired_limit_mb`), which persists across process restart
   until reboot. If the app is SIGKILLed mid-request before the
   ticket's `end()` unwinds, the kernel wired limit stays elevated.
   Fix belongs upstream in mlx-swift (signal handler) or in Osaurus
   (conservative ticket sizing + on-startup reset). vmlx-swift-lm
   only consumes tickets, never creates them.
2. **Gemma looping on some variants.** Parser test coverage is now
   75/75 green across A1-A8 + B1-B5 + harmony edge cases. If looping
   reproduces on a specific Gemma-4 prompt, that's a parser edge case
   we can pin with the exact byte sequence — but it is not caused by
   this fix. Tracked separately.
3. **DiskCache single-entry cap.** Added in the reverted commit (the
   only guard that was actually correct). Re-ship as a narrow
   belt-and-suspenders change if needed.
