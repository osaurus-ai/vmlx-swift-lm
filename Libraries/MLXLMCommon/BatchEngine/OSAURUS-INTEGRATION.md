# vmlx-swift-lm ↔ osaurus integration notes

**Link this file** from osaurus issues / Discord / PRs. Standalone, short, directly addresses the two referenced osaurus docs.

**Status** (2026-04-19): **production-ready** to flip `mlxBatchEngine=YES` as default. Verified 121 engine unit tests (0 failures), 25 bench scenarios across dense / hybrid-SSM / sliding-window / VL JANG / VL mlx-community model families.

---

## Addressing tpae's two references

### 1. `osaurus/docs/INFERENCE_RUNTIME.md`

| INFERENCE_RUNTIME.md concern | vmlx-swift-lm resolution |
|---|---|
| "`mlxBatchEngine` — default OFF until blockers close" | All `BatchEnginePlan.openBlockers` closed — see §2 below. Ready to flip default ON. |
| "`mlxBatchEngineMaxBatchSize` default 4" | Verified at B=4 AND B=8: slot 0 byte-identical to solo-B=1 reference on Qwen3-0.6B. |
| "`mlxAllowConcurrentStreams` — caution, MLX-vs-MLX unvalidated" | vmlx-side thread-safety covered: `DiskCache.store` now locks across `MLX.eval` + `save` + SQLite (iter 61 fix). Coordinator paged cache + SSM state + setHybrid flag all thread-safe under 128-way concurrent fuzz. MLX-vs-MLX at the Metal layer is still osaurus's operational call per its own doc. |
| "`cooperativeYield` — StreamAccumulator yield between tokens" | Osaurus-side only. vmlx emits tokens via `AsyncStream`; osaurus's `StreamAccumulator` chooses yield policy. |
| Freeze / Metal crash classes | Nine production bug classes fixed — listed in §3 below. |

### 2. `osaurus/Packages/OsaurusCore/Services/ModelRuntime/BatchEnginePlan.swift`

Both `Blocker` cases closed upstream:

```swift
public enum Blocker: String, CaseIterable {
    case kvQuantization    // CLOSED by vmlx iter 17 (Stage 0 TurboQuant)
    case compileSupport    // CLOSED by vmlx iters 9-22 (Stages 1A-5 compile path)
}
```

**Proposed osaurus change**: `public static var openBlockers: [Blocker] { [] }`.

**Why closed**:

- **kvQuantization** → `Libraries/MLXLMCommon/BatchEngine/BatchQuantize.swift` runs `wrapNewCacheIfNeeded` at admission + `maybeCompress` post-prefill. Supported `kvMode: .turboQuant(keyBits:, valueBits:)`. Legacy affine (`kvBits`, `kvMode: .affine`) deliberately not supported under batch — requires quantized-tuple attention sites out of Stage-0 scope; affine requests run with float KV and log a warning. Verified by `BENCH_BATCH_TQ_B2`: slot 0 plain with a concurrent TQ(4,4) neighbour is byte-identical to solo reference (no cross-slot contamination).
- **compileSupport** → `Libraries/MLXLMCommon/BatchEngine/BatchCompile.swift` classifies per-slot cache topology (`.simple` / `.turboQuant` / `.rotating` / `.cacheList` / `.mamba` / `.heterogeneous`); eligible families promote via `maybePromoteToCompiledDecode` at prefill end. Heterogeneous (Gemma-4 SWA, Qwen3.5-MoE mix) and `.mamba` fall through to uncompiled — correct-by-design. Verified by `CompilableKVCache|TurboQuant|Rotating|CacheList|Mamba` probe suites at 5e-7 abs diff, and `BENCH_BATCH_CHAT` compile ON ≡ compile OFF byte-identity on every tested model.

---

## What osaurus integrators need to know

### Load-time env-var shims (opt-in, zero impact when unset)

- **`VMLX_CHAT_TEMPLATE_OVERRIDE=/path/to/template.jinja`** — tokenizer bridge substitutes the shipped `chat_template.jinja` with this file's contents. Needed for **Gemma-4** because its native template trips a swift-jinja 1.3.0 interaction bug. Ship two compatible templates:
  - `Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja` — text + image/video/audio content parts
  - `Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja` — adds `tool_calls` + `tool_responses`
- **`VMLX_TOKENIZER_CLASS_OVERRIDE=Qwen2Tokenizer`** — auto-rewrites `tokenizer_class` at load. Default map includes `TokenizersBackend` → `Qwen2Tokenizer` (unblocks `mlx-community/Qwen3.5-VL-9B-8bit`).

### Auto-detected behaviour (no env var needed)

- **`CacheCoordinator.isHybrid` auto-flip**: when the first slot's cache contains a Mamba/SSM layer, `BatchEngine.admitPendingRequests` calls `coordinator.setHybrid(true)` automatically. Osaurus no longer needs to remember per-model.
- **JANG weights-only tokenizer fallback**: `JangLoader.resolveTokenizerDirectory` redirects to the cached source-model snapshot when a JANG bundle ships without tokenizer files (MiniMax JANGTQ et al.).
- **VL / hybrid SSM partial cache-hit rollback**: when a prefix-extend cache hit would split the vision-token region or interrupt the SSM recurrence, the engine rolls back to full prefill instead of producing corrupted output. Log line: `rolling back to full prefill (VL vision-token region can't be split)` or `(hybrid SSM recurrence path-dependent on full prefix)`.

### Multi-turn cache behaviour osaurus should expect

| Scenario | Outcome |
|---|---|
| Turn-2 tokens = Turn-1 tokens (session replay) | Full hit. Dense: 40-70% prefill speedup. VL: vision tower skipped. Hybrid SSM: SSM state restored via `ssmStateCache`. |
| Turn-2 = Turn-1 + new tokens, dense | Paged hit on shared prefix; remaining tokens prefill normally. 62% observed speedup on Qwen3-0.6B test harness. |
| Turn-2 = Turn-1 + new tokens, VL or hybrid SSM | Partial hit reported by coordinator, **engine rolls back to full prefill** for correctness. No speedup; no corruption. |

---

## Production bugs fixed this session (iters 28-64)

Each would crash or silently corrupt under osaurus production load.

| Bug class | Closing commit |
|---|---|
| `BatchEngine.generate()` hung across turns under real HF tokenizer (`while let detokenizer.next()` infinite loop) | `16b72d7` (iter 28) |
| `UserInput(prompt:, images:)` silently dropped images — `didSet` doesn't fire in init | `16b72d7` (iter 45) |
| VL partial cache-hit crashed MLX vision-feature merge (`SmallVector out of range`) | `16b72d7` (iter 48) |
| JANGTQ4 bundles crashed at first forward (`JANGTQ runtime sidecar not loaded` — bits=2 default on VL-wrapped configs) | `16b72d7` (iter 49) |
| Hybrid SSM partial cache-hit silently degraded output | `16b72d7` (iter 57) |
| Coordinator `isHybrid` had to be manually set per hybrid model | `16b72d7` (iter 57) |
| JANG weights-only bundles had no chat template | `bca0786` (iter 29) |
| `mlx-community/Qwen3.5-VL-9B-8bit` unsupported `TokenizersBackend` tokenizer class | `cc1ee54` (iter 59) |
| `DiskCache.store` not thread-safe across `MLX.eval` + `save` + SQLite (MTL command-buffer crash under concurrent writers) | `30e00a1` (iter 61) |

---

## How to verify before flipping the default

From this repo's root:

```bash
# Unit tests (~20s) — 121 expected, 4 skipped, 0 failed
./scripts/verify-engine.sh --tests-only

# Quick model sweep (~5 min) — skips 35B hybrid
./scripts/verify-engine.sh --quick

# Full sweep (~15 min) — 25 scenarios across 5 families
./scripts/verify-engine.sh

# 1-hour rotating soak (manual op) — flags any crash / hang / silent regression
./scripts/soak-engine.sh --duration 3600
```

Osaurus-side integration smoke (after flag flip):

1. `/v1/chat/completions` × 1 request — warm path.
2. `/v1/chat/completions` × 4 concurrent (same model) — stress `mlxBatchEngineMaxBatchSize`.
3. Close chat window mid-stream — stress `ModelLease.wait-for-release`.
4. Model swap under `strictSingleModel=true` mid-stream — stress lease eviction deferral.

All four should complete without Metal crashes — those are the crash classes osaurus's `ModelLease` + `MetalGate` layers exist to close; vmlx-swift-lm's `cancel()` path and per-actor serialization match osaurus's assumptions.

---

## Pointer to the detailed iter log

Architectural decisions, per-iteration addenda, and the full real-model verification log live at `Libraries/MLXLMCommon/BatchEngine/BATCH_ENGINE.md`. ~2100 lines, chronological. Read this file for a 10-minute overview; read `BATCH_ENGINE.md` when you need the why behind a specific design choice.
