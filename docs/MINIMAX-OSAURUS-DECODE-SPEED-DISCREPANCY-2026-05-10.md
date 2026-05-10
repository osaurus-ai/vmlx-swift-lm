# MiniMax Osaurus Decode Speed Discrepancy

Date: 2026-05-10
Scope: `vmlx-swift-lm` at `a5a0e37` and Osaurus PR #1057 at `3f8e3cce`.

This is a proof note for the MiniMax M2.7 JANGTQ report where Osaurus shows
about 30 tok/s while some local Swift CLI logs say 46-49 tok/s.

## Short Verdict

The original 30 tok/s Osaurus row was real and should not be dismissed as
Debug-mode noise. The root cause was two-part:

1. The source pinned by Osaurus did not contain the documented
   `BatchEngine.generate` single-slot fast path, so app traffic went through
   the actor-managed `submit(...)` loop even when `maxBatchSize == 1`.
2. The current source also lacked the JANGTQ kernel/meta optimizations that
   earlier notes claimed were already in tree: the Hadamard multiblock kernel
   still used `float newv[64]`, and JANGTQ launch metadata was rebuilt as a new
   `MLXArray` on every dispatch.

Both issues are fixed in this working tree:

- `BatchEngine.generate(...)` now uses a cache-safe TokenIterator-backed
  single-slot fast path when the engine is idle and `maxBatchSize == 1`.
- `JANGTQKernels.swift` now uses `newv[8]` and cached meta tensors for the
  Hadamard, fused gate/up/SwiGLU, and gather launchers.

Fresh Release `RunBench` proof after those fixes:

| Bundle | Path | Coordinator | Result |
|---|---|---:|---:|
| `MiniMax-M2.7-Small-JANGTQ` | TokenIterator | no | 47.4 tok/s |
| `MiniMax-M2.7-Small-JANGTQ` | `BatchEngine.generate` | no | 46.8 tok/s |
| `MiniMax-M2.7-JANGTQ` | `BatchEngine.generate` | no | 46.6 tok/s |
| `MiniMax-M2.7-JANGTQ` | `BatchEngine.generate` | yes | 46.4 tok/s |

All four rows were coherent, stopped by `max_tokens`, and had no loop or marker
leak. The first measured run after load still tends to land around 40-41 tok/s;
the second warmed run is the sustained row.

`MiniMax-M2.7-JANGTQ_K` / CRACK are still awaiting the same re-run in a clear
memory window because active Hy3 uploads left about 60 GB free while those
bundles are 74 GB on disk.

## Proven Evidence

### Fast local rows

Logs under `build/minimax-speed-sweep/20260509T1505-python-jang-reference/`:

| Bundle | Log | Result |
|---|---|---|
| `MiniMax-M2.7-JANGTQ` | `swift-minimax-jangtq-solo-fastpath.log` | 47.9 median / 48.0 best tok/s |
| `MiniMax-M2.7-JANGTQ_K` | `swift-minimax-jangtq-k-solo-fastpath.log` | 46.8 / 46.9 tok/s |
| `MiniMax-M2.7-Small-JANGTQ` | `swift-minimax-small-jangtq-solo-fastpath.log` | 48.7 / 49.1 tok/s |
| `MiniMax-M2.7-JANGTQ_K-CRACK` | `swift-crack-k-batch-newv8-solo-fastpath-stopfix2.log` | 46.4 / 46.6 tok/s |

All of these rows reported coherent output, no loop, and no marker leak.

### Slow local rows that match Osaurus better

| Log | Result |
|---|---|
| `/tmp/vmlx_minimax_m27_jangtqk_speed_dynamic_cache_warm_20260509.log` | 29.8 tok/s |
| `/tmp/vmlx_minimax_k_all_three_rerun_20260509.log` | 27.8 tok/s |
| `build/minimax-speed-sweep/20260509T214602Z-crack-check/logs/runbench-crack-k-batch.log` | 30.5 tok/s |
| `build/minimax-speed-sweep/20260509T1505-python-jang-reference/swift-crack-k-batch-newv8-baseline-final.log` | 40.5 tok/s |
| `build/minimax-speed-sweep/20260509T1505-python-jang-reference/swift-crack-k-batch-newv8-final.log` | 41.1 tok/s |

These are the important rows for app debugging because they prove that the
Swift CLI can also reproduce the slow band when it is not on the faster
single-sequence decode loop.

### Current source does not contain the claimed solo fast path

Historical root cause at `a5a0e37`: `BatchEngine.generate(...)` in
`Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift` calls:

```swift
let (requestId, tokenStream) = submit(input: input, parameters: parameters)
```

Searches for the documented fast-path symbols are empty in source and git
history:

```sh
rg "soloFastPath|soloFastPathInFlight|finishSoloFastPath" Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift
git log --all -S "soloFastPathInFlight" -- Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift
```

No committed implementation was present at `a5a0e37`.

The current working tree adds that implementation with:

- `soloFastPathTask` / `soloFastPathID` lifecycle state;
- queued-work exclusion while the single-slot stream owns the model;
- shutdown cancellation;
- `CacheCoordinator` KV-policy propagation;
- a mapping from `enableCompiledBatchDecode` to TokenIterator's
  `enableCompiledDecode` only inside this path.

Regression coverage:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
  --filter "testGenerateSoloFastPathCompletesWithoutQueuedWork|testSubmitQueuesBehindActiveSoloFastPath|testGenerateUsesSoloFastPathWhenEngineIsIdleAndBOne"
```

Result: all three tests pass.

### Claimed JANGTQ speed fixes were missing from source

Earlier local docs said `newv[8]` and `cachedJANGTQMeta` were already landed.
That was stale. The source still had:

```metal
float newv[64];
```

and no `JANGTQMetaCacheKey` / `cachedJANGTQMeta(...)` helper.

The current working tree restores the real kernel/meta changes:

- `float newv[64]` -> `float newv[8]` in the Hadamard multiblock kernel;
- cached meta tensors for Hadamard, fused gate/up/SwiGLU, `gatherTQ`, and
  `gatherTQTopK`;
- launcher-specific cache keys so meta arrays do not alias across kernels.

Regression coverage:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
  --filter "JANGTQKernelsTests"
```

Result: 13 tests pass, including Hadamard shape, norm, rank-three, and Python
reference parity.

### Osaurus always installs a cache coordinator

Osaurus calls `installCacheCoordinator(on:)` after model load, and its
`MLXBatchAdapter.Registry.engine(...)` constructs:

```swift
await container.makeBatchEngine(maxBatchSize: maxBatchSize)
```

`ModelContainer.makeBatchEngine(...)` captures `self.cacheCoordinator` and
passes it into `BatchEngine(...)`. This is correct for cache features, but it
means Osaurus is not exercising a "no coordinator" path.

### Osaurus compile flag is not the full explanation

Osaurus currently sets:

```swift
enableCompiledBatchDecode: maxBatchSize == 1
```

That is the right flag for the committed actor-managed BatchEngine path.
There is a separate TokenIterator flag (`enableCompiledDecode`), but current
`BatchEngine.generate` does not route into TokenIterator, so simply setting
that flag in Osaurus would not fix this discrepancy by itself.

## Resolved Root Cause

The app was slow because the production Osaurus path was:

```text
ModelRuntime -> MLXBatchAdapter -> ModelContainer.makeBatchEngine
  -> BatchEngine.generate -> submit -> actor-managed prefill/decode
```

The fast proof rows were generated by a different path:

```text
single-sequence TokenIterator-style decode loop
```

The committed source and Osaurus integration did not line up with the docs that
said `BatchEngine.generate` would choose the fast TokenIterator-backed path
under `maxBatchSize == 1`. After restoring that path and restoring the missing
JANGTQ kernel/meta changes, the production-style path reaches the same speed
band.

## Secondary Finding: Reasoning-Only Output Skews UI/API Accounting

The controlled Osaurus streaming probe for MiniMax with thinking enabled
returned a long `reasoning_content` stream and zero content tokens in usage.
That explains confusing "completion_tokens=0" style API output for
thinking-only generations, but it does not explain the low engine tok/s by
itself.

The UI screenshot's 30.8 tok/s is consistent with the actor-managed decode
loop speed band.

## What Not To Do

- Do not call the 48 tok/s row general CLI truth; it is path-specific.
- Do not claim Debug mode is the explanation until a Release-vs-Debug A/B is
  run on the same path.
- Do not blindly set `enableCompiledDecode` in Osaurus as a speed fix. Current
  `BatchEngine.generate` does not use TokenIterator, so that flag is not
  sufficient.
- Do not disable the cache coordinator globally. It owns reasoning/media
  cache identity, L2 disk restore, path-dependent SSM/Zaya CCA handling, and
  Osaurus integration contracts.

## New Proof Logs

All logs are under `build/minimax-osaurus-discrepancy-20260510/`.

| Log | Result |
|---|---|
| `small-iter-newv8-meta.log` | `MiniMax-M2.7-Small-JANGTQ`, TokenIterator, 47.4 tok/s |
| `small-batch-newv8-meta.log` | `MiniMax-M2.7-Small-JANGTQ`, `BatchEngine.generate`, 46.8 tok/s |
| `full-batch-newv8-meta.log` | `MiniMax-M2.7-JANGTQ`, `BatchEngine.generate`, 46.6 tok/s |
| `full-batch-newv8-meta-prodcoord.log` | `MiniMax-M2.7-JANGTQ`, `BatchEngine.generate` + production coordinator, 46.4 tok/s |

## Remaining Proof Steps

1. Commit and pin this vmlx-swift-lm change in Osaurus, then re-run the app
   Release path from the UI/API.
2. Re-run the 74 GB `MiniMax-M2.7-JANGTQ_K` and CRACK bundles when memory is
   clear.
3. Add an optional diagnostic path label to `BatchEngine.generate` / `finishSlot`
   so future UI speed reports can distinguish `tokenIteratorFastPath` from
   actor-managed B>1 decode.
4. Keep B>1 MiniMax rows gated separately. The single-slot fast path does not
   replace the actor-managed continuous batching path when multiple requests
   overlap.

## Fix Boundary

This is not an app-layer monkeypatch. It is an engine-level single-slot fast
path plus the missing JANGTQ kernel/meta optimization. The fix preserves:

- no concurrent model mutation;
- cancellation and shutdown semantics;
- cache coordinator media/reasoning salt;
- prompt-boundary cache storage through the coordinator path;
- hybrid SSM/Zaya CCA disk/SSM state handling;
- no bypass of B>1 continuous batching when more than one request is active.
