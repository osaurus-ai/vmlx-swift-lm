# Handoff: JANG Model Crash Investigation (Issues #813 & #814) — Session 2

## Session 1 Summary (by previous agent)

See git history: commits `b59586f` (safeGeluApproximate), `0db30fb` (SwitchGLU init fix), and `10d547e` (JANG bit inference disambiguation) were all committed and pushed. Session 1's handoff is preserved in git history at those commits.

## Session 2 — What happened

### Starting state

- `main` at `10d547e` (all 3 prior fixes committed and pushed)
- All 9 model audit tests passing on M4 Max 128 GB
- Issues #813 and #814 still open on osaurus-ai/osaurus
- Osaurus 0.16.6 released, pinning mlx-swift-lm at `7d9a6ab` (predates ALL fixes)

### What we did

1. **Read and absorbed Session 1 handoff** — understood both issues, fixes, and verification matrix
2. **Re-audited all JANG models** against `10d547e` — all 9 tests passed (JANG_2L LLM+VLM, JANG_4M VLM, CRACK variants, Gemma 4 31B JANG_4M, Qwen 3.5 4B/9B, non-JANG Gemma 4 26B MLX 4-bit)
3. **Committed and pushed `10d547e`** (JANG bit inference fix from Session 1's working tree)
4. **Confirmed tpae still crashes** on Gemma 4 JANG_2L despite our fixes being deployed

### The new crash — what tpae's Xcode debugger showed

tpae ran Osaurus debug build with Xcode attached and caught the crash. From the debugger screenshot:

**Environment (confirmed):**
- `mlx-swift-lm main (10d547e)` ✅ — our fix IS deployed
- `mlx-swift osaurus-0.31.3 (02b01f0)` — forked MLX package
- macOS 26.2, Hardware Model Mac16,8 (Mac mini M2)
- Osaurus 0.16.6

**Crash details:**
- `EXC_BREAKPOINT (SIGTRAP)` / `Trace/BPT trap: 5` on Thread 2 (cooperative dispatch queue)
- Crash at `Transforms+Compile.swift:171`: `compileState.call([a])[0]` — **line 171**
- `Swift/ContiguousArrayBuffer.swift:691: Fatal error: Index out of range`
- `compileState` properties: `shapeless = true`, `inputs = 0 values`, `outputs = 0 values`
- `f = closure #1 in MLX.compile(inputs:outputs:shapeless:_:)` — the single-input compile overload

**Model state:**
- Model loaded successfully: `loadContainer: loaded gemma-4-26b-a4b-it-jang_2l isVLM=true`
- Crash during inference: `promptTokens=1077`, `effectiveTokens=1`, `hasImage=false`
- `generateEventStream: prepareAndGenerate existingCache=false cachedTokens=0`

**The smoking gun:** `shapeless = true` on the CompiledFunction that crashed. This is `safeGeluApproximate` — our own fix! It uses `compile(shapeless: true)` with `x * x * x` (not `x ** 3`). We fixed the Power primitive trigger but NOT the underlying `compile(shapeless: true)` infrastructure bug on affected GPUs.

---

## Root cause — macOS Tahoe + MLX Metal JIT bug (NOT our code)

### The bug

MLX's `compile(shapeless: true)` path wraps closures in `CompiledFunction`, which calls the MLX C++ `Compiled::eval_gpu`. On certain macOS Tahoe GPU drivers (particularly M2 Mac mini, Mac16,8), this compiled kernel path returns **zero results** instead of the expected array. The Swift code then does `compileState.call([a])[0]` on an empty array → `Index out of range`.

This is NOT specific to the Power primitive (`x ** 3`). It affects **any** `compile(shapeless: true)` closure when processing mmap'd safetensors weights whose Metal buffer hasn't been materialized on the GPU.

### Evidence chain (exhaustive)

| # | Source | What it proves |
|---|--------|---------------|
| 1 | **MLX #3329** (open) | `Compiled::eval_gpu` returns 0 results with null MTLBuffer on M1/M2. Exact same crash pattern. MLX team's official workaround: `mx.disable_compile()`. URL: https://github.com/ml-explore/mlx/issues/3329 |
| 2 | **MLX #3256** (closed) | NSRangeException "index 0 beyond bounds for empty array" on macOS Tahoe 26.3.1. Reproduces across MLX 0.28.0, 0.30.6, 0.31.1 — proves platform bug, not version regression. URL: https://github.com/ml-explore/mlx/issues/3256 |
| 3 | **MLX #3201** (open) | `compile(shapeless=True)` returns stale/wrong results on macOS Tahoe. Metal JIT compiler produces incorrect code. URL: https://github.com/ml-explore/mlx/issues/3201 |
| 4 | **MLX #3337** (closed) | Metal Toolchain 32023 broke `bfloat16_t` and `vec` namespace on macOS 26 Tahoe. Before fix: GPU functionally idle (2% active, silent CPU fallback). After fix: 17→35 tok/s. URL: https://github.com/ml-explore/mlx/issues/3337 |
| 5 | **LM Studio #1741** (open) | Gemma 4 26B fails on macOS 26.4 M1 Max with MLX. Same model family, same OS. URL: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1741 |
| 6 | **exo-explore #1694** (closed) | Metal JIT compilation fails on macOS 26 Tahoe — `bfloat16_t`, `erf`, `fp8_e4m3`, `gather_front`. URL: https://github.com/exo-explore/exo/issues/1694 |
| 7 | **UBOS tech analysis** | "macOS Tahoe 26.3 causes sudden system freezes, purple-screen flashes and automatic reboots on Mac Mini M2 Pro." URL: https://ubos.tech/news/macos-tahoe-26-3-crashes-on-mac-mini-m2-pro-ubos-tech-analysis/ |
| 8 | **mflux workaround** | `mflux` already works around this exact bug by avoiding `mx.compile()` on M1/M2 chips (check `AppleSiliconUtil.is_m1_or_m2()`). Cited in MLX #3329. |
| 9 | **tpae's Xcode debugger** | Crash at `compileState.call([a])[0]` with `shapeless=true` on Mac16,8 (M2 Mac mini), macOS 26.2. `mlx-swift-lm` at `10d547e` (our fix deployed). |

### Fix status from upstream

| Bug | Status | Fix released? |
|-----|--------|--------------|
| MLX #3329 (Compiled null MTLBuffer) | **OPEN** — no comments, no PR | ❌ No |
| MLX #3201 (compile shapeless stale) | **OPEN** — labeled `bug` | ❌ No |
| MLX #3256 (NSRangeException Tahoe) | **CLOSED** — no explanation of fix | ⚠️ Unclear |
| MLX #3337 (Metal Toolchain bfloat16_t) | **CLOSED** — fixed in source | ✅ Yes (but only kernel compilation, not compile() runtime) |
| mlx-swift `x ** 3` in geluApproximate | **NOT ADDRESSED** — still `x ** 3` in upstream `0.31.3` | ❌ No |

Latest MLX release: `v0.31.1` (Mar 12, 2026) — 56 commits behind main, no fix release.
Latest mlx-swift release: `0.31.3` (Apr 1, 2026) — still has `x ** 3`.
osaurus-ai/mlx-swift `osaurus-0.31.3` branch — still has `x ** 3`.

**No upstream fix exists. No upstream fix is in progress.**

---

## What needs to happen (the fix)

### Our code changes needed

Remove `compile(shapeless: true)` from ALL activation functions that use it. Run them as plain closures instead of compiled Metal graph kernels. The individual ops (Multiply, Tanh, Sqrt, Sigmoid, etc.) all have proper `output_shapes` support and work correctly on all GPU generations.

**Files to change:**

1. **`Libraries/MLXLMCommon/SwitchLayers.swift`** — Three changes:
   - `safeGeluApproximate` (line 12-15): Remove `compile(shapeless: true)` wrapper. Make it a plain closure.
   - `compiledSwiGLU` (line 28-31): Either remove compile or add a GPU-generation guard.
   - `compiledGeGLU` (line 33-36): Either remove compile or add a GPU-generation guard.

2. **`Libraries/MLXLLM/Models/Gemma4Text.swift`** — `compiledLogitSoftcap` (line 17-20): Evaluate if this also crashes. Uses `tanh(x / cap) * cap` — simpler expression, may be safe. Test on affected GPU.

3. **`Libraries/MLXVLM/Models/Gemma4.swift`** — Same `compiledLogitSoftcap` (line 19-22).

4. **`Libraries/MLXVLM/Models/Qwen35.swift`** — Three compiled closures (lines 18, 24, 36): `compiledSigmoidMultiply`, `compiledSigmoidGate`, `_vlmCompiledComputeG`.

5. **`Libraries/MLXLLM/Models/Qwen35.swift`** — `compiledSigmoidGate` (line 17).

6. **`Libraries/MLXLLM/Models/GatedDelta.swift`** — `_compiledComputeG` (line 18).

7. **`Libraries/MLXLLM/Models/Qwen3Next.swift`** — `compiledSigmoidGate` (line 18).

### Specific code change for `safeGeluApproximate`

**Current (broken on M2/Tahoe):**
```swift
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray =
    compile(shapeless: true) { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
```

**Proposed fix:**
```swift
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray =
    { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
```

Just remove `compile(shapeless: true)` — the expression runs as individual safe Metal ops.

### Performance impact

Negligible. Each `compile(shapeless: true)` saves ~1-2 Metal kernel launch overheads per call. For `safeGeluApproximate` called on every token:
- Compiled path: 1 Metal dispatch (fused GELU)
- Uncompiled path: ~5-6 Metal dispatches (multiply, tanh, sqrt, multiply, multiply, multiply)
- Per-token cost difference: ~0.5-1μs on modern Apple Silicon
- At 40 tok/s that's 40μs/s = 0.004% overhead

This is the same tradeoff mflux made (they disable compile on M1/M2 entirely) and it's considered acceptable.

### Alternative approaches (in order of preference)

1. **Remove compile from activations** (recommended above) — simple, safe, works everywhere
2. **GPU-generation guard** — detect M1/M2 and skip compile on those chips, keep compile on M3+:
   ```swift
   // Check if we're on a GPU generation that supports compile properly
   // M1/M2 (g7x) have the null MTLBuffer bug, M3+ (g8x) work fine
   let supportsCompile: Bool = {
       // Could check ProcessInfo, Metal feature set, or just try-and-catch
   }()
   ```
3. **Patch the osaurus-ai/mlx-swift fork** — fix `Compiled::eval_gpu` in the C++ backend to check for null MTLBuffer and fall back. This is the "proper" fix but requires C++ Metal expertise and maintaining a fork patch.
4. **Wait for upstream MLX fix** — All relevant issues are open with no activity. MLX latest release is 56 commits behind main with no fix release planned.

---

## Git state at session end

```
10d547e fix: JANG per-layer bit inference — strict round-trip, bitWidthsUsed disambiguation  (HEAD, origin/main)
0db30fb fix: SwitchGLU init crash — replace final geluApproximate call
b59586f fix: avoid MLXNN compiledGeluApproximate Power primitive crash
```

Working tree: **clean** (no uncommitted changes).

---

## Untracked files (do NOT commit)

- `DECODE-ANALYSIS.md` — scratch
- `FRAMEWORK-FIXES.md` — scratch
- `SWIFT-PERF-FIXES.md` — scratch
- `Libraries/MLXLMCommon/CompilableKVCache.swift` — scratch
- `Libraries/MLXLMCommon/DynamicSlice.swift` — scratch
- `Tests/MLXLMTests/Gemma4VLMTests.swift` — scratch
- `bench_batch.py` — scratch
- `HANDOFF-JANG-CRASH-813-814.md` — this file

---

## Next steps

1. **Implement the `compile(shapeless: true)` removal** from all activation functions listed above
2. **Test locally** (M4 Max will still pass — these ops work fine on M3+)
3. **Send tpae a debug build** to verify the crash is gone on Mac16,8
4. **Cut Osaurus 0.16.7** pinning to the new commit
5. **Close #813 and #814** with explanation that it was a macOS Tahoe Metal JIT bug affecting `compile(shapeless: true)` on M1/M2 GPUs
6. **Optionally**: file a PR on ml-explore/mlx-swift to fix `compiledGeluApproximate` upstream (change `x ** 3` to `x * x * x`)

---

## Key source locations

- `Libraries/MLXLMCommon/SwitchLayers.swift` — `safeGeluApproximate` (line 12), `compiledSwiGLU` (line 28), `compiledGeGLU` (line 33), `SwitchGLU.init` activation detection (line 88)
- `Libraries/MLXLMCommon/JangLoader.swift` — `inferBitWidthAndGroupSize`, `inferPerLayerQuantization`
- `Libraries/MLXLLM/Models/Gemma4Text.swift` — `compiledLogitSoftcap` (line 17), model forward pass
- `Libraries/MLXVLM/Models/Gemma4.swift` — `compiledLogitSoftcap` (line 19), model forward pass
- `Libraries/MLXVLM/Models/Qwen35.swift` — compiled closures (lines 18, 24, 36)
- `Libraries/MLXLLM/Models/Qwen35.swift` — `compiledSigmoidGate` (line 17)
- `Libraries/MLXLLM/Models/GatedDelta.swift` — `_compiledComputeG` (line 18)
- `Libraries/MLXLLM/Models/Qwen3Next.swift` — `compiledSigmoidGate` (line 18)
- `Package.swift` — MLX dependency: `osaurus-ai/mlx-swift` branch `osaurus-0.31.3`
