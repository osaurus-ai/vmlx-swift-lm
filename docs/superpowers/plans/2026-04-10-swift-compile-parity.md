# Swift compile() Parity with Python — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 3x decode speed gap between Swift mlx-swift-lm (44 tok/s) and Python mlx_lm (123 tok/s) on Qwen3.5-35B-A3B-4bit by making compile() actually eliminate FFI bridge overhead.

**Architecture:** The Swift-to-C FFI bridge costs ~1-2us per MLX op. Qwen3.5 has ~1000+ ops per decode step, so bridge overhead alone is ~1-2ms/step. Python pybind11 is ~3x faster. The ONLY way to close this gap is to make compile() truly fuse the compute graph into a single C-level dispatch, eliminating per-op bridge crossings. This requires fixing the compile infrastructure in mlx-swift itself (Transforms+Compile.swift), fixing cache types to be compile-compatible, and wiring compile into the generation loop.

**Tech Stack:** Swift 6, MLX Swift (osaurus-ai fork of ml-explore/mlx-swift 0.31.3), Metal, C interop via Cmlx

---

## Critical Context for All Agents

### The Performance Gap (Measured 2026-04-10, M4 Max 128GB)

| Implementation | Qwen3.5-35B tok/s | How |
|---|---|---|
| Python mlx_lm 0.31.1 | **123.6** | Uncompiled, pybind11 bridge |
| Inferencer.app (MLX 0.24.2 fork) | **107.4** | Proprietary, unknown optimizations |
| Swift mlx-swift-lm (uncompiled) | **44** | C bridge overhead per op |
| Swift mlx-swift-lm (with compile()) | **47** | compile() barely helps (1.1x) |

### Why compile() Doesn't Help Right Now

The Swift compile() wrapper in Transforms+Compile.swift (line 39-119) has massive per-call overhead:

1. **Two locks** per call: `lock.withLock` (line 40) + `evalLock.lock()` (line 89)
2. **Multiple allocations** per call: `flatMap { $0.innerState() }` creates new arrays each time (line 46, 73, 110)
3. **Closure creation** per call: `new_mlx_closure(inner)` allocates a C closure every call (line 84)
4. **Vector conversion** per call: `new_mlx_vector_array(innerInputs)` (line 98)

The C++ side (mlx_detail_compile + mlx_closure_apply) IS the fused graph replay. But the Swift wrapper around it does so much work that the savings are eaten. Net result: 1.1x instead of expected 2-3x.

### What Python Does Differently

Python pybind11 bridge is inherently lighter (~3x less overhead per op). Python mlx_lm does NOT compile the model forward for generation -- it just calls `model(tokens, cache=cache)` directly. The 123 tok/s is achieved through a faster bridge alone.

Target for Swift: make compile() work efficiently so per-step cost drops from 24ms (uncompiled) to ~8-10ms (matching Python 123 tok/s throughput).

### Past Attempts (What Failed and Why)

1. **CompilableKVCache + compile wrapping model forward (2026-04-09/10):** Works for dense models (Llama +26%, IDENTICAL output). Fails for Qwen3.5 because MambaCache subscript replaces array references, breaking compile tracing.

2. **MambaCache _updateInternal in subscript (2026-04-10):** Added `existing._updateInternal(newValue)` to ArraysCache subscript setter. Compile then activates on Qwen3.5 but gives only 1.1x speedup because the compile wrapper overhead is too high. Output was IDENTICAL on short generation (12 tokens).

3. **Compiled full step including argmax (2026-04-10):** Wrapped model + logit slicing + argmax in one compile closure. Still only 1.1x -- proves the bottleneck is the compile WRAPPER overhead, not the scope of what's compiled.

### Key Files

| File | What It Does | Needs Changes |
|---|---|---|
| `.build/checkouts/mlx-swift/Source/MLX/Transforms+Compile.swift` | Swift compile() wrapper -- THE bottleneck | **YES -- critical** |
| `.build/checkouts/mlx-swift/Source/MLX/Protocols.swift` | Updatable protocol, innerState() | Maybe |
| `Libraries/MLXLMCommon/Evaluate.swift` | TokenIterator, generation loop, setupCompiledDecode | YES |
| `Libraries/MLXLMCommon/KVCache.swift` | Cache types -- ArraysCache subscript needs _updateInternal | YES |
| `Libraries/MLXLMCommon/CompilableKVCache.swift` | Overflow bin cache for compile-safe KV | Already exists |
| `Libraries/MLXLMCommon/DynamicSlice.swift` | C API for compile-safe slice ops | Already exists |
| `Libraries/MLXLMCommon/HardwareInfo.swift` | M3+ GPU gate for compile safety | Already exists |
| `Libraries/MLXLMCommon/RoPEApplication.swift` | MLXArray offset path for compile | Already exists |
| `TestRunner/Sources/main.swift` | Benchmark harness with --raw-bench, --compiled | YES |

### Reference: How compile() Works at the C Level

```
// What compile() does at the C level (simplified):
// 1. First call: trace function -> capture op graph -> compile to Metal
// 2. Subsequent calls: skip tracing, replay compiled graph
// The compiled graph dispatches Metal kernels directly -- no per-op FFI crossing
mlx_detail_compile(&compiled, innerClosure, id, shapeless, [], 0)
mlx_closure_apply(&resultVector, compiled, innerInputsVector)
```

### Reference: Current compile() Swift Wrapper (THE BOTTLENECK)

```swift
// Transforms+Compile.swift lines 39-119
func call(_ arguments: [MLXArray]) -> [MLXArray] {
    lock.withLock {                                          // LOCK 1
        innerCall(arguments)
    }
}

func innerCall(_ arguments: [MLXArray]) -> [MLXArray] {
    let stateInputs = inputs.flatMap { $0.innerState() }    // ALLOC per call
    let argumentsCount = arguments.count

    func inner(tracers: [MLXArray]) -> [MLXArray] {
        let savedStateInputs = stateInputs.map { $0.copyContext() }  // ALLOC per call
        for (s, tracer) in zip(stateInputs, tracers[argumentsCount...]) {
            s._updateInternal(tracer)
        }
        let result = f(tracerArguments)
        let stateOutputTracers = outputs.flatMap { $0.innerState() }.map { $0.copyContext() } // ALLOC
        for (s, saved) in zip(stateInputs, savedStateInputs) {
            s._updateInternal(saved)
        }
        return result + stateOutputTracers
    }

    let innerClosure = new_mlx_closure(inner(tracers:))      // C ALLOC per call
    defer { mlx_closure_free(innerClosure) }

    evalLock.lock()                                           // LOCK 2
    var compiled = mlx_closure_new()
    mlx_detail_compile(&compiled, innerClosure, id, shapeless, [], 0)
    defer { mlx_closure_free(compiled); evalLock.unlock() }

    let innerInputs = arguments + stateInputs                 // ALLOC per call
    let innerInputsVector = new_mlx_vector_array(innerInputs) // C ALLOC per call
    defer { mlx_vector_array_free(innerInputsVector) }

    var resultVector = mlx_vector_array_new()
    mlx_closure_apply(&resultVector, compiled, innerInputsVector) // THE ACTUAL WORK
    defer { mlx_vector_array_free(resultVector) }

    let resultsPlusStateOutput = mlx_vector_array_values(resultVector)
    let stateOutput = outputs.flatMap { $0.innerState() }     // ALLOC per call

    for (s, newValues) in zip(stateOutput, resultsPlusStateOutput.suffix(stateOutput.count)) {
        s._updateInternal(newValues)
    }
    return Array(resultsPlusStateOutput.prefix(resultLength))
}
```

### Benchmark Commands

```bash
# Python reference (123 tok/s target):
python3 -m mlx_lm.generate --model ~/models/Qwen3.5-35B-A3B-4bit --prompt "Hello" --max-tokens 200

# Swift raw bench (current 44 tok/s):
cd TestRunner && swift build -c release && .build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench

# Swift with TokenIterator (--compiled flag):
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit "Hello" --compiled
```

---

## Task 1: Optimize the compile() Wrapper in Transforms+Compile.swift

**This is the highest-impact change.** The current wrapper adds ~5-10ms of overhead per call through locks, allocations, and C API bookkeeping. The C++ compiled graph replay itself is fast -- the wrapper is the bottleneck.

**Files:**
- Modify: `.build/checkouts/mlx-swift/Source/MLX/Transforms+Compile.swift:39-119`

**What to change:**

The `CompiledFunction.innerCall` method (line 45-119) does this per call:
- `inputs.flatMap { $0.innerState() }` -- allocates new array every call
- `stateInputs.map { $0.copyContext() }` -- allocates N copies
- `zip + _updateInternal` loop -- N iterations
- `new_mlx_closure(inner)` -- C heap allocation
- `new_mlx_vector_array(innerInputs)` -- C vector allocation
- `mlx_vector_array_values(resultVector)` -- Swift array from C
- `outputs.flatMap { $0.innerState() }` -- second allocation
- `zip + _updateInternal` loop -- another N iterations

Optimizations:

- [ ] **Step 1: Cache stateInputs/stateOutputs arrays across calls**

The `inputs.flatMap { $0.innerState() }` and `outputs.flatMap { $0.innerState() }` return the SAME array references on every call (the whole point of _updateInternal is that references don't change). Cache them at construction time instead of recomputing per call.

```swift
// In CompiledFunction, add cached fields:
private let cachedStateInputs: [MLXArray]
private let cachedStateOutputs: [MLXArray]

// Set in init (after inputs/outputs are stored):
cachedStateInputs = inputs.flatMap { $0.innerState() }
cachedStateOutputs = outputs.flatMap { $0.innerState() }
```

Then replace `inputs.flatMap { $0.innerState() }` on line 46 with `cachedStateInputs`, and `outputs.flatMap { $0.innerState() }` on line 110 with `cachedStateOutputs`.

- [ ] **Step 2: Pre-allocate the C closure and reuse it**

`new_mlx_closure(inner)` on line 84 creates a new C heap object every call. The `inner` function captures `stateInputs` which are now cached. Create the closure ONCE at construction and reuse.

- [ ] **Step 3: Pre-allocate the input vector and reuse it**

`new_mlx_vector_array(innerInputs)` on line 98 allocates a C vector every call. Pre-allocate at construction time and update array contents in-place per call.

- [ ] **Step 4: Remove the instance lock or replace with lighter synchronization**

Line 40: `lock.withLock { innerCall(arguments) }` -- this is an NSLock. For single-threaded decode (which is the hot path), this lock is unnecessary. Replace with `os_unfair_lock` or remove entirely for the decode case.

- [ ] **Step 5: Remove evalLock from compile path**

Line 89: `evalLock.lock()` was already removed from regular `MLX.eval`/`asyncEval` (commit 5f4f0ae) because the C++ scheduler has its own mutex. Same argument applies to compile -- `mlx_detail_compile` is already thread-safe. Remove the redundant Swift lock.

- [ ] **Step 6: Benchmark after each change**

```bash
cd TestRunner && swift build -c release
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench
# Target: compiled avg should drop from 22ms to <15ms
```

- [ ] **Step 7: Commit**

```bash
git -C .build/checkouts/mlx-swift add Source/MLX/Transforms+Compile.swift
git -C .build/checkouts/mlx-swift commit -m "perf: optimize compile() wrapper -- cache state, reduce allocations, remove locks"
```

---

## Task 2: Make ArraysCache/MambaCache Compile-Compatible

**Files:**
- Modify: `Libraries/MLXLMCommon/KVCache.swift` (ArraysCache subscript, ~line 1099)

**What to change:**

ArraysCache subscript setter currently does `cache[index] = newValue` which replaces the MLXArray reference. compile() tracks references via innerState() and needs them to stay stable. Change to _updateInternal when the slot is already populated.

- [ ] **Step 1: Read the current ArraysCache subscript**

```bash
grep -n "subscript" Libraries/MLXLMCommon/KVCache.swift
```

- [ ] **Step 2: Modify the subscript setter**

```swift
public subscript(index: Int) -> MLXArray? {
    get { cache[index] }
    set {
        if let existing = cache[index], let newValue {
            existing._updateInternal(newValue)
        } else {
            cache[index] = newValue
        }
    }
}
```

This preserves reference identity after prefill (when slots are populated). During prefill (slots nil), it does normal assignment.

- [ ] **Step 3: Verify Qwen3.5 uncompiled baseline is unchanged**

```bash
cd TestRunner && swift build -c release
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench
# Should still show ~44 tok/s uncompiled
```

- [ ] **Step 4: Commit**

```bash
git add Libraries/MLXLMCommon/KVCache.swift
git commit -m "fix: ArraysCache subscript uses _updateInternal for compile compatibility"
```

---

## Task 3: Enable Compiled Decode for Heterogeneous Caches

**Files:**
- Modify: `Libraries/MLXLMCommon/Evaluate.swift` (~line 737, setupCompiledDecode)

**What to change:**

setupCompiledDecode currently bails if ANY cache is not KVCacheSimple. With Task 2's fix, MambaCache/ArraysCache is now compile-safe. Update the guard to allow ArraysCache through.

- [ ] **Step 1: Read current setupCompiledDecode**

```bash
grep -A30 "mutating func setupCompiledDecode" Libraries/MLXLMCommon/Evaluate.swift
```

- [ ] **Step 2: Update the cache conversion loop**

```swift
mutating func setupCompiledDecode(maxCacheLength: Int) throws {
    guard HardwareInfo.isCompiledDecodeSupported else { return }
    guard state == nil else { return }

    // Materialize cache before conversion
    MLX.eval(cache)  // Note: MLX.eval, not bare eval

    // KVCacheSimple -> CompilableKVCache. ArraysCache/MambaCache stays as-is.
    // RotatingKVCache, QuantizedKVCache -- bail.
    var converted = 0
    for i in 0..<cache.count {
        if cache[i] is KVCacheSimple {
            cache[i] = CompilableKVCache(from: cache[i], maxLength: maxCacheLength)
            converted += 1
        } else if cache[i] is ArraysCache {
            continue
        } else {
            return
        }
    }
    guard converted > 0 else { return }

    let capturedModel = model
    let cacheRef = cache

    self.compiledForward = compile(inputs: cacheRef, outputs: cacheRef) {
        (args: [MLXArray]) -> [MLXArray] in
        let result = capturedModel(
            LMInput.Text(tokens: args[0])[text: .newAxis],
            cache: cacheRef.isEmpty ? nil : cacheRef,
            state: nil)
        return [result.logits]
    }
}
```

- [ ] **Step 3: Test with Qwen3.5**

```bash
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench
# After Task 1 optimizations + this: target >80 tok/s compiled
```

- [ ] **Step 4: Test with Llama (regression check)**

```bash
.build/release/TestRunner ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912 "What is 2+2?" --compiled
# Must show: Output: IDENTICAL
```

- [ ] **Step 5: Commit**

```bash
git add Libraries/MLXLMCommon/Evaluate.swift
git commit -m "feat: enable compiled decode for models with heterogeneous caches"
```

---

## Task 4: Compile the Full Decode Step (Model + Argmax)

**Files:**
- Modify: `Libraries/MLXLMCommon/Evaluate.swift` (setupCompiledDecode and step function)

**What to change:**

Currently the compile closure wraps only the model forward and returns logits. The argmax/sampling happens OUTSIDE the compile boundary, adding FFI crossings. Move argmax inside the compile closure for greedy decode.

- [ ] **Step 1: Update the compile closure to include argmax**

```swift
self.compiledForward = compile(inputs: cacheRef, outputs: cacheRef) {
    (args: [MLXArray]) -> [MLXArray] in
    let result = capturedModel(
        LMInput.Text(tokens: args[0])[text: .newAxis],
        cache: cacheRef.isEmpty ? nil : cacheRef,
        state: nil)
    let token = result.logits[0..., -1, 0...].argMax(axis: -1)
    return [token]
}
```

- [ ] **Step 2: Update step() to use the token directly**

The compiled step now returns a token, not logits. Update the step function's compiled path:

```swift
if self.compiledForward != nil {
    let result = self.compiledForward!([previous.tokens])
    if result.count > 0 {
        self.state = nil
        if needsCacheQuantization {
            maybeQuantizeKVCache(
                cache: &cache, kvBits: kvBits,
                kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart,
                kvMode: kvMode)
        }
        return result[0]  // Already a token, not logits
    }
    self.compiledForward = nil
}
```

Note: This only works for greedy decode (temperature=0). For non-greedy sampling, fall back to uncompiled path or compile the sampler separately.

- [ ] **Step 3: Benchmark**

```bash
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench
```

- [ ] **Step 4: Commit**

```bash
git add Libraries/MLXLMCommon/Evaluate.swift
git commit -m "perf: compile full decode step including argmax"
```

---

## Task 5: Direct C API Decode Loop (Nuclear Option)

If Tasks 1-4 don't close the gap sufficiently, this task bypasses the Swift compile() wrapper entirely and calls the MLX C API directly for the decode loop.

**Files:**
- Create: `Libraries/MLXLMCommon/CompiledDecodeLoop.swift`

**What to build:**

A minimal decode loop that:
1. Traces the model forward ONCE using mlx_detail_compile
2. Caches the compiled closure as a raw C handle (mlx_closure)
3. On each step, calls mlx_closure_apply directly -- no Swift wrapper, no locks, no allocations
4. Reads the result token via mlx_array_item_int32

This eliminates ALL Swift overhead in the hot loop. The only Swift code per token is the loop counter and C function calls.

- [ ] **Step 1: Create CompiledDecodeLoop.swift with C API imports**

```swift
import Cmlx
import MLX
import MLXNN

/// Zero-overhead compiled decode loop using direct C API.
/// Bypasses Transforms+Compile.swift wrapper entirely.
public struct CompiledDecodeLoop {
    private var compiledClosure: mlx_closure
    private var inputVector: mlx_vector_array
    private let stateArrays: [MLXArray]  // references tracked by compile
    
    public init(model: any LanguageModel, cache: [KVCache]) {
        // Build the trace closure, compile it, cache the handle
    }
    
    public mutating func step(token: MLXArray) -> MLXArray {
        // 1. Set token in input vector
        // 2. mlx_closure_apply (THE compiled graph)
        // 3. Extract result, update state
        // Zero allocations. Zero locks.
    }
    
    public func cleanup() {
        mlx_closure_free(compiledClosure)
        mlx_vector_array_free(inputVector)
    }
}
```

- [ ] **Step 2: Implement trace and compile using mlx_detail_compile directly**

The key C API calls:
```c
int mlx_detail_compile(mlx_closure* res, mlx_closure fun, uintptr_t id, bool shapeless, const uint64_t* constants, size_t num_constants);
int mlx_closure_apply(mlx_vector_array* res, mlx_closure cls, const mlx_vector_array inputs);
```

- [ ] **Step 3: Implement step function with zero allocations**

Update the mlx_vector_array in-place, call apply, read result.

- [ ] **Step 4: Wire into TestRunner --raw-bench for testing**

- [ ] **Step 5: Benchmark**

```bash
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench
# Target: >100 tok/s (matching Python 123)
```

- [ ] **Step 6: Commit**

```bash
git add Libraries/MLXLMCommon/CompiledDecodeLoop.swift
git commit -m "feat: direct C API decode loop for zero-overhead compiled inference"
```

---

## Task 6: Wire Everything Into TokenIterator

**Files:**
- Modify: `Libraries/MLXLMCommon/Evaluate.swift`

After the compile optimizations are validated in the raw bench, integrate them into the production TokenIterator so all callers benefit.

- [ ] **Step 1: Update setupCompiledDecode to use the optimized compile path**

Whether that's the optimized Transforms+Compile.swift (Task 1) or the direct C API loop (Task 5), wire it into the compiledForward field on TokenIterator.

- [ ] **Step 2: Test with --compiled flag**

```bash
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit "Hello" --compiled
# Must show: Output: IDENTICAL, speed >80 tok/s
```

- [ ] **Step 3: Test Llama regression**

```bash
.build/release/TestRunner ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912 "What is 2+2?" --compiled
# Must show: Output: IDENTICAL
```

- [ ] **Step 4: Commit**

```bash
git add Libraries/MLXLMCommon/Evaluate.swift
git commit -m "feat: integrate optimized compile into TokenIterator"
```

---

## Success Criteria

| Metric | Current | Target | Method |
|---|---|---|---|
| Qwen3.5-35B decode tok/s | 44 | **>90** | --raw-bench |
| Llama 3.2 1B output correctness | IDENTICAL | IDENTICAL | --compiled |
| Uncompiled path regression | 44 tok/s | 44 tok/s | --raw-bench without compile |
| swift build passes | Yes | Yes | swift build |

## Priority Order

**Task 1 (compile wrapper optimization) is the CRITICAL PATH.** If the wrapper overhead drops from ~5ms to ~0.5ms, that alone could give 2x+ speedup. Tasks 2-4 are enablers. Task 5 is the nuclear option if Task 1 doesn't yield enough.

Do Tasks 1 -> 2 -> 3 -> benchmark -> decide if Task 4/5 needed -> Task 6.
