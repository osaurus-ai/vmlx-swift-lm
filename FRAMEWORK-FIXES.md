# Framework-Level Fixes: Closing the 14% Swift-Python Gap

## Current State (clean GPU, M4 Max 128GB, 200 tokens greedy)

| Model | Python mlx_lm | Swift mlx-swift-lm | Gap |
|-------|:---:|:---:|:---:|
| Gemma4 26B MLX-4bit | 80 tok/s | 97 tok/s | Swift +21% |
| Qwen3.5-35B JANG_4K | 71 tok/s | 61 tok/s | 14% behind |
| Qwen3.5-35B MLX uniform 4-bit | 118 tok/s | ~100+ (est) | ~15% |

The remaining 14% gap (~2-3ms/token overhead) is distributed across 3 framework-level causes.
All model-level optimizations are done (compiled sub-blocks, asType removals, needsCacheQuantization guard).

---

## Fix 1: Eliminate Existential Protocol Dispatch

**Impact: ~0.5-0.7ms/token (~3-5%)**

### Problem
`TokenIterator` stores `model: any LanguageModel` (line 566 of Evaluate.swift).
Every `model(...)` call goes through existential dispatch (vtable lookup + heap allocation for return value).
This happens once per token, but the model internally does 64 layer dispatches.
`SpeculativeTokenIterator` has the same issue (lines 794-795).

### Where
- `Libraries/MLXLMCommon/Evaluate.swift`
  - Line 566: `let model: any LanguageModel`
  - Line 794-795: `let mainModel: any LanguageModel` / `let draftModel: any LanguageModel`
  - Line 1719: `iterator: consuming any TokenIteratorProtocol`

### Implementation

**Option A: Genericize TokenIterator (API-breaking)**
```swift
// BEFORE
public struct TokenIterator: TokenIteratorProtocol {
    let model: any LanguageModel
    var cache: [KVCache]
    ...
}

// AFTER
public struct TokenIterator<M: LanguageModel>: TokenIteratorProtocol {
    let model: M
    var cache: [KVCache]  // KVCache is already protocol-typed in cache array
    ...
}
```
Then every function that creates a `TokenIterator` needs to become generic:
```swift
// BEFORE
public func generate(input:, model: any LanguageModel, ...) -> ...

// AFTER
public func generate<M: LanguageModel>(input:, model: M, ...) -> ...
```
Similarly for `generateLoopTask` (line 1715) and all public `generate()` overloads (lines 1219, 1260, 1322, 1419, 1583, 1668).

**Scope of change:**
- All 8 `generate()` function signatures in Evaluate.swift
- `TokenIterator` struct
- `SpeculativeTokenIterator` struct
- Any downstream code that stores `TokenIterator` (e.g. in osa-jang's ChatEngine)

**Option B: @_specialize attribute (non-breaking, partial)**
```swift
@_specialize(where M == Gemma4)
@_specialize(where M == Qwen35Model)
mutating func step(...) -> MLXArray { ... }
```
Less invasive but requires listing every model type and only helps known models.

**Recommendation:** Option A. The API break is worth it for a permanent 3-5% improvement.

### Cache Array
`var cache: [KVCache]` stores protocol-typed values. This means `cache[i]` also does existential dispatch
(64 times per token). However, this is harder to genericize because different layers can have different
cache types (KVCache vs RotatingKVCache). The existential overhead here is smaller since cache
operations are simple (just `update(keys:values:)` + property reads).

---

## Fix 2: Dedicated Generation Stream

**Impact: ~1-2ms/token (~7-10%)**

### Problem
Python uses `mx.stream(generation_stream)` in `_step()` -- this sets a C++ thread-local
`default_stream` so ALL ops inside the model forward run on a dedicated Metal command queue.
This allows GPU to pipeline: while `item()` materializes the previous token on one queue,
the next forward pass can begin building on the dedicated queue.

Swift currently runs everything on the default stream. Prior attempts:
1. `@TaskLocal` (Stream.withNewDefaultStream) -- adds ~2ms overhead per scope entry/exit,
   and Swift Tasks hop threads between suspension points, breaking C++ thread-local state.
2. `Stream.setDefault()` (direct C++) -- works on paper, but `item()` and `asyncEval()` then
   run on the same stream, eliminating the pipelining benefit. The Python pattern works because
   the stream context manager sets/restores the default around `_step()` only.
3. `stream.runWith {}` (mlx_stream_run_with callback) -- tested, same issue.

### Where
- `.build/checkouts/mlx-swift/Source/MLX/Stream.swift` -- Stream class, lines 88-206
- `Libraries/MLXLMCommon/Evaluate.swift` -- `next()` at line 742, `step()` at line 724

### Why Python Works
In Python's `generate_step` (generate.py:452-466):
```python
while True:
    next_y, next_logprobs = _step(y)       # Build graph on generation_stream
    mx.async_eval(next_y, next_logprobs)   # Submit to generation_stream
    yield y.item(), logprobs               # Materialize previous on default stream
```
The key: `_step()` runs inside `with mx.stream(generation_stream)`. The C++ thread-local
is set before entering `_step` and restored after. Since Python's GIL ensures single-threaded
execution, the thread-local is reliable.

### Implementation Strategy

The core issue is that Swift's `async/await` and `Task` system moves code between OS threads
at each suspension point. C++ thread-local state doesn't follow.

**Option A: Serial DispatchQueue (bypass Swift concurrency)**
```swift
let generateQueue = DispatchQueue(label: "mlx.generate", qos: .userInteractive)
let genStream = Stream(Device.defaultDevice())

generateQueue.sync {
    // This closure runs on a single OS thread for its entire duration
    Stream.setDefault(genStream)
    
    for token in iterator {
        // step() runs on genStream (thread-local is set)
        // item() also runs on genStream -- but that's fine if we accept
        // sequential execution (no pipelining between streams)
        ...
    }
    
    Stream.restoreDefault()
}
```
Problem: This gives us a dedicated stream but NOT pipelining. For pipelining we need
`item()` to run on a DIFFERENT stream than `step()`.

**Option B: Two DispatchQueues**
```swift
let modelQueue = DispatchQueue(label: "mlx.model")
let materialQueue = DispatchQueue(label: "mlx.material")
let genStream = Stream(Device.defaultDevice())

// On modelQueue: run forward pass on genStream
modelQueue.async {
    Stream.setDefault(genStream)
    let token = step(previous: y)
    asyncEval(token)
}

// On materialQueue: materialize on default stream
materialQueue.async {
    // item() uses default stream
    let value = previousToken.item(Int.self)
}
```
Problem: TokenIterator is a value type (struct), can't be shared across queues.
Would need to refactor to class or use unsafe pointers.

**Option C: MLX C++ level fix**
Add a `mlx_async_eval_on_stream(array, stream)` function that overrides which stream
the async submission uses, regardless of which OS thread calls it. This would let Swift do:
```swift
asyncEval(token, stream: genStream)  // Submit to genStream
let value = previousToken.item(Int.self)  // Materialize on default
```
This requires changes to mlx C++ backend.

**Recommendation:** Option C is the cleanest but requires upstream MLX changes.
Option A is achievable now but won't give pipelining benefit.
For now, this fix is **deferred** unless someone can contribute the C++ change.

### Key Insight
On a single-user Mac with no competing Metal work, the default stream has zero contention.
The pipelining benefit from a dedicated stream is only ~1-2ms/token. The stream overhead in
Swift (from any of the attempted approaches) was 1-2ms/token -- washing out the benefit.
This fix has the highest potential but the hardest implementation path.

---

## Fix 3: Compile Full Model Forward (KV Cache Growth)

**Impact: ~0.5-1ms/token (~3-5%)**

### Problem
Python's VMLX engine uses `mx.compile(self.model.__call__)` to compile the entire model
forward pass. This fuses hundreds of ops per token into fewer Metal dispatches.

In Swift, `compile()` crashes or retraces expensively because:
1. KV cache grows by 1 token each decode step (shape changes)
2. Without `shapeless: true`, the compile tracer retraces every step
3. With `shapeless: true`, some ops crash (e.g. `argPartition` in MoE routing)

### Where
- `.build/checkouts/mlx-swift/Source/MLX/Transforms+Compile.swift` -- compile infrastructure
- Python reference: `vmlx_engine/model_runner.py` -- `mx.compile(self.model.__call__)`

### How Python Handles This
Standard `mlx_lm generate` does NOT compile the full model forward.
Only the VMLX engine compiles it. The VMLX compile works because:
- Python's compile retraces on shape change but the C++ CompiledFunction caches efficiently
- The retrace cost for decode (graph structure identical, only constants differ) is low

### What We Already Compile
Sub-block compilations that DO work in Swift (shapes fixed during decode):
- `compiledLogitSoftcap` -- Gemma4 (3 ops -> 1)
- `compiledSigmoidGate` -- Qwen3.5 MoE shared expert (2 ops -> 1)
- `_compiledSigmoidMultiply` -- Qwen3Next GatedDeltaNet output (2 ops -> 1)
- `_compiledComputeG` -- GatedDelta decay (5 ops -> 1)
- `_compiledStepOps` -- GatedDelta step (10 ops -> 1, decode only)

These already capture the biggest op-fusion wins. The remaining ops in a model forward
are primarily:
- Linear/QuantizedLinear (already single Metal kernels)
- RMSNorm (already fused in MLXFast)
- ScaledDotProductAttention (already fused in MLXFast)
- Residual additions (single ops, not worth fusing)

### Implementation Path
If we wanted to compile the full forward:

1. **Investigate retrace cost**: Add timing around `CompiledFunction.call()` to measure
   whether the retrace per shape change is actually expensive
2. **Shapeless compile with workarounds**: Use `shapeless: true` but replace problematic
   ops (argPartition) with compile-safe alternatives
3. **Static KV cache**: Pre-allocate KV cache to max_seq_len, use a position counter
   instead of growing. This eliminates shape changes entirely.

**Recommendation:** Low priority. Sub-block compilation already captures most of the benefit.

---

## Priority Order

1. **Existential dispatch elimination** -- Highest ROI, well-understood, just engineering work
2. **Dedicated generation stream** -- Highest potential but hardest, may need upstream MLX changes
3. **Full model compile** -- Lowest priority, sub-block compilation covers most benefit

## Files to Read Before Starting

### Core (MUST read)
- `Libraries/MLXLMCommon/Evaluate.swift` -- TokenIterator, generate functions, the hot path
- `DECODE-ANALYSIS.md` -- Complete per-token trace, Python vs Swift comparison
- `.build/checkouts/mlx-swift/Source/MLX/Transforms+Compile.swift` -- compile infrastructure
- `.build/checkouts/mlx-swift/Source/MLX/Stream.swift` -- Stream API, prior stream experiments

### Model references (read for context)
- `Libraries/MLXLLM/Models/Gemma4Text.swift` -- MoE model, compiled softcap
- `Libraries/MLXLLM/Models/Qwen35.swift` -- MoE + GatedDeltaNet, compiled sigmoid gate
- `Libraries/MLXLLM/Models/Qwen3Next.swift` -- GatedDeltaNet attention, compiled sigmoid multiply
- `Libraries/MLXLLM/Models/GatedDelta.swift` -- Metal kernel + compiled ops fallback

### Python references
- `/opt/homebrew/lib/python3.14/site-packages/mlx_lm/generate.py` -- generate_step loop (lines 392-466)
- `/Users/eric/mlx/vllm-mlx/vmlx_engine/model_runner.py` -- VMLX compiled forward pattern

## What NOT to Touch
- Model-level code (GatedDelta.swift, Gemma4Text.swift, Qwen35.swift, Qwen3Next.swift) -- all optimized
- VLM versions -- already mirror LLM optimizations
- Load.swift / JangLoader.swift -- weight loading is correct
- asyncEval pattern -- current single-array submit is optimal
- Memory.clearCache() cadence -- 256 tokens is correct

## What Was Already Tried and Failed
- Stream.withNewDefaultStream (@TaskLocal) -- -11% regression
- Stream.setDefault() in generate loop -- 0% improvement (no pipelining without split)
- stream.runWith {} callback -- same as setDefault
- compile(shapeless: true) on full model -- crashes on argPartition (MoE routing)
- compile() without shapeless on full model -- retraces every step, overhead > benefit
- asyncEval with cache state arrays -- 0% benefit, adds per-token overhead
- Dedicated stream with item() on different stream -- not possible with struct TokenIterator
