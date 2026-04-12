# Decode Path Analysis: Swift vs Python MLX

## Benchmark Numbers (M4 Max 128GB, clean GPU, 200 tokens greedy)

| Model | Python mlx_lm | Swift mlx-swift-lm | Gap |
|-------|:---:|:---:|:---:|
| Qwen3.5-35B MLX uniform 4-bit | 118 tok/s | ~100+ (est) | ~15% |
| Qwen3.5-35B JANG_4K (3.98-bit mixed) | 71 tok/s | 61 tok/s | 14% |
| Gemma4 26B MLX uniform 4-bit | 80 tok/s (vmlx) | 97 tok/s | Swift +21% |

JANG format costs ~40% vs uniform 4-bit in BOTH Python and Swift (mixed bitwidth Metal kernel penalty).

## Per-Token Decode Path Comparison

### Python (generate_step, generate.py:452-466)

```
while True:
    next_y, next_logprobs = _step(y)       # 1. Build graph
    mx.async_eval(next_y, next_logprobs)   # 2. Submit 2 arrays
    yield y.item(), logprobs               # 3. Materialize previous
    if n % 256: mx.clear_cache()           # 4. Periodic cleanup
    y = next_y                             # 5. Pointer swap (zero cost)
```

Inside `_step` (generate.py:392-418):
```python
with mx.stream(generation_stream):      # Sets C++ thread-local (0 overhead)
    logits = model(y[None], cache=cache) # Direct pybind11 call
    logits = logits[:, -1, :]            # 1 slice op (2 indices)
    logprobs = logits - mx.logsumexp(logits)  # 2 ops
    sampled = sampler(logprobs)          # 1 op (argmax)
    return sampled, logprobs
```

### Swift (TokenIterator.next, Evaluate.swift:741-760)

```swift
let token = step(previous: previousY)   // 1. Build graph
y = .init(tokens: token)                // 2. Struct creation (LMInput.Text)
asyncEval(token)                        // 3. Submit 1 array
tokenCount += 1                         // 4. Counter
if tokenCount % 256: Memory.clearCache()// 5. Periodic cleanup
return previousY.tokens.item(Int.self)  // 6. Materialize previous
```

Inside `step` (Evaluate.swift:723-738):
```swift
let result = model(                      // Existential dispatch (any LanguageModel)
    previous[text: .newAxis],            // subscript creates LMInput.Text + tokens[.newAxis]
    cache: cache.isEmpty ? nil : cache,  // isEmpty check every token
    state: state)                        // passed through
self.state = result.state                // struct assignment
// needsCacheQuantization computed property check
return convertToToken(logits: result.logits)
```

Inside `convertToToken` (Evaluate.swift:712-722):
```swift
var logits = logits[0..., -1, 0...]      // 3 index ops (Python: 2)
logits = processor?.process(logits:) ?? logits  // Optional dispatch + nil coalescing
let y = sampler.sample(logits: logits)   // Protocol dispatch
processor?.didSample(token: y)           // Optional dispatch (usually nil)
return y
```

## Identified Overheads (Swift vs Python)

### 1. Existential Protocol Dispatch (~0.5ms/token?)
Swift: `model(...)` through `any LanguageModel` -> runtime vtable lookup per call.
Python: Direct `model(...)` through pybind11 -> C++ virtual call (faster).

### 2. `logits[0..., -1, 0...]` vs `logits[:, -1, :]` (~0.1ms/token?)
Swift creates 3 MLX graph nodes (3 index operations).
Python creates 2 MLX graph nodes (2 index operations).
1 extra op per token.

### 3. `processor?.process() -> sampler.sample() -> processor?.didSample()` (~0.2ms/token?)
3 protocol dispatches per token. When processor is nil, the optional chain
still runs. Python does 1 function call: `sampler(logprobs)`.

### 4. `previous[text: .newAxis]` wrapper overhead (~0.1ms/token?)
Creates `LMInput.Text(tokens: tokens[.newAxis], mask: nil)` struct.
Python: `y[None]` -- single MLX op, no wrapper.

### 5. `with mx.stream(generation_stream)` vs no stream (~1-2ms/token?)
Python sets C++ thread-local default stream via `StreamContext` constructor.
All ops in _step run on dedicated Metal command queue.
Swift: All ops run on default stream. Prior attempts to add dedicated stream
in Swift (via TaskLocal or Stream.setDefault) gave 0% or negative results
due to Swift Task thread hopping and cross-stream synchronization overhead.

### 6. `logprobs = logits - mx.logsumexp(logits)` in Python (~0.1ms/token?)
Python computes logprobs for every token (used by some samplers).
Swift does NOT compute logprobs (saves 2 ops). This is a SWIFT ADVANTAGE.

### 7. `mx.async_eval(next_y, next_logprobs)` vs `asyncEval(token)` (~0ms)
Python submits 2 arrays, Swift submits 1. Negligible difference.

## Per-Layer Overhead (64 layers per token)

### 8. `cache` array indexing
Python: `cache[i]` -- direct list index (C-level).
Swift: `cache[i]` -- Array subscript (bounds-checked).
64 accesses per token.

### 9. `for (layer, c) in zip(layers, cache)` vs `for (i, layer) in layers.enumerated()`
Python: `zip()` is a C-level iterator, zero allocation.
Swift: `enumerated()` creates IndexSequence wrapper, but compiler should optimize.

### 10. Mask creation
Python: Creates `fa_mask` and `ssm_mask` ONCE before the layer loop.
Swift: Same pattern -- creates masks once. No overhead here.

## Total Estimated Overhead
- Protocol dispatch: ~0.5ms
- Extra graph ops: ~0.3ms
- Stream difference: ~1-2ms
- Per-layer indexing: ~0.2ms
- **Total: ~2-3ms/token -> at 61 tok/s baseline = ~14% overhead**

## Optimizations Applied
1. Remove `.asType(.float32)` before softmax (softmax promotes internally)
2. Remove `.asType(h.dtype)` on scalar multipliers (MLX broadcasts)
3. Compile logit softcap (tanh+div+mul -> 1 Metal dispatch)
4. Compile sigmoidMultiply for GatedDeltaNet output gate
5. Compile sigmoidGate for shared expert gate
6. Compile computeGatedDeltaG (exp+exp+softplus+mul+neg -> 1 dispatch)
7. Compile gatedDeltaStepOps for ops fallback (decode uses Metal kernel)
8. Skip maybeQuantizeKVCache when kvBits==nil
9. Minimal asyncEval submit
10. Memory.clearCache() every 256 tokens

## What Would Close the Gap (framework-level changes)
1. **Eliminate existential dispatch**: Replace `any LanguageModel` with generic
   `<M: LanguageModel>` in TokenIterator. Requires API change.
2. **Simplify logit extraction**: Use `logits[-1]` instead of `logits[0..., -1, 0...]`.
3. **Inline sampler for greedy**: When temperature=0, use `logits.argMax()` directly
   instead of going through LogitSampler protocol.
4. **Make generation stream work**: Needs Swift-level solution that does not rely on
   TaskLocal (thread-hopping issues). Direct `mlx_set_default_stream` at start of
   generate loop, restore at end. Must ensure single-thread execution.
5. **Compile full model forward**: Needs MLX compile to handle KV cache growth.
   Python's compile retraces on shape change; Swift's crashes. Apple needs to fix
   the Swift compile tracer.
