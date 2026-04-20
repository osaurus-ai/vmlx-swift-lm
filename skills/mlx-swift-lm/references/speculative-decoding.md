# Speculative decoding (DFlash + DDTree)

## Overview

Block-diffusion speculative decoding in Swift/MLX. Three strategies, one opt-in field on `GenerateParameters`.

| Strategy | Drafter | Verify | Paper speedup |
|---|---|---|---|
| `.autoregressive` | small target-family model | sequential in block | 29-79% |
| `.dflash` | block-diffusion model | linear over block | ~6× over AR (paper) |
| `.ddtree` | block-diffusion model | best-first heap tree + ancestor mask | ~7.5× (paper) |

**Files:**
- `Libraries/MLXLMCommon/SpecDec/DraftStrategy.swift` — public enum
- `Libraries/MLXLMCommon/SpecDec/DDTree.swift`, `TreeBuilder.swift`, `TreeCompile.swift`, `TreeVerify.swift` — DDTree pipeline
- `Libraries/MLXLMCommon/SpecDec/DFlashDraftModel.swift` + `DFlashDrafterLoader.swift` — drafter arch + HF snapshot load
- `Libraries/MLXLMCommon/SpecDec/HiddenStateCapture.swift` + `TokenEmbedderModel.swift` — target-model protocols
- `Libraries/MLXLMCommon/SpecDec/SpecDecRuntime.swift` — `SpecDecRuntimeLinear.run` + `SpecDecRuntimeDDTree.run`
- `Libraries/MLXLMCommon/SpecDec/SpecDecStream.swift` — `AsyncStream<Generation>` adapter
- `Libraries/MLXLMCommon/SpecDec/SpecDecDrafterResolver.swift` — by-path cache + `.shared` singleton

## One-line opt-in

```swift
var params = GenerateParameters(maxTokens: 256, temperature: 0)
params.draftStrategy = .ddtree(
    drafterPath: URL(fileURLWithPath: "…/z-lab/Qwen3.5-27B-DFlash"),
    branchingBudget: 32,
    blockSize: 16)
let stream = try generate(input: input, parameters: params, context: ctx)
```

Set the field → same `AsyncStream<Generation>` as the non-speculative path. Events: `.chunk(String)`, `.toolCall(ToolCall)`, `.info(GenerateCompletionInfo)`.

When `draftStrategy` is `.none` / `nil` / `.autoregressive`, existing code path runs byte-identically.

## DraftStrategy enum

```swift
public enum DraftStrategy: Sendable {
    case none
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)
    case dflash(drafterPath: URL, blockSize: Int)
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)
}
```

- `blockSize` — must match drafter's training `block_size` from its `config.json`.
- `branchingBudget` — tree node cap (excluding root). 32-64 for greedy, 16-24 for sampling.

## Byte-parity invariant

**At `temperature: 0`, output matches greedy autoregressive decode byte-for-byte**. Drafter affects speed only, not tokens.

Pinned tests:
- `Tests/MLXLMTests/DFlashLinearByteParityTests.swift` — DFlash linear ≡ greedy AR
- `Tests/MLXLMTests/DDTreeEndToEndTests.swift` — DDTree ≡ greedy AR across budgets 1/2/4/6/8

Reasoning: the tree walker (`followVerifiedTree`) only accepts nodes where `posteriorTokens[node_i]` == target's argmax at that path. So committed sequence == AR sequence regardless of drafter proposals or tree shape.

## Target model requirements

A target must conform to **both** protocols for block-diffusion strategies to activate:

```swift
public protocol HiddenStateCaptureModel: LanguageModel {
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>
    ) -> (logits: MLXArray, capturedHiddenStates: [Int: MLXArray])
}

public protocol TokenEmbedderModel: LanguageModel {
    func embed(_ tokenIds: MLXArray) -> MLXArray
    func projectToLogits(_ hidden: MLXArray) -> MLXArray
}
```

Current conformance: **Qwen3** ✅. Other families roll out in Phase 3+.

If a target doesn't conform, `SpecDecStream.streamViaStrategy` returns `nil` and the dispatch in `Evaluate.generate` / `BatchEngine.generate` falls through to the non-speculative path. No error, no regression — the `draftStrategy` field is silently ignored.

## Drafter checkpoints

Load via HuggingFace. Public `z-lab` drafters as of 2026-04-20:

| Repo | Target | Size |
|---|---|---|
| `z-lab/gpt-oss-20b-DFlash` | gpt-oss-20b | 1.5 GB |
| `z-lab/Qwen3.5-27B-DFlash` | Qwen 3.5-27B (hybrid SSM) | 3.2 GB |
| `z-lab/Kimi-K2.5-DFlash` | Kimi K2.5 | — |

Download:
```
hf download z-lab/Qwen3.5-27B-DFlash --local-dir ~/drafters/Qwen3.5-27B-DFlash
```

Then:
```swift
params.draftStrategy = .ddtree(
    drafterPath: URL(fileURLWithPath: "~/drafters/Qwen3.5-27B-DFlash"),
    branchingBudget: 32,
    blockSize: 16)
```

Drafter is **cached by path** via `SpecDecDrafterResolver.shared`. Subsequent calls with the same path skip the 3.2 GB safetensors load.

## Architecture deep-dive

### DFlash drafter (`DFlashDraftModel`)

Small transformer (5-8 layers) that reads:
1. Target's shared embedding of `[bonus, mask, mask, …]` — the drafter proposes positions 1…block-1.
2. Target's per-layer hidden states at `dflash_config.target_layer_ids` — concatenated along the hidden dim and projected via drafter's `fc` layer.

Attention's K/V come from **concat(target_hidden, hidden_states)** — the "KV injection" of the DFlash paper.

Output: `(1, block_size, hidden)` passed through target's `lm_head` → `(1, block_size-1, vocab)` draft logits.

### DDTree (`TreeBuilder`, `TreeCompile`, `TreeVerify`)

Algorithm 1 best-first heap on drafter's per-position top-K log-probs → `DDTree` of up to `branchingBudget` nodes. Visibility matrix is ancestor-only attention mask.

`TreeCompile.compile` produces MLX tensors: `inputIds` uint32 `(1, T)`, `positionIds` int32, `attentionMask` float32 `(1, 1, T, T)`, `dfsOrder` / `invDfsOrder`.

`TreeVerify.verifyForward` runs target model to produce `posteriorTokens[i]` = target's argmax at each tree node. v1 is correct-but-slow (O(N) forwards); single-forward optimisation pending Phase 2 iter 13.

`TreeBuilder.followVerifiedTree` walks root → leaf greedily against `posteriorTokens`, returns `(acceptedIndices, bonusToken)`. Walk stops at first mismatch.

### Streaming (`SpecDecStream`)

Two ways to consume:

1. **Strategy-driven (recommended)**:
```swift
SpecDecStream.streamViaStrategy(
    strategy: params.draftStrategy!,
    inputIds: input.text.tokens,
    context: ctx,
    maxNewTokens: 256)
```
Resolves the drafter via `SpecDecDrafterResolver.shared`, dispatches to the appropriate runtime, returns `AsyncStream<Generation>`.

2. **Stateless runtime + explicit adapter**:
```swift
let drafter = try DFlashDrafterLoader.load(from: drafterPath)
let args = DDTreeArgs(target: myTarget, drafter: drafter, …)
let stream = SpecDecStream.streamDDTree(
    args: args, tokenizer: ctx.tokenizer, toolCallFormat: .xmlFunction)
```

Both paths:
- Run on a detached `Task`
- Pipe each committed batch through `NaiveStreamingDetokenizer`
- Strip `<think>…</think>` via optional `ReasoningParser`
- Extract tool calls via `ToolCallProcessor`
- Emit final `.info(GenerateCompletionInfo)` with `stopReason = .length`

## Performance expectations

Paper-reported speedups on CUDA. Swift/MLX numbers pending real-checkpoint benchmarks (iter 14+):

| Scenario | DFlash | DDTree |
|---|---|---|
| Dense, short block | 3× over AR | 4-5× over AR |
| Dense, long block | 4-6× over AR | 5-7× over AR |
| Hybrid SSM | 1.3× over AR | 1.5× over AR (ceiling until Phase 3) |

**Current state**: iter 12 ships integration, not speedup. `TreeVerify.verifyForward` v1 does O(N) forwards per verify round. Single-forward optimisation with combined `(1, 1, T, prefix_len + T)` attention mask + per-token RoPE lands Phase 2 iter 13+.

## Verification

```bash
swift test --filter 'DDTree|DFlash|SpecDec'  # 65 SpecDec tests
./scripts/verify-engine.sh --tests-only      # 121 baseline + all SpecDec
```

## osaurus integration

See `Libraries/MLXLMCommon/SpecDec/OSAURUS-SPECDEC.md` for the complete integration guide with checkpoint map, resolver usage, protocol requirements, and gap analysis.
