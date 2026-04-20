# DFlash + DDTree speculative decoding — design

Native Swift/MLX implementation of block-diffusion speculative decoding per [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) (DFlash) and [arXiv 2604.12989](https://arxiv.org/abs/2604.12989) (DDTree), ported from the Python MLX reference [humanrouter/ddtree-mlx](https://github.com/humanrouter/ddtree-mlx).

**Status** (2026-04-20, branch `feature/ddtree-spec-dec`, iter 1): Phase 0 — design + stubs. No user-visible behaviour change yet.

## 1. Why this, not autoregressive speculative decoding

| Approach | Draft cost | Speedup ceiling | Blocker for us |
|---|---|---|---|
| Classic autoregressive draft model (our existing `SpeculativeTokenIterator`) | Draft model runs *D* sequential forward passes to propose *D* tokens | ~1.3-1.8× typical | Sequential draft = main bottleneck. Every op dispatch on the draft model kills TTFT. |
| Medusa / EAGLE multi-head drafters | One draft forward but heavy head logic and tree search | ~2-3× typical | Each head retrains — no unified drafter checkpoint. Harder to maintain multiple families. |
| **DFlash** (block diffusion drafter) | ONE forward produces whole block of draft logits via block diffusion | ~6× reported on CUDA pure-attention | Needs a published block-diffusion drafter per target family — z-lab hosts these. |
| **DDTree** (DFlash + best-first tree) | Same drafter forward; verification runs over a tree of branches | ~7.5× reported (strict superset of DFlash) | Tree verify needs ancestor-mask SDPA + per-node state forking. |

DFlash is the strict subset of DDTree (tree with branching factor 1). We build DDTree; DFlash comes along for free as a degenerate case.

## 2. Reference stack and source-of-truth

- Paper 1: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) — DFlash: Block Diffusion for Flash Speculative Decoding.
- Paper 2: [arXiv 2604.12989](https://arxiv.org/abs/2604.12989) — Accelerating Speculative Decoding with Block Diffusion Draft Trees.
- Python CUDA: [z-lab/dflash](https://github.com/z-lab/dflash), [liranringel/ddtree](https://github.com/liranringel/ddtree).
- **Port target** (line-for-line when in doubt): [humanrouter/ddtree-mlx](https://github.com/humanrouter/ddtree-mlx). Pure Python MLX. Structure:
  - `tree.py` (234 lines) — `DDTree` namedtuple + `build_ddtree_tree` (Algorithm 1 best-first heap) + `follow_verified_tree` + `compute_dfs_order`.
  - `compile.py` (109 lines) — `CompiledTree` namedtuple + `compile_tree(tree, root_token_id, prefix_len)` → MLX tensors; `is_dfs_prefix` fast-path check.
  - `verify.py` (810 lines) — `tree_verify_forward` over attention + recurrent layers; attention mask `(1, 1, T, prefix_len + T)`; per-token RoPE via batch-reshape.
  - `kernels.py` (234 lines) — custom Metal kernels for GatedDeltaNet recurrent layers.
  - `cache.py` (188 lines) — `snapshot_caches` / `restore_caches` / `fast_path_commit` (DFS prefix) / `tree_aware_path_commit` / `slow_path_commit` (re-forward on non-DFS path).
  - `runtime.py` (711 lines) — main draft → tree-build → verify → walk-and-commit loop.

## 3. Data types (Swift mirror of humanrouter)

### 3.1 `DDTree` (tree.py → DDTree.swift)

```swift
public struct DDTree: Sendable {
    /// Token ID for each tree node. Shape: (N,), int32. Root is NOT in this array
    /// (the root token is passed separately to `compileTree`).
    public let nodeTokenIds: MLXArray
    /// Depth of each node; root's children are depth 1. Shape: (N,), int32.
    public let nodeDepths: MLXArray
    /// Parent index for each node. Size N+1; parents[0] = -1 (root).
    public let parents: [Int32]
    /// Per-node {tokenID: childIndex} maps. Size N+1.
    public let childMaps: [[Int32: Int32]]
    /// Ancestor-only attention mask. Shape: (N+1, N+1), Bool.
    public let visibility: MLXArray
    /// Number of drafted nodes (excluding root).
    public let nodeCount: Int
}
```

### 3.2 `CompiledTree` (compile.py → TreeCompile.swift)

```swift
public struct CompiledTree: Sendable {
    public let inputIds: MLXArray       // (1, N+1) uint32
    public let positionIds: MLXArray    // (N+1,) int32 — absolute positions for RoPE
    public let attentionMask: MLXArray  // (1, 1, N+1, N+1) float32 — tree-only additive mask
    public let dfsOrder: MLXArray       // (N+1,) int32
    public let invDfsOrder: MLXArray    // (N+1,) int32
    public let parents: [Int32]          // (N+1,)
    public let depths: [Int32]           // (N+1,) — root=0, drafted=1..L
    public let treeSize: Int             // N+1
}
```

### 3.3 `DraftStrategy` (public API on `GenerateParameters`)

```swift
public enum DraftStrategy: Sendable {
    /// No speculative decoding. Default — preserves existing behaviour byte-for-byte.
    case none

    /// Classic autoregressive draft model (existing SpeculativeTokenIterator path).
    /// Kept for backward compatibility; maps to the existing code path.
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)

    /// DFlash linear verification — block diffusion drafter, verifies one
    /// trajectory per round.
    case dflash(drafterPath: URL, blockSize: Int)

    /// DDTree tree verification — block diffusion drafter, best-first heap tree,
    /// ancestor-mask SDPA verification of the full tree in one target forward.
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)
}
```

The enum lives on `GenerateParameters.draftStrategy: DraftStrategy?` (optional) with `nil` default — zero API churn for callers that don't set it.

## 4. Runtime loop (`runtime.py` → `SpecDecRuntime.swift`)

Per generation round:

1. **Draft**. Drafter forward pass reads the last-layer hidden state(s) from the target model (captured during the previous target step) and produces `(L, vocab)` per-position logits in ONE MLX forward. This is the block-diffusion core: no autoregressive sub-loop on the drafter.
2. **Build tree**. `buildDDTree(draftLogits:, budget:)` runs Algorithm 1 best-first heap to produce a `DDTree` of up to `branchingBudget` nodes. The heap picks the highest-log-probability prefixes across the block.
3. **Compile tree**. `compileTree(tree:, rootTokenID:, prefixLen:)` produces the `CompiledTree` tensors for the verify forward pass.
4. **Snapshot caches**. `snapshotCaches(cacheEntries:)` saves minimal state: offsets for `KVCache`, state-refs for SSM/recurrent layers. Lazy — no array copies unless an in-place write is later attempted.
5. **Tree-verify**. `treeVerifyForward(target:, compiledTree:, cacheEntries:)` runs ONE target forward with the tree attention mask. Returns posterior (argmax) token for each tree node.
6. **Walk**. `followVerifiedTree(childMaps:, posteriorTokens:)` walks the tree greedily against the target's argmax. Returns `(acceptedIndices, bonusToken)`. The walk terminates at the first mismatch; the mismatch token becomes the bonus for the next round.
7. **Commit**. Three strategies, picked in order:
   - **Fast path** (DFS prefix): accepted path == `dfsOrder[:n]` → trim KV offsets + replay recurrent tape. No re-forward.
   - **Tree-aware path**: accepted path is arbitrary-depth → pack accepted KV entries with `mx.take`, install the captured per-node recurrent state for the final accepted node.
   - **Slow path**: rare — restore cache snapshot and re-forward accepted tokens sequentially. Guarantees lossless cache state.

## 5. KV injection (drafter receives target hidden states)

The DFlash drafter is not a standalone model — it reads the target's penultimate-layer hidden state as input conditioning. Hook point:

```swift
// Target decoder, last step of decode loop:
let (targetLogits, penultimateHidden) = target.forward(
    inputIds, cache: cache, captureLayer: target.numLayers - 2)

// Pass penultimateHidden to drafter as context
let draftLogits = drafter.forward(
    inputIds: lastTargetToken,   // bonus token after walk
    targetHidden: penultimateHidden)
```

The hidden state is captured during the target forward pass and passed into the drafter during the next draft step. Requires exposing a `captureLayerIDs:` param on the target model's `forward`. We'll reuse the existing `captureLayerIDs` Evaluate.swift plumbing when present.

## 6. Attention mask shape (tree verify)

The tree verify forward's SDPA mask is `(1, 1, T, prefix_len + T)` where `T = N + 1`:

- Left block (prefix): all zeros (every tree node attends to entire prefix — it's causal).
- Right block (tree-to-tree): the ancestor-only visibility mask, with `-inf` off-mask and `0.0` on-mask. This is what `compile.py` builds.

Construction (inside verify):

```swift
let prefixMask = MLXArray.zeros((1, 1, treeSize, prefixLen), dtype: .float32)
let fullMask = concatenated([prefixMask, compiledTree.attentionMask], axis: -1)
```

Per-token positions come from `compiledTree.positionIds` and apply through a batch-reshape RoPE trick (NOT `rope.withOffset(pos)` which only supports scalar offsets).

## 7. Hybrid SSM + recurrent layer handling

humanrouter's Python DDTree does not fully accelerate GatedDeltaNet/Mamba recurrent layers — it reorders tokens to DFS before the recurrent pass which serializes the branches. Our vmlx fork has native hybrid SSM (Qwen 3.5/3.6 JANGTQ, Nemotron-H), so Phase 3 explicitly builds per-node recurrent state forking:

- Each tree node gets its own recurrent state snapshot keyed by `parent → child`.
- `tree_aware_path_commit` installs the final-accepted-node's recurrent state into the cache entry.
- Custom Metal kernel (port of `kernels.py`) computes per-position gated-delta-net updates in parallel across tree nodes.

Pin: Phase 3 acceptance requires byte-identical output vs `.none` at temp 0 on Qwen 3.5-27B AND a tree-verify speedup — otherwise the fork-and-serialize approach is no better than humanrouter's.

## 8. Integration points

### 8.1 `Evaluate.generate(input:cache:parameters:context:)`

- Reads `parameters.draftStrategy`. If `nil` or `.none` → current path, byte-identical output.
- If `.autoregressive(draftModel:, numDraftTokens:)` → current `SpeculativeTokenIterator` path (zero behaviour change).
- If `.dflash` or `.ddtree` → new `SpecDecRuntime.run(strategy:)` loop owns the decode loop; emits the same `Generation` events.

### 8.2 `BatchEngine.generate(input:parameters:)` and `submit`

- Per-slot `GenerateParameters` already flows through — each slot can pick its own strategy.
- Drafter is **per-model**, not per-slot: one `BatchEngine` holds a shared drafter context that every slot using `.dflash`/`.ddtree` shares. Memory cost: drafter weights loaded once.
- TurboQuant compatibility: drafter runs on plain fp16 KV (drafters are ~0.5-2B params, not worth quantising). Target may still use TurboQuant — `SpecDecRuntime` treats target cache as opaque and calls into `CacheCoordinator` / `BatchQuantize` the same way the current path does.

### 8.3 `ToolCallProcessor` + `ReasoningParser` pipeline

Unchanged. Accepted tokens flow through the detokenizer → reasoning-strip → tool-call-parse pipeline exactly as in the non-spec-dec path. Osaurus sees no API surface change.

## 9. JANG capability stamp extension (Phase 5)

Extend `JangCapabilities`:

```swift
public let draftStrategy: String?  // "dflash" / "ddtree" / nil
public let drafterPath: String?    // relative to jang_config.json; usually "drafter/"
public let branchingBudget: Int?   // for .ddtree
```

`ParserResolution.draftStrategy(capabilities:modelKey:)` resolves to a concrete `DraftStrategy` enum with the HF drafter path if osaurus installed the drafter alongside the target.

## 10. Acceptance criteria (per phase — duplicates `.claude/ralph-loop.local.md`)

| Phase | Minimum evidence |
|---|---|
| 0 | Stubs compile; `DraftStrategy` enum added; `DDTreeDesignTests` passes; 121 existing tests stay green. |
| 1 | DFlash drafter forward byte-identical to humanrouter reference on 10 fixed (prompt, hidden) pairs at temp 0; linear-verify byte-identical to autoregressive on Qwen 3-8B (when drafter available) or gpt-oss-20b; ≥ 3× wall-clock speedup. |
| 2 | Tree builder byte-identical to reference on synthetic logits; `compileTree` byte-identical; tree-verify byte-identical accepted-set vs reference; end-to-end byte-identical vs autoregressive + ≥ 1.5× over DFlash. |
| 3 | Hybrid SSM byte-identical vs autoregressive on Qwen 3.5-27B; tree-verify beats autoregressive measurably. |
| 4 | BatchEngine per-slot draft strategies compose with TurboQuant without KV corruption; `.chunk`/`.toolCall` stream unchanged; `BENCH_BATCH_SPECDEC` scenario green. |
| 5 | JANG stamp auto-pickup; OSAURUS-API-SURFACE.md §7 updated; osaurus can flip to `.ddtree` with one field. |

## 11. Drafter availability (probed 2026-04-20)

Public `z-lab/<name>-DFlash` checkpoints that return a `config.json` (unauthenticated):

| Repo | Target | Drafter size | Status |
|---|---|---|---|
| `z-lab/gpt-oss-20b-DFlash` | gpt-oss-20b (dense) | **1.5 GB** | ✅ downloaded |
| `z-lab/Qwen3.5-27B-DFlash` | Qwen 3.5-27B (hybrid SSM) | **3.2 GB** | ✅ downloaded |
| `z-lab/Kimi-K2.5-DFlash` | Kimi K2.5 | ~? | public (target model too large to test locally) |
| `z-lab/Qwen3-8B-DFlash` | Qwen 3-8B | — | **401** — gated or unreleased |
| `z-lab/Llama-3.1-8B-Instruct-DFlash` | Llama 3.1 8B | — | **401** — gated or unreleased |
| Others | | | 401 across the board |

### Concrete drafter architectures (from downloaded `config.json`)

**gpt-oss-20b-DFlash** (our Phase 1 primary, dense target):
- Architecture: `DFlashDraftModel` (transformer with `auto_map` to `dflash.DFlashDraftModel`)
- 8 layers, all `full_attention`
- `hidden_size`: 2880 · `num_attention_heads`: ? · `head_dim`: 64 · `intermediate_size`: 7680
- `block_size`: 8 (positions emitted per drafter forward)
- `dflash_config.mask_token_id`: 200000 (drafter-specific sentinel)
- `dflash_config.target_layer_ids`: `[1, 6, 11, 16, 21]` — 5 layers of the target model whose hidden states the drafter injects
- `dtype`: bfloat16 · `max_position_embeddings`: 131072

**Qwen3.5-27B-DFlash** (Phase 1 secondary, Phase 3 hybrid SSM primary):
- Architecture: `DFlashDraftModel`
- 5 layers, all `full_attention`
- `hidden_size`: 5120 · `num_attention_heads`: 32 · `head_dim`: 128 · `num_key_value_heads`: 8 · `intermediate_size`: 17408
- `block_size`: 16
- `dflash_config.mask_token_id`: 248070
- `dflash_config.target_layer_ids`: `[1, 16, 31, 46, 61]` (5 layers of 62-layer target)
- `dtype`: bfloat16 · `max_position_embeddings`: 262144 · `model_type`: qwen3

### Drafter → target binding rules

- The drafter is a **small-ish transformer** (5-8 layers) that takes the bonus token + a block of `mask_token_id` placeholders, plus target hidden states at `target_layer_ids`. Its output logits span `block_size` positions.
- **KV injection hook**: the target model must expose a capture-multiple-hidden-states API. Our existing `captureLayerIDs` param in `Evaluate.swift` already supports this for single-layer capture; Phase 1 extends it to capture a list.
- **Tokenizer compatibility**: drafter and target must share a tokenizer. For gpt-oss-20b-DFlash paired with `mlx-community/gpt-oss-20b-MXFP4-Q4` target, both use the harmony tokenizer. For Qwen 3.5-27B, the drafter is 27B-class and matches the target tokenizer exactly.

Phase 1 work targets gpt-oss-20b first (dense, smallest public drafter, pure-attention). Phase 3's hybrid SSM story then brings in Qwen 3.5-27B where the speedup vs autoregressive is ceiling-limited until per-node SSM fork lands.

## 12. Iter log (commit SHAs)

- **Iter 1 (89ea00f)** — Phase 0 scaffolding: this doc + stub SpecDec/*.swift files + `DraftStrategy` enum + `DDTreeDesignTests` (14 tests). No runtime behaviour change.
- **Iter 2 (d292b2a)** — Phase 1 kickoff: `DFlashDraftModel.swift` (drafter architecture port of `dflash.py`) + `DFlashDrafterLoader.swift` (safetensors load from local HF snapshot) + `DFlashDrafterForwardTests.swift` (6 tests). Both `z-lab/gpt-oss-20b-DFlash` (1.5 GB) and `z-lab/Qwen3.5-27B-DFlash` (3.2 GB) load cleanly; drafter-specific `fc.weight` and `hidden_norm.weight` populate. Forward pass shape matches Python reference.
- **Iter 3 (TBD)** — Phase 1 middle: `HiddenStateCapture.swift` protocol + `extractContextFeature(captured:targetLayerIDs:)` helper. `Qwen3Model` + `Qwen3ModelInner` conform; forward with empty `captureLayerIDs` byte-identical to plain `callAsFunction(_:cache:)`; capture is side-effect-free on logits. 5/5 `HiddenStateCaptureTests` green (serialized to avoid MLX Metal concurrency issue). Target is now ready to feed DFlash drafters their `target_hidden` conditioning. Next iter: wire the drafter + target into `SpecDecRuntime.runDflashLinear` and validate byte-identical output vs autoregressive at temp 0.
