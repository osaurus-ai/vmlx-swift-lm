// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/tree.py
//
// Phase 0 stub — public API only. Phase 2 ports Algorithm 1 (best-first heap).

import Foundation
import MLX

/// Builds a ``DDTree`` from the DFlash drafter's per-position logits.
///
/// Phase 0 stub — every method throws ``SpecDecError/notImplemented`` until
/// Phase 2 ports Algorithm 1 (best-first heap search) from
/// [humanrouter/ddtree-mlx `tree.py`](https://github.com/humanrouter/ddtree-mlx/blob/main/ddtree_mlx/tree.py).
public enum TreeBuilder {

    /// Build a ``DDTree`` from `(L, vocab)` draft logits.
    ///
    /// Picks the top-`budget` highest-log-probability prefixes greedily via
    /// a max-heap keyed by accumulated log-weight. Produces a tree of up to
    /// `budget` nodes; siblings do not see each other in the attention mask.
    ///
    /// - Parameters:
    ///   - draftLogits: `(L, vocab)` float32 — per-position logits from the
    ///     DFlash drafter for positions 1...L after the bonus token.
    ///   - budget: maximum number of tree nodes (excluding root). `0` → an
    ///     empty tree (single-step fallback).
    /// - Returns: ``DDTree`` ready for ``TreeCompile/compile(tree:rootTokenID:prefixLen:)``.
    public static func build(draftLogits: MLXArray, budget: Int) throws -> DDTree {
        throw SpecDecError.notImplemented(
            "TreeBuilder.build — Phase 2 will port Algorithm 1 from humanrouter/ddtree-mlx tree.py"
        )
    }

    /// Build a ``DDTree`` from precomputed top-K log-probabilities.
    ///
    /// Faster variant when the caller already ran top-K on the drafter
    /// logits (useful when the drafter emits top-K directly).
    ///
    /// - Parameters:
    ///   - topTokenIds: `(L, K)` Int32 — token IDs sorted by descending
    ///     log-prob.
    ///   - topLogProbs: `(L, K)` float32 — log-probabilities aligned with
    ///     `topTokenIds`.
    ///   - budget: maximum number of tree nodes.
    public static func buildFromTopK(
        topTokenIds: MLXArray,
        topLogProbs: MLXArray,
        budget: Int
    ) throws -> DDTree {
        throw SpecDecError.notImplemented("TreeBuilder.buildFromTopK — Phase 2")
    }

    /// Walk the verified tree greedily against the target model's argmax
    /// tokens.
    ///
    /// Starting at the root (index 0), check if the target model's chosen
    /// token matches a child of the current node. If so, accept the child
    /// and continue. Terminates at the first mismatch.
    ///
    /// - Parameters:
    ///   - childMaps: from ``DDTree/childMaps``.
    ///   - posteriorTokens: target's greedy argmax for each tree node.
    ///     Length must equal `tree_size` (N+1).
    /// - Returns: `(acceptedIndices, bonusToken)`. `acceptedIndices[0]` is
    ///   always `0` (the root). `bonusToken` is the first target token that
    ///   didn't match any child — it becomes the root of the next round's
    ///   tree.
    public static func followVerifiedTree(
        childMaps: [[Int32: Int32]],
        posteriorTokens: [Int32]
    ) throws -> (acceptedIndices: [Int32], bonusToken: Int32) {
        throw SpecDecError.notImplemented("TreeBuilder.followVerifiedTree — Phase 2")
    }

    /// Depth-first traversal order for linear / recurrent layer processing.
    ///
    /// Heap-pop order already produces nodes roughly in probability order,
    /// but recurrent (SSM) layers need strict DFS so parent state feeds
    /// into children before siblings.
    ///
    /// - Returns: `(dfsOrder, invDfsOrder)` where `dfsOrder[i]` is the
    ///   tree-node index at DFS position `i`, and `invDfsOrder[j]` is the
    ///   DFS position of tree-node `j`.
    public static func computeDfsOrder(
        tree: DDTree
    ) throws -> (dfsOrder: [Int32], invDfsOrder: [Int32]) {
        throw SpecDecError.notImplemented("TreeBuilder.computeDfsOrder — Phase 2")
    }
}
