// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/tree.py
//
// Phase 0 stub — types only. Phase 2 lands the best-first heap builder.

import Foundation
import MLX

/// A tree of drafted continuations produced by the block-diffusion drafter.
///
/// Built by ``TreeBuilder/build(draftLogits:budget:)`` from the DFlash
/// drafter's per-position logits. The tree encodes up to `budget` distinct
/// continuations that share prefixes, selected greedily by log-probability
/// via a max-heap (Algorithm 1 in [arXiv 2604.12989](https://arxiv.org/abs/2604.12989)).
///
/// Indexing: node index 0 is the root (the "bonus token" from the previous
/// round). Indices 1...N are the drafted nodes in heap-pop order.
public struct DDTree: @unchecked Sendable {

    /// Token ID for each drafted node. Shape: (N,), Int32. Does NOT include
    /// the root — the root token is passed separately to
    /// ``TreeCompile/compile(tree:rootTokenID:prefixLen:)``.
    public let nodeTokenIds: MLXArray

    /// Depth of each node. Root's children are depth 1; root is depth 0
    /// (not stored here). Shape: (N,), Int32.
    public let nodeDepths: MLXArray

    /// Parent tree-index for each tree position, including root.
    /// Size N+1. `parents[0] == -1` (root has no parent).
    public let parents: [Int32]

    /// Per-node `[tokenID: childIndex]` maps. Size N+1. Used by
    /// ``TreeBuilder/followVerifiedTree(childMaps:posteriorTokens:)``
    /// to walk the tree against the target's argmax.
    public let childMaps: [[Int32: Int32]]

    /// Ancestor-only attention visibility matrix. Shape: (N+1, N+1), Bool.
    /// `visibility[i, j] == true` means node `i` can attend to node `j`.
    /// Every node attends to itself and all its ancestors; siblings do not
    /// see each other. Converted to the additive SDPA mask in
    /// ``TreeCompile/compile(tree:rootTokenID:prefixLen:)``.
    public let visibility: MLXArray

    /// Number of drafted nodes. Equal to `nodeTokenIds.count`.
    public let nodeCount: Int

    public init(
        nodeTokenIds: MLXArray,
        nodeDepths: MLXArray,
        parents: [Int32],
        childMaps: [[Int32: Int32]],
        visibility: MLXArray,
        nodeCount: Int
    ) {
        self.nodeTokenIds = nodeTokenIds
        self.nodeDepths = nodeDepths
        self.parents = parents
        self.childMaps = childMaps
        self.visibility = visibility
        self.nodeCount = nodeCount
    }

    /// Build an empty tree (root only) — returned when `budget <= 0` or the
    /// drafter produced no positions. The caller falls back to single-step
    /// greedy decode on the bonus token.
    public static func empty() -> DDTree {
        let visibility = MLXArray.ones([1, 1], type: Bool.self)
        return DDTree(
            nodeTokenIds: MLXArray([] as [Int32]),
            nodeDepths: MLXArray([] as [Int32]),
            parents: [-1],
            childMaps: [[:]],
            visibility: visibility,
            nodeCount: 0
        )
    }
}
