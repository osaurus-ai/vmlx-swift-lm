// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/compile.py
//
// Phase 0 stub — types + API only. Phase 2 ports the real compile pass.

import Foundation
import MLX

/// A ``DDTree`` compiled into MLX tensors, ready for
/// ``TreeVerify/verifyForward(target:compiled:cache:prefixLen:)``.
public struct CompiledTree: @unchecked Sendable {

    /// Token IDs for every tree position including the root. Shape:
    /// `(1, N+1)`, UInt32. `inputIds[0, 0]` is the root (bonus) token.
    public let inputIds: MLXArray

    /// Absolute positions for per-token RoPE. Shape: `(N+1,)`, Int32.
    /// Root is at `prefix_len`; drafted node `i` at `prefix_len +
    /// node_depths[i]`.
    public let positionIds: MLXArray

    /// Tree-to-tree additive SDPA mask. Shape: `(1, 1, N+1, N+1)`, float32.
    /// `0.0` on ancestor cells, `-inf` elsewhere. The verifier extends this
    /// with a `(1, 1, N+1, prefix_len)` all-zeros prefix block at call time.
    public let attentionMask: MLXArray

    /// DFS traversal order for linear / recurrent layers. Shape: `(N+1,)`,
    /// Int32.
    public let dfsOrder: MLXArray

    /// Inverse of ``dfsOrder`` — mapping tree-index → DFS position.
    public let invDfsOrder: MLXArray

    /// Parent tree-index for each node (including root at `-1`). Size N+1.
    public let parents: [Int32]

    /// Depth for each tree position. Root is `0`; drafted nodes are
    /// `1...L`. Size N+1.
    public let depths: [Int32]

    /// Total number of tree positions (N+1).
    public let treeSize: Int

    public init(
        inputIds: MLXArray,
        positionIds: MLXArray,
        attentionMask: MLXArray,
        dfsOrder: MLXArray,
        invDfsOrder: MLXArray,
        parents: [Int32],
        depths: [Int32],
        treeSize: Int
    ) {
        self.inputIds = inputIds
        self.positionIds = positionIds
        self.attentionMask = attentionMask
        self.dfsOrder = dfsOrder
        self.invDfsOrder = invDfsOrder
        self.parents = parents
        self.depths = depths
        self.treeSize = treeSize
    }
}

/// Compile a ``DDTree`` into MLX tensors for ``TreeVerify``.
///
/// Phase 0 stub — Phase 2 ports the real `compile_tree` from humanrouter's
/// `compile.py`.
public enum TreeCompile {

    /// Produce a ``CompiledTree`` from the Python/Swift-side tree structure.
    ///
    /// - Parameters:
    ///   - tree: from ``TreeBuilder/build(draftLogits:budget:)``.
    ///   - rootTokenID: the bonus token from the previous round; sits at
    ///     tree position 0.
    ///   - prefixLen: number of tokens already in the target's KV cache
    ///     (context length). Used to offset ``CompiledTree/positionIds``.
    public static func compile(
        tree: DDTree,
        rootTokenID: Int32,
        prefixLen: Int
    ) throws -> CompiledTree {
        throw SpecDecError.notImplemented(
            "TreeCompile.compile — Phase 2 will port humanrouter/ddtree-mlx compile.py"
        )
    }

    /// `True` when the accepted path matches a DFS prefix.
    ///
    /// When true, ``SpecDecCache/fastPathCommit(cacheEntries:prefixLen:nAccepted:)``
    /// can trim KV offsets + replay the recurrent tape without re-forwarding.
    /// When false, the tree-aware commit path must pack accepted KV entries
    /// and install the final-accepted recurrent state directly.
    public static func isDfsPrefix(
        acceptedIndices: [Int32],
        dfsOrder: [Int32]
    ) -> Bool {
        let n = acceptedIndices.count
        guard n <= dfsOrder.count else { return false }
        for i in 0..<n where acceptedIndices[i] != dfsOrder[i] {
            return false
        }
        return true
    }
}
