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
/// Port of humanrouter/ddtree-mlx `compile.py` (iter 7).
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
        let treeSize = 1 + tree.nodeCount

        // 1. Input IDs: [root_token, node_0_token, ..., node_{N-1}_token]
        var tokenIdsBuf: [Int32] = Array(repeating: 0, count: treeSize)
        tokenIdsBuf[0] = rootTokenID
        if tree.nodeCount > 0 {
            let nodeTokens = tree.nodeTokenIds.asArray(Int32.self)
            for (i, t) in nodeTokens.enumerated() {
                tokenIdsBuf[1 + i] = t
            }
        }
        // Python wraps in uint32 via `mx.array(..., dtype=mx.uint32)[None]`.
        let inputIdsInt32 = MLXArray(tokenIdsBuf).reshaped(1, treeSize)
        let inputIds = inputIdsInt32.asType(.uint32)

        // 2. Position IDs: root at prefix_len; each node at prefix_len + depth.
        var positionsBuf: [Int32] = Array(repeating: 0, count: treeSize)
        positionsBuf[0] = Int32(prefixLen)
        var depthsBuf: [Int32] = [0]
        depthsBuf.reserveCapacity(treeSize)
        if tree.nodeCount > 0 {
            let depths = tree.nodeDepths.asArray(Int32.self)
            for (i, d) in depths.enumerated() {
                positionsBuf[1 + i] = Int32(prefixLen) + d
                depthsBuf.append(d)
            }
        }
        let positionIds = MLXArray(positionsBuf)

        // 3. Attention mask: tree-to-tree visibility only — 0.0 where
        //    the child can attend to the ancestor, -inf elsewhere.
        //    Shape: (1, 1, tree_size, tree_size), float32.
        //    Python: `np.where(tree.visibility, 0.0, -np.inf)` then
        //    `[None, None, :, :]`.
        let visibilityBool = tree.visibility.asArray(Bool.self)
        var maskBuf: [Float] = Array(
            repeating: -Float.infinity, count: treeSize * treeSize)
        for i in 0..<(treeSize * treeSize) where visibilityBool[i] {
            maskBuf[i] = 0.0
        }
        let attentionMask = MLXArray(maskBuf)
            .reshaped(treeSize, treeSize)
            .expandedDimensions(axes: [0, 1])  // (1, 1, T, T)

        // 4. DFS ordering for linear / recurrent layers.
        let (dfs, invDfs) = try TreeBuilder.computeDfsOrder(tree: tree)
        let dfsOrder = MLXArray(dfs)
        let invDfsOrder = MLXArray(invDfs)

        return CompiledTree(
            inputIds: inputIds,
            positionIds: positionIds,
            attentionMask: attentionMask,
            dfsOrder: dfsOrder,
            invDfsOrder: invDfsOrder,
            parents: tree.parents,
            depths: depthsBuf,
            treeSize: treeSize
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
