// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 2 iter 7 — pin TreeCompile.compile output byte-for-byte against
// hand-traced reference values on the same branching tree used by
// DDTreeBuilderTests.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("DDTree compile — Phase 2", .serialized)
struct DDTreeCompileTests {

    /// Reproduce the hand-traced branching tree used in
    /// DDTreeBuilderTests. 5 nodes + root, nodes:
    ///   index 1 token=10 parent=0 depth=1
    ///   index 2 token=30 parent=1 depth=2
    ///   index 3 token=40 parent=1 depth=2
    ///   index 4 token=20 parent=0 depth=1
    ///   index 5 token=30 parent=4 depth=2
    private func branchingTree() throws -> DDTree {
        let tokenIds = MLXArray([Int32(10), 20, 30, 40]).reshaped(2, 2)
        let logProbs = MLXArray([Float(0), -1, 0, -1]).reshaped(2, 2)
        return try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 5)
    }

    @Test("input_ids starts with rootTokenID then tree-node tokens")
    func testInputIdsOrdering() throws {
        let tree = try branchingTree()
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 10)
        #expect(compiled.treeSize == 6)
        #expect(compiled.inputIds.ndim == 2)
        #expect(compiled.inputIds.dim(0) == 1)
        #expect(compiled.inputIds.dim(1) == 6)
        // Python ships uint32 ids.
        #expect(compiled.inputIds.dtype == .uint32)
        // Cast to Int32 for equality comparison; values fit.
        let ids = compiled.inputIds.asType(.int32).asArray(Int32.self)
        #expect(ids == [99, 10, 30, 40, 20, 30])
    }

    @Test("position_ids = prefix_len + depths with root at prefix_len")
    func testPositionIds() throws {
        let tree = try branchingTree()
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 10)
        let pos = compiled.positionIds.asArray(Int32.self)
        // Root depth=0 → prefix_len; node depths = [1, 2, 2, 1, 2] → [11, 12, 12, 11, 12].
        #expect(pos == [10, 11, 12, 12, 11, 12])
    }

    @Test("depths mirror positionIds - prefixLen")
    func testDepthsVector() throws {
        let tree = try branchingTree()
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 10)
        #expect(compiled.depths == [0, 1, 2, 2, 1, 2])
    }

    @Test("attention mask has 0.0 on ancestors, -inf elsewhere")
    func testAttentionMaskPattern() throws {
        let tree = try branchingTree()
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 10)
        #expect(compiled.attentionMask.ndim == 4)
        #expect(compiled.attentionMask.dim(0) == 1)
        #expect(compiled.attentionMask.dim(1) == 1)
        #expect(compiled.attentionMask.dim(2) == 6)
        #expect(compiled.attentionMask.dim(3) == 6)

        // Flatten to 6x6 and check a handful of positions.
        let flat = compiled.attentionMask
            .squeezed(axis: 0)
            .squeezed(axis: 0)
            .asArray(Float.self)
        // Helper: value at (row, col).
        func at(_ r: Int, _ c: Int) -> Float { flat[r * 6 + c] }
        // Diagonal is always 0.0 (self-visible).
        for i in 0..<6 {
            #expect(at(i, i) == 0.0, "diagonal at \(i) must be 0.0")
        }
        // Row 1 (parent 0) sees columns 0, 1.
        #expect(at(1, 0) == 0.0)
        #expect(at(1, 1) == 0.0)
        #expect(at(1, 2) == -Float.infinity)
        // Row 5 (parent 4, parent-of-parent 0) sees columns 0, 4, 5.
        #expect(at(5, 0) == 0.0)
        #expect(at(5, 4) == 0.0)
        #expect(at(5, 5) == 0.0)
        #expect(at(5, 1) == -Float.infinity)
        #expect(at(5, 3) == -Float.infinity)
    }

    @Test("dfs_order + inv_dfs_order are consistent permutations")
    func testDfsOrder() throws {
        let tree = try branchingTree()
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 10)
        let dfs = compiled.dfsOrder.asArray(Int32.self)
        let inv = compiled.invDfsOrder.asArray(Int32.self)
        #expect(dfs.count == 6)
        #expect(inv.count == 6)
        // Invariant: inv[dfs[i]] == i for all i.
        for i in 0..<6 {
            #expect(inv[Int(dfs[i])] == Int32(i))
        }
        // For this specific tree (children appended in heap-pop order),
        // DFS = [0, 1, 2, 3, 4, 5] (matches the builder hand-trace).
        #expect(dfs == [0, 1, 2, 3, 4, 5])
    }

    @Test("parents + treeSize pass through unchanged")
    func testParentsPassThrough() throws {
        let tree = try branchingTree()
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 10)
        #expect(compiled.parents == [-1, 0, 1, 1, 0, 4])
        #expect(compiled.treeSize == 6)
    }

    @Test("Empty tree compiles to a single root position")
    func testEmptyTree() throws {
        let empty = DDTree.empty()
        let compiled = try TreeCompile.compile(
            tree: empty, rootTokenID: 42, prefixLen: 7)
        #expect(compiled.treeSize == 1)
        #expect(compiled.inputIds.dim(1) == 1)
        let ids = compiled.inputIds.asType(.int32).asArray(Int32.self)
        #expect(ids == [42])
        let pos = compiled.positionIds.asArray(Int32.self)
        #expect(pos == [7])
        #expect(compiled.depths == [0])
        // 1x1 mask with just the self-visible entry.
        let flat = compiled.attentionMask
            .squeezed(axis: 0).squeezed(axis: 0)
            .asArray(Float.self)
        #expect(flat == [Float(0)])
    }

    @Test("prefix_len offset propagates to every position")
    func testPrefixLenOffset() throws {
        let tree = try branchingTree()
        let c0 = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 0)
        let c50 = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 50)
        let pos0 = c0.positionIds.asArray(Int32.self)
        let pos50 = c50.positionIds.asArray(Int32.self)
        for i in 0..<6 {
            #expect(pos50[i] - pos0[i] == 50)
        }
    }
}
