// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 2 iter 8 — pin the tree-verify correctness invariant on
// random-weight Qwen3 targets.
//
// Invariant: `verifyForward.posteriorTokens[i]` MUST equal plain
// greedy-AR's argmax at absolute position `prefix_len + depth[i]`
// when the target is fed `(prefix + path_from_root_to_node_i)`.
//
// This is the tree-verify analogue of iter 5's DFlash byte-parity —
// same underlying invariant (greedy argmax on a canonical prefix), now
// extended over tree paths rather than a linear block.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DDTree verify — Phase 2", .serialized)
struct DDTreeVerifyTests {

    private static let hiddenSize = 128
    private static let numAttentionHeads = 4
    private static let numKVHeads = 2
    private static let headDim = 32
    private static let vocabSize = 512
    private static let targetLayers = 4

    private func targetConfig() -> Qwen3Configuration {
        let json = """
        {
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.targetLayers),
          "intermediate_size": 256,
          "num_attention_heads": \(Self.numAttentionHeads),
          "rms_norm_eps": 1e-6,
          "vocab_size": \(Self.vocabSize),
          "num_key_value_heads": \(Self.numKVHeads),
          "rope_theta": 1000000,
          "head_dim": \(Self.headDim),
          "tie_word_embeddings": false,
          "max_position_embeddings": 512
        }
        """
        return try! JSONDecoder().decode(
            Qwen3Configuration.self, from: json.data(using: .utf8)!)
    }

    /// Greedy argmax reference — target's prediction for the next token
    /// given the full sequence, no cache.
    private func targetArgmax(
        _ target: Qwen3Model, _ seq: [Int32]
    ) -> Int32 {
        let input = MLXArray(seq).reshaped(1, seq.count)
        let logits = target(input, cache: nil)
        let last = logits[0, logits.dim(1) - 1, 0...]
        return argMax(last, axis: -1).asType(.int32).item(Int32.self)
    }

    // MARK: - Linear tree (branching=1) equivalent to plain AR

    @Test("Linear tree (budget=1) posterior matches greedy AR argmax")
    func testLinearTreePosteriorMatchesAR() throws {
        MLXRandom.seed(0x1234)
        let target = Qwen3Model(targetConfig())

        // Build a linear tree by giving budget=1 on top-K=1 per position.
        // Node tokens are arbitrary — verify's job is to tell us what the
        // target WOULD predict at each position regardless of input.
        let linearTokens = MLXArray([Int32(42), 43, 44, 45]).reshaped(4, 1)
        let linearLogProbs = MLXArray([Float(0), 0, 0, 0]).reshaped(4, 1)
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: linearTokens,
            topLogProbs: linearLogProbs,
            budget: 3)
        #expect(tree.nodeCount == 3)

        // Compile with a bonus-token root and no prefix.
        let rootToken: Int32 = 7
        let prefix: [Int32] = [1, 2, 3]
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: rootToken, prefixLen: prefix.count)

        let result = try TreeVerify.verifyForward(
            target: target,
            compiled: compiled,
            prefixTokens: prefix)

        // Check posterior[i] == target-argmax over (prefix + path-to-i).
        let inputIds = compiled.inputIds.asType(.int32).asArray(Int32.self)
        for i in 0..<compiled.treeSize {
            // Walk parents to build path.
            var path: [Int32] = []
            var cursor = i
            while cursor != 0 {
                path.append(inputIds[cursor])
                cursor = Int(compiled.parents[cursor])
            }
            path.append(inputIds[0])
            path.reverse()
            let full = prefix + path
            let expected = targetArgmax(target, full)
            #expect(result.posteriorTokens[i] == expected,
                "posterior mismatch at tree node \(i)")
        }
    }

    // MARK: - Branching tree per-node posterior consistency

    @Test("Branching tree per-node posterior matches direct path argmax")
    func testBranchingTreePerNodePosterior() throws {
        MLXRandom.seed(0xBABE)
        let target = Qwen3Model(targetConfig())

        // 2-depth branching tree (same shape as DDTreeBuilderTests).
        let tokenIds = MLXArray([Int32(10), 20, 30, 40]).reshaped(2, 2)
        let logProbs = MLXArray([Float(0), -1, 0, -1]).reshaped(2, 2)
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 5)
        let rootToken: Int32 = 99
        let prefix: [Int32] = [5, 5, 5]

        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: rootToken, prefixLen: prefix.count)

        let result = try TreeVerify.verifyForward(
            target: target, compiled: compiled, prefixTokens: prefix)

        #expect(result.posteriorTokens.count == 6)
        let inputIds = compiled.inputIds.asType(.int32).asArray(Int32.self)

        // Hand-compute path for each node, compare argmax.
        for i in 0..<6 {
            var path: [Int32] = []
            var cursor = i
            while cursor != 0 {
                path.append(inputIds[cursor])
                cursor = Int(compiled.parents[cursor])
            }
            path.append(inputIds[0])
            path.reverse()
            let expected = targetArgmax(target, prefix + path)
            #expect(result.posteriorTokens[i] == expected,
                "per-node mismatch at tree index \(i)")
        }
    }

    // MARK: - Empty tree edge case

    @Test("Empty tree returns single posterior for the root only")
    func testEmptyTreeSingleRootPosterior() throws {
        MLXRandom.seed(0xFACE)
        let target = Qwen3Model(targetConfig())

        let empty = DDTree.empty()
        let compiled = try TreeCompile.compile(
            tree: empty, rootTokenID: 7, prefixLen: 2)
        let prefix: [Int32] = [1, 2]
        let result = try TreeVerify.verifyForward(
            target: target, compiled: compiled, prefixTokens: prefix)
        #expect(result.posteriorTokens.count == 1)
        let expected = targetArgmax(target, prefix + [7])
        #expect(result.posteriorTokens[0] == expected)
    }

    // MARK: - Logits tensor shape

    @Test("verify result logits has shape (1, treeSize, vocab)")
    func testLogitsShape() throws {
        MLXRandom.seed(0xA5A5)
        let target = Qwen3Model(targetConfig())
        let tokenIds = MLXArray([Int32(10), 20]).reshaped(1, 2)
        let logProbs = MLXArray([Float(0), -1]).reshaped(1, 2)
        let tree = try TreeBuilder.buildFromTopK(
            topTokenIds: tokenIds, topLogProbs: logProbs, budget: 2)
        let compiled = try TreeCompile.compile(
            tree: tree, rootTokenID: 99, prefixLen: 0)
        let result = try TreeVerify.verifyForward(
            target: target, compiled: compiled, prefixTokens: [])
        #expect(result.logits.ndim == 3)
        #expect(result.logits.dim(0) == 1)
        #expect(result.logits.dim(1) == compiled.treeSize)
        #expect(result.logits.dim(2) == Self.vocabSize)
    }
}
