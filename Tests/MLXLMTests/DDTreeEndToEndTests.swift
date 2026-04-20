// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 2 iter 9 — end-to-end DDTree byte-parity pin.
//
// Same invariant as iter 5's DFlash byte-parity, extended over trees:
// at temperature 0, `SpecDecRuntimeDDTree.run` must produce output
// byte-identical to plain greedy autoregressive decode. The tree
// `follow_verified_tree` walker only accepts nodes that match target
// argmax, so the committed token sequence is exactly what AR would
// produce regardless of drafter / tree branching.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DDTree end-to-end byte-parity vs AR — Phase 2", .serialized)
struct DDTreeEndToEndTests {

    private static let hiddenSize = 128
    private static let numAttentionHeads = 4
    private static let numKVHeads = 2
    private static let headDim = 32
    private static let vocabSize = 512
    private static let targetLayers = 4
    private static let draftLayers = 2
    private static let targetBlockIDs = [0, 2]
    private static let blockSize = 4
    private static let maskTokenID: Int32 = 500

    private func tokens(_ values: [Int32]) -> MLXArray {
        MLXArray(values).reshaped(1, values.count)
    }

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

    private func drafterConfig() -> DFlashDrafterConfiguration {
        let hfStampIDs = Self.targetBlockIDs.map { $0 + 1 }
        let json = """
        {
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.draftLayers),
          "num_attention_heads": \(Self.numAttentionHeads),
          "num_key_value_heads": \(Self.numKVHeads),
          "head_dim": \(Self.headDim),
          "intermediate_size": 256,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "max_position_embeddings": 512,
          "attention_bias": false,
          "hidden_act": "silu",
          "block_size": \(Self.blockSize),
          "dflash_config": {
            "mask_token_id": \(Self.maskTokenID),
            "target_layer_ids": \(hfStampIDs)
          }
        }
        """
        return try! JSONDecoder().decode(
            DFlashDrafterConfiguration.self, from: json.data(using: .utf8)!)
    }

    private func argmaxGreedyAR(
        target: Qwen3Model,
        promptTokens: [Int32],
        maxNewTokens: Int
    ) -> [Int32] {
        var out = promptTokens
        for _ in 0..<maxNewTokens {
            let input = tokens(out)
            let logits = target(input, cache: nil)
            let last = logits[0, logits.dim(1) - 1, 0...]
            let next = argMax(last, axis: -1).asType(.int32).item(Int32.self)
            out.append(next)
        }
        return out
    }

    @Test("DDTree (branching=4) output == greedy AR output")
    func testBranching4ByteParity() throws {
        MLXRandom.seed(0xD1D7E2)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())

        let promptTokens: [Int32] = [3, 5, 7, 11, 13]
        let maxNewTokens = 6

        let arTokens = argmaxGreedyAR(
            target: target, promptTokens: promptTokens,
            maxNewTokens: maxNewTokens)

        let args = DDTreeArgs(
            target: target,
            drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(promptTokens),
            maxNewTokens: maxNewTokens,
            stopTokenIDs: [],
            temperature: 0,
            branchingBudget: 4)
        let ddResult = try SpecDecRuntimeDDTree.run(args)

        let truncated = Array(ddResult.tokenIds.prefix(arTokens.count))
        #expect(truncated == arTokens,
            "DDTree (branching=4) must byte-match greedy AR.")
    }

    @Test("DDTree (branching=1, linear) output == greedy AR output")
    func testBranching1ByteParity() throws {
        MLXRandom.seed(0xDEADBEEF)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())

        let promptTokens: [Int32] = [2, 4, 6, 8]
        let maxNewTokens = 6

        let arTokens = argmaxGreedyAR(
            target: target, promptTokens: promptTokens,
            maxNewTokens: maxNewTokens)

        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(promptTokens),
            maxNewTokens: maxNewTokens,
            stopTokenIDs: [], temperature: 0,
            branchingBudget: 1)
        let ddResult = try SpecDecRuntimeDDTree.run(args)

        let truncated = Array(ddResult.tokenIds.prefix(arTokens.count))
        #expect(truncated == arTokens,
            "DDTree (branching=1) must byte-match greedy AR.")
    }

    @Test("DDTree byte-parity holds across multiple prompts + budgets")
    func testMultipleConfigsByteParity() throws {
        let configs: [(prompt: [Int32], maxNew: Int, budget: Int, seed: UInt64)] = [
            ([1, 2, 3], 5, 2, 0xAAAA),
            ([100, 200], 4, 8, 0xBBBB),
            ([42, 43, 44, 45, 46], 7, 6, 0xCCCC),
        ]
        for (idx, c) in configs.enumerated() {
            MLXRandom.seed(c.seed)
            let target = Qwen3Model(targetConfig())
            let drafter = DFlashDraftModel(drafterConfig())

            let arTokens = argmaxGreedyAR(
                target: target, promptTokens: c.prompt,
                maxNewTokens: c.maxNew)
            let args = DDTreeArgs(
                target: target, drafter: drafter,
                targetBlockIDs: Self.targetBlockIDs,
                maskTokenID: Self.maskTokenID,
                inputIds: tokens(c.prompt),
                maxNewTokens: c.maxNew,
                stopTokenIDs: [], temperature: 0,
                branchingBudget: c.budget)
            let ddResult = try SpecDecRuntimeDDTree.run(args)
            let truncated = Array(ddResult.tokenIds.prefix(arTokens.count))
            #expect(truncated == arTokens,
                "config idx \(idx): DDTree != AR")
        }
    }

    @Test("DDTree result exposes valid acceptance metric")
    func testAcceptanceMetric() throws {
        MLXRandom.seed(0xABC)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())

        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens([1, 2, 3]),
            maxNewTokens: 4, stopTokenIDs: [], temperature: 0,
            branchingBudget: 4)
        let ddResult = try SpecDecRuntimeDDTree.run(args)
        #expect(ddResult.rounds >= 1)
        // Each round accepts 0..budget tokens.
        for len in ddResult.acceptanceLengths {
            #expect(len >= 0 && len <= 4)
        }
        let mean = ddResult.meanAcceptanceLength
        #expect(mean >= 0.0 && mean <= 4.0)
    }
}
