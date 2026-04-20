// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Edge cases for SpecDecRuntimeLinear + SpecDecRuntimeDDTree:
//  - maxNewTokens less than blockSize (runtime commits a full block
//    but must trim back to exactly the requested budget)
//  - stop token committed mid-block (result must truncate at the first
//    stop and not leak any tokens committed after it in the same round)
//  - prompt length 1 (degenerate prefill)
//  - maxNewTokens == 1 (only the prefill bonus, no decode rounds)
//
// These used to silently overshoot / leak before the iter-after-merge
// fixes — tests pin the contract going forward.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DFlash/DDTree edge cases", .serialized)
struct DFlashEdgeCaseTests {

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
          "model_type": "qwen3",
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.targetLayers),
          "intermediate_size": 256,
          "num_attention_heads": \(Self.numAttentionHeads),
          "num_key_value_heads": \(Self.numKVHeads),
          "head_dim": \(Self.headDim),
          "rms_norm_eps": 1e-6,
          "vocab_size": \(Self.vocabSize),
          "rope_theta": 1000000,
          "max_position_embeddings": 512,
          "tie_word_embeddings": false,
          "attention_bias": false
        }
        """
        return try! JSONDecoder().decode(
            Qwen3Configuration.self, from: json.data(using: .utf8)!)
    }

    private func drafterConfig() -> DFlashDrafterConfiguration {
        let hfStampIDs = Self.targetBlockIDs
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

    // MARK: - maxNewTokens shorter than blockSize

    @Test("DFlash respects maxNewTokens < blockSize")
    func testMaxNewTokensSmallerThanBlock() throws {
        MLXRandom.seed(0xED_E1)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [3, 5, 7]
        let maxNew = 2  // blockSize = 4, so must trim

        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0)
        let result = try SpecDecRuntimeLinear.run(args)

        // Result must be exactly prompt_len + maxNew tokens — no overshoot.
        #expect(result.tokenIds.count == prompt.count + maxNew)
        // Prompt prefix preserved.
        #expect(Array(result.tokenIds.prefix(prompt.count)) == prompt)
    }

    @Test("DDTree respects maxNewTokens < blockSize")
    func testDDTreeMaxNewTokensSmallerThanBlock() throws {
        MLXRandom.seed(0xED_E2)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [3, 5, 7]
        let maxNew = 2

        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0,
            branchingBudget: 4)
        let result = try SpecDecRuntimeDDTree.run(args)
        #expect(result.tokenIds.count == prompt.count + maxNew)
        #expect(Array(result.tokenIds.prefix(prompt.count)) == prompt)
    }

    // MARK: - Very small / degenerate cases

    @Test("DFlash with maxNewTokens = 1 only emits prefill bonus")
    func testMaxNewTokensOne() throws {
        MLXRandom.seed(0xED_E3)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [3, 5, 7]

        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: 1,
            stopTokenIDs: [], temperature: 0)
        let result = try SpecDecRuntimeLinear.run(args)
        #expect(result.tokenIds.count == prompt.count + 1)
        // No decode rounds should have run — loop exits immediately.
        #expect(result.acceptanceLengths.isEmpty)
    }

    @Test("DFlash with prompt length 1")
    func testShortPromptLength() throws {
        MLXRandom.seed(0xED_E4)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())

        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens([42]), maxNewTokens: 4,
            stopTokenIDs: [], temperature: 0)
        let result = try SpecDecRuntimeLinear.run(args)
        #expect(result.tokenIds.count == 1 + 4)
        #expect(result.tokenIds[0] == 42)
    }

    // MARK: - Stop token truncation

    @Test("DFlash truncates output at first stop token")
    func testStopTokenTruncation() throws {
        MLXRandom.seed(0xED_E5)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [1, 2]

        // First compute what AR would emit to pick a stop token that
        // we'll deterministically hit.
        var arTokens = prompt
        for _ in 0..<8 {
            let input = tokens(arTokens)
            let logits = target(input, cache: nil)
            let last = logits[0, logits.dim(1) - 1, 0...]
            let t = argMax(last, axis: -1).asType(.int32).item(Int32.self)
            arTokens.append(t)
        }
        // Pick the token at position prompt.count + 2 as the stop.
        let stop = arTokens[prompt.count + 2]

        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: 8,
            stopTokenIDs: [stop], temperature: 0)
        let result = try SpecDecRuntimeLinear.run(args)

        // Must terminate at or before maxNew, and result must END on the
        // stop token (no tokens leaked after the stop).
        #expect(result.tokenIds.count <= prompt.count + 8)
        #expect(result.tokenIds.last == stop)
        // And there can only be ONE stop token in the output.
        let stopCount = result.tokenIds.filter { $0 == stop }.count
        #expect(stopCount == 1)
    }
}
