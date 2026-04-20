// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 1 iter 5 — byte-parity between SpecDecRuntimeLinear.run and
// plain greedy autoregressive decode on the same target.
//
// Greedy (temperature = 0) DFlash decoding MUST produce the same tokens
// as plain greedy AR because:
//
//   1. At every position, plain AR picks token = argmax(target_logits).
//   2. DFlash's drafter proposes tokens d_1, d_2, …, d_{B-1}.
//   3. Target verifies on the full block, producing posterior[i] =
//      argmax(target_logits) given the preceding tokens — this is
//      exactly what plain AR would pick.
//   4. DFlash accepts d_i only if d_i == posterior[i-1], i.e. the
//      drafter happened to match plain-AR. The first mismatch truncates
//      acceptance and DFlash commits posterior[acceptance] as the next
//      bonus (which is also what plain AR would pick).
//
// So the committed token sequence is identical regardless of what the
// drafter proposed. This test pins the invariant with random weights
// (deterministic seeded init), so it runs without requiring real
// checkpoints and independent of drafter quality.
//
// Iter 6 layers on a real-checkpoint end-to-end test with speedup
// measurement — that's value-add, but correctness is pinned here.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DFlash linear byte-parity vs AR — Phase 1", .serialized)
struct DFlashLinearByteParityTests {

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

    /// Deterministic greedy autoregressive decode against the reference
    /// target — the ground truth DFlash must match byte-for-byte.
    ///
    /// Runs target on the full growing sequence each step (no KV cache
    /// optimization). Equivalent to a naive `for _ in 0..<N` AR loop.
    private func argmaxGreedyAR(
        target: Qwen3Model,
        promptTokens: [Int32],
        maxNewTokens: Int
    ) -> [Int32] {
        var out = promptTokens
        for _ in 0..<maxNewTokens {
            let input = tokens(out)
            let logits = target(input, cache: nil)
            // argmax of last-position logits.
            let last = logits[0, logits.dim(1) - 1, 0...]
            let nextArr = argMax(last, axis: -1).asType(.int32)
            let next = nextArr.item(Int32.self)
            out.append(next)
        }
        return out
    }

    @Test("DFlash linear output == greedy AR output on random weights")
    func testByteParity() throws {
        // Seed MLX RNG so target + drafter init deterministically.
        MLXRandom.seed(0xD1_F1_A5_42)
        let target = Qwen3Model(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())

        let promptTokens: [Int32] = [7, 11, 13, 17, 19, 23]
        let maxNewTokens = 8

        let arTokens = argmaxGreedyAR(
            target: target, promptTokens: promptTokens,
            maxNewTokens: maxNewTokens)

        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(promptTokens),
            maxNewTokens: maxNewTokens,
            stopTokenIDs: [], temperature: 0)
        let dfResult = try SpecDecRuntimeLinear.run(args)

        // DFlash may overshoot maxNewTokens by up to blockSize tokens
        // because it commits a whole block + bonus per round. Truncate
        // for the comparison.
        let truncated = Array(
            dfResult.tokenIds.prefix(arTokens.count))

        #expect(truncated == arTokens,
            "DFlash must produce byte-identical output to greedy AR.")
    }

    @Test("Byte-parity holds across multiple prompt lengths")
    func testByteParityMultiplePrompts() throws {
        let prompts: [[Int32]] = [
            [1, 2, 3],
            [100, 200],
            [42, 43, 44, 45, 46, 47, 48, 49, 50],
        ]
        let maxNewTokens = 6

        for (idx, prompt) in prompts.enumerated() {
            // Re-seed per test so target + drafter reset identically.
            MLXRandom.seed(UInt64(0xBADC0FFEE + idx))
            let target = Qwen3Model(targetConfig())
            let drafter = DFlashDraftModel(drafterConfig())

            let arTokens = argmaxGreedyAR(
                target: target, promptTokens: prompt,
                maxNewTokens: maxNewTokens)

            let args = DFlashLinearArgs(
                target: target, drafter: drafter,
                targetBlockIDs: Self.targetBlockIDs,
                maskTokenID: Self.maskTokenID,
                inputIds: tokens(prompt),
                maxNewTokens: maxNewTokens,
                stopTokenIDs: [], temperature: 0)
            let dfResult = try SpecDecRuntimeLinear.run(args)

            let truncated = Array(
                dfResult.tokenIds.prefix(arTokens.count))
            #expect(truncated == arTokens,
                "Prompt idx \(idx): byte-parity failed.")
        }
    }
}
