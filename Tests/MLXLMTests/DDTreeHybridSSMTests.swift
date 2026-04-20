// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 3 iter 15 — pin DDTree byte-parity on hybrid SSM targets.
// Closes the final test-matrix row so completion criterion 1 is
// satisfied.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@inline(__always)
private func materialize(_ a: MLXArray) { MLX.eval(a) }

@Suite("DDTree hybrid SSM byte-parity — Phase 3", .serialized)
struct DDTreeHybridSSMTests {

    private static let hiddenSize = 128
    private static let numAttentionHeads = 4
    private static let numKVHeads = 2
    private static let vocabSize = 512
    private static let hiddenLayers = 4
    private static let draftLayers = 2
    private static let targetBlockIDs = [0, 2]  // mix of SSM + attention blocks
    private static let blockSize = 4
    private static let maskTokenID: Int32 = 500

    private func tokens(_ values: [Int32]) -> MLXArray {
        MLXArray(values).reshaped(1, values.count)
    }

    /// Tiny Qwen35 config — interleaved linear + full-attention layers
    /// via `full_attention_interval=2` (every other layer is attention).
    private func targetConfig() -> Qwen35TextConfiguration {
        let json = """
        {
          "model_type": "qwen3_5",
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.hiddenLayers),
          "intermediate_size": 256,
          "num_attention_heads": \(Self.numAttentionHeads),
          "num_key_value_heads": \(Self.numKVHeads),
          "linear_num_value_heads": 4,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 32,
          "linear_value_head_dim": 32,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": \(Self.vocabSize),
          "rope_theta": 100000.0,
          "partial_rotary_factor": 0.25,
          "max_position_embeddings": 512,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "head_dim": 32,
          "full_attention_interval": 2
        }
        """
        return try! JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    private func drafterConfig() -> DFlashDrafterConfiguration {
        let hfStampIDs = Self.targetBlockIDs.map { $0 + 1 }
        let json = """
        {
          "hidden_size": \(Self.hiddenSize),
          "num_hidden_layers": \(Self.draftLayers),
          "num_attention_heads": \(Self.numAttentionHeads),
          "num_key_value_heads": \(Self.numKVHeads),
          "head_dim": 32,
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
        target: Qwen35TextModel,
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

    // MARK: - Protocol conformance

    @Test("Qwen35TextModel conforms to HiddenStateCaptureModel + TokenEmbedderModel")
    func testConformance() {
        MLXRandom.seed(0x31_31_35)
        let target = Qwen35TextModel(targetConfig())
        let asHC: Any = target
        let asEmb: Any = target
        #expect(asHC is HiddenStateCaptureModel)
        #expect(asEmb is TokenEmbedderModel)
    }

    @Test("Qwen35 empty captureLayerIDs is byte-identical to plain forward")
    func testEmptyCaptureByteIdentical() {
        MLXRandom.seed(0x31_31_36)
        let target = Qwen35TextModel(targetConfig())
        let input = tokens([1, 2, 3, 4, 5])
        let plain = target(input, cache: nil)
        let (captured, states) = target(
            input, cache: nil, captureLayerIDs: [])
        #expect(states.isEmpty)
        let eq = equal(plain, captured).all()
        materialize(eq)
        #expect(eq.item(Bool.self))
    }

    @Test("Qwen35 capturing the mixed-layer set fills expected keys")
    func testCaptureAcrossSSMAndAttention() {
        MLXRandom.seed(0x31_31_37)
        let target = Qwen35TextModel(targetConfig())
        let input = tokens([1, 2, 3])
        let (_, states) = target(
            input, cache: nil, captureLayerIDs: [0, 1])
        #expect(Set(states.keys) == Set([0, 1]))
        for (_, h) in states {
            #expect(h.ndim == 3)
            #expect(h.dim(0) == 1)
            #expect(h.dim(1) == 3)
            #expect(h.dim(2) == Self.hiddenSize)
        }
    }

    // MARK: - End-to-end byte-parity

    @Test("DDTree on Qwen35 hybrid SSM == greedy AR")
    func testDDTreeHybridSSMByteParity() throws {
        MLXRandom.seed(0x9A_9B_9C)
        let target = Qwen35TextModel(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [3, 5, 7, 11]
        let maxNew = 5

        let arTokens = argmaxGreedyAR(
            target: target, promptTokens: prompt, maxNewTokens: maxNew)

        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0,
            branchingBudget: 4)
        let ddResult = try SpecDecRuntimeDDTree.run(args)
        let truncated = Array(ddResult.tokenIds.prefix(arTokens.count))
        #expect(truncated == arTokens)
    }

    @Test("DFlash linear on Qwen35 hybrid SSM == greedy AR")
    func testDFlashLinearHybridSSMByteParity() throws {
        MLXRandom.seed(0x9A_9B_9D)
        let target = Qwen35TextModel(targetConfig())
        let drafter = DFlashDraftModel(drafterConfig())
        let prompt: [Int32] = [2, 4, 6]
        let maxNew = 4

        let arTokens = argmaxGreedyAR(
            target: target, promptTokens: prompt, maxNewTokens: maxNew)

        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: tokens(prompt), maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0)
        let dfResult = try SpecDecRuntimeLinear.run(args)
        let truncated = Array(dfResult.tokenIds.prefix(arTokens.count))
        #expect(truncated == arTokens)
    }
}
