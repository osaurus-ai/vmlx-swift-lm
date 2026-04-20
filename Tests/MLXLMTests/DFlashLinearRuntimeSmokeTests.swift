// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 1 iter 4 smoke test — wire a tiny Qwen3 target to a tiny
// DFlashDraftModel through `SpecDecRuntimeLinear.run` and verify the
// loop completes without crashing on random weights.
//
// Byte-parity vs autoregressive is iter 5 work with real checkpoints.
// This iter just proves the plumbing — shapes, slicing, protocol
// conformance, control flow — is correct.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("DFlash linear runtime — Phase 1 smoke", .serialized)
struct DFlashLinearRuntimeSmokeTests {

    /// Tiny Qwen3 + matching DFlash drafter config pair. hidden_size,
    /// vocab_size, num_attention_heads, num_kv_heads, head_dim are
    /// shared so target's embedding / LM head sizes match what the
    /// drafter expects when its output flows through
    /// `target.projectToLogits(...)`.
    private static let hiddenSize = 128
    private static let numAttentionHeads = 4
    private static let numKVHeads = 2
    private static let headDim = 32
    private static let vocabSize = 512
    private static let targetLayers = 4
    private static let draftLayers = 2
    private static let targetBlockIDs = [0, 2]  // 2 of the 4 target blocks
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
        // targetBlockIDs is 0-based vmlx convention; HF stamp is 1-based.
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

    private func makeTarget() -> Qwen3Model {
        // MLX lazy evaluation materializes params on first forward;
        // no explicit eval needed here.
        Qwen3Model(targetConfig())
    }

    private func makeDrafter() -> DFlashDraftModel {
        DFlashDraftModel(drafterConfig())
    }

    @Test("SpecDecRuntimeLinear.run completes one loop on random weights")
    func testOneLoopCompletes() throws {
        let target = makeTarget()
        let drafter = makeDrafter()
        let prompt = tokens([1, 2, 3, 4, 5])
        let args = DFlashLinearArgs(
            target: target,
            drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: prompt,
            maxNewTokens: 4,
            stopTokenIDs: [],
            temperature: 0)

        let result = try SpecDecRuntimeLinear.run(args)

        // Prompt prefix must survive.
        for (i, t) in [1, 2, 3, 4, 5].enumerated() {
            #expect(result.tokenIds[i] == Int32(t))
        }
        // At least one round executed.
        #expect(result.rounds >= 1)
        // Acceptance lengths are in range [0, block_size - 1].
        for len in result.acceptanceLengths {
            #expect(len >= 0 && len <= Self.blockSize - 1)
        }
        // No mask tokens leak through.
        #expect(!result.tokenIds.contains(Self.maskTokenID))
    }

    @Test("runDflash produces at least maxNewTokens tokens past prompt")
    func testGeneratesEnoughTokens() throws {
        let target = makeTarget()
        let drafter = makeDrafter()
        let prompt = tokens([10, 20, 30])
        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: prompt, maxNewTokens: 6,
            stopTokenIDs: [], temperature: 0)
        let result = try SpecDecRuntimeLinear.run(args)
        let suffixLen = result.tokenIds.count - 3
        #expect(suffixLen >= 6,
            "Must produce at least maxNewTokens new tokens, got \(suffixLen)")
    }

    @Test("Stop tokens terminate generation")
    func testStopTokenHalts() throws {
        let target = makeTarget()
        let drafter = makeDrafter()
        let prompt = tokens([1, 2, 3])
        // Stop on any vocab token — random weights will land on SOME
        // token within the first round, so the loop must exit early.
        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: prompt, maxNewTokens: 100,
            stopTokenIDs: Set((0..<Int32(Self.vocabSize)).map { $0 }),
            temperature: 0)
        let result = try SpecDecRuntimeLinear.run(args)
        #expect(result.tokenIds.count < 3 + 100)
    }

    @Test("SpecDecRuntime actor delegates to SpecDecRuntimeLinear")
    func testActorDelegate() async throws {
        let target = makeTarget()
        let drafter = makeDrafter()
        let prompt = tokens([1, 2])
        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: Self.targetBlockIDs,
            maskTokenID: Self.maskTokenID,
            inputIds: prompt, maxNewTokens: 4,
            stopTokenIDs: [], temperature: 0)

        let runtime = SpecDecRuntime(config: .init(
            strategy: .dflash(
                drafterPath: URL(fileURLWithPath: "/placeholder"),
                blockSize: Self.blockSize),
            parameters: GenerateParameters()))

        let r1 = try await runtime.runDflash(args)
        let r2 = try SpecDecRuntime.executeDflashLinear(args)
        #expect(r1.tokenIds.count > 0)
        #expect(r2.tokenIds.count > 0)
    }
}
