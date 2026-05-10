// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Recognition gate + Hy3Configuration decode regression for Tencent
// Hunyuan v3. Mirrors the ZAYA1-VL Phase A test pattern (`.serialized`
// suite, real-bundle `.enabled(if:)` decode).
//
// Native Hy3Model + Hy3Attention + Hy3MoE support is expected to exist.
// This suite locks the contract: bundle config decodes cleanly and factory
// dispatch returns a Hy3 model instead of silently routing through Qwen /
// DSV3 / Zaya or throwing the old Phase-A recognition-gate error.

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

private enum Hy3LocalBundles {
    private static let modelRoot =
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models")

    static let jangtq2Candidates = [
        modelRoot.appendingPathComponent("JANGQ/Hy3-preview-JANGTQ"),
        modelRoot.appendingPathComponent("JANGQ/Hy3-preview-JANGTQ2"),
    ]

    static let sourceBundle = modelRoot.appendingPathComponent("Tencent/Hy3-preview")
    static let jangtq1Bundle = modelRoot.appendingPathComponent("JANGQ/Hy3-preview-JANGTQ1")

    static var jangtq2: URL? {
        jangtq2Candidates.first { bundle in
            FileManager.default.fileExists(
                atPath: bundle.appending(component: "config.json").path)
        }
    }

    static var hasJangtq2Config: Bool { jangtq2 != nil }

    static var hasJangtq2Index: Bool {
        guard let bundle = jangtq2 else { return false }
        return FileManager.default.fileExists(
            atPath: bundle.appending(component: "model.safetensors.index.json").path)
    }
}

@Suite("Hy3 (Tencent Hunyuan v3) recognition and config decode", .serialized)
struct Hy3RegistrationTests {

    @Test("LLM registry creates a native Hy3 model for hy_v3")
    func registryRecognizesHy3() async throws {
        let data = """
            {
              "model_type": "hy_v3",
              "architectures": ["HYV3ForCausalLM"],
              "hidden_size": 4096,
              "num_hidden_layers": 80,
              "num_attention_heads": 64,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "intermediate_size": 13312,
              "moe_intermediate_size": 1536,
              "expert_hidden_dim": 1536,
              "first_k_dense_replace": 1,
              "num_experts": 192,
              "num_experts_per_tok": 8,
              "num_shared_experts": 1,
              "qk_norm": true,
              "rms_norm_eps": 1e-5,
              "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
              "max_position_embeddings": 262144,
              "route_norm": true,
              "router_scaling_factor": 2.826,
              "moe_router_enable_expert_bias": true,
              "moe_router_use_sigmoid": true,
              "tie_word_embeddings": false,
              "vocab_size": 120832,
              "num_nextn_predict_layers": 1
            }
            """.data(using: .utf8)!

        let modelTypeName = try await MLXMetalTestLock.withLock {
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "hy_v3")
            return String(describing: type(of: model))
        }
        #expect(modelTypeName.contains("Hy3"))
    }

    @Test("Hy3Configuration decodes the canonical field set without defaults")
    func configDecodesCanonicalFields() throws {
        let data = """
            {
              "model_type": "hy_v3",
              "architectures": ["HYV3ForCausalLM"],
              "hidden_size": 4096,
              "num_hidden_layers": 80,
              "num_attention_heads": 64,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "intermediate_size": 13312,
              "moe_intermediate_size": 1536,
              "expert_hidden_dim": 1536,
              "first_k_dense_replace": 1,
              "num_experts": 192,
              "num_experts_per_tok": 8,
              "num_shared_experts": 1,
              "qk_norm": true,
              "rms_norm_eps": 1e-5,
              "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
              "max_position_embeddings": 262144,
              "route_norm": true,
              "router_scaling_factor": 2.826,
              "moe_router_enable_expert_bias": true,
              "moe_router_use_sigmoid": true,
              "tie_word_embeddings": false,
              "vocab_size": 120832,
              "hidden_act": "silu",
              "num_nextn_predict_layers": 1
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Hy3Configuration.self, from: data)

        #expect(config.modelType == "hy_v3")
        #expect(config.architectures == ["HYV3ForCausalLM"])
        #expect(config.hiddenSize == 4096)
        #expect(config.numHiddenLayers == 80)
        #expect(config.numAttentionHeads == 64)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.numKeyValueGroups == 8)  // 64 / 8
        #expect(config.intermediateSize == 13312)
        #expect(config.moeIntermediateSize == 1536)
        #expect(config.expertHiddenDim == 1536)
        #expect(config.firstKDenseReplace == 1)
        #expect(config.numExperts == 192)
        #expect(config.numExpertsPerTok == 8)
        #expect(config.numSharedExperts == 1)
        #expect(config.qkNorm)
        #expect(abs(config.rmsNormEps - 1e-5) < 1e-9)
        #expect(config.ropeParameters.ropeTheta == 11158840.0)
        #expect(config.ropeParameters.ropeType == "default")
        #expect(config.maxPositionEmbeddings == 262144)
        #expect(config.routeNorm)
        #expect(abs(config.routerScalingFactor - 2.826) < 1e-6)
        #expect(config.moeRouterEnableExpertBias)
        #expect(config.moeRouterUseSigmoid)
        #expect(!config.tieWordEmbeddings)
        #expect(config.vocabSize == 120832)
        #expect(config.numNextnPredictLayers == 1)
        #expect(config.hiddenAct == "silu")
    }

    @Test("Real local Hy3 source bundle decodes if present",
          .enabled(if: FileManager.default.fileExists(
              atPath: Hy3LocalBundles.sourceBundle.appending(component: "config.json").path)))
    func realSourceBundleDecodes() throws {
        let url = Hy3LocalBundles.sourceBundle.appending(component: "config.json")
        let data = try Data(contentsOf: url)
        let config = try JSONDecoder.json5().decode(Hy3Configuration.self, from: data)

        #expect(config.modelType == "hy_v3")
        #expect(config.architectures == ["HYV3ForCausalLM"])
        #expect(config.numHiddenLayers == 80)
        #expect(config.numAttentionHeads == 64)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.numExperts == 192)
        #expect(config.numExpertsPerTok == 8)
        #expect(config.numSharedExperts == 1)
        #expect(config.firstKDenseReplace == 1)
        #expect(config.qkNorm)
        #expect(config.routeNorm)
        #expect(config.moeRouterEnableExpertBias)
        #expect(config.moeRouterUseSigmoid)
        #expect(abs(config.routerScalingFactor - 2.826) < 1e-6)
        #expect(config.ropeParameters.ropeTheta == 11158840.0)
        #expect(config.maxPositionEmbeddings == 262144)
        #expect(config.numNextnPredictLayers == 1)
    }

    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    @Test("Real local Hy3 JANGTQ2 bundle safetensors index matches Phase B native adapter requirements",
          .enabled(if: Hy3LocalBundles.hasJangtq2Index))
    func jangtq2SafetensorsIndexStructureMatchesPhaseBRequirements() throws {
        let bundle = try #require(Hy3LocalBundles.jangtq2)
        let data = try Data(contentsOf: bundle.appending(
            component: "model.safetensors.index.json"))
        let index = try JSONDecoder.json5().decode(SafetensorsIndex.self, from: data)
        let keys = Set(index.weightMap.keys)

        // Top-level layout
        #expect(keys.contains("model.embed_tokens.weight"))
        #expect(keys.contains("lm_head.weight"))
        #expect(keys.contains("model.norm.weight"))

        // Layer count: 80 declared (num_hidden_layers=80) + 1 MTP at layer 80
        // (num_nextn_predict_layers=1) — total 81 distinct layer indices.
        // Phase B decoder must allocate 81 layer slots.
        let layerIndices = Set(keys.compactMap { key -> Int? in
            guard key.hasPrefix("model.layers.") else { return nil }
            let suffix = key.dropFirst("model.layers.".count)
            guard let dotIndex = suffix.firstIndex(of: ".") else { return nil }
            return Int(suffix[suffix.startIndex..<dotIndex])
        })
        #expect(layerIndices.min() == 0)
        #expect(layerIndices.max() == 80)
        #expect(layerIndices.count == 81)

        // Layer 0 is dense FFN per first_k_dense_replace=1 — no experts.
        let layer0Keys = keys.filter { $0.hasPrefix("model.layers.0.mlp.") }
        #expect(!layer0Keys.contains { $0.contains(".experts.") },
            "Layer 0 must be dense FFN, no experts (first_k_dense_replace=1)")
        #expect(layer0Keys.contains { $0.hasPrefix("model.layers.0.mlp.gate_proj.") })
        #expect(layer0Keys.contains { $0.hasPrefix("model.layers.0.mlp.up_proj.") })
        #expect(layer0Keys.contains { $0.hasPrefix("model.layers.0.mlp.down_proj.") })

        // Sparse MoE layers 1..79: 192 experts, JANGTQ2 routed sidecars
        // (tq_packed/tq_norms/tq_bits) AND shared_mlp branch.
        for layer in 1..<80 {
            let prefix = "model.layers.\(layer).mlp."
            let layerKeys = keys.filter { $0.hasPrefix(prefix) }
            #expect(layerKeys.contains { $0.contains(".experts.0.gate_proj.tq_packed") },
                "Layer \(layer) MoE must have JANGTQ2 routed-expert tq_packed sidecar")
            #expect(layerKeys.contains { $0.contains(".experts.191.down_proj.tq_packed") },
                "Layer \(layer) must have all 192 experts (top-1 indexed up to 191)")
            #expect(layerKeys.contains { $0.contains(".shared_mlp.") },
                "Layer \(layer) must have shared_mlp branch (num_shared_experts=1)")
        }

        // Layer 80 = MTP layer — affine 8-bit (not tq_packed) per
        // mxtq_bits.mtp=8 in jang_config. Has MoE structure but no JANGTQ2.
        let layer80Keys = keys.filter { $0.hasPrefix("model.layers.80.") }
        #expect(!layer80Keys.contains { $0.contains("tq_packed") },
            "Layer 80 (MTP) must use affine 8-bit, not tq_packed routed quant")
        #expect(layer80Keys.contains { $0.contains("experts.0.") },
            "Layer 80 retains MoE expert structure even at 8-bit")

        // Attention layers: every layer 0..80 has self_attn with q_norm + k_norm
        // (qk_norm=true) before RoPE.
        for layer in 0...80 {
            let attnPrefix = "model.layers.\(layer).self_attn."
            #expect(keys.contains { $0.hasPrefix(attnPrefix + "q_proj.") })
            #expect(keys.contains { $0.hasPrefix(attnPrefix + "k_proj.") })
            #expect(keys.contains { $0.hasPrefix(attnPrefix + "v_proj.") })
            #expect(keys.contains { $0.hasPrefix(attnPrefix + "o_proj.") })
        }
        let qkNormKeys = keys.filter { $0.contains("q_norm") || $0.contains("k_norm") }
        #expect(!qkNormKeys.isEmpty, "qk_norm=true requires q_norm + k_norm tensors")

        // Router + expert bias present in sparse layers.
        let routerKeys = keys.filter { $0.contains("router") }
        let expertBiasKeys = keys.filter { $0.contains("expert_bias") }
        #expect(!routerKeys.isEmpty, "router weights expected for sigmoid+bias top-k routing")
        #expect(!expertBiasKeys.isEmpty,
            "expert_bias expected for moe_router_enable_expert_bias=true")

        // Hy3 stores MTP in layer-80, NOT in a top-level mtp/nextn namespace.
        // This contract-level invariant prevents a future loader from looking
        // for `model.mtp.*` or `model.nextn.*` keys that don't exist.
        #expect(!keys.contains { $0.contains("model.mtp.") },
            "Hy3 does NOT use a top-level model.mtp namespace; MTP is layer 80")
        #expect(!keys.contains { $0.contains("model.nextn.") })
        #expect(!keys.contains { $0.lowercased().contains("predict") })
    }

    @Test("Real local Hy3 JANGTQ2 bundle decodes if config exists",
          .enabled(if: Hy3LocalBundles.hasJangtq2Config))
    func jangtq2BundleDecodes() throws {
        let bundle = try #require(Hy3LocalBundles.jangtq2)
        let data = try Data(contentsOf: bundle.appending(component: "config.json"))
        let config = try JSONDecoder.json5().decode(Hy3Configuration.self, from: data)

        // Architecture sanity: same shape as source bundle.
        #expect(config.modelType == "hy_v3")
        #expect(config.architectures == ["HYV3ForCausalLM"])
        #expect(config.numHiddenLayers == 80)
        #expect(config.numAttentionHeads == 64)
        #expect(config.numKeyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.numExperts == 192)
        #expect(config.numExpertsPerTok == 8)
        #expect(config.numSharedExperts == 1)
        #expect(config.firstKDenseReplace == 1)
        #expect(config.numNextnPredictLayers == 1)

        // Quant routing: weight_format=mxtq + dict-form mxtq_bits.
        #expect(config.weightFormat == "mxtq")
        // Real JANGTQ2 bundles ship `mxtq_bits` as a dict
        // (routed_expert / attention / shared_expert / dense_ffn / mtp /
        // embed_tokens / lm_head / norms_router_biases). Hy3Configuration
        // routes through `decodeRoutedExpertBits` which prefers
        // `routed_expert` for kernel routing.
        #expect(config.routedExpertBits == 2,
            "JANGTQ2 routed_expert bits must decode to 2, got \(config.routedExpertBits as Any)")
    }

    @Test("Real local Hy3 JANGTQ1 bundle decodes one-bit routed experts if present",
          .enabled(if: FileManager.default.fileExists(
              atPath: Hy3LocalBundles.jangtq1Bundle.appending(component: "config.json").path)
              && FileManager.default.fileExists(
                  atPath: Hy3LocalBundles.jangtq1Bundle.appending(component: "jangtq_runtime.safetensors").path)))
    func jangtq1BundleDecodesOneBitRuntimeContract() throws {
        let bundle = Hy3LocalBundles.jangtq1Bundle
        let data = try Data(contentsOf: bundle.appending(component: "config.json"))
        let config = try JSONDecoder.json5().decode(Hy3Configuration.self, from: data)
        let sidecarBits = JANGTQRuntimeCache.sniffCodebookBits(
            at: bundle.appending(component: "jangtq_runtime.safetensors"))

        #expect(config.modelType == "hy_v3")
        #expect(config.weightFormat == "mxtq")
        #expect(config.routedExpertBits == 1,
            "JANGTQ1 routed_expert bits must decode to 1, got \(config.routedExpertBits as Any)")
        #expect(sidecarBits == 1,
            "JANGTQ1 sidecar codebook bits must sniff as 1, got \(sidecarBits as Any)")

        let indexData = try Data(contentsOf: bundle.appending(
            component: "model.safetensors.index.json"))
        let index = try JSONDecoder.json5().decode(SafetensorsIndex.self, from: indexData)
        #expect(Set(index.weightMap.values).count == 50)
        #expect(index.weightMap.keys.contains("model.layers.1.mlp.experts.0.gate_proj.tq_packed"))
        #expect(index.weightMap.keys.contains("model.layers.79.mlp.experts.191.down_proj.tq_packed"))
    }
}
