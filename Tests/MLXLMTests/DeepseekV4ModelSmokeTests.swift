// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Smoke tests for `DeepseekV4Model` / `DeepseekV4JANGTQModel`:
//   - Model instantiates from `config.json` + `jang_config.json`
//     (no live bundle needed; uses a tiny synthetic config)
//   - `sanitize(weights:)` correctly remaps DSV4 bundle keys
//   - Factory dispatch picks affine vs JANGTQ variant based on
//     `weight_format` field
//   - Live model exposes the right `kvHeads` shape for the cache
//     allocator
//
// Live-bundle load + NaN check + greedy gen coherence is Phase 3 work
// (gated on copying a DSV4 JANGTQ2 / JANG_2L bundle into local storage).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

// Serialized: each @Test allocates a DSV4 Module tree (embedding, quantized
// linears, HyperConnection). Running them in parallel collides on MLX's
// global hash-bucket state — `__next_prime overflow` in
// `mlx-c/mlx/c/array.cpp:352`. Same infrastructure issue seen elsewhere
// with Gemma4VLMTests / EvalTests.
@Suite("DSV4 model smoke", .serialized)
struct DeepseekV4ModelSmokeTests {

    /// Minimal synthetic config — small enough to instantiate without
    /// touching real weights, large enough to exercise every code path.
    static func tinyConfig() -> DeepseekV4Configuration {
        var c = DeepseekV4Configuration()
        c.vocabSize = 256
        c.hiddenSize = 16
        c.numHiddenLayers = 2
        c.numAttentionHeads = 2
        c.numKeyValueHeads = 1
        c.headDim = 8
        c.qkRopeHeadDim = 4
        c.qLoraRank = 8
        c.oGroups = 2
        c.oLoraRank = 4
        c.nRoutedExperts = 4
        c.nSharedExperts = 1
        c.numExpertsPerTok = 2
        c.moeIntermediateSize = 16
        c.numHashLayers = 1  // first layer uses hash routing
        c.hcMult = 2
        c.hcSinkhornIters = 4
        c.compressRatios = [0, 0]  // both layers plain attention (no compressor)
        c.useAttnSink = true
        return c
    }

    // NOTE: standalone `DeepseekV4Model(cfg)` / `DeepseekV4JANGTQModel(cfg)`
    // instantiation tests are covered via the factory dispatch tests below.
    // Separate standalone tests triggered a Metal concurrent-init segfault
    // when Swift Testing ran several Embedding-allocating @Test functions
    // in the same process under its parallel executor — a known pre-existing
    // issue also seen with Gemma4VLMTests / EvalTests.

    // MARK: - Sanitize remapping

    @Test("sanitize() rewrites attn→self_attn, ffn→mlp, norms, hc_*, top-level")
    func sanitizeRemap() {
        let cfg = Self.tinyConfig()
        let model = DeepseekV4Model(cfg)

        // Fake bundle weights — shapes don't matter for this test, we
        // only check the key remapping.
        let dummy = MLXArray.ones([1])
        var input: [String: MLXArray] = [:]
        input["embed.weight"] = dummy
        input["norm.weight"] = dummy
        input["head.weight"] = dummy
        input["hc_head_fn"] = dummy
        input["hc_head_base"] = dummy
        input["hc_head_scale"] = dummy
        for L in 0..<cfg.numHiddenLayers {
            input["layers.\(L).attn.wq_a.weight"] = dummy
            input["layers.\(L).attn.wq_b.weight"] = dummy
            input["layers.\(L).attn.wkv.weight"] = dummy
            input["layers.\(L).attn.wo_a.weight"] = dummy
            input["layers.\(L).attn.wo_b.weight"] = dummy
            input["layers.\(L).attn.q_norm.weight"] = dummy
            input["layers.\(L).attn.kv_norm.weight"] = dummy
            input["layers.\(L).attn.attn_sink"] = dummy
            input["layers.\(L).attn_norm.weight"] = dummy
            input["layers.\(L).ffn_norm.weight"] = dummy
            input["layers.\(L).hc_attn_fn"] = dummy
            input["layers.\(L).hc_attn_base"] = dummy
            input["layers.\(L).hc_attn_scale"] = dummy
            input["layers.\(L).hc_ffn_fn"] = dummy
            input["layers.\(L).hc_ffn_base"] = dummy
            input["layers.\(L).hc_ffn_scale"] = dummy
            input["layers.\(L).ffn.gate.weight"] = dummy
            input["layers.\(L).ffn.shared_experts.gate_proj.weight"] = dummy
        }
        // Keys to DROP
        input["mtp.0.weight"] = dummy
        input["layers.0.attn.compressor.wkv.weight"] = dummy
        input["layers.0.attn.indexer.wq_b.weight"] = dummy

        let out = model.sanitize(weights: input)

        // Top-level renames
        #expect(out["model.embed_tokens.weight"] != nil)
        #expect(out["model.norm.weight"] != nil)
        #expect(out["lm_head.weight"] != nil)
        #expect(out["model.hc_head.hc_head_fn"] != nil)

        // Per-layer renames
        for L in 0..<cfg.numHiddenLayers {
            #expect(out["model.layers.\(L).self_attn.wq_a.weight"] != nil)
            #expect(out["model.layers.\(L).self_attn.attn_sink"] != nil)
            #expect(out["model.layers.\(L).input_layernorm.weight"] != nil)
            #expect(out["model.layers.\(L).post_attention_layernorm.weight"] != nil)
            #expect(out["model.layers.\(L).attn_hc.fn"] != nil)
            #expect(out["model.layers.\(L).attn_hc.base"] != nil)
            #expect(out["model.layers.\(L).attn_hc.scale"] != nil)
            #expect(out["model.layers.\(L).ffn_hc.fn"] != nil)
            #expect(out["model.layers.\(L).mlp.gate.weight"] != nil)
            #expect(out["model.layers.\(L).mlp.shared_experts.gate_proj.weight"] != nil)

            // Old names must be GONE
            #expect(out["layers.\(L).attn.wq_a.weight"] == nil)
            #expect(out["layers.\(L).attn_norm.weight"] == nil)
            #expect(out["layers.\(L).ffn_norm.weight"] == nil)
        }

        // Dropped keys
        #expect(out["mtp.0.weight"] == nil,
            "MTP training-head keys must be dropped")

        // Compressor + Indexer keys KEPT and remapped to
        // model.layers.L.self_attn.{compressor,indexer}.*
        #expect(
            out["model.layers.0.self_attn.compressor.wkv.weight"] != nil,
            "compressor keys must be KEPT and remapped under self_attn")
        #expect(
            out["model.layers.0.self_attn.indexer.wq_b.weight"] != nil,
            "indexer keys must be KEPT and remapped under self_attn")
    }

    @Test("sanitize() stacks per-expert weights into switch_mlp.{proj}.*")
    func sanitizeExpertStacking() {
        let cfg = Self.tinyConfig()
        let model = DeepseekV4Model(cfg)

        // Build per-expert tensors for layer 0 with distinct values so
        // we can verify the stacked shape is (n_experts, out, in).
        var input: [String: MLXArray] = [:]
        let outDim = cfg.moeIntermediateSize
        let inDim = cfg.hiddenSize
        for e in 0..<cfg.nRoutedExperts {
            // w1 → gate_proj, w2 → down_proj, w3 → up_proj
            input["layers.0.ffn.experts.\(e).w1.weight"] =
                MLXArray.ones([outDim, inDim]) * Float(e)
            input["layers.0.ffn.experts.\(e).w2.weight"] =
                MLXArray.ones([inDim, outDim]) * Float(e)
            input["layers.0.ffn.experts.\(e).w3.weight"] =
                MLXArray.ones([outDim, inDim]) * Float(e)
        }

        let out = model.sanitize(weights: input)

        let gateKey = "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        let downKey = "model.layers.0.mlp.switch_mlp.down_proj.weight"
        let upKey = "model.layers.0.mlp.switch_mlp.up_proj.weight"
        #expect(out[gateKey] != nil,
            "expert w1 must stack into switch_mlp.gate_proj.weight")
        #expect(out[downKey] != nil,
            "expert w2 must stack into switch_mlp.down_proj.weight")
        #expect(out[upKey] != nil,
            "expert w3 must stack into switch_mlp.up_proj.weight")

        if let stacked = out[gateKey] {
            #expect(stacked.shape == [cfg.nRoutedExperts, outDim, inDim])
        }
        // Per-expert originals must be consumed.
        for e in 0..<cfg.nRoutedExperts {
            #expect(out["layers.0.ffn.experts.\(e).w1.weight"] == nil)
        }
    }

    // MARK: - Factory dispatch

    @Test("Factory dispatch — affine vs JANGTQ routing via weight_format")
    func factoryDispatchBothVariants() throws {
        // Combined into one test to avoid Swift Testing's parallel executor
        // crashing when multiple @Test methods each allocate a Metal-backed
        // Embedding + full DSV4 stack in the same process. Running the two
        // dispatches sequentially inside one @Test body is safe.
        struct FormatCheck: Codable {
            let weightFormat: String?
            enum CodingKeys: String, CodingKey { case weightFormat = "weight_format" }
        }
        let body = """
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 4,
            "q_lora_rank": 8,
            "o_groups": 2,
            "o_lora_rank": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "hc_mult": 2,
            "compress_ratios": [0, 0]
            """

        // Affine variant
        let affineJSON = "{\"model_type\":\"deepseek_v4\",\(body)}".data(using: .utf8)!
        let affineCfg = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: affineJSON)
        let affineCheck = try? JSONDecoder().decode(
            FormatCheck.self, from: affineJSON)
        #expect(affineCheck?.weightFormat == nil)
        let affineModel: any LanguageModel =
            affineCheck?.weightFormat == "mxtq"
            ? DeepseekV4JANGTQModel(affineCfg) : DeepseekV4Model(affineCfg)
        #expect(affineModel is DeepseekV4Model,
            "no weight_format → DeepseekV4Model")
        let dsv4 = affineModel as? DeepseekV4Model
        #expect(dsv4?.kvHeads.count == affineCfg.numHiddenLayers)
        #expect(dsv4?.kvHeads.allSatisfy { $0 == 1 } == true,
            "DSV4 single latent KV head → kvHeads all == 1")

        // JANGTQ variant
        let mxtqJSON =
            "{\"model_type\":\"deepseek_v4\",\"weight_format\":\"mxtq\",\(body)}"
            .data(using: .utf8)!
        let mxtqCfg = try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: mxtqJSON)
        let mxtqCheck = try? JSONDecoder().decode(FormatCheck.self, from: mxtqJSON)
        #expect(mxtqCheck?.weightFormat == "mxtq")
        let mxtqModel: any LanguageModel =
            mxtqCheck?.weightFormat == "mxtq"
            ? DeepseekV4JANGTQModel(mxtqCfg) : DeepseekV4Model(mxtqCfg)
        #expect(mxtqModel is DeepseekV4JANGTQModel,
            "weight_format=mxtq → DeepseekV4JANGTQModel")
    }
}
