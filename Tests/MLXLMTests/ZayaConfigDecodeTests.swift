// ZayaConfiguration JSON decode — verifies bundle config.json shapes
// (flat int mxtq_bits, post-jang-merge per-projection, real bundle config).

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("ZayaConfiguration decode")
struct ZayaConfigDecodeTests {

    @Test("Decodes minimal flat config with cca + tied embeddings")
    func minimalDecode() throws {
        let json = """
        {
            "model_type": "zaya",
            "hidden_size": 2048,
            "num_hidden_layers": 80,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_query_groups": 2,
            "cca": true,
            "cca_num_q_heads": 8,
            "kv_channels": 128,
            "num_experts": 16,
            "moe_router_topk": 1,
            "max_position_embeddings": 131072,
            "rope_theta": 5000000,
            "partial_rotary_factor": 0.5,
            "vocab_size": 262272,
            "norm_epsilon": 0.000001,
            "ffn_hidden_size": 2048,
            "tie_word_embeddings": true,
            "weight_format": "mxtq",
            "mxtq_bits": 2,
            "mxtq_seed": 42,
            "zaya_expert_layout": "split_switch_mlp"
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: json)
        #expect(cfg.modelType == "zaya")
        #expect(cfg.textConfig.hiddenSize == 2048)
        #expect(cfg.textConfig.numHiddenLayers == 80)
        #expect(cfg.textConfig.ccaNumQHeads == 8)
        #expect(cfg.textConfig.numExperts == 16)
        #expect(cfg.textConfig.tieWordEmbeddings == true)
        #expect(cfg.textConfig.weightFormat == "mxtq")
        #expect(cfg.textConfig.mxtqBits == 2)
        #expect(cfg.textConfig.mxtqSeed == 42)
        #expect(cfg.textConfig.zayaExpertLayout == "split_switch_mlp")
    }

    @Test("Per-role mxtq_bits dict decodes via routed_expert key")
    func perRoleDictDecode() throws {
        let json = """
        {
          "model_type": "zaya",
          "weight_format": "mxtq",
          "mxtq_bits": {
            "routed_expert": 2,
            "attention": 8,
            "router": 16,
            "embed_tokens": 8,
            "lm_head": 8,
            "cca_conv": 16,
            "norms_residual": 16
          }
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: json)
        #expect(cfg.textConfig.mxtqBits == 2)
    }

    @Test("Per-projection mxtq_gate_up_bits / mxtq_down_bits decode independently")
    func perProjectionBitsDecode() throws {
        // Simulates LLMModelFactory:1046–1057 pre-merging the nested
        // {gate_proj/up_proj/down_proj} dict into top-level scalars.
        let json = """
        {
          "model_type": "zaya",
          "weight_format": "mxtq",
          "mxtq_bits": 2,
          "mxtq_gate_up_bits": 2,
          "mxtq_down_bits": 4
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: json)
        #expect(cfg.textConfig.mxtqBits == 2)
        #expect(cfg.textConfig.mxtqGateUpBits == 2)
        #expect(cfg.textConfig.mxtqDownBits == 4)
    }

    @Test("Defaults apply when fields are missing")
    func defaultsApply() throws {
        let json = """
        { "model_type": "zaya" }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: json)
        #expect(cfg.textConfig.ropeTheta == 5_000_000)
        #expect(cfg.textConfig.partialRotaryFactor == 0.5)
        #expect(cfg.textConfig.numExperts == 16)
        #expect(cfg.textConfig.ccaNumQHeads == 8)
        #expect(cfg.textConfig.tieWordEmbeddings == true)
    }

    @Test("Real ZAYA1-8B-JANGTQ2 config.json decodes",
          .enabled(if: FileManager.default.fileExists(
              atPath: "/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ2/config.json")))
    func realJANGTQ2Decode() throws {
        let url = URL(fileURLWithPath:
            "/Users/eric/jang/models/Zyphra/ZAYA1-8B-JANGTQ2/config.json")
        let data = try Data(contentsOf: url)
        let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: data)
        #expect(cfg.textConfig.hiddenSize == 2048)
        #expect(cfg.textConfig.numHiddenLayers == 80)
        #expect(cfg.textConfig.numExperts == 16)
        #expect(cfg.textConfig.weightFormat == "mxtq")
        #expect(cfg.textConfig.mxtqBits == 2)
        #expect(cfg.textConfig.tieWordEmbeddings == true)
    }
}
