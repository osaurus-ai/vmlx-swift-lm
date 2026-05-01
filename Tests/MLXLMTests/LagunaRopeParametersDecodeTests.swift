// Copyright © 2026 osaurus.
//
// Regression coverage for `LagunaConfiguration`'s `rope_parameters` decode.
// Real Laguna bundles ship a mixed-shape object — a top-level scalar
// `original_max_position_embeddings` sits alongside the per-layer-type
// sub-dicts (`full_attention`, `sliding_attention`). Decoding strictly as
// `[String: [String: StringOrNumber]]` fails on the scalar; we filter to
// dict-valued entries instead so loading proceeds.

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import Testing

@Suite("Laguna rope_parameters — mixed-shape decode")
struct LagunaRopeParametersDecodeTests {

    /// Minimum-viable Laguna config covering all required fields plus the
    /// problematic `rope_parameters` shape. Only the rope decode is
    /// asserted — the rest of the config is supplied so JSON decoding
    /// doesn't fail on missing keys.
    private static let configWithMixedRope = #"""
    {
      "model_type": "laguna",
      "hidden_size": 64,
      "intermediate_size": 128,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 16,
      "max_position_embeddings": 4096,
      "vocab_size": 1024,
      "rms_norm_eps": 1.0e-5,
      "tie_word_embeddings": true,
      "layer_types": ["sliding_attention", "full_attention"],
      "moe_intermediate_size": 64,
      "num_experts_per_tok": 2,
      "num_local_experts": 4,
      "num_shared_experts": 1,
      "use_qk_norm": true,
      "rope_parameters": {
        "full_attention": {
          "rope_theta": 500000.0,
          "rope_type": "default"
        },
        "sliding_attention": {
          "rope_theta": 500000.0,
          "rope_type": "default"
        },
        "original_max_position_embeddings": 4096
      }
    }
    """#

    @Test("Mixed-shape rope_parameters decodes without throwing")
    func decodeMixedShape() throws {
        let data = Self.configWithMixedRope.data(using: .utf8)!
        // The bug: this decode used to throw `Type mismatch at
        // rope_parameters.original_max_position_embeddings`.
        let cfg = try JSONDecoder().decode(LagunaConfiguration.self, from: data)
        // The two per-layer-type sub-dicts must survive; the top-level
        // scalar is dropped (it's available via `maxPositionEmbeddings`).
        #expect(cfg.ropeParameters.keys.contains("full_attention"))
        #expect(cfg.ropeParameters.keys.contains("sliding_attention"))
        #expect(!cfg.ropeParameters.keys.contains("original_max_position_embeddings"))
    }

    /// Pure (no top-level scalar) shape must continue to decode — the
    /// permissive path mustn't regress the strict shape for older bundles.
    @Test("Pure-dict rope_parameters still decodes")
    func decodePureDictShape() throws {
        let json = #"""
        {
          "model_type": "laguna",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "max_position_embeddings": 4096,
          "vocab_size": 1024,
          "rms_norm_eps": 1.0e-5,
          "tie_word_embeddings": true,
          "layer_types": ["full_attention"],
          "moe_intermediate_size": 64,
          "num_experts_per_tok": 2,
          "num_local_experts": 4,
          "num_shared_experts": 1,
          "use_qk_norm": true,
          "rope_parameters": {
            "full_attention": {
              "rope_theta": 500000.0,
              "rope_type": "default"
            }
          }
        }
        """#
        let cfg = try JSONDecoder().decode(LagunaConfiguration.self, from: json.data(using: .utf8)!)
        #expect(cfg.ropeParameters.keys.contains("full_attention"))
    }

    /// Missing rope_parameters entirely → empty dict, no crash.
    @Test("Absent rope_parameters → empty dict")
    func decodeAbsentRopeParameters() throws {
        let json = #"""
        {
          "model_type": "laguna",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "max_position_embeddings": 4096,
          "vocab_size": 1024,
          "rms_norm_eps": 1.0e-5,
          "tie_word_embeddings": true,
          "layer_types": ["full_attention"],
          "moe_intermediate_size": 64,
          "num_experts_per_tok": 2,
          "num_local_experts": 4,
          "num_shared_experts": 1,
          "use_qk_norm": true
        }
        """#
        let cfg = try JSONDecoder().decode(LagunaConfiguration.self, from: json.data(using: .utf8)!)
        #expect(cfg.ropeParameters.isEmpty)
    }
}
