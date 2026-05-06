// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import XCTest

@testable import MLXLLM

final class MiniMaxJANGTQConfigTests: XCTestCase {
    func testUniformMxtqBitsDecode() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "mxtq_bits": 2
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertNil(config.mxtqGateUpBits)
        XCTAssertNil(config.mxtqDownBits)
    }

    func testJANGTQKPerProjectionBitsDecode() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "mxtq_bits": {
              "routed_expert": {
                "gate_proj": 2,
                "up_proj": 2,
                "down_proj": 4
              }
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertEqual(config.mxtqGateUpBits, 2)
        XCTAssertEqual(config.mxtqDownBits, 4)
    }

    func testQuantizationRoutedExpertBitsFallback() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "quantization": {
              "routed_expert_bits": 4
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 4)
        XCTAssertNil(config.mxtqGateUpBits)
        XCTAssertNil(config.mxtqDownBits)
    }

    func testQuantizationNestedMxtqBitsFallback() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "quantization": {
              "mxtq_bits": {
                "routed_expert": {
                  "gate_proj": 2,
                  "up_proj": 2,
                  "down_proj": 4
                }
              }
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertEqual(config.mxtqGateUpBits, 2)
        XCTAssertEqual(config.mxtqDownBits, 4)
    }

    func testExplicitProjectionBitFieldsWinOverNestedBits() throws {
        let config = try decodeMiniMaxConfig(extra: #"""
            "mxtq_gate_up_bits": 4,
            "mxtq_down_bits": 4,
            "mxtq_bits": {
              "routed_expert": {
                "gate_proj": 2,
                "up_proj": 2,
                "down_proj": 2
              }
            }
            """#)

        XCTAssertEqual(config.mxtqBits, 2)
        XCTAssertEqual(config.mxtqGateUpBits, 4)
        XCTAssertEqual(config.mxtqDownBits, 4)
    }

    private func decodeMiniMaxConfig(extra: String) throws -> MiniMaxJANGTQConfiguration {
        let suffix = extra.trimmingCharacters(in: .whitespacesAndNewlines)
        let optionalExtra = suffix.isEmpty ? "" : ",\n\(suffix)"
        let json = """
            {
              "model_type": "minimax_m2",
              "hidden_size": 3072,
              "intermediate_size": 1536,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "max_position_embeddings": 262144,
              "num_experts_per_tok": 8,
              "num_local_experts": 256,
              "shared_intermediate_size": 0,
              "num_hidden_layers": 62,
              "rms_norm_eps": 0.000001,
              "rope_theta": 1000000.0,
              "rotary_dim": 96,
              "vocab_size": 200064
              \(optionalExtra)
            }
            """
        let data = Data(json.utf8)
        return try JSONDecoder().decode(MiniMaxJANGTQConfiguration.self, from: data)
    }
}
