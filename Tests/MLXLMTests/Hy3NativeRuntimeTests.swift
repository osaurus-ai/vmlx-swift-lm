import MLX
@testable import MLXLLM
import MLXNN
import XCTest

final class Hy3NativeRuntimeTests: XCTestCase {
    private func makeConfig(
        numHiddenLayers: Int = 2,
        firstKDenseReplace: Int = 1,
        numExperts: Int = 2
    ) throws -> Hy3Configuration {
        let json = """
            {
              "model_type": "hy_v3",
              "architectures": ["HYV3ForCausalLM"],
              "hidden_size": 8,
              "num_hidden_layers": \(numHiddenLayers),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "intermediate_size": 16,
              "moe_intermediate_size": 4,
              "expert_hidden_dim": 4,
              "first_k_dense_replace": \(firstKDenseReplace),
              "num_experts": \(numExperts),
              "num_experts_per_tok": 1,
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
              "vocab_size": 32,
              "mxtq_seed": 42,
              "mxtq_bits": 2,
              "num_nextn_predict_layers": 1
            }
            """
        return try JSONDecoder.json5().decode(Hy3Configuration.self, from: Data(json.utf8))
    }

    func testNativeModelCreatesOneKVCachePerBaseLayerAndDropsMTPFromDecodePath() throws {
        try MLXMetalTestLock.withLock {
            let config = try makeConfig(numHiddenLayers: 2)
            let model = Hy3Model(config)

            XCTAssertEqual(model.kvHeads, [1, 1])
            XCTAssertEqual(model.newCache(parameters: nil).count, 2)
            XCTAssertEqual(model.loraLayers.count, 2)
        }
    }

    func testSanitizeRemapsHy3JANGTQBundleKeysAndDropsPreservedDisabledMTP() throws {
        try MLXMetalTestLock.withLock {
            let config = try makeConfig(numHiddenLayers: 2, firstKDenseReplace: 1, numExperts: 2)
            let model = Hy3Model(config)
            var weights: [String: MLXArray] = [:]

            weights["model.layers.2.self_attn.q_proj.weight"] = MLXArray.ones([8, 8])
            weights["model.layers.1.mlp.router.gate.weight"] = MLXArray.ones([2, 8])
            weights["model.layers.1.mlp.expert_bias"] = MLXArray.ones([2])
            weights["model.layers.1.mlp.shared_mlp.gate_proj.weight"] = MLXArray.ones([4, 8])
            weights["model.layers.1.mlp.shared_mlp.up_proj.weight"] = MLXArray.ones([4, 8])
            weights["model.layers.1.mlp.shared_mlp.down_proj.weight"] = MLXArray.ones([8, 4])

            for expert in 0..<2 {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    weights["model.layers.1.mlp.experts.\(expert).\(projection).tq_packed"] =
                        MLXArray.zeros([2, 1], dtype: .uint32)
                    weights["model.layers.1.mlp.experts.\(expert).\(projection).tq_norms"] =
                        MLXArray.ones([2], dtype: .float16)
                    weights["model.layers.1.mlp.experts.\(expert).\(projection).tq_bits"] =
                        MLXArray(2)
                }
            }

            let sanitized = model.sanitize(weights: weights)

            XCTAssertNil(sanitized["model.layers.2.self_attn.q_proj.weight"])
            XCTAssertNil(sanitized["model.layers.1.mlp.router.gate.weight"])
            XCTAssertNil(sanitized["model.layers.1.mlp.expert_bias"])
            XCTAssertNil(sanitized["model.layers.1.mlp.shared_mlp.gate_proj.weight"])
            XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.tq_packed"])
            XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.tq_bits"])

            XCTAssertNotNil(sanitized["model.layers.1.mlp.gate.weight"])
            XCTAssertNotNil(sanitized["model.layers.1.mlp.gate.e_score_correction_bias"])
            XCTAssertNotNil(sanitized["model.layers.1.mlp.shared_experts.gate_proj.weight"])
            XCTAssertEqual(
                sanitized["model.layers.1.mlp.switch_mlp.gate_proj.tq_packed"]?.shape,
                [2, 2, 1])
            XCTAssertEqual(
                sanitized["model.layers.1.mlp.switch_mlp.down_proj.tq_norms"]?.shape,
                [2, 2])
        }
    }

    func testSanitizeFusesHy3AttentionQKVProjections() throws {
        try MLXMetalTestLock.withLock {
            let config = try makeConfig(numHiddenLayers: 1, firstKDenseReplace: 1, numExperts: 2)
            let model = Hy3Model(config)
            let prefix = "model.layers.0.self_attn"
            let weights: [String: MLXArray] = [
                "\(prefix).q_proj.weight": MLXArray.ones([8, 4]),
                "\(prefix).k_proj.weight": MLXArray.ones([4, 4]) * 2,
                "\(prefix).v_proj.weight": MLXArray.ones([4, 4]) * 3,
                "\(prefix).q_proj.scales": MLXArray.ones([8, 1]),
                "\(prefix).k_proj.scales": MLXArray.ones([4, 1]) * 2,
                "\(prefix).v_proj.scales": MLXArray.ones([4, 1]) * 3,
                "\(prefix).q_proj.biases": MLXArray.ones([8, 1]),
                "\(prefix).k_proj.biases": MLXArray.ones([4, 1]) * 2,
                "\(prefix).v_proj.biases": MLXArray.ones([4, 1]) * 3,
            ]

            let sanitized = model.sanitize(weights: weights)

            XCTAssertEqual(sanitized["\(prefix).qkv_proj.weight"]?.shape, [16, 4])
            XCTAssertEqual(sanitized["\(prefix).qkv_proj.scales"]?.shape, [16, 1])
            XCTAssertEqual(sanitized["\(prefix).qkv_proj.biases"]?.shape, [16, 1])
            XCTAssertNil(sanitized["\(prefix).q_proj.weight"])
            XCTAssertNil(sanitized["\(prefix).k_proj.weight"])
            XCTAssertNil(sanitized["\(prefix).v_proj.weight"])
        }
    }

    func testLMHeadProjectionUsesFP32ForDenseWeights() throws {
        try MLXMetalTestLock.withLock {
            let hidden = MLXArray(
                [Float(1.0), -2.0, 0.5, 4.0],
                [1, 1, 4]
            ).asType(.float16)
            let head = Linear(
                weight: MLXArray(
                    [
                        Float(0.25), -0.5, 1.0, 0.125,
                        1.5, 0.25, -0.75, 0.5,
                    ],
                    [2, 4]
                ).asType(.float16),
                bias: nil)

            let actual = hy3LMHeadFP32(hidden, head)
            let expected = matmul(
                hidden.asType(.float32),
                head.weight.asType(.float32).transposed())

            XCTAssertEqual(actual.dtype, .float32)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    func testLMHeadProjectionDequantizesQuantizedWeightsBeforeFP32Matmul() throws {
        try MLXMetalTestLock.withLock {
            let hidden = MLXArray(
                (0..<32).map { Float($0 % 7) * 0.25 - 0.75 },
                [1, 1, 32]
            ).asType(.float16)
            let base = MLXArray(
                (0..<64).map { Float(($0 % 11) - 5) * 0.125 },
                [2, 32]
            ).asType(.float32)
            let head = QuantizedLinear(
                weight: base,
                bias: nil,
                groupSize: 32,
                bits: 4)

            let actual = hy3LMHeadFP32(hidden, head)
            let expected = matmul(
                hidden.asType(.float32),
                dequantized(
                    head.weight,
                    scales: head.scales,
                    biases: head.biases,
                    groupSize: head.groupSize,
                    bits: head.bits,
                    mode: head.mode,
                    dtype: .float32)
                    .transposed())

            XCTAssertEqual(actual.dtype, .float32)
            assertClose(actual, expected, tolerance: 1e-5)
        }
    }

    private func assertClose(
        _ actual: MLXArray,
        _ expected: MLXArray,
        tolerance: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        MLX.eval(actual, expected)
        let actualValues = actual.asArray(Float.self)
        let expectedValues = expected.asArray(Float.self)
        XCTAssertEqual(actualValues.count, expectedValues.count, file: file, line: line)
        for (a, e) in zip(actualValues, expectedValues) {
            XCTAssertLessThanOrEqual(abs(a - e), tolerance, file: file, line: line)
        }
    }
}
