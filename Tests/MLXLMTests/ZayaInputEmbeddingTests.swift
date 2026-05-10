import Foundation
import MLX
@testable import MLXLLM
import Testing

@Suite("Zaya input embedding hook", .serialized)
struct ZayaInputEmbeddingTests {
    @Test("Inner trunk can consume precomputed embeddings for VLM image merge")
    func innerTrunkUsesInputEmbeddingWhenProvided() throws {
        try MLXMetalTestLock.withLock {
            let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: """
            {
              "model_type": "zaya",
              "hidden_size": 4,
              "num_hidden_layers": 0,
              "vocab_size": 16,
              "tie_word_embeddings": true,
              "scale_residual_merge": false,
              "residual_in_fp32": false,
              "norm_epsilon": 0.000001
            }
            """.data(using: .utf8)!)
            let trunk = ZayaModelInner(cfg.textConfig, context: nil)
            let inputIds = MLXArray([1, 2]).reshaped(1, 2)
            let inputEmbedding = MLXArray([
                Float(0.25), -0.5, 1.0, 2.0,
                Float(-1.5), 0.75, 0.5, -0.25,
            ]).reshaped(1, 2, 4)

            let actual = trunk(inputIds, cache: nil, inputEmbedding: inputEmbedding)
            let expected = ZayaRMSNorm(dimensions: 4, eps: cfg.textConfig.normEpsilon)(
                inputEmbedding)

            let maxDelta = (actual.asType(.float32) - expected.asType(.float32))
                .abs().max().item(Float.self)
            #expect(maxDelta < 1e-6)
            #expect(actual.shape == inputEmbedding.shape)
        }
    }

    @Test("Top-level tied Zaya model exposes the same embedding-entry path")
    func topLevelModelAcceptsInputEmbedding() throws {
        try MLXMetalTestLock.withLock {
            let cfg = try JSONDecoder().decode(ZayaConfiguration.self, from: """
            {
              "model_type": "zaya",
              "hidden_size": 4,
              "num_hidden_layers": 0,
              "vocab_size": 16,
              "tie_word_embeddings": true,
              "scale_residual_merge": false,
              "residual_in_fp32": false
            }
            """.data(using: .utf8)!)
            let model = ZayaModel(cfg, moe: nil)
            let inputIds = MLXArray([1, 2]).reshaped(1, 2)
            let firstEmbedding = MLXArray([
                Float(0.25), -0.5, 1.0, 2.0,
                Float(-1.5), 0.75, 0.5, -0.25,
            ]).reshaped(1, 2, 4)
            let secondEmbedding = MLXArray([
                Float(2.0), 1.0, -0.5, 0.25,
                Float(0.5), -0.25, -1.5, 0.75,
            ]).reshaped(1, 2, 4)

            let first = model(inputIds, cache: nil, inputEmbedding: firstEmbedding)
            let second = model(inputIds, cache: nil, inputEmbedding: secondEmbedding)

            #expect(first.shape == [1, 2, 16])
            #expect(second.shape == [1, 2, 16])
            let delta = (first.asType(.float32) - second.asType(.float32))
                .abs().max().item(Float.self)
            #expect(delta > 1e-6)
        }
    }
}
