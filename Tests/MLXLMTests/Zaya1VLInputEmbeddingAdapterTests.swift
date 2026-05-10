import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("ZAYA1-VL input embedding adapter", .serialized)
struct Zaya1VLInputEmbeddingAdapterTests {
    @Test("Vision tower output is merged into image-token embeddings with reusable image mask")
    func visionFeaturesReplaceImageTokenEmbeddings() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Zaya1VLConfiguration.self,
                from: """
                {
                  "model_type": "zaya1_vl",
                  "architectures": ["Zaya1VLForConditionalGeneration"],
                  "hidden_size": 32,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 1,
                  "head_dim": 8,
                  "vocab_size": 64,
                  "image_token_id": 7,
                  "vision_start_token_id": 5,
                  "vision_end_token_id": 6,
                  "vision_config": {
                    "depth": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "out_hidden_size": 32,
                    "num_heads": 4,
                    "patch_size": 2,
                    "spatial_patch_size": 2,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "window_size": 4,
                    "fullatt_block_indexes": [0],
                    "tokens_per_second": 4,
                    "in_chans": 3
                  }
                }
                """.data(using: .utf8)!)

            let adapter = try Zaya1VLInputEmbeddingAdapter(config)
            let inputIds = MLXArray([1, 7, 2]).reshaped(1, 3)
            let inputEmbeds = MLXArray([
                Float(0.25), -0.5, 1.0, 2.0, 0.0, 0.1, 0.2, 0.3,
                Float(0.4), 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                Float(1.2), 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                Float(2.0), 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                Float(9.0), 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7,
                Float(9.8), 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5,
                Float(10.6), 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3,
                Float(11.4), 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1,
                Float(-1.0), -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7,
                Float(-1.8), -1.9, -2.0, -2.1, -2.2, -2.3, -2.4, -2.5,
                Float(-2.6), -2.7, -2.8, -2.9, -3.0, -3.1, -3.2, -3.3,
                Float(-3.4), -3.5, -3.6, -3.7, -3.8, -3.9, -4.0, -4.1,
            ]).reshaped(1, 3, 32)
            let pixelPatches = MLXArray.zeros([4, 12], dtype: .float32)
            let frames = [THW(1, 2, 2)]

            let imageFeatures = adapter.projectImageFeatures(
                pixelValues: pixelPatches, frames: frames)
            let result = try adapter.mergeImageFeatures(
                inputIds: inputIds,
                inputEmbeds: inputEmbeds,
                pixelValues: pixelPatches,
                frames: frames)

            #expect(imageFeatures.shape == [1, 32])
            #expect(result.embeddings.shape == inputEmbeds.shape)
            #expect(result.imageMask?.shape == inputIds.shape)
            #expect(result.imageMask?.asArray(Bool.self) == [false, true, false])

            let firstTextDelta = (
                result.embeddings[0..., 0, 0...] - inputEmbeds[0..., 0, 0...]
            ).abs().max().item(Float.self)
            let lastTextDelta = (
                result.embeddings[0..., 2, 0...] - inputEmbeds[0..., 2, 0...]
            ).abs().max().item(Float.self)
            let imageDelta = (
                result.embeddings[0, 1, 0...] - imageFeatures[0, 0...]
            ).abs().max().item(Float.self)

            #expect(firstTextDelta < 1e-6)
            #expect(lastTextDelta < 1e-6)
            #expect(imageDelta < 1e-6)
        }
    }
}
