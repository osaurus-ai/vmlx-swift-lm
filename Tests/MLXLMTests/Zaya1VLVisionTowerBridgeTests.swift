import MLX
import MLXLMCommon
@testable import MLXVLM
import Foundation
import Testing

@Suite("ZAYA1-VL Qwen2.5 vision tower bridge", .serialized)
struct Zaya1VLVisionTowerBridgeTests {
    @Test("Qwen2.5-VL vision tower is reusable without instantiating Qwen25 language model")
    func qwen25VisionTowerCanBeCalledStandalone() throws {
        try MLXMetalTestLock.withLock {
            let config = try JSONDecoder.json5().decode(
                Qwen25VLConfiguration.VisionConfiguration.self,
                from: """
                {
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
                """.data(using: .utf8)!)

            let tower = Qwen25Vision.VisionModel(config)
            let pixelPatches = MLXArray.zeros([4, 12], dtype: .float32)
            let features = tower(pixelPatches, frames: [THW(1, 2, 2)])

            #expect(features.shape == [1, 32])
        }
    }
}
