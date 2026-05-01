// Copyright © 2026 osaurus.
//
// Factory-dispatch tests for the LLM-side `mistral3` closure with
// `weight_format == "mxtq"`. Verifies:
//
//   - Mistral3TextJANGTQModel is constructed (not vanilla
//     Mistral3TextModel) when weight_format=mxtq
//   - mxtq_bits / mxtq_seed flow through to the model's JANGTQDenseLinear
//     instances
//   - Vanilla Mistral3TextModel stays the dispatch target for non-mxtq
//     bundles (mxfp4, full precision) — no behavior regression
//
// Tests run against synthetic JSON config + skip the safetensor weight
// load (constructed model has zero-initialized .tq_packed / .tq_norms;
// loading real weights is gated on a real bundle and tested separately).
//

import Foundation
@testable import MLXLLM
@testable import MLXLMCommon
import XCTest

final class Mistral3TextJANGTQDispatchTests: XCTestCase {

    // Minimal config matching Mistral3TextConfiguration's required JSON
    // keys. Smaller than a real Mistral 3.5 (128 layers, hidden 12288)
    // because we're only testing dispatch — model gets constructed but
    // not run. Matching field names exactly so JSONDecoder.json5() binds.
    private func minimalConfig(weightFormat: String?, mxtqBits: Int? = nil) -> Data {
        var dict: [String: Any] = [
            "model_type": "ministral3",  // inner type for Mistral 3.5
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000.0,
            "head_dim": 128,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": false,
            "layer_types": Array(repeating: "full_attention", count: 4),
        ]
        if let weightFormat {
            dict["weight_format"] = weightFormat
        }
        if let mxtqBits {
            dict["mxtq_bits"] = mxtqBits
        }
        return try! JSONSerialization.data(withJSONObject: dict)
    }

    // MARK: - mxtq dispatch

    func testMxtqDispatchRoutesToJANGTQModel() async throws {
        let configData = minimalConfig(weightFormat: "mxtq", mxtqBits: 2)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "mistral3")
        XCTAssertTrue(model is Mistral3TextJANGTQModel,
            "mxtq Mistral 3 must route to JANGTQ variant; got \(type(of: model))")
        let m = model as! Mistral3TextJANGTQModel
        XCTAssertEqual(m.model.layers.count, 4)
    }

    func testMxtq4bDispatchPropagatesBits() async throws {
        let configData = minimalConfig(weightFormat: "mxtq", mxtqBits: 4)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "mistral3")
        XCTAssertTrue(model is Mistral3TextJANGTQModel)
        let m = model as! Mistral3TextJANGTQModel
        let firstLayer = m.model.layers[0]
        let qPacked = firstLayer.attention.wq.packed
        // bits=4 → vals_per_u32=8 → packed_in = 4096/8 = 512
        XCTAssertEqual(qPacked.dim(-1), 512,
            "bits=4 packed_in must be 4096/8=512; got \(qPacked.dim(-1))")
    }

    func testMxtqDefaultBitsIs2() async throws {
        let configData = minimalConfig(weightFormat: "mxtq", mxtqBits: nil)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "mistral3")
        let m = model as! Mistral3TextJANGTQModel
        let qPacked = m.model.layers[0].attention.wq.packed
        XCTAssertEqual(qPacked.dim(-1), 256,
            "default bits=2 packed_in must be 4096/16=256; got \(qPacked.dim(-1))")
    }

    // MARK: - Non-mxtq paths must NOT route to JANGTQ

    func testMxfp4DispatchRoutesToVanillaModel() async throws {
        let configData = minimalConfig(weightFormat: "mxfp4")
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "mistral3")
        XCTAssertTrue(model is Mistral3TextModel,
            "mxfp4 must NOT route to JANGTQ; got \(type(of: model))")
        XCTAssertFalse(model is Mistral3TextJANGTQModel)
    }

    func testNoWeightFormatRoutesToVanillaModel() async throws {
        let configData = minimalConfig(weightFormat: nil)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "mistral3")
        XCTAssertTrue(model is Mistral3TextModel)
        XCTAssertFalse(model is Mistral3TextJANGTQModel)
    }

    func testWeightFormatCaseInsensitive() async throws {
        let configData = minimalConfig(weightFormat: "MXTQ", mxtqBits: 2)
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: configData, modelType: "mistral3")
        XCTAssertTrue(model is Mistral3TextJANGTQModel,
            "uppercase MXTQ must still route to JANGTQ; got \(type(of: model))")
    }
}
