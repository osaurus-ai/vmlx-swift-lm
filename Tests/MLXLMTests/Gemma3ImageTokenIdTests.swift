// Pin the Gemma3 `image_token_id` decode contract.
//
// 2026-05-10 audit: `Gemma3.swift` previously hardcoded `imageTokenId = 262144`
// in three places (model `prepareInputsForMultimodal`, processor
// `Gemma3Processor.prepare`, and `Gemma3ProcessorConfiguration` as a
// non-decoded `let imageTokenId: Int = 262144`). A future Gemma3 variant
// shipping a different `image_token_id` in `config.json` would have been
// silently ignored, leaving the post-expansion image-token mask
// mis-aligned with the bundle's tokenizer.
//
// Fix: `Gemma3Configuration` now decodes `image_token_id` (private
// `_imageTokenId: Int?` + public computed `imageTokenId: Int { _imageTokenId ?? 262144 }`)
// matching the existing `pad_token_id` pattern. The model and processor
// read through `config.imageTokenId` instead of re-hardcoding 262144.

import Foundation
@testable import MLXVLM
import Testing

@Suite("Gemma3 image_token_id decode contract", .serialized)
struct Gemma3ImageTokenIdTests {

    private static let baseTextConfig = """
    "text_config": {
      "model_type": "gemma3_text",
      "vocab_size": 262208,
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "intermediate_size": 64,
      "num_attention_heads": 4,
      "head_dim": 64,
      "rms_norm_eps": 1e-5,
      "num_key_value_heads": 4,
      "rope_theta": 1000000.0,
      "rope_local_base_freq": 10000.0,
      "rope_traditional": false,
      "query_pre_attn_scalar": 256,
      "sliding_window": 512,
      "sliding_window_pattern": 6,
      "max_position_embeddings": 32768
    }
    """

    private static let baseVisionConfig = """
    "vision_config": {
      "model_type": "siglip_vision_model",
      "num_hidden_layers": 2,
      "hidden_size": 64,
      "intermediate_size": 64,
      "num_attention_heads": 4,
      "patch_size": 14,
      "image_size": 224
    }
    """

    /// Default fallback fires when `image_token_id` is absent. This is
    /// the case for every Gemma3 bundle currently shipped (the field
    /// has always been implicit).
    @Test("Gemma3Configuration falls back to 262144 when image_token_id is absent")
    func defaultsTo262144WhenAbsent() throws {
        let json = """
        {
          \(Self.baseTextConfig),
          \(Self.baseVisionConfig),
          "model_type": "gemma3",
          "mm_tokens_per_image": 256
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma3Configuration.self, from: json)
        #expect(config.imageTokenId == 262144)
    }

    /// Override path: a future bundle stamping a different
    /// `image_token_id` flows through unchanged so model + processor
    /// stay aligned with the tokenizer's actual special-token mapping.
    @Test("Gemma3Configuration decodes a custom image_token_id from config.json")
    func decodesCustomImageTokenId() throws {
        let json = """
        {
          \(Self.baseTextConfig),
          \(Self.baseVisionConfig),
          "model_type": "gemma3",
          "mm_tokens_per_image": 256,
          "image_token_id": 999999
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Gemma3Configuration.self, from: json)
        #expect(config.imageTokenId == 999_999)
    }

    /// Source-coverage guard: prevent re-introduction of the
    /// `let imageTokenId = 262144` literal in `Gemma3.swift`. Any
    /// future caller must read `config.imageTokenId` instead.
    @Test("Gemma3.swift no longer hardcodes 262144 inline")
    func sourceContainsNoInlineHardcode() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repo
            .appendingPathComponent("Libraries")
            .appendingPathComponent("MLXVLM")
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma3.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        // The decoded fallback in the computed accessor is the one
        // legitimate place a literal lives; anything else suggests a
        // regression of the audit fix.
        let inlineModelHardcode = "let imageTokenId = 262144"
        let inlineProcessorHardcode = "imageTokenId = 262144  // Image token used after expansion"
        #expect(!source.contains(inlineModelHardcode),
            "Gemma3.swift contains the prior model-side hardcoded `let imageTokenId = 262144` — read `config.imageTokenId` instead.")
        #expect(!source.contains(inlineProcessorHardcode),
            "Gemma3.swift contains the prior processor-side hardcoded `imageTokenId = 262144  // Image token used after expansion` — read `config.imageTokenId` instead.")
        // Sanity: the fix landed.
        #expect(source.contains("config.imageTokenId"),
            "Gemma3.swift should read `config.imageTokenId` after the audit fix.")
        #expect(source.contains("_imageTokenId ?? 262144"),
            "Gemma3Configuration should expose `imageTokenId` via `_imageTokenId ?? 262144` fallback.")
    }
}
