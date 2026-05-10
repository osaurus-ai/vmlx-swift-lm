// Pin Gemma4 PLE (per-layer-embedding) config-coherence contract for both
// the LLM (`Gemma4TextConfiguration`) and VLM (`Gemma4.Gemma4TextConfiguration`) decoders.
//
// Background (see `docs/GEMMA4-DEEP-TRACE-2026-05-10.md` §7.6):
//
// Gemma4 E2B / E4B variants opt into Per-Layer Embedding by setting BOTH
// `hidden_size_per_layer_input` AND `vocab_size_per_layer_input` to non-zero.
// Setting only one of them is structurally invalid:
//   * vocab=0, hidden>0 → `Embedding(embeddingCount: 0, ...)` → garbage rows
//   * vocab>0, hidden=0 → zero-dim hidden state → silent dim collapse
// Both fields default to 0 in the decoder (PLE off — base E27 / 31B models).
//
// Before today, the decoder accepted the incoherent state silently and the
// model would either crash deeper in init or silently produce wrong output.
// Now `init(from:)` throws `DecodingError.dataCorrupted` so a malformed
// config.json fails fast at load time with a precise error.
//
// Source-coverage style — no MLX runtime needed.

import Foundation
@testable import MLXLLM
import Testing

@Suite("Gemma4 PLE config-coherence contract")
struct Gemma4PLECoherenceTests {

    private static func decode(
        _ json: String,
        file: String = #filePath, line: Int = #line
    ) throws -> Gemma4TextConfiguration {
        let data = json.data(using: .utf8)!
        return try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
    }

    /// Both fields zero (PLE off) — this is the base 27B / 31B path and
    /// must continue to decode cleanly.
    @Test("Gemma4 LLM config decodes when PLE is off (both fields zero)")
    func plOffDecodesCleanly() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 2304,
              "num_hidden_layers": 4,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "intermediate_size": 9216,
              "vocab_size": 262144,
              "rms_norm_eps": 1e-6,
              "sliding_window": 1024,
              "layer_types": ["sliding","full","sliding","full"]
            }
            """
        let config = try Self.decode(json)
        #expect(config.hiddenSizePerLayerInput == 0)
        #expect(config.vocabSizePerLayerInput == 0)
    }

    /// Both fields positive (PLE on) — the E2B/E4B path.
    @Test("Gemma4 LLM config decodes when PLE is on (both fields positive)")
    func plOnDecodesCleanly() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size": 2304,
              "num_hidden_layers": 4,
              "hidden_size_per_layer_input": 256,
              "vocab_size_per_layer_input": 262144
            }
            """
        let config = try Self.decode(json)
        #expect(config.hiddenSizePerLayerInput == 256)
        #expect(config.vocabSizePerLayerInput == 262144)
    }

    /// Hidden positive but vocab zero — must throw.
    @Test("Gemma4 LLM config rejects hidden>0 with vocab=0")
    func plHiddenWithoutVocabThrows() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size_per_layer_input": 256,
              "vocab_size_per_layer_input": 0
            }
            """
        #expect(throws: DecodingError.self) {
            _ = try Self.decode(json)
        }
    }

    /// Vocab positive but hidden zero — must throw (mirror case).
    @Test("Gemma4 LLM config rejects vocab>0 with hidden=0")
    func plVocabWithoutHiddenThrows() throws {
        let json = """
            {
              "model_type": "gemma4_text",
              "hidden_size_per_layer_input": 0,
              "vocab_size_per_layer_input": 262144
            }
            """
        #expect(throws: DecodingError.self) {
            _ = try Self.decode(json)
        }
    }

    /// Source-coverage guard for the VLM-side config (Gemma4.swift's
    /// nested `Gemma4TextConfiguration`). The VLM type is fileprivate to MLXVLM, so
    /// we pin its source contract directly rather than constructing the
    /// type — same recipe used for Zaya1VL adapters.
    @Test("Gemma4 VLM (MLXVLM) Gemma4TextConfiguration has the same PLE coherence guard")
    func vlmConfigHasMatchingGuard() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repo.appendingPathComponent("Libraries/MLXVLM/Models/Gemma4.swift")
        let source = try String(contentsOf: url, encoding: .utf8)

        // The guard is present.
        #expect(
            source.contains("Gemma4 PLE config incoherent"),
            "Gemma4.swift (VLM) must include the same PLE coherence DecodingError as the LLM side.")
        // The xor pattern is present (matches LLM side phrasing).
        #expect(
            source.contains("(hiddenSizePerLayerInput == 0) != (vocabSizePerLayerInput == 0)"),
            "Gemma4.swift (VLM) must use the same xor-on-zero pattern.")
    }
}
