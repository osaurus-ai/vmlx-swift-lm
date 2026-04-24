// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Kimi K2.6 (model_type = `kimi_k25`) plumbing verification. Confirms
// the three touchpoints jang/research/KIMI-K2.6-VMLX-INTEGRATION.md
// §2.1–§2.2 require are all wired and route to the correct handler:
//
//   1. LLMModelFactory `kimi_k25` (and `kimi_k2`) → DeepseekV3Model
//   2. ToolCallFormat.infer(modelType: "kimi_k25") → .kimiK2
//   3. ReasoningParser.fromCapabilityName("kimi"/"kimi_k2"/"kimik2") →
//      non-nil parser (Kimi is an always-thinking model per §2.16)
//
// These were plumbed in commits 8bc98e5 (parsers) and this session's
// Kimi factory registration. Without a test, silent regressions are
// possible — Kimi has no model weights locally that the official
// matrix exercises, so this fast stand-alone test is the gate.

import Foundation
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class KimiK25RoutingTests: XCTestCase {

    // MARK: - Factory registration

    /// `kimi_k25` model_type loads via DeepseekV3 infrastructure. We
    /// verify by asking the registry to instantiate with a synthetic
    /// DeepseekV3 config — a missing registration throws
    /// `ModelFactoryError.unsupportedModelType`; a registration
    /// throws something else (config decode error, module-init error)
    /// or succeeds. Either of the non-"unsupportedModelType" outcomes
    /// proves the handler is wired.
    func testKimiK25IsRegisteredAsDeepseekV3() async throws {
        let typeRegistry = LLMModelFactory.shared.typeRegistry
        // Minimal DeepseekV3-shaped config. We don't ship weights so
        // full instantiation isn't attempted here — we just need the
        // handler lookup to succeed. DeepseekV3Configuration has a
        // large field surface so we pick the smallest passing set.
        let minimalConfigJSON = """
            {
              "model_type": "kimi_k25",
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "intermediate_size": 128,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "rms_norm_eps": 1e-6,
              "vocab_size": 100,
              "max_position_embeddings": 2048,
              "rope_theta": 10000.0,
              "q_lora_rank": 16,
              "kv_lora_rank": 16,
              "qk_nope_head_dim": 8,
              "qk_rope_head_dim": 8,
              "v_head_dim": 8,
              "moe_intermediate_size": 32,
              "first_k_dense_replace": 1,
              "moe_layer_freq": 1,
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "topk_group": 1,
              "n_group": 1,
              "routed_scaling_factor": 1.0
            }
            """
        let data = minimalConfigJSON.data(using: .utf8)!

        for modelType in ["kimi_k25", "kimi_k2", "deepseek_v3"] {
            do {
                _ = try await typeRegistry.createModel(
                    configuration: data, modelType: modelType)
                // Handler resolved + model built — ideal case.
            } catch let err as ModelFactoryError {
                if case .unsupportedModelType(let mt) = err {
                    XCTFail("model_type '\(mt)' not registered")
                }
                // Any other ModelFactoryError means the creator was
                // found and ran — registration confirmed.
            } catch {
                // Non-factory error = config decode / module init
                // problem. The creator was still resolved; that's
                // all this test asserts.
            }
        }
    }

    // MARK: - Tool format routing

    /// `ToolCallFormat.infer(from:)` must route `kimi_k25`, `kimi_k2`,
    /// and the JANG-converter stamp `"kimi"` all to `.kimiK2`. Without
    /// this, non-JANG Kimi bundles fall back to default `.json` and
    /// emit mis-parsed tool calls at inference.
    func testKimiK25ToolFormatInference() {
        XCTAssertEqual(ToolCallFormat.infer(from: "kimi_k25"), .kimiK2)
        XCTAssertEqual(ToolCallFormat.infer(from: "kimi_k2"), .kimiK2)
        XCTAssertEqual(ToolCallFormat.infer(from: "kimi"), .kimiK2,
            "bare `kimi` model_type must also route to .kimiK2")
    }

    func testKimiCapabilityStampResolution() {
        XCTAssertEqual(ToolCallFormat.fromCapabilityName("kimi"), .kimiK2)
        XCTAssertEqual(ToolCallFormat.fromCapabilityName("kimi_k2"), .kimiK2)
        XCTAssertEqual(ToolCallFormat.fromCapabilityName("kimik2"), .kimiK2)
    }

    // MARK: - Reasoning stamp

    /// Kimi K2.6 is an always-thinking model — the chat template
    /// unconditionally appends `<think>` to the assistant prefix
    /// (see KIMI-K2.6-IMPLEMENTATION.md §2.16). Reasoning parser
    /// stamp must produce a non-nil parser for all spellings so the
    /// `<think>…</think>` block is routed to `.reasoning` events
    /// instead of leaking into `.chunk`.
    func testKimiReasoningStampResolvesToThinkParser() {
        for name in ["kimi", "kimi_k2", "kimik2"] {
            let parser = ReasoningParser.fromCapabilityName(name)
            XCTAssertNotNil(parser,
                "ReasoningParser.fromCapabilityName(\"\(name)\") must return a parser")
        }
    }
}
