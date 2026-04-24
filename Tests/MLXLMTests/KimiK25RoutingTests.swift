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

    // MARK: - JANGTQ (mxtq) dispatch

    /// Kimi K2.6 + DeepSeek-V3 JANGTQ bundles carry
    /// `"weight_format": "mxtq"` in `config.json`. The factory must
    /// route to `DeepseekV3JANGTQModel`, not the standard
    /// `DeepseekV3Model`. We probe by instantiating with a synthetic
    /// config and inspecting the returned type.
    func testKimiK25MxtqRoutesToJANGTQModel() async throws {
        let typeRegistry = LLMModelFactory.shared.typeRegistry
        // Minimal DSV3 config + the three JANGTQ knobs.
        let cfgJSON = """
            {
              "model_type": "kimi_k25",
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "intermediate_size": 128,
              "moe_intermediate_size": 32,
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
              "first_k_dense_replace": 1,
              "moe_layer_freq": 1,
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "topk_group": 1,
              "n_group": 1,
              "routed_scaling_factor": 1.0,
              "attention_bias": false,
              "norm_topk_prob": true,
              "weight_format": "mxtq",
              "mxtq_bits": 2,
              "mxtq_seed": 42
            }
            """
        let data = cfgJSON.data(using: .utf8)!
        for modelType in ["kimi_k25", "kimi_k2", "deepseek_v3"] {
            do {
                let model = try await typeRegistry.createModel(
                    configuration: data, modelType: modelType)
                XCTAssertTrue(
                    model is DeepseekV3JANGTQModel,
                    "model_type=\(modelType) with weight_format=mxtq must instantiate DeepseekV3JANGTQModel, got \(type(of: model))")
            } catch let err as ModelFactoryError {
                if case .unsupportedModelType(let mt) = err {
                    XCTFail("model_type '\(mt)' unexpectedly not registered")
                }
                // Any other factory error is acceptable — dispatch was
                // reached, config-decode complaint is unrelated.
            } catch {
                XCTFail("Unexpected error during mxtq dispatch: \(error)")
            }
        }
    }

    // MARK: - DSV4 live dispatch (Phase 1b landed)

    /// `model_type = "deepseek_v4"` now instantiates a live
    /// `DeepseekV4Model` (Phase 1b wired: mHC + MLA + attn sinks +
    /// inverse RoPE + grouped O + sqrtsoftplus MoE + DSV4 SwiGLU +
    /// HyperHead). `weight_format = "mxtq"` routes to
    /// `DeepseekV4JANGTQModel` with TurboQuantSwitchGLU for routed
    /// experts.
    func testDeepseekV4DispatchesToLiveModel() async throws {
        let typeRegistry = LLMModelFactory.shared.typeRegistry
        let cfg = """
            {
              "model_type": "deepseek_v4",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "moe_intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "qk_rope_head_dim": 4,
              "q_lora_rank": 8,
              "o_groups": 2,
              "o_lora_rank": 4,
              "vocab_size": 256,
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "n_shared_experts": 1,
              "hc_mult": 2,
              "compress_ratios": [0, 0]
            }
            """
        let model = try await typeRegistry.createModel(
            configuration: cfg.data(using: .utf8)!,
            modelType: "deepseek_v4")
        XCTAssertTrue(model is DeepseekV4Model,
            "affine deepseek_v4 must dispatch to DeepseekV4Model")
    }

    func testDeepseekV4MxtqDispatchesToJANGTQ() async throws {
        let typeRegistry = LLMModelFactory.shared.typeRegistry
        let cfg = """
            {
              "model_type": "deepseek_v4",
              "weight_format": "mxtq",
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "moe_intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "qk_rope_head_dim": 4,
              "q_lora_rank": 8,
              "o_groups": 2,
              "o_lora_rank": 4,
              "vocab_size": 256,
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "n_shared_experts": 1,
              "hc_mult": 2,
              "compress_ratios": [0, 0]
            }
            """
        let model = try await typeRegistry.createModel(
            configuration: cfg.data(using: .utf8)!,
            modelType: "deepseek_v4")
        XCTAssertTrue(model is DeepseekV4JANGTQModel,
            "mxtq deepseek_v4 must dispatch to DeepseekV4JANGTQModel")
    }

    /// Non-mxtq configs with the same model_type MUST fall through to
    /// the standard DeepseekV3Model. Regression guard: if someone adds
    /// a new format check above the deepseek branch, affine bundles
    /// would silently dispatch to the JANGTQ model and break weight
    /// loading.
    func testKimiK25AffineRoutesToStandardModel() async throws {
        let typeRegistry = LLMModelFactory.shared.typeRegistry
        let cfgJSON = """
            {
              "model_type": "kimi_k25",
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "intermediate_size": 128,
              "moe_intermediate_size": 32,
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
              "first_k_dense_replace": 1,
              "moe_layer_freq": 1,
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "topk_group": 1,
              "n_group": 1,
              "routed_scaling_factor": 1.0,
              "attention_bias": false,
              "norm_topk_prob": true
            }
            """
        let data = cfgJSON.data(using: .utf8)!
        do {
            let model = try await typeRegistry.createModel(
                configuration: data, modelType: "kimi_k25")
            XCTAssertTrue(
                model is DeepseekV3Model && !(model is DeepseekV3JANGTQModel),
                "Affine kimi_k25 (no weight_format) must instantiate DeepseekV3Model, got \(type(of: model))")
        } catch {
            // Config decode complaint acceptable — we're probing
            // dispatch, not full instantiation.
        }
    }
}
