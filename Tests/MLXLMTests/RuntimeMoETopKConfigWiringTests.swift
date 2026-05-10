// Copyright © 2026 osaurus.

import Foundation
import XCTest

final class RuntimeMoETopKConfigWiringTests: XCTestCase {
    func testCompatibleTextConfigsCallRuntimeTopKOverride() throws {
        let expectations: [(String, String)] = [
            ("Libraries/MLXLLM/Models/MiniMax.swift", "numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift", "numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/Hy3.swift", "self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/BailingHybrid.swift", "self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/NemotronH.swift", "numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/Qwen3MoE.swift", "self.numExpertsPerToken = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/Qwen35.swift", "self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/Qwen35JANGTQ.swift", "self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK"),
            ("Libraries/MLXLLM/Models/Gemma4Text.swift", "topKExperts = RuntimeMoETopKOverride.effectiveTopK"),
        ]

        for (path, needle) in expectations {
            let source = try readRepositoryFile(path)
            XCTAssertTrue(source.contains(needle), "\(path) missing \(needle)")
        }
    }

    func testCompatibleVLMConfigsCallRuntimeTopKOverride() throws {
        let source = try readRepositoryFile("Libraries/MLXVLM/Models/Gemma4.swift")
        XCTAssertTrue(
            source.contains("topKExperts = RuntimeMoETopKOverride.effectiveTopK"),
            "Gemma4 VLM text_config top_k_experts must share the top-k override helper")
    }

    func testZayaTopOneConfigsAreNotWiredToRuntimeTopKOverride() throws {
        let textSource = try readRepositoryFile("Libraries/MLXLLM/Models/Zaya.swift")
        let vlSource = try readRepositoryFile("Libraries/MLXVLM/Models/Zaya1VL.swift")

        XCTAssertFalse(textSource.contains("moeRouterTopk = RuntimeMoETopKOverride"))
        XCTAssertFalse(textSource.contains("moeRouterTopk = RuntimeMoETopKOverride.effectiveTopK"))
        XCTAssertFalse(vlSource.contains("moeRouterTopk = RuntimeMoETopKOverride"))
        XCTAssertFalse(vlSource.contains("moeRouterTopk = RuntimeMoETopKOverride.effectiveTopK"))
    }

    func testModelContainerScopesCacheKeysForRuntimeTopKOverride() throws {
        let source = try readRepositoryFile("Libraries/MLXLMCommon/ModelContainer.swift")
        XCTAssertGreaterThanOrEqual(
            source.components(separatedBy: "RuntimeMoETopKOverride.cacheScopedModelKey").count - 1,
            2,
            "Both explicit enableCaching(config:) and enableCachingAsync() must scope cache modelKey by top-k override")
    }

    func testMiniMaxRouterCompileReadsCanonicalAndLegacyEnvNames() throws {
        let source = try readRepositoryFile("Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift")
        XCTAssertTrue(source.contains("VMLX_MINIMAX_ROUTER_COMPILE"))
        XCTAssertTrue(source.contains("VMLINUX_MINIMAX_ROUTER_COMPILE"))
        XCTAssertTrue(
            source.contains("miniMaxJANGTQRouterCompileEnabled(environment:"),
            "Router compile env parsing must stay in a testable helper")
    }

    private func readRepositoryFile(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }
}
