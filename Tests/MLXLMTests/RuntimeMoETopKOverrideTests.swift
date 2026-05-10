// Copyright © 2026 osaurus.

import XCTest
@testable import MLXLMCommon

final class RuntimeMoETopKOverrideTests: XCTestCase {
    func testMissingEnvironmentKeepsCurrentTopK() {
        let decision = RuntimeMoETopKOverride.resolve(
            currentTopK: 8,
            modelType: "minimax_m2",
            field: "num_experts_per_tok",
            environment: [:])

        XCTAssertEqual(decision.effectiveTopK, 8)
        XCTAssertFalse(decision.applied)
        XCTAssertEqual(decision.reason, .unset)
    }

    func testCanonicalEnvironmentLowersTopK() {
        let decision = RuntimeMoETopKOverride.resolve(
            currentTopK: 8,
            modelType: "minimax_m2",
            field: "num_experts_per_tok",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"])

        XCTAssertEqual(decision.effectiveTopK, 4)
        XCTAssertTrue(decision.applied)
        XCTAssertEqual(decision.reason, .lowered)
    }

    func testLegacyTypoEnvironmentAliasLowersTopK() {
        let decision = RuntimeMoETopKOverride.resolve(
            currentTopK: 8,
            modelType: "hy_v3",
            field: "num_experts_per_tok",
            environment: ["VMLINUX_MOE_TOPK_OVERRIDE": "4"])

        XCTAssertEqual(decision.effectiveTopK, 4)
        XCTAssertTrue(decision.applied)
        XCTAssertEqual(decision.reason, .lowered)
    }

    func testCanonicalEnvironmentWinsOverLegacyAlias() {
        let decision = RuntimeMoETopKOverride.resolve(
            currentTopK: 8,
            modelType: "qwen3_moe",
            field: "num_experts_per_tok",
            environment: [
                "VMLX_MOE_TOPK_OVERRIDE": "4",
                "VMLINUX_MOE_TOPK_OVERRIDE": "2",
            ])

        XCTAssertEqual(decision.effectiveTopK, 4)
        XCTAssertTrue(decision.applied)
    }

    func testInvalidAndNonPositiveValuesKeepCurrentTopK() {
        for value in ["banana", "0", "-1"] {
            let decision = RuntimeMoETopKOverride.resolve(
                currentTopK: 8,
                modelType: "minimax_m2",
                field: "num_experts_per_tok",
                environment: ["VMLX_MOE_TOPK_OVERRIDE": value])

            XCTAssertEqual(decision.effectiveTopK, 8)
            XCTAssertFalse(decision.applied)
            XCTAssertEqual(decision.reason, .invalidRequestedTopK)
        }
    }

    func testNeverRaisesLowTopKFamilies() {
        let decision = RuntimeMoETopKOverride.resolve(
            currentTopK: 1,
            modelType: "zaya",
            field: "moe_router_topk",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"])

        XCTAssertEqual(decision.effectiveTopK, 1)
        XCTAssertFalse(decision.applied)
        XCTAssertEqual(decision.reason, .requestedTopKAboveCurrent)
    }

    func testNoopsWhenRequestedEqualsCurrentTopK() {
        let decision = RuntimeMoETopKOverride.resolve(
            currentTopK: 4,
            modelType: "gemma4",
            field: "top_k_experts",
            environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"])

        XCTAssertEqual(decision.effectiveTopK, 4)
        XCTAssertFalse(decision.applied)
        XCTAssertEqual(decision.reason, .requestedTopKAlreadySatisfied)
    }

    func testValidOverrideAddsCacheKeyComponent() {
        XCTAssertEqual(
            RuntimeMoETopKOverride.cacheKeyComponent(
                environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"]),
            "moeTopK=4")
        XCTAssertEqual(
            RuntimeMoETopKOverride.cacheKeyComponent(
                environment: ["VMLINUX_MOE_TOPK_OVERRIDE": "4"]),
            "moeTopK=4")
    }

    func testInvalidOverrideDoesNotAddCacheKeyComponent() {
        for value in ["", "banana", "0", "-1"] {
            XCTAssertNil(RuntimeMoETopKOverride.cacheKeyComponent(
                environment: ["VMLX_MOE_TOPK_OVERRIDE": value]))
        }
    }

    func testCacheScopedModelKeyIncludesOverrideOnlyWhenValid() {
        XCTAssertEqual(
            RuntimeMoETopKOverride.cacheScopedModelKey(
                "org/model",
                environment: ["VMLX_MOE_TOPK_OVERRIDE": "4"]),
            "org/model|moeTopK=4")
        XCTAssertEqual(
            RuntimeMoETopKOverride.cacheScopedModelKey(
                "org/model",
                environment: ["VMLINUX_MOE_TOPK_OVERRIDE": "4"]),
            "org/model|moeTopK=4")
        XCTAssertEqual(
            RuntimeMoETopKOverride.cacheScopedModelKey(
                "org/model",
                environment: ["VMLX_MOE_TOPK_OVERRIDE": "nope"]),
            "org/model")
    }
}
