// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@testable import MLXLMCommon
import XCTest

final class CacheCoordinatorModeKeyIsolationTests: XCTestCase {

    func testReasoningModeChangesEveryCoordinatorHashTier() {
        let tokens = [101, 102, 103, 104]
        let reasoningOff = "bundle-a|kv=fp16|reasoning=off"
        let reasoningOn = "bundle-a|kv=fp16|reasoning=on"

        XCTAssertNotEqual(
            DiskCache.hashTokens(tokens, modelKey: reasoningOff),
            DiskCache.hashTokens(tokens, modelKey: reasoningOn))
        XCTAssertNotEqual(
            CacheBlock.computeBlockHash(
                parentHash: nil, tokenIds: tokens, modelKey: reasoningOff),
            CacheBlock.computeBlockHash(
                parentHash: nil, tokenIds: tokens, modelKey: reasoningOn))
        XCTAssertNotEqual(
            SSMStateCache.makeKey(
                tokens: tokens, boundary: tokens.count, modelKey: reasoningOff),
            SSMStateCache.makeKey(
                tokens: tokens, boundary: tokens.count, modelKey: reasoningOn))
    }

    func testMediaSaltChangesModeScopedHashes() {
        let tokens = [201, 202, 203, 204]
        let modeKey = "bundle-a|kv=fp16|reasoning=off"
        let imageA = "media:image:a"
        let imageB = "media:image:b"

        XCTAssertNotEqual(
            DiskCache.hashTokens(tokens, modelKey: modeKey, mediaSalt: imageA),
            DiskCache.hashTokens(tokens, modelKey: modeKey, mediaSalt: imageB))
        XCTAssertNotEqual(
            CacheBlock.computeBlockHash(
                parentHash: nil, tokenIds: tokens, modelKey: modeKey, mediaSalt: imageA),
            CacheBlock.computeBlockHash(
                parentHash: nil, tokenIds: tokens, modelKey: modeKey, mediaSalt: imageB))
        XCTAssertNotEqual(
            SSMStateCache.makeKey(
                tokens: tokens, boundary: tokens.count, mediaSalt: imageA, modelKey: modeKey),
            SSMStateCache.makeKey(
                tokens: tokens, boundary: tokens.count, mediaSalt: imageB, modelKey: modeKey))
    }

    func testDiskCoordinatorDoesNotShareEntriesAcrossReasoningModes() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mode-key-isolation-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = [301, 302, 303, 304]
        let off = makeDiskCoordinator(dir: dir, modeKey: "bundle-a|kv=fp16|reasoning=off")
        let on = makeDiskCoordinator(dir: dir, modeKey: "bundle-a|kv=fp16|reasoning=on")

        off.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil,
            cache: nil,
            mediaSalt: nil)

        guard case .hit(let matched, _, let detail, _, _, _) = off.fetch(tokens: tokens) else {
            return XCTFail("writer coordinator should read back its own disk entry")
        }
        XCTAssertEqual(matched, tokens.count)
        XCTAssertEqual(detail, .disk)

        if case .hit = on.fetch(tokens: tokens) {
            XCTFail("reasoning=on coordinator must not read reasoning=off disk entry")
        }
    }

    func testDiskCoordinatorDoesNotShareEntriesAcrossMediaSalt() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mode-key-media-isolation-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = [401, 402, 403, 404]
        let coordinator = makeDiskCoordinator(
            dir: dir,
            modeKey: "bundle-a|kv=fp16|reasoning=off")

        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil,
            cache: nil,
            mediaSalt: "image-a")

        guard case .hit(let matched, _, let detail, _, _, _) =
                coordinator.fetch(tokens: tokens, mediaSalt: "image-a")
        else {
            return XCTFail("same media salt should read back its own disk entry")
        }
        XCTAssertEqual(matched, tokens.count)
        XCTAssertEqual(detail, .disk)

        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: "image-b") {
            XCTFail("different media salt must not read the stored disk entry")
        }
        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: nil) {
            XCTFail("text-only nil media salt must not read a salted disk entry")
        }
    }

    func testSingleCoordinatorDoesNotShareEntriesAcrossDynamicReasoningSalt() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamic-mode-salt-isolation-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let tokens = [501, 502, 503, 504]
        let tokenArray = MLXArray(tokens.map(Int32.init)).expandedDimensions(axis: 0)
        let text = LMInput.Text(tokens: tokenArray)
        let reasoningOffSalt = computeCacheSalt(for: LMInput(
            text: text,
            cacheScopeSalt: "reasoning=off"))
        let reasoningOnSalt = computeCacheSalt(for: LMInput(
            text: text,
            cacheScopeSalt: "reasoning=on"))
        XCTAssertNotNil(reasoningOffSalt)
        XCTAssertNotNil(reasoningOnSalt)
        XCTAssertNotEqual(reasoningOffSalt, reasoningOnSalt)

        let coordinator = makeDiskCoordinator(dir: dir, modeKey: "bundle-a|kv=fp16")
        coordinator.storeAfterGeneration(
            promptTokens: tokens,
            perLayerData: fakeLayerData(tokenCount: tokens.count),
            ssmStates: nil,
            cache: nil,
            mediaSalt: reasoningOffSalt)

        guard case .hit(let matched, _, let detail, _, _, _) =
                coordinator.fetch(tokens: tokens, mediaSalt: reasoningOffSalt)
        else {
            return XCTFail("same dynamic reasoning salt should read back its own disk entry")
        }
        XCTAssertEqual(matched, tokens.count)
        XCTAssertEqual(detail, .disk)

        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: reasoningOnSalt) {
            XCTFail("reasoning=on request must not read reasoning=off cache entry")
        }
        if case .hit = coordinator.fetch(tokens: tokens, mediaSalt: nil) {
            XCTFail("unsalted request must not read dynamic reasoning cache entry")
        }
    }

    func testCacheScopeSaltIncludesReasoningEffortOnlyForSemanticKeys() {
        XCTAssertEqual(
            cacheScopeSalt(from: ["reasoning_effort": "high"]),
            "effort=high")
        XCTAssertEqual(
            cacheScopeSalt(from: ["reasoning_effort": " No_Think "]),
            "effort=no_think")
        XCTAssertEqual(
            cacheScopeSalt(from: [
                "enable_thinking": true,
                "reasoning_effort": "low",
            ]),
            "reasoning=on|effort=low")
        XCTAssertEqual(
            cacheScopeSalt(from: [
                "enable_thinking": false,
                "reasoning_effort": "max",
            ]),
            "reasoning=off|effort=max")
        XCTAssertNil(cacheScopeSalt(from: [
            "ui_panel": "visible",
            "temperature_source": "default",
        ]))
    }

    func testReasoningEffortChangesDynamicCacheSalt() {
        let tokenArray = MLXArray([Int32(601), Int32(602), Int32(603)])
            .expandedDimensions(axis: 0)
        let text = LMInput.Text(tokens: tokenArray)

        let noThink = computeCacheSalt(for: LMInput(
            text: text,
            cacheScopeSalt: cacheScopeSalt(from: ["reasoning_effort": "no_think"])))
        let high = computeCacheSalt(for: LMInput(
            text: text,
            cacheScopeSalt: cacheScopeSalt(from: ["reasoning_effort": "high"])))
        let unrelated = computeCacheSalt(for: LMInput(
            text: text,
            cacheScopeSalt: cacheScopeSalt(from: ["unrelated": "value"])))

        XCTAssertNotNil(noThink)
        XCTAssertNotNil(high)
        XCTAssertNotEqual(noThink, high)
        XCTAssertNil(unrelated)
    }

    private func makeDiskCoordinator(dir: URL, modeKey: String) -> CacheCoordinator {
        CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            enableDiskCache: true,
            diskCacheMaxGB: 1.0,
            diskCacheDir: dir,
            modelKey: modeKey))
    }

    private func fakeLayerData(tokenCount: Int) -> [(keys: MLXArray, values: MLXArray)?] {
        let keys = MLXArray.ones([1, 1, tokenCount, 4], dtype: .bfloat16)
        let values = MLXArray.ones([1, 1, tokenCount, 4], dtype: .bfloat16) * Float(0.5)
        MLX.eval(keys, values)
        return [(keys: keys, values: values)]
    }
}
