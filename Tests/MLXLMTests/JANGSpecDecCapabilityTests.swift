// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 5 iter 13 — pin the JANG capability stamp → DraftStrategy
// mapping. Closes the `JANGSpecDecCapabilityTests.swift` row in the
// test matrix.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("JANG SpecDec capability — Phase 5", .serialized)
struct JANGSpecDecCapabilityTests {

    private let modelDir = URL(fileURLWithPath: "/tmp/vmlx-fake-model")

    // MARK: - JangCapabilities decoding

    @Test("JangCapabilities has draftStrategy fields")
    func testFieldsExist() {
        let cap = JangCapabilities(
            draftStrategy: "ddtree",
            drafterPath: "drafter/",
            branchingBudget: 32,
            blockSize: 16)
        #expect(cap.draftStrategy == "ddtree")
        #expect(cap.drafterPath == "drafter/")
        #expect(cap.branchingBudget == 32)
        #expect(cap.blockSize == 16)
    }

    @Test("JangCapabilities default-init has no draft strategy")
    func testDefaultNoDraftStrategy() {
        let cap = JangCapabilities()
        #expect(cap.draftStrategy == nil)
        #expect(cap.drafterPath == nil)
        #expect(cap.branchingBudget == nil)
        #expect(cap.blockSize == nil)
    }

    // MARK: - ParserResolution.draftStrategy

    @Test("nil capabilities returns nil strategy")
    func testNilCapabilities() {
        let (s, source) = ParserResolution.draftStrategy(
            capabilities: nil, modelDirectory: modelDir)
        #expect(s == nil)
        #expect(source == .none)
    }

    @Test("Missing draftStrategy stamp returns nil")
    func testMissingStamp() {
        let cap = JangCapabilities(toolParser: "qwen")
        let (s, source) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(s == nil)
        #expect(source == .none)
    }

    @Test("\"none\" stamp returns nil strategy")
    func testNoneStamp() {
        let cap = JangCapabilities(
            draftStrategy: "none",
            drafterPath: "drafter/",
            blockSize: 16)
        let (s, _) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(s == nil)
    }

    @Test("dflash stamp resolves to .dflash with absolute URL")
    func testDflashStamp() {
        let cap = JangCapabilities(
            draftStrategy: "dflash",
            drafterPath: "drafter/",
            blockSize: 8)
        let (s, source) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(source == .jangStamped)
        guard case let .dflash(url, blockSize) = s else {
            Issue.record("Expected .dflash, got \(String(describing: s))")
            return
        }
        #expect(blockSize == 8)
        #expect(url.path.hasSuffix("/drafter"))
    }

    @Test("ddtree stamp resolves to .ddtree with defaults")
    func testDdtreeStamp() {
        let cap = JangCapabilities(
            draftStrategy: "ddtree",
            drafterPath: "drafter/",
            branchingBudget: 64,
            blockSize: 16)
        let (s, source) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(source == .jangStamped)
        guard case let .ddtree(url, budget, blockSize) = s else {
            Issue.record("Expected .ddtree, got \(String(describing: s))")
            return
        }
        #expect(budget == 64)
        #expect(blockSize == 16)
        #expect(url.path.hasSuffix("/drafter"))
    }

    @Test("ddtree without branchingBudget defaults to 32")
    func testDdtreeDefaultBudget() {
        let cap = JangCapabilities(
            draftStrategy: "ddtree",
            drafterPath: "drafter/",
            blockSize: 16)
        let (s, _) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        guard case .ddtree(_, let budget, _) = s else {
            Issue.record("Expected .ddtree")
            return
        }
        #expect(budget == 32)
    }

    @Test("Unknown stamp returns nil")
    func testUnknownStamp() {
        let cap = JangCapabilities(
            draftStrategy: "custom_magic",
            drafterPath: "drafter/",
            blockSize: 8)
        let (s, source) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(s == nil)
        #expect(source == .none)
    }

    @Test("Missing blockSize returns nil (invalid stamp)")
    func testMissingBlockSize() {
        let cap = JangCapabilities(
            draftStrategy: "dflash",
            drafterPath: "drafter/")
        // blockSize is required for both .dflash + .ddtree
        let (s, _) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(s == nil)
    }

    @Test("Missing drafterPath returns nil")
    func testMissingDrafterPath() {
        let cap = JangCapabilities(
            draftStrategy: "dflash",
            blockSize: 8)
        let (s, _) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: modelDir)
        #expect(s == nil)
    }

    // MARK: - JSON decoder path

    @Test("jang_config.json capabilities block decodes draftStrategy fields")
    func testJSONDecodesDraftCapabilities() throws {
        // Build a jang_config.json fragment with all SpecDec fields set.
        let json = """
        {
          "format": "jang",
          "format_version": "2.0",
          "quantization": {},
          "source_model": {},
          "architecture": {},
          "runtime": {},
          "capabilities": {
            "reasoning_parser": "qwen3",
            "tool_parser": "qwen",
            "draft_strategy": "ddtree",
            "drafter_path": "drafter/",
            "branching_budget": 48,
            "block_size": 16
          }
        }
        """
        // Write to a temp dir so JangLoader.loadConfig can read it.
        let tmp = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("vmlx-jang-specdec-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let cfgURL = tmp.appendingPathComponent("jang_config.json")
        try json.write(to: cfgURL, atomically: true, encoding: .utf8)

        let loaded = try JangLoader.loadConfig(at: tmp)
        let cap = try #require(loaded.capabilities)
        #expect(cap.draftStrategy == "ddtree")
        #expect(cap.drafterPath == "drafter/")
        #expect(cap.branchingBudget == 48)
        #expect(cap.blockSize == 16)

        // And the resolver turns it into a .ddtree case with correct
        // absolute URL.
        let (strategy, source) = ParserResolution.draftStrategy(
            capabilities: cap, modelDirectory: tmp)
        #expect(source == .jangStamped)
        guard case .ddtree(let url, let budget, let block) = strategy else {
            Issue.record("Expected .ddtree from JSON")
            return
        }
        #expect(budget == 48)
        #expect(block == 16)
        #expect(url.path.hasSuffix("/drafter"))
    }
}
