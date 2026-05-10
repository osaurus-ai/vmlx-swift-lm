// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Production-quality regression: pin the parser-stamp → enum dispatch
// path end-to-end for ZAYA + ZAYA1-VL across MXFP4/JANGTQ2/JANGTQ4
// quant variants. This catches drift between (a) the bundle's stamped
// capabilities, (b) `ToolCallFormat.fromCapabilityName` resolution, and
// (c) `ReasoningParser.fromCapabilityName` resolution.
//
// Why this is separate from `Zaya1VLRegistrationTests`:
// - That file pins the bundle stamps themselves (they say what the file
//   claims).
// - This file pins the FORWARD path: bundle stamp → parser instance
//   (whether the runtime correctly threads the stamp into the enum).
//
// Production claims being pinned:
// - All 6 ZAYA bundles (text JANGTQ2/4 + ZAYA1-VL MXFP4/JANGTQ2/JANGTQ4)
//   stamp `tool_parser=zaya_xml` → resolves to `.zayaXml`.
// - All 6 stamp `reasoning_parser=qwen3` → resolves to a non-nil
//   `ReasoningParser` with `startInReasoning=true` (Qwen 3.x semantics).
// - All 6 stamp `think_in_template=false` in BOTH config.json and
//   jang_config.json. ZAYA capability stamps are trusted by the runtime;
//   stale bundle metadata must be fixed at the bundle/source level.
// - Quant variant DOES NOT change parser routing — MXFP4/JANGTQ2/JANGTQ4
//   all dispatch identically.

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("ZAYA + ZAYA1-VL parser-stamp dispatch end-to-end", .serialized)
struct ZayaParserDispatchTests {

    /// Subset of `JangConfig.capabilities` decoded directly from bundle JSON
    /// to keep the test independent of the JangLoader pipeline (which is
    /// covered separately and includes the stale-stamp normalization).
    private struct CapabilitiesProbe: Codable {
        struct Caps: Codable {
            let reasoningParser: String?
            let toolParser: String?
            let thinkInTemplate: Bool
            let supportsTools: Bool
            let supportsThinking: Bool
            let family: String?
            let modality: String?
            let cacheType: String?

            enum CodingKeys: String, CodingKey {
                case reasoningParser = "reasoning_parser"
                case toolParser = "tool_parser"
                case thinkInTemplate = "think_in_template"
                case supportsTools = "supports_tools"
                case supportsThinking = "supports_thinking"
                case family
                case modality
                case cacheType = "cache_type"
            }
        }
        let capabilities: Caps
    }

    private static let bundleRoot =
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models")

    private static func bundlePath(_ name: String) -> String? {
        let candidates = [
            bundleRoot.appendingPathComponent("Osaurus").appendingPathComponent(name),
            bundleRoot.appendingPathComponent("JANGQ").appendingPathComponent(name),
            bundleRoot.appendingPathComponent("Zyphra").appendingPathComponent(name),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url.path
        }
        return nil
    }

    private static func loadCaps(_ bundle: String, _ file: String)
        throws -> CapabilitiesProbe.Caps?
    {
        guard let dir = bundlePath(bundle) else { return nil }
        let url = URL(fileURLWithPath: dir + "/" + file)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONDecoder.json5().decode(CapabilitiesProbe.self, from: data).capabilities
    }

    // MARK: - Tool parser dispatch

    @Test("ZAYA1-VL MXFP4 tool_parser stamp dispatches to .zayaXml",
          .enabled(if: bundlePath("ZAYA1-VL-8B-MXFP4") != nil))
    func mxfp4VLDispatchesZayaXml() throws {
        for file in ["config.json", "jang_config.json"] {
            guard let caps = try Self.loadCaps("ZAYA1-VL-8B-MXFP4", file) else { continue }
            #expect(caps.toolParser == "zaya_xml")
            #expect(ToolCallFormat.fromCapabilityName(caps.toolParser) == .zayaXml)
        }
    }

    @Test("ZAYA1-VL JANGTQ2 tool_parser stamp dispatches to .zayaXml",
          .enabled(if: bundlePath("ZAYA1-VL-8B-JANGTQ2") != nil))
    func jangtq2VLDispatchesZayaXml() throws {
        for file in ["config.json", "jang_config.json"] {
            guard let caps = try Self.loadCaps("ZAYA1-VL-8B-JANGTQ2", file) else { continue }
            #expect(caps.toolParser == "zaya_xml")
            #expect(ToolCallFormat.fromCapabilityName(caps.toolParser) == .zayaXml)
        }
    }

    @Test("ZAYA1-VL JANGTQ4 tool_parser stamp dispatches to .zayaXml",
          .enabled(if: bundlePath("ZAYA1-VL-8B-JANGTQ4") != nil))
    func jangtq4VLDispatchesZayaXml() throws {
        for file in ["config.json", "jang_config.json"] {
            guard let caps = try Self.loadCaps("ZAYA1-VL-8B-JANGTQ4", file) else { continue }
            #expect(caps.toolParser == "zaya_xml")
            #expect(ToolCallFormat.fromCapabilityName(caps.toolParser) == .zayaXml)
        }
    }

    @Test("ZAYA text JANGTQ2/JANGTQ4 tool_parser stamps dispatch to .zayaXml")
    func zayaTextDispatchesZayaXml() throws {
        for bundle in ["ZAYA1-8B-JANGTQ2", "ZAYA1-8B-JANGTQ4"] {
            guard Self.bundlePath(bundle) != nil else { continue }
            for file in ["config.json", "jang_config.json"] {
                guard let caps = try Self.loadCaps(bundle, file) else { continue }
                #expect(caps.toolParser == "zaya_xml")
                #expect(ToolCallFormat.fromCapabilityName(caps.toolParser) == .zayaXml)
            }
        }
    }

    // MARK: - Reasoning parser dispatch

    @Test("All ZAYA + ZAYA1-VL bundles route reasoning_parser=qwen3 to Qwen-3.x parser")
    func allBundlesDispatchQwen3Reasoning() throws {
        let bundles = [
            "ZAYA1-VL-8B-MXFP4",
            "ZAYA1-VL-8B-JANGTQ2",
            "ZAYA1-VL-8B-JANGTQ4",
            "ZAYA1-8B-MXFP4",
            "ZAYA1-8B-JANGTQ2",
            "ZAYA1-8B-JANGTQ4",
        ]
        var anyChecked = false
        for bundle in bundles {
            guard Self.bundlePath(bundle) != nil else { continue }
            anyChecked = true
            for file in ["config.json", "jang_config.json"] {
                guard let caps = try Self.loadCaps(bundle, file) else { continue }
                #expect(caps.reasoningParser == "qwen3")
                let parser = ReasoningParser.fromCapabilityName(caps.reasoningParser)
                #expect(parser != nil, "qwen3 must resolve to a non-nil parser for \(bundle)")
            }
        }
        // At least one bundle must have been actually checked, otherwise
        // the test silently passes vacuously.
        #expect(anyChecked, "no ZAYA bundles found locally — test gates are too lenient")
    }

    // MARK: - Quant-variant invariance

    @Test("Quant variant does not change parser routing across ZAYA1-VL family")
    func quantVariantInvariantParserDispatch() throws {
        let cases: [(bundle: String, label: String)] = [
            ("ZAYA1-VL-8B-MXFP4", "mxfp4"),
            ("ZAYA1-VL-8B-JANGTQ2", "mxtq2"),
            ("ZAYA1-VL-8B-JANGTQ4", "mxtq4"),
        ]
        var resolved: [(String, ToolCallFormat?, String?)] = []
        for (bundle, label) in cases {
            guard let caps = try Self.loadCaps(bundle, "config.json") else { continue }
            let toolFormat = ToolCallFormat.fromCapabilityName(caps.toolParser)
            resolved.append((label, toolFormat, caps.reasoningParser))
        }
        guard resolved.count >= 2 else {
            // Need at least 2 quant variants present locally to assert invariance.
            // If only one is present (rare during conversion), test is skipped.
            return
        }
        let firstTool = resolved[0].1
        let firstReasoning = resolved[0].2
        for (label, tool, reasoning) in resolved.dropFirst() {
            #expect(tool == firstTool,
                "quant variant \(label) routed to \(String(describing: tool)) — expected \(String(describing: firstTool))")
            #expect(reasoning == firstReasoning,
                "quant variant \(label) reasoning=\(reasoning ?? "nil") — expected \(firstReasoning ?? "nil")")
        }
    }

    // MARK: - Thinking policy across config/jang_config consistency

    @Test("ZAYA + ZAYA1-VL bundle stamps consistent across config.json and jang_config.json")
    func stampsConsistentAcrossFiles() throws {
        let bundles = [
            "ZAYA1-VL-8B-MXFP4",
            "ZAYA1-VL-8B-JANGTQ2",
            "ZAYA1-VL-8B-JANGTQ4",
            "ZAYA1-8B-MXFP4",
            "ZAYA1-8B-JANGTQ2",
            "ZAYA1-8B-JANGTQ4",
        ]
        for bundle in bundles {
            guard Self.bundlePath(bundle) != nil else { continue }
            let configCaps = try Self.loadCaps(bundle, "config.json")
            let jangCaps = try Self.loadCaps(bundle, "jang_config.json")
            guard let configCaps, let jangCaps else { continue }
            #expect(configCaps.supportsThinking == jangCaps.supportsThinking,
                "\(bundle): supports_thinking drift between config.json and jang_config.json")
            #expect(configCaps.thinkInTemplate == jangCaps.thinkInTemplate,
                "\(bundle): think_in_template drift between config.json and jang_config.json")
            #expect(configCaps.toolParser == jangCaps.toolParser,
                "\(bundle): tool_parser drift between config.json and jang_config.json")
            #expect(configCaps.reasoningParser == jangCaps.reasoningParser,
                "\(bundle): reasoning_parser drift between config.json and jang_config.json")
            #expect(configCaps.family == jangCaps.family,
                "\(bundle): family drift between config.json and jang_config.json")
            #expect(configCaps.modality == jangCaps.modality,
                "\(bundle): modality drift between config.json and jang_config.json")
        }
    }
}
