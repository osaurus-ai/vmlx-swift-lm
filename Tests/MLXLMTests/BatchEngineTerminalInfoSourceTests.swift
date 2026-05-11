// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import Testing

@Suite("BatchEngine terminal info source coverage")
struct BatchEngineTerminalInfoSourceTests {
    @Test("public generate wrapper synthesizes terminal info if token stream closes without info")
    func generateWrapperSynthesizesTerminalInfoForEarlyClosedTokenStream() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(source.contains("var sawTerminalInfo = false"))
        #expect(source.contains("if !sawTerminalInfo"))
        #expect(source.contains("flush()"))
        #expect(source.contains("unclosedReasoning: unclosed"))
        #expect(source.contains("continuation.yield(.info(finalInfo))"))
    }

    @Test("MiniMax blank content after reasoning is stopped on both public generate paths")
    func miniMaxBlankContentAfterReasoningStopsBothGeneratePaths() throws {
        let batchSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let evalSource = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        for source in [batchSource, evalSource] {
            #expect(source.contains("blankContentAfterReasoningLimit"))
            #expect(source.contains("toolCallFormat == .minimaxM2"))
            #expect(source.contains("heldPostReasoningWhitespace.count >= limit"))
            #expect(source.contains("sawReasoningText"))
            #expect(source.contains("!sawContentText"))
        }
    }
}
