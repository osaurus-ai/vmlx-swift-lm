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
}
