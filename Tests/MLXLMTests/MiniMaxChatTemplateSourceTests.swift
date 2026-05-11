// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("MiniMax chat template source")
struct MiniMaxChatTemplateSourceTests {

    @Test("MiniMax template file and fallback both wire tool schemas, calls, results, and thinking")
    func minimaxTemplateSourcesWireToolConversation() throws {
        let templatePath =
            "Libraries/MLXLMCommon/ChatTemplates/MiniMaxM2Minimal.jinja"
        let template = try String(
            contentsOfFile: templatePath,
            encoding: .utf8)
        let fallback = ChatTemplateFallbacks.minimaxM2Minimal

        for (label, source) in [
            ("template file", template),
            ("fallback", fallback),
        ] {
            #expect(source.contains("<minimax:tool_call>"), "\(label) must define MiniMax tool-call opener")
            #expect(source.contains("</minimax:tool_call>"), "\(label) must define MiniMax tool-call closer")
            #expect(source.contains("When making tool calls, use XML format"), "\(label) must teach the output format")
            #expect(source.contains("message.tool_calls"), "\(label) must render prior assistant tool calls")
            #expect(source.contains("message.role == 'tool'"), "\(label) must render tool result turns")
            #expect(source.contains("<response>"), "\(label) must wrap tool results")
            #expect(source.contains("_enable_thinking"), "\(label) must honor enable_thinking")
        }
    }
}
