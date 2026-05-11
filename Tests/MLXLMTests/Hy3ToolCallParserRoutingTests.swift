// Hy3 / Tencent Hunyuan tool-call parser — additive coverage to
// `Hy3ParserDispatchTests`. Covers what that suite doesn't:
//   - case-insensitive `infer` / `fromCapabilityName`
//   - explicit "does not silently route to other formats" guard
//   - parser tag identity (startTag / endTag)
//   - malformed-input rejection (no <tool_sep> in body)
//
// The basic routing alias coverage + multi-call parser coverage live in
// `Hy3ParserDispatchTests.hy3ToolParserAliasesResolveToHunyuan` and
// `hy3HunyuanParserExtractsMultipleCalls`. This file does NOT duplicate
// either; it only fills gaps.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("Hy3 / Tencent Hunyuan tool-call parser routing — additive coverage")
struct Hy3ToolCallParserRoutingTests {

    @Test("ToolCallFormat.infer is case-insensitive for hy3 model_type variants")
    func inferIsCaseInsensitive() throws {
        // The dispatch suite covers `hy_v3` lowercase; pin upper/mixed too so a
        // future bundle stamping the model_type with mixed case routes
        // correctly.
        for input in ["HY_V3", "Hy_V3", "Hy3", "HY3", "hY3_PrEvIeW"] {
            #expect(
                ToolCallFormat.infer(from: input) == .hunyuan,
                "Hy3 model_type \(input) must route to .hunyuan regardless of case")
        }
    }

    @Test("ToolCallFormat.infer does not silently route Hy3 to other formats")
    func inferDoesNotLeakToOtherFormats() throws {
        // A Hy3 bundle must NOT be matched by the zaya/qwen/xml_function/
        // glm4/mistral catch-alls in `infer`. This pins branch ordering.
        for input in ["hy_v3", "hy3", "hy3_preview", "hy-v3"] {
            let result = ToolCallFormat.infer(from: input)
            #expect(result == .hunyuan)
            #expect(result != .zayaXml)
            #expect(result != .xmlFunction)
            #expect(result != .glm4)
            #expect(result != .json)
            #expect(result != .mistral)
        }
    }

    @Test("HunyuanToolCallParser exposes the canonical Hy3 wrapper tags")
    func parserTagIdentity() throws {
        let parser = ToolCallFormat.hunyuan.createParser()
        #expect(parser.startTag == "<tool_calls>")
        #expect(parser.endTag == "</tool_calls>")
    }

    @Test("HunyuanToolCallParser rejects a tool_call body missing the <tool_sep> separator")
    func parserRejectsMalformedNoSeparator() throws {
        let parser = HunyuanToolCallParser()
        // No <tool_sep> → not a valid Hunyuan call. The parser must NOT
        // invent a function name from the raw body.
        let payload = """
            <tool_calls>
            <tool_call>just_a_name_no_separator</tool_call>
            </tool_calls>
            """
        #expect(parser.parseEOS(payload, tools: nil).isEmpty)
    }

    @Test("HunyuanToolCallParser returns empty for an empty tool_calls block")
    func parserHandlesEmptyBlock() throws {
        let parser = HunyuanToolCallParser()
        #expect(parser.parseEOS("<tool_calls></tool_calls>", tools: nil).isEmpty)
    }

    @Test("ToolCallProcessor extracts every Hunyuan call when the wrapper closes normally")
    func processorExtractsEveryClosedWrapperCall() throws {
        let processor = ToolCallProcessor(format: .hunyuan)
        let payload = """
            before
            <tool_calls>
            <tool_call>first<tool_sep>
            <arg_key>x</arg_key><arg_value>1</arg_value>
            </tool_call>
            <tool_call>second<tool_sep>
            <arg_key>y</arg_key><arg_value>"two"</arg_value>
            </tool_call>
            </tool_calls>
            after
            """

        var visible = ""
        for scalar in payload {
            if let chunk = processor.processChunk(String(scalar)) {
                visible += chunk
            }
        }
        processor.processEOS()

        #expect(processor.toolCalls.map(\.function.name) == ["first", "second"])
        #expect(processor.toolCalls[0].function.arguments["x"] == .int(1))
        #expect(processor.toolCalls[1].function.arguments["y"] == .string("two"))
        #expect(visible.contains("before"))
        #expect(visible.contains("after"))
        #expect(!visible.contains("<tool_calls>"))
        #expect(!visible.contains("<tool_call>"))
    }

    @Test("ToolCallProcessor flushes invalid Hunyuan tool-call prose during stream")
    func processorFlushesInvalidWrapperProseDuringStream() throws {
        let processor = ToolCallProcessor(format: .hunyuan)
        let text = "Answer text <tool_calls>I'm not calling a tool; this is prose."
        var visible = ""

        for ch in text {
            if let chunk = processor.processChunk(String(ch)) {
                visible += chunk
            }
        }

        #expect(
            visible == text,
            "Invalid Hy3/Hunyuan tool-call-looking prose must not stay buffered until EOS"
        )
        #expect(processor.toolCalls.isEmpty)
        #expect(processor.processEOS() == nil)
    }
}
