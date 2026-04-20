// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import MLXLMCommon
import Testing

/// Mini driver that mirrors the post-detokenizer pipeline shared by
/// Evaluate.swift / BatchEngine.swift / SpecDecStream.swift (ReasoningParser →
/// ToolCallProcessor → emit). Used only by the `Generation.reasoning` event
/// regression tests below — keeps them free of model-loading machinery.
private func driveGenerationPipeline(
    chunks: [String],
    reasoningParserStartTag: String = "<think>",
    reasoningParserEndTag: String = "</think>",
    toolCallFormat: ToolCallFormat = .json
) -> [Generation] {
    var reasoningParser: ReasoningParser? = ReasoningParser(
        startTag: reasoningParserStartTag, endTag: reasoningParserEndTag)
    let toolCallProcessor = ToolCallProcessor(format: toolCallFormat)
    var events: [Generation] = []

    func pump(_ raw: String) {
        let pieces: [String]
        if var parser = reasoningParser {
            var kept: [String] = []
            for segment in parser.feed(raw) {
                switch segment {
                case .content(let c):
                    kept.append(c)
                case .reasoning(let r):
                    events.append(.reasoning(r))
                }
            }
            reasoningParser = parser
            pieces = kept
        } else {
            pieces = [raw]
        }
        for piece in pieces {
            if let textToYield = toolCallProcessor.processChunk(piece) {
                events.append(.chunk(textToYield))
            }
            if let toolCall = toolCallProcessor.toolCalls.popLast() {
                events.append(.toolCall(toolCall))
            }
        }
    }

    func flush() {
        if var parser = reasoningParser {
            for segment in parser.flush() {
                switch segment {
                case .content(let c):
                    if let textToYield = toolCallProcessor.processChunk(c) {
                        events.append(.chunk(textToYield))
                    }
                    if let toolCall = toolCallProcessor.toolCalls.popLast() {
                        events.append(.toolCall(toolCall))
                    }
                case .reasoning(let r):
                    events.append(.reasoning(r))
                }
            }
            reasoningParser = parser
        }
        toolCallProcessor.processEOS()
        for toolCall in toolCallProcessor.toolCalls {
            events.append(.toolCall(toolCall))
        }
    }

    for chunk in chunks { pump(chunk) }
    flush()
    return events
}

@Suite("Generation.reasoning event")
struct GenerationReasoningEventTests {

    @Test("reasoning block surfaces as .reasoning events, answer as .chunk")
    func testReasoningAndContentSeparated() {
        let events = driveGenerationPipeline(chunks: [
            "<think>weighing options</think>final answer"
        ])

        let reasoningPieces = events.compactMap { $0.reasoning }
        let contentPieces = events.compactMap { $0.chunk }
        #expect(reasoningPieces.joined() == "weighing options")
        #expect(contentPieces.joined() == "final answer")
    }

    @Test("reasoning streams char-by-char across chunk boundaries")
    func testReasoningStreamsAcrossChunks() {
        // Simulate the token-by-token delivery osaurus sees from the
        // detokenizer. The parser must still emit a coherent .reasoning
        // stream even when the start/end tags straddle chunks.
        let prompt = "pre<think>A B C</think>post"
        let chunks = prompt.map { String($0) }
        let events = driveGenerationPipeline(chunks: chunks)

        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()
        #expect(reasoning == "A B C")
        #expect(content == "prepost")
    }

    @Test("unclosed <think> flushes trailing reasoning at end-of-stream")
    func testUnclosedReasoningFlushesOnEOS() {
        let events = driveGenerationPipeline(chunks: [
            "answer<think>truncated mid-thought"
        ])
        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()
        // Matches ReasoningParser.split semantics: post-start bytes are
        // reasoning, even if no closing tag arrived.
        #expect(reasoning == "truncated mid-thought")
        #expect(content == "answer")
    }

    @Test("no reasoning tag → zero .reasoning events")
    func testNoReasoningEmitsNoReasoningEvents() {
        let events = driveGenerationPipeline(chunks: ["just an answer"])
        let reasoningCount = events.filter { $0.reasoning != nil }.count
        #expect(reasoningCount == 0)
        #expect(events.compactMap { $0.chunk }.joined() == "just an answer")
    }

    @Test("reasoning + tool-call coexist — both surfaced, .chunk is clean")
    func testReasoningAndToolCallCoexist() {
        // ToolCallProcessor(format: .json) extracts inline JSON tool calls.
        let events = driveGenerationPipeline(chunks: [
            "<think>need to call weather</think>",
            "Here you go: ",
            #"{"name": "get_weather", "arguments": {"city": "SF"}}"#,
        ])
        let reasoning = events.compactMap { $0.reasoning }.joined()
        let content = events.compactMap { $0.chunk }.joined()
        let toolCalls = events.compactMap { $0.toolCall }

        #expect(reasoning == "need to call weather")
        #expect(content == "Here you go: ")
        #expect(toolCalls.count == 1)
        #expect(toolCalls.first?.function.name == "get_weather")
    }

    @Test("Generation enum has .reasoning case + reasoning computed property")
    func testGenerationEnumSurface() {
        // Pure enum smoke test — pin the public API surface osaurus
        // depends on.
        let r: Generation = .reasoning("thinking out loud")
        #expect(r.reasoning == "thinking out loud")
        #expect(r.chunk == nil)
        #expect(r.toolCall == nil)
        #expect(r.info == nil)

        let c: Generation = .chunk("answer")
        #expect(c.reasoning == nil)
        #expect(c.chunk == "answer")
    }
}

@Suite("ReasoningParser")
struct ReasoningParserTests {

    // MARK: - Whole-string split

    @Test func splitEmpty() {
        let (r, c) = ReasoningParser.split("")
        #expect(r.isEmpty)
        #expect(c.isEmpty)
    }

    @Test func splitNoTags() {
        let (r, c) = ReasoningParser.split("hello world")
        #expect(r.isEmpty)
        #expect(c == "hello world")
    }

    @Test func splitSingleReasoningBlock() {
        let (r, c) = ReasoningParser.split("<think>weighing options</think>final answer")
        #expect(r == "weighing options")
        #expect(c == "final answer")
    }

    @Test func splitReasoningOnly() {
        let (r, c) = ReasoningParser.split("prefix<think>only thinking</think>")
        #expect(r == "only thinking")
        #expect(c == "prefix")
    }

    @Test func splitMultipleReasoningBlocks() {
        // Two interleaved think blocks — accumulate.
        let (r, c) = ReasoningParser.split(
            "first<think>r1</think>middle<think>r2</think>last")
        #expect(r == "r1r2")
        #expect(c == "firstmiddlelast")
    }

    @Test func splitUnclosedReasoning() {
        // No `</think>` → everything after `<think>` is reasoning.
        let (r, c) = ReasoningParser.split("answer<think>truncated...")
        #expect(r == "truncated...")
        #expect(c == "answer")
    }

    // MARK: - Streaming

    @Test func streamCharByCharSimpleBlock() {
        // Drip-feed every character to verify partial-tag holdback works.
        var parser = ReasoningParser()
        let input = "<think>hi</think>ok"
        var reasoning = ""
        var content = ""
        for ch in input {
            for seg in parser.feed(String(ch)) {
                switch seg {
                case .reasoning(let r): reasoning.append(r)
                case .content(let c): content.append(c)
                }
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "hi")
        #expect(content == "ok")
    }

    @Test func streamTagSplitAcrossChunks() {
        // The opening tag `<think>` arrives as `<thi` then `nk>`.
        var parser = ReasoningParser()
        let chunks = ["pre<thi", "nk>thoughts</thi", "nk>post"]
        var reasoning = ""
        var content = ""
        for ch in chunks {
            for seg in parser.feed(ch) {
                switch seg {
                case .reasoning(let r): reasoning.append(r)
                case .content(let c): content.append(c)
                }
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "thoughts")
        #expect(content == "prepost")
    }

    @Test func streamAdjacentBlocks() {
        // `</think>` immediately followed by `<think>` — back-to-back.
        var parser = ReasoningParser()
        let input = "<think>a</think><think>b</think>tail"
        var reasoning = ""
        var content = ""
        for seg in parser.feed(input) {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        #expect(reasoning == "ab")
        #expect(content == "tail")
    }

    @Test func streamFlushDrainsBufferedPartial() {
        // Stream ends mid-pretag — flush must still emit the buffered text
        // as content rather than dropping it.
        var parser = ReasoningParser()
        var content = ""
        for seg in parser.feed("hello<thi") {
            if case .content(let c) = seg { content.append(c) }
        }
        for seg in parser.flush() {
            if case .content(let c) = seg { content.append(c) }
        }
        #expect(content == "hello<thi")
    }

    // MARK: - Custom tags

    @Test func customTags() {
        let (r, c) = ReasoningParser.split(
            "[REASON]inner[/REASON]visible",
            startTag: "[REASON]", endTag: "[/REASON]")
        #expect(r == "inner")
        #expect(c == "visible")
    }

    // MARK: - Capability-name resolution

    @Test func capabilityAliasesQwen3() {
        for name in ["qwen3", "qwen3_5", "qwen3_6", "think_xml", "deepseek_r1"] {
            #expect(
                ReasoningParser.fromCapabilityName(name) != nil,
                "\(name) should resolve to a parser")
        }
    }

    @Test func capabilityNoneAliases() {
        for name in ["none", "off", "disabled", "mistral", "gemma4"] {
            #expect(
                ReasoningParser.fromCapabilityName(name) == nil,
                "\(name) should resolve to no parser")
        }
    }

    @Test func capabilityUnknownReturnsNil() {
        #expect(ReasoningParser.fromCapabilityName(nil) == nil)
        #expect(ReasoningParser.fromCapabilityName("") == nil)
        #expect(ReasoningParser.fromCapabilityName("madeup") == nil)
    }

    // MARK: - ParserResolution precedence

    @Test func reasoningStampedQwenWinsOverHeuristic() {
        let cap = JangCapabilities(reasoningParser: "qwen3")
        let (parser, source) = ParserResolution.reasoning(
            capabilities: cap, modelType: "mistral4")
        #expect(parser != nil, "stamp must override mistral heuristic")
        #expect(source == .jangStamped)
    }

    @Test func reasoningStampedNoneWinsOverHeuristic() {
        let cap = JangCapabilities(reasoningParser: "none")
        let (parser, source) = ParserResolution.reasoning(
            capabilities: cap, modelType: "qwen3_5_moe")
        #expect(parser == nil, "stamp `none` must suppress qwen heuristic")
        #expect(source == .jangStamped)
    }

    @Test func reasoningHeuristicQwenFallback() {
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: "qwen3_5_moe")
        #expect(parser != nil)
        #expect(source == .modelTypeHeuristic)
    }

    @Test func reasoningHeuristicQwen36TextConfigVariant() {
        // Qwen 3.6 sometimes surfaces model_type=qwen3_5_moe_text from text_config.
        // Heuristic must still return a reasoning parser.
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: "qwen3_5_moe_text")
        #expect(parser != nil)
        #expect(source == .modelTypeHeuristic)
    }

    @Test func reasoningHeuristicMistralReturnsNone() {
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: "mistral4")
        #expect(parser == nil)
        #expect(source == .modelTypeHeuristic)
    }

    @Test func reasoningEmptyInputsReturnNone() {
        let (parser, source) = ParserResolution.reasoning(
            capabilities: nil, modelType: nil)
        #expect(parser == nil)
        #expect(source == .none)
    }
}

// MARK: - ToolCallFormat capability resolution

@Suite("ToolCallFormat capability")
struct ToolCallFormatCapabilityTests {

    @Test func directRawValueWins() {
        #expect(ToolCallFormat.fromCapabilityName("xml_function") == .xmlFunction)
        #expect(ToolCallFormat.fromCapabilityName("minimax_m2") == .minimaxM2)
        #expect(ToolCallFormat.fromCapabilityName("kimi_k2") == .kimiK2)
    }

    @Test func qwenAliases() {
        for name in ["qwen", "qwen3", "qwen3_5", "qwen3_6", "qwen3_coder"] {
            #expect(
                ToolCallFormat.fromCapabilityName(name) == .xmlFunction,
                "\(name) should map to xml_function")
        }
    }

    @Test func minimaxAlias() {
        #expect(ToolCallFormat.fromCapabilityName("minimax") == .minimaxM2)
    }

    @Test func glmAndDeepseekAliases() {
        #expect(ToolCallFormat.fromCapabilityName("glm47") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("glm4_moe") == .glm4)
        #expect(ToolCallFormat.fromCapabilityName("deepseek") == .glm4)
    }

    @Test func nemotronAlias() {
        #expect(ToolCallFormat.fromCapabilityName("nemotron") == .xmlFunction)
        #expect(ToolCallFormat.fromCapabilityName("nemotron_h") == .xmlFunction)
    }

    @Test func unknownReturnsNil() {
        #expect(ToolCallFormat.fromCapabilityName(nil) == nil)
        #expect(ToolCallFormat.fromCapabilityName("") == nil)
        #expect(ToolCallFormat.fromCapabilityName("zzzunknown") == nil)
    }

    @Test func resolutionStampedWinsOverHeuristic() {
        let cap = JangCapabilities(toolParser: "qwen3_coder")
        let (fmt, src) = ParserResolution.toolCall(
            capabilities: cap, modelType: "mistral3")
        #expect(fmt == .xmlFunction, "stamp must override mistral heuristic")
        #expect(src == .jangStamped)
    }

    @Test func resolutionHeuristicFallback() {
        let (fmt, src) = ParserResolution.toolCall(
            capabilities: nil, modelType: "qwen3_5_moe")
        #expect(fmt == .xmlFunction)
        #expect(src == .modelTypeHeuristic)
    }

    @Test func resolutionEmptyReturnsNone() {
        let (fmt, src) = ParserResolution.toolCall(
            capabilities: nil, modelType: nil)
        #expect(fmt == nil)
        #expect(src == .none)
    }
}
