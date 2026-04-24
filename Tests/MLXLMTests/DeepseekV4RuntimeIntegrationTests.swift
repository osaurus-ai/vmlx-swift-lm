// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// End-to-end runtime integration for DeepSeek-V4: chain
// `DeepseekV4ChatEncoder` → `ReasoningParser.forPrompt(stampName:
// promptTail:)` → `ToolCallProcessor(format: .dsml)` across all three
// DSV4 runtime modes and prove each mode routes content correctly.
//
// Three DSV4 modes (per research/DSV4-RUNTIME-ARCHITECTURE.md §4 +
// §23 of the DSV-FAMILY-RUNTIME-GUIDE):
//
//   1. `instruct` / chat mode — encoder emits closed `</think>` tail;
//      runtime parser must start in CONTENT state; `.chunk` events
//      only, 0 `.reasoning` leakage.
//   2. `reasoning` (thinking + effort=nil or "high") — encoder emits
//      open `<think>` tail; parser must start in REASONING state;
//      bytes before the model's `</think>` route to `.reasoning`,
//      after to `.chunk`.
//   3. `max reasoning` (thinking + effort="max") — same as mode 2
//      PLUS the REASONING_EFFORT_MAX preface is prepended at index 0.
//
// Tool-call overlay: all three modes MUST route DSML tool-call blocks
// (`<｜DSML｜tool_calls>`…) to `Generation.toolCall(ToolCall)` events
// and strip them from the visible content stream.

import Foundation
import MLXLMCommon
import Testing

@Suite("DSV4 runtime end-to-end (encoder → parser → tool processor)")
struct DeepseekV4RuntimeIntegrationTests {

    typealias Msg = DeepseekV4ChatEncoder.Message

    /// Take the last 256 chars of the prompt — that's what BatchEngine
    /// decodes via `_decodePromptTail(..., tokens: 64)` (a 64-token
    /// tail is ~200 chars for most tokenizers).
    static func promptTail(_ prompt: String) -> String {
        let idx =
            prompt.index(
                prompt.endIndex,
                offsetBy: -256,
                limitedBy: prompt.startIndex) ?? prompt.startIndex
        return String(prompt[idx...])
    }

    /// Stream a simulated model output through the full runtime
    /// pipeline (reasoning parser + tool-call processor) and return
    /// the aggregated event channels — just like `BatchEngine` and
    /// `Evaluate.generate`. Mirrors the canonical production order:
    /// reasoning parser first, tool processor second.
    ///
    /// We stream the model output char-by-char to reflect the actual
    /// runtime behavior (NaiveStreamingDetokenizer emits a few chars
    /// per token). Feeding the whole output as one chunk exposes edge
    /// cases in ToolCallProcessor that never fire in production.
    static func pipeline(
        stampName: String,
        promptTail: String,
        modelOutput: String,
        toolFormat: ToolCallFormat = .dsml
    ) -> (reasoning: String, content: String, toolCalls: [ToolCall]) {
        var reasoningParser = ReasoningParser.forPrompt(
            stampName: stampName, promptTail: promptTail)
        let tcProcessor = ToolCallProcessor(format: toolFormat)

        var reasoning = ""
        var content = ""

        func handle(_ segs: [ReasoningSegment]) {
            for seg in segs {
                switch seg {
                case .reasoning(let r):
                    reasoning += r
                case .content(let c):
                    if let visible = tcProcessor.processChunk(c) {
                        content += visible
                    }
                }
            }
        }

        // Stream char-by-char — matches production detokenizer output.
        for ch in modelOutput {
            if var parser = reasoningParser {
                let segs = parser.feed(String(ch))
                reasoningParser = parser
                handle(segs)
            } else {
                if let visible = tcProcessor.processChunk(String(ch)) {
                    content += visible
                }
            }
        }
        if var parser = reasoningParser {
            handle(parser.flush())
            reasoningParser = parser
        }
        tcProcessor.processEOS()

        return (reasoning, content, tcProcessor.toolCalls)
    }

    // MARK: - Mode 1: instruct (chat, no thinking)

    @Test("Mode 1 instruct: closed </think> tail → content-only output")
    func instructMode() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "What is 2+2?")],
            thinkingMode: .chat)
        // Sanity: prompt ends with </think>
        #expect(prompt.hasSuffix("</think>"))

        // Model emits a plain answer (no think markers).
        let modelOutput = "The answer is 4."
        let (reasoning, content, calls) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)

        #expect(reasoning.isEmpty,
            "instruct mode must produce zero .reasoning bytes")
        #expect(content == "The answer is 4.")
        #expect(calls.isEmpty)
    }

    @Test("Mode 1 instruct + tool call: DSML extracted, no reasoning leak")
    func instructModeWithToolCall() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(
                    role: .system,
                    content: "Helpful.",
                    tools: [[
                        "function": [
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": ["location": "string"]
                                as [String: any Sendable],
                        ] as [String: any Sendable]
                    ] as [String: any Sendable]]),
                Msg(role: .user, content: "Weather in Paris?"),
            ],
            thinkingMode: .chat)
        #expect(prompt.hasSuffix("</think>"))

        // Model output: direct tool call (no visible content before).
        let dsml = DeepseekV4Tokens.dsml
        let modelOutput =
            "<\(dsml)tool_calls>\n<\(dsml)invoke name=\"get_weather\">\n<\(dsml)parameter name=\"location\" string=\"true\">Paris</\(dsml)parameter>\n</\(dsml)invoke>\n</\(dsml)tool_calls>"
        let (reasoning, _, calls) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)

        #expect(reasoning.isEmpty,
            "instruct mode tool call must not leak to .reasoning")
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["location"] == .string("Paris"))
    }

    // MARK: - Mode 2: reasoning (thinking, no effort)

    @Test("Mode 2 reasoning: open <think> tail → output splits at </think>")
    func reasoningMode() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "Explain 2+2=4 step-by-step.")],
            thinkingMode: .thinking)
        #expect(prompt.hasSuffix("<think>"))

        // Model output: reasoning body, then closer, then answer.
        let modelOutput = "First, two plus two is four.</think>The answer is 4."
        let (reasoning, content, calls) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)

        #expect(reasoning == "First, two plus two is four.",
            "reasoning mode must route pre-</think> bytes to .reasoning")
        #expect(content == "The answer is 4.")
        #expect(calls.isEmpty)
    }

    @Test("Mode 2 reasoning + tool call AFTER </think>: split correctly")
    func reasoningModeWithToolAfterThinking() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(
                    role: .system,
                    content: "Helpful.",
                    tools: [[
                        "function": [
                            "name": "search",
                            "description": "search",
                            "parameters": ["q": "string"] as [String: any Sendable],
                        ] as [String: any Sendable]
                    ] as [String: any Sendable]]),
                Msg(role: .user, content: "search the latest news"),
            ],
            thinkingMode: .thinking,
            // tools force drop_thinking=false, but tail is still <think>
            // because index >= lastUserIdx.
            dropEarlierReasoning: true)
        #expect(prompt.hasSuffix("<think>"))

        let dsml = DeepseekV4Tokens.dsml
        let modelOutput =
            "I should search for news.</think>Let me call the tool.\n\n<\(dsml)tool_calls>\n<\(dsml)invoke name=\"search\">\n<\(dsml)parameter name=\"q\" string=\"true\">news</\(dsml)parameter>\n</\(dsml)invoke>\n</\(dsml)tool_calls>"
        let (reasoning, content, calls) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)

        #expect(reasoning == "I should search for news.")
        #expect(content.contains("Let me call the tool."),
            "post-</think> content must be in .chunk before the DSML block")
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "search")
        #expect(calls.first?.function.arguments["q"] == .string("news"))
    }

    // MARK: - Mode 3: max reasoning (thinking + effort=max)

    @Test("Mode 3 max reasoning: preface prepended, same tail semantics")
    func maxReasoningMode() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "Hard question.")],
            thinkingMode: .thinking,
            reasoningEffort: .max)

        // Preface IS prepended.
        #expect(prompt.contains("Reasoning Effort: Absolute maximum"),
            "max effort must inject the preface at turn 0")
        // Tail is still open <think> — the preface changes the prefix,
        // not the tail shape.
        #expect(prompt.hasSuffix("<think>"))

        // Stream a long reasoning block (emulating "max" budget),
        // then closer, then answer.
        let longThink = String(repeating: "step-by-step... ", count: 20)
        let modelOutput = longThink + "</think>Final answer."
        let (reasoning, content, calls) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)

        #expect(reasoning == longThink,
            "max reasoning must route the entire pre-</think> body to .reasoning")
        #expect(content == "Final answer.")
        #expect(calls.isEmpty)
    }

    @Test("Mode 3 max reasoning + multi-turn: drop_earlier strips prior turn")
    func maxReasoningMultiTurnDropEarlier() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [
                Msg(role: .user, content: "Turn 1."),
                Msg(
                    role: .assistant,
                    content: "Answer 1.",
                    reasoningContent: "Long turn 1 CoT that must be stripped."),
                Msg(role: .user, content: "Turn 2."),
            ],
            thinkingMode: .thinking,
            reasoningEffort: .max,
            dropEarlierReasoning: true)

        // Preface at turn 0.
        #expect(prompt.contains("Reasoning Effort: Absolute maximum"))
        // Prior turn's CoT is stripped.
        #expect(!prompt.contains("Long turn 1 CoT"))
        // Prior turn's answer survives.
        #expect(prompt.contains("Answer 1."))
        // Tail is open <think> for turn 2.
        #expect(prompt.hasSuffix("<think>"))

        let modelOutput = "turn 2 reasoning</think>Turn 2 answer."
        let (reasoning, content, _) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)
        #expect(reasoning == "turn 2 reasoning")
        #expect(content == "Turn 2 answer.")
    }

    // MARK: - Cross-mode: verify every mode emits DSML that the
    // DSMLToolCallParser can decode back, byte-consistent with encoding.

    @Test("DSML encode/decode round-trips identically in all three modes")
    func dsmlRoundTripAllModes() throws {
        let encoded = DeepseekV4ChatEncoder.renderToolCallInvoke(
            name: "set_config",
            arguments: "{\"retries\":3,\"enabled\":true,\"label\":\"ready\"}")

        let parser = DSMLToolCallParser()
        let wrapped =
            "<\(DeepseekV4Tokens.dsml)tool_calls>\n\(encoded)\n</\(DeepseekV4Tokens.dsml)tool_calls>"
        let calls = parser.parseEOS(wrapped, tools: nil)
        #expect(calls.count == 1)
        let args = calls.first!.function.arguments
        #expect(args["retries"] == .int(3))
        #expect(args["enabled"] == .bool(true))
        #expect(args["label"] == .string("ready"))
    }

    // MARK: - Stamp routing: model_type = deepseek_v4 → correct stamp

    @Test("Factory stamp resolution for deepseek_v4 model_type")
    func stampResolutionDeepseekV4() {
        // reasoningStampFromModelType is the explicit allowlist helper
        // every factory uses; deepseek_v4 must resolve to think_xml.
        #expect(reasoningStampFromModelType("deepseek_v4") == "think_xml")
        #expect(reasoningStampFromModelType("deepseek_v4_flash") == "think_xml")
        #expect(reasoningStampFromModelType("deepseek_v4_pro") == "think_xml")
        // Tool format resolver: deepseek_v4 → DSML.
        #expect(ToolCallFormat.infer(from: "deepseek_v4") == .dsml)
        #expect(ToolCallFormat.fromCapabilityName("deepseek_v4") == .dsml)
    }

    // MARK: - Edge: chat mode + stray <think> in model output
    // (belt-and-suspenders against an instruct-mode model emitting a
    // leaked think block)

    @Test("instruct mode with stray <think> in model output: still extracts visibly")
    func instructModeStrayThink() {
        let encoder = DeepseekV4ChatEncoder()
        let prompt = encoder.encode(
            messages: [Msg(role: .user, content: "Question?")],
            thinkingMode: .chat)
        #expect(prompt.hasSuffix("</think>"))

        // The "hardening" fix in 7ed9e1d makes this case start in
        // content because tail has closer only (no opener AFTER the
        // closer). Mid-stream <think>…</think> still latches via the
        // interleaved-think state machine.
        let modelOutput = "Here is the answer: <think>second thought</think> 42."
        let (reasoning, content, _) = Self.pipeline(
            stampName: "think_xml",
            promptTail: Self.promptTail(prompt),
            modelOutput: modelOutput)

        #expect(reasoning == "second thought",
            "mid-stream <think>…</think> must still be routed to .reasoning")
        #expect(content == "Here is the answer:  42.")
    }
}
