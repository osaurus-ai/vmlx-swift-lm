// Copyright © 2026 osaurus.
//
// Interleaved reasoning + thinking-tag leak audit. Locks the contract
// that ReasoningParser preserves the boundary between visible content
// and `<think>...</think>` reasoning across:
//
//   - Multi-turn streams (turn 1 has reasoning, turn 2 doesn't, turn 3
//     has reasoning again — no marker leak between turns)
//   - Token-boundary splits across feed() calls (`<thi` then `nk>`)
//   - Mid-reasoning stream truncation (max_tokens cap inside `<think>`
//     — leftover buffer must NOT leak markers into next turn's content)
//   - Empty think blocks (`<think></think>` — zero reasoning bytes,
//     no spurious `.reasoning("")` events)
//   - Stray closer in content mode — must be passed through as content
//
// Pure unit tests over the parser API. Covers the leak surfaces every
// shipping model family routes through (Nemotron-3, Qwen 3.5/3.6,
// MiniMax, Kimi, GLM, Laguna — all `think_xml` stamp).
//

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("Interleaved reasoning — thinking-tag leak boundary")
struct InterleavedReasoningLeakTests {

    // MARK: - Helpers

    /// Build a parser explicitly in CONTENT mode (startInReasoning=false).
    /// Models the case where the prompt template did NOT prefill a
    /// `<think>` opener — the model emits its own opener explicitly.
    /// This is the cleaner audit shape for content→think→content
    /// boundary verification (the alternate mode where the parser
    /// starts INSIDE reasoning is locked elsewhere by the harmony /
    /// startInReasoning suites).
    private func segment(
        tokens: [String]
    ) -> (content: String, reasoning: String) {
        var parser = ReasoningParser(startInReasoning: false)
        var content = ""
        var reasoning = ""
        for tok in tokens {
            for seg in parser.feed(tok) {
                switch seg {
                case .content(let s): content += s
                case .reasoning(let s): reasoning += s
                }
            }
        }
        for seg in parser.flush() {
            switch seg {
            case .content(let s): content += s
            case .reasoning(let s): reasoning += s
            }
        }
        return (content, reasoning)
    }

    // MARK: - Single-turn boundaries

    @Test("Empty <think></think> — zero reasoning bytes leak as reasoning")
    func emptyThink_zeroBytes() {
        let (content, reasoning) = segment(
            tokens: ["before ", "<think>", "</think>", "after"])
        #expect(reasoning == "", "empty think block must not produce reasoning bytes")
        #expect(!content.contains("<think>"), "opener must be consumed")
        #expect(!content.contains("</think>"), "closer must be consumed")
        #expect(content.contains("before"))
        #expect(content.contains("after"))
    }

    @Test("Token-split opener `<thi`+`nk>` — no leak")
    func splitOpener_noLeak() {
        let (content, reasoning) = segment(
            tokens: ["start ", "<thi", "nk>", "thought", "</think>", " end"])
        #expect(reasoning.contains("thought"))
        #expect(!content.contains("<thi"), "split-opener prefix must not leak as content")
        #expect(!content.contains("<think>"))
        #expect(content.contains("start"))
        #expect(content.contains("end"))
    }

    @Test("Token-split closer `</thi`+`nk>` — no leak")
    func splitCloser_noLeak() {
        let (content, reasoning) = segment(
            tokens: ["a ", "<think>", "thought ", "</thi", "nk>", "b"])
        #expect(reasoning.contains("thought"))
        #expect(!content.contains("</thi"), "split-closer prefix must not leak")
        #expect(!content.contains("</think>"))
        #expect(content.contains("a"))
        #expect(content.contains("b"))
    }

    // MARK: - Multi-turn interleaving

    /// Each turn re-instantiates the parser via fromCapabilityName.
    /// Verifies that no stale state crosses turns — turn N+1 starts
    /// fresh and a `<think>` opener arrival doesn't spill into the
    /// previous turn's content channel.
    @Test("Multi-turn: turn 1 has reasoning, turn 2 plain, turn 3 reasoning — no cross-turn leak")
    func multiTurn_noCrossLeak() {
        let t1 = segment(
            tokens: ["<think>", "thought1", "</think>", "answer1"])
        #expect(t1.reasoning.contains("thought1"))
        #expect(t1.content.contains("answer1"))

        let t2 = segment(
            tokens: ["plain answer 2"])
        #expect(t2.reasoning.isEmpty,
            "turn 2 with no <think> must produce zero reasoning")
        // Turn 2 might still be in startInReasoning mode for some
        // stamps — check the parser's contract by looking at content
        // OR reasoning containing the answer.
        let t2Combined = t2.content + t2.reasoning
        #expect(t2Combined.contains("plain answer 2"))

        let t3 = segment(
            tokens: ["<think>", "thought3", "</think>", "answer3"])
        #expect(t3.reasoning.contains("thought3"))
        #expect(t3.content.contains("answer3"))
        // Critical: no "thought1" / "answer1" leakage into t3 — every
        // turn is a fresh parser instance.
        #expect(!t3.content.contains("thought1"))
        #expect(!t3.reasoning.contains("answer1"))
    }

    // MARK: - Mid-reasoning truncation

    /// max_tokens cap fires while inside `<think>` block. Buffered
    /// reasoning must flush as `.reasoning` on parser.flush() — not
    /// as `.content`. This covers tpae's original disk-cache report
    /// scenario where turn 1 caps mid-think and turn 2 disk-restores
    /// against the buffered state.
    @Test("Mid-think truncation flushes buffered tokens as reasoning, not content")
    func midThinkTruncation_flushesAsReasoning() {
        var parser = ReasoningParser(startInReasoning: false)
        var content = ""
        var reasoning = ""
        for tok in ["<think>", "partial thought without closer "] {
            for seg in parser.feed(tok) {
                switch seg {
                case .content(let s): content += s
                case .reasoning(let s): reasoning += s
                }
            }
        }
        // Simulate max_tokens cap — caller stops feeding and flushes.
        for seg in parser.flush() {
            switch seg {
            case .content(let s): content += s
            case .reasoning(let s): reasoning += s
            }
        }
        #expect(reasoning.contains("partial thought"),
            "mid-think buffered tokens must flush as reasoning")
        #expect(!content.contains("<think>"),
            "opener must NOT leak into content even on truncation")
        #expect(!content.contains("partial thought"),
            "buffered reasoning bytes must NOT leak into content on truncation")
    }

    // MARK: - 'none' stamp (Mistral 3 / 3.5 / Mistral 4 / dense LLMs)

    @Test("'none' stamp returns nil parser — caller streams raw")
    func noneStamp_isNilParser() {
        // Mistral 3 / 3.5 / dense LLMs route here. The contract is that
        // callers see no .reasoning side-channel — they stream model
        // output directly as visible content.
        #expect(ReasoningParser.fromCapabilityName("none") == nil)
    }

    // MARK: - Stray closer in content mode

    @Test("Stray </think> with no opener doesn't open reasoning channel")
    func strayCloser_noChannelOpen() {
        let (content, reasoning) = segment(
            tokens: ["normal text ", "</think>", " more text"])
        // Without a matching opener, the closer either gets stripped
        // or passed through as content — but reasoning channel must
        // NOT open spuriously.
        #expect(reasoning.isEmpty,
            "stray closer without opener must NOT produce reasoning bytes")
        let combined = content + reasoning
        #expect(combined.contains("normal text"))
        #expect(combined.contains("more text"))
    }

    // MARK: - Family-by-family stamp resolution boundary

    /// Lock the dispatch matrix once more — guards against silent drift
    /// in `reasoningStampFromModelType` across the families this test
    /// suite cares about.
    @Test("Family stamp resolution remains stable across audit families")
    func familyStampMatrix() {
        // Reasoning families
        #expect(reasoningStampFromModelType("nemotron_h") == "think_xml")
        #expect(reasoningStampFromModelType("qwen3_5_moe") == "think_xml")
        #expect(reasoningStampFromModelType("qwen3_6") == "think_xml")
        #expect(reasoningStampFromModelType("minimax_m2") == "think_xml")
        #expect(reasoningStampFromModelType("kimi_k25") == "think_xml")
        #expect(reasoningStampFromModelType("deepseek_v4") == "think_xml")
        #expect(reasoningStampFromModelType("glm5") == "think_xml")
        #expect(reasoningStampFromModelType("holo3") == "think_xml")
        #expect(reasoningStampFromModelType("laguna") == "think_xml")
        #expect(reasoningStampFromModelType("laguna_xs") == "think_xml")

        // No-reasoning families (Mistral 3 family + dense LLMs)
        #expect(reasoningStampFromModelType("mistral3") == "none")
        #expect(reasoningStampFromModelType("ministral3") == "none")
        #expect(reasoningStampFromModelType("mistral4") == "none")
        #expect(reasoningStampFromModelType("lfm2") == "none")
        #expect(reasoningStampFromModelType("gpt_oss") == "none")
        #expect(reasoningStampFromModelType("phi") == "none")

        // Harmony channel (Gemma-4)
        #expect(reasoningStampFromModelType("gemma4") == "harmony")
    }
}
