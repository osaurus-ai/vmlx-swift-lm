// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Verifies `NaiveStreamingDetokenizer` defers mid-grapheme-cluster
// emits so streaming output never splits a multi-codepoint emoji.
// Reproduces the user-visible MiniMax-M2.7-Small JANGTQ symptom
// (osaurus title `American History ❓国旗`) where the streaming
// detokenizer yielded the first regional-indicator scalar of a
// flag emoji alone (rendered as broken-box `❓`) before its
// sibling arrived.

import Foundation
import MLXLMCommon
import Testing

@Suite("NaiveStreamingDetokenizer multi-codepoint emoji deferral")
struct NaiveStreamingDetokenizerEmojiTests {

    /// Stand-in tokenizer that just decodes a token-id list to a
    /// fixed string the test controls — bypasses real BPE so we can
    /// step the streamer through a synthetic decode timeline.
    struct StubTokenizer: Tokenizer {
        let timeline: [String]  // index by token count; segment after N
                                // tokens = timeline[N-1].
        public func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        public func decode(tokenIds: [Int]) -> String {
            guard !tokenIds.isEmpty else { return "" }
            return timeline[min(tokenIds.count - 1, timeline.count - 1)]
        }
        public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            decode(tokenIds: tokenIds)
        }
        public func convertTokenToId(_ token: String) -> Int? { nil }
        public func convertIdToToken(_ id: Int) -> String? { nil }
        public var bosToken: String? { nil }
        public var eosToken: String? { nil }
        public var unknownToken: String? { nil }
        public func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// Step the streamer through the timeline and collect each
    /// non-empty yield. Empty-string emits (pre-existing contract
    /// when newSegment.count == segment.count) are no-ops from the
    /// consumer's perspective, so filter them.
    static func run(_ timeline: [String]) -> [String] {
        var det = NaiveStreamingDetokenizer(tokenizer: StubTokenizer(timeline: timeline))
        var emitted: [String] = []
        for tok in 0..<timeline.count {
            det.append(token: tok)
            if let s = det.next(), !s.isEmpty { emitted.append(s) }
        }
        return emitted
    }

    @Test("US flag emoji 🇺🇸 streams atomically (no mid-pair render)")
    func usFlagAtomic() {
        // Decode timeline:
        //   step 1: model emits regional-indicator U → "🇺" alone
        //   step 2: model emits regional-indicator S → flag complete
        let timeline = ["🇺", "🇺🇸"]
        let emitted = Self.run(timeline)
        // Streamer must defer step 1 (unpaired regional indicator)
        // and emit the full flag only after step 2 completes.
        #expect(emitted == ["🇺🇸"],
            "got \(emitted) — flag must not stream as two halves")
    }

    @Test("Trailing ZWJ at end-of-chunk defers (no \\u200D leaks alone)")
    func zwjMidStream() {
        // 👨‍🦰 = man + ZWJ + red-hair (compound emoji).
        // step 1: "👨"        — base, emits as-is
        // step 2: "👨\u{200D}" — base + dangling ZWJ → DEFER
        //         (in Swift this happens to be ONE grapheme cluster
        //         with count==1, so the suffix is empty and nothing
        //         gets to the lastChar check anyway — but the defer
        //         contract still holds: we must NOT yield a bare
        //         U+200D to the consumer.)
        // step 3: "👨\u{200D}🦰" — full grapheme cluster (count==1
        //         relative to segment of count==1, suffix is empty).
        //         This is a deeper limitation: when a token EXTENDS a
        //         previously-yielded grapheme into a compound, the
        //         streamer cannot retroactively replace the prior
        //         emit. The "👨" already left the streamer. Consumer
        //         sees "👨", not "👨\u{200D}🦰".
        //
        // What this test guards: bare ZWJ never reaches the consumer.
        // The compound-emoji-extends-prior-grapheme case is a known
        // limitation tracked elsewhere — not what this test covers.
        let timeline = ["👨", "👨\u{200D}", "👨\u{200D}🦰"]
        let emitted = Self.run(timeline)
        // Verify: no bare U+200D string in emitted.
        for s in emitted {
            #expect(
                !s.unicodeScalars.contains { $0.value == 0x200D && s.count == 1 },
                "emit \(s.debugDescription) leaks a bare ZWJ to consumer")
        }
        // First emit is the base.
        #expect(emitted.first == "👨")
    }

    @Test("Plain ASCII / non-emoji chars unaffected")
    func plainAsciiUnaffected() {
        let timeline = ["H", "Hello", "Hello world"]
        let emitted = Self.run(timeline)
        #expect(emitted == ["H", "ello", " world"])
    }

    @Test("Trailing replacement char (legacy contract) still defers")
    func legacyFFFDDeferral() {
        let timeline = ["abc", "abc\u{FFFD}", "abc🌍"]
        let emitted = Self.run(timeline)
        // Step 2 ends in FFFD → defer (existing contract).
        // Step 3 completes the emoji — emit the full delta.
        #expect(emitted == ["abc", "🌍"])
    }
}
