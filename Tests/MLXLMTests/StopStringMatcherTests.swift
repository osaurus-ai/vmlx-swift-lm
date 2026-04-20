// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Streaming text-level stop-sequence matcher.
// See `Libraries/MLXLMCommon/BatchEngine/STOP-SEQUENCES-CONTRACT.md`.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("StopStringMatcher")
struct StopStringMatcherTests {

    // MARK: - Disabled matcher (no stop strings)

    @Test("empty stop list → pass-through")
    func testEmptyIsPassThrough() {
        var m = StopStringMatcher(stopStrings: [])
        #expect(!m.isEnabled)
        switch m.feed("hello world") {
        case .streaming(let out):
            #expect(out == "hello world")
        case .stopped:
            Issue.record("Should not stop when no strings are configured")
        }
        #expect(m.flush() == "")
    }

    @Test("all-empty-string input is filtered, matcher disabled")
    func testFiltersEmptyStrings() {
        var m = StopStringMatcher(stopStrings: ["", "", ""])
        #expect(!m.isEnabled)
        if case .streaming(let out) = m.feed("anything") {
            #expect(out == "anything")
        } else {
            Issue.record("Disabled matcher should always stream")
        }
    }

    @Test("duplicates are de-duplicated at init")
    func testDeduplicates() {
        let m = StopStringMatcher(stopStrings: ["END", "END", "STOP", "END"])
        #expect(m.stopStrings.count == 2)
        #expect(m.stopStrings == ["END", "STOP"])
    }

    // MARK: - Single feed, single stop string

    @Test("stop string in the middle of a single feed splits correctly")
    func testSingleFeedMiddleMatch() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        switch m.feed("answer STOP ignored tail") {
        case .stopped(let out):
            #expect(out == "answer ")
        case .streaming:
            Issue.record("Expected a stop match")
        }
    }

    @Test("stop string at start of single feed — empty emit")
    func testSingleFeedStartMatch() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        switch m.feed("STOP trailing") {
        case .stopped(let out):
            #expect(out.isEmpty)
        case .streaming:
            Issue.record("Expected stop match at offset 0")
        }
    }

    @Test("stop string at end of single feed — full emit")
    func testSingleFeedEndMatch() {
        var m = StopStringMatcher(stopStrings: ["END"])
        switch m.feed("some text END") {
        case .stopped(let out):
            #expect(out == "some text ")
        case .streaming:
            Issue.record("Expected stop match")
        }
    }

    // MARK: - Streaming across chunks

    @Test("stop string split across two chunks")
    func testSplitAcrossChunks() {
        var m = StopStringMatcher(stopStrings: ["STOP"])

        // First feed contains a partial match — the last (maxStopLen-1 == 3)
        // characters "ST " are held as potential prefix, so emit is
        // everything before them ("before").
        let r1 = m.feed("before ST")
        switch r1 {
        case .streaming(let out):
            #expect(out == "before")
        case .stopped:
            Issue.record("Not yet a full match")
        }

        // Second feed completes the match. The buffer at this point is
        // " ST" + "OP after" = " STOP after" — match lands at offset 1,
        // so the pre-match emit is the single space.
        let r2 = m.feed("OP after")
        switch r2 {
        case .stopped(let out):
            #expect(out == " ")
        case .streaming:
            Issue.record("Expected stopped on second chunk")
        }
    }

    @Test("char-by-char streaming holds all bytes until disambiguated")
    func testCharByCharStreaming() {
        var m = StopStringMatcher(stopStrings: ["</end>"])
        var emitted = ""
        var stopped = false

        // The matcher holds up to 5 chars (maxStopLen-1 = 5) waiting.
        for ch in "hello</end>tail" {
            let res = m.feed(String(ch))
            switch res {
            case .streaming(let out):
                emitted += out
            case .stopped(let out):
                emitted += out
                stopped = true
                break
            }
            if stopped { break }
        }
        #expect(stopped)
        // Only "hello" is safely emittable — everything after "hello" is
        // part of or preceding the match.
        #expect(emitted == "hello")
    }

    @Test("flush drains held tail when no match arrives")
    func testFlushDrainsTail() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        // "partial match potential" doesn't contain STOP; 3 chars at the
        // end ("ial") are held as potential prefix.
        switch m.feed("partial match potential") {
        case .streaming(let out):
            // Emitted all but last (maxStopLen - 1 = 3) characters
            #expect(out == "partial match potent")
        case .stopped:
            Issue.record("No stop string present")
        }
        // Flush releases the held tail.
        #expect(m.flush() == "ial")
    }

    // MARK: - Multiple stop strings

    @Test("earliest match across multiple stop strings wins")
    func testEarliestMatchAcrossMultiple() {
        var m = StopStringMatcher(stopStrings: ["ENDING", "STOP", "FIN"])
        // "FIN" appears first (offset 9 vs STOP at 13) — earliest wins.
        switch m.feed("something FIN then STOP") {
        case .stopped(let out):
            #expect(out == "something ")
        case .streaming:
            Issue.record("Expected a stop match")
        }
    }

    @Test("different-length stop strings respect longest hold size")
    func testDifferentLengthHold() {
        var m = StopStringMatcher(stopStrings: ["X", "LONGSTOP"])
        // After feeding "partial LONGSTO", the last 7 chars ("LONGSTO")
        // are potentially a prefix of "LONGSTOP" — must be held.
        let r = m.feed("partial LONGSTO")
        switch r {
        case .streaming(let out):
            #expect(out == "partial ")
        case .stopped:
            Issue.record("No full match yet")
        }
        // Complete match → stop.
        switch m.feed("P rest") {
        case .stopped(let out):
            #expect(out.isEmpty)
        case .streaming:
            Issue.record("Expected stop")
        }
    }

    // MARK: - No match, natural termination

    @Test("stream finishes with no match → flush returns all held text")
    func testNaturalTerminationFlushesHeld() {
        var m = StopStringMatcher(stopStrings: ["STOP"])
        _ = m.feed("a") // holds "a"
        _ = m.feed("b") // holds "ab"
        // No more text. Flush.
        let tail = m.flush()
        #expect(tail == "ab")
    }

    @Test("repeated feeds then flush reassemble the original text")
    func testRoundTripByFlush() {
        var m = StopStringMatcher(stopStrings: ["SOMETHING_THAT_NEVER_APPEARS"])
        var emitted = ""
        let input = "the quick brown fox jumps over the lazy dog"
        for ch in input {
            switch m.feed(String(ch)) {
            case .streaming(let out):
                emitted += out
            case .stopped:
                Issue.record("Stop string not in text")
            }
        }
        emitted += m.flush()
        #expect(emitted == input)
    }

    // MARK: - GenerateParameters plumbing

    @Test("GenerateParameters.extraStopStrings round-trips through init")
    func testGenerateParametersField() {
        let params = GenerateParameters(extraStopStrings: ["END", "STOP"])
        #expect(params.extraStopStrings == ["END", "STOP"])

        let defaults = GenerateParameters()
        #expect(defaults.extraStopStrings.isEmpty)
    }
}
