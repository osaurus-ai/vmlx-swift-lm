// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation

// MARK: - ReasoningSegment

/// A segment of model output classified as either visible content or hidden
/// chain-of-thought reasoning.
public enum ReasoningSegment: Sendable, Equatable {
    /// Visible content the user should see.
    case content(String)
    /// Reasoning the application may want to display in a separate UI affordance
    /// (think pane, foldable section, etc.) — *not* the visible answer.
    case reasoning(String)
}

// MARK: - ReasoningParser

/// Streaming-safe parser that splits a token-by-token model stream into
/// `.content(...)` and `.reasoning(...)` segments based on tag delimiters.
///
/// **Why this lives in vmlx-swift-lm rather than osaurus:**
/// Models like Qwen 3.5/3.6, DeepSeek-R1, and others mark reasoning blocks
/// with literal vocabulary tokens (e.g. `<think>` / `</think>`) that they
/// have **deliberately marked as `special: false`** in their tokenizer config,
/// so every consumer (osaurus, llm-tool, anything else built on
/// vmlx-swift-lm) sees them as plain text. Each consumer would otherwise
/// re-implement the same boundary tracking — and get edge cases wrong.
/// Centralising it here keeps streaming behaviour consistent and lets
/// consumers choose to either show, hide, or relabel reasoning.
///
/// Default tags match Qwen 3.5 / Qwen 3.6 / DeepSeek-R1 (`<think>...</think>`).
/// Override `startTag`/`endTag` for models that use different markers.
///
/// ## Streaming contract
///
/// Token streams arrive in fragments — a single tag may be split across
/// several `feed(...)` calls (e.g. `<thi`, `nk>`). The parser buffers
/// **only** the portion that could be a partial tag prefix; everything
/// else is emitted immediately as `.content` or `.reasoning`.
///
/// On end-of-sequence, call `flush()` once to drain any remaining buffered
/// text. Anything still buffered after a final `flush()` is emitted as
/// `.content` (we never lose tokens to the parser).
///
/// ## Example
///
/// ```swift
/// var parser = ReasoningParser()  // defaults to <think>/</think>
/// for chunk in stream {
///     for segment in parser.feed(chunk) {
///         switch segment {
///         case .content(let text):   appendToVisibleAnswer(text)
///         case .reasoning(let text): appendToThinkPane(text)
///         }
///     }
/// }
/// for segment in parser.flush() { ... }
/// ```
public struct ReasoningParser: Sendable {

    // MARK: Configuration

    /// The tag that starts a reasoning block. Default `<think>`.
    public let startTag: String

    /// The tag that ends a reasoning block. Default `</think>`.
    public let endTag: String

    // MARK: State

    /// Text not yet emitted because it might be a partial tag prefix.
    private var buffer: String = ""

    /// Whether we're currently inside a reasoning block.
    private var insideReasoning: Bool = false

    // MARK: Init

    public init(startTag: String = "<think>", endTag: String = "</think>") {
        self.startTag = startTag
        self.endTag = endTag
    }

    // MARK: Streaming API

    /// Feed an incoming token-stream chunk. Returns zero or more segments.
    public mutating func feed(_ chunk: String) -> [ReasoningSegment] {
        guard !chunk.isEmpty else { return [] }
        buffer.append(chunk)
        return drain()
    }

    /// Call once when the stream ends. Flushes any buffered partial text
    /// as `.content` (so we never silently drop tokens).
    public mutating func flush() -> [ReasoningSegment] {
        var out = drain(allowPartialTagAtEnd: false)
        if !buffer.isEmpty {
            // Anything left over after the final drain is plain text — emit
            // as content (or as reasoning if we never saw a closing tag).
            out.append(insideReasoning ? .reasoning(buffer) : .content(buffer))
            buffer.removeAll(keepingCapacity: false)
        }
        insideReasoning = false
        return out
    }

    // MARK: Internals

    /// Process the buffer, peeling off as many complete segments as possible.
    /// `allowPartialTagAtEnd` keeps a tail of up to `max(startTag, endTag).count - 1`
    /// characters in the buffer when streaming, so a tag split across
    /// chunks isn't mistakenly emitted as content.
    private mutating func drain(allowPartialTagAtEnd: Bool = true)
        -> [ReasoningSegment]
    {
        var out: [ReasoningSegment] = []

        while !buffer.isEmpty {
            let lookFor = insideReasoning ? endTag : startTag
            if let range = buffer.range(of: lookFor) {
                // Emit everything before the tag in the current mode.
                let before = String(buffer[..<range.lowerBound])
                if !before.isEmpty {
                    out.append(insideReasoning ? .reasoning(before) : .content(before))
                }
                // Consume the tag itself (don't emit it).
                buffer.removeSubrange(buffer.startIndex..<range.upperBound)
                insideReasoning.toggle()
                continue
            }

            // No complete tag in the buffer. If we might still be assembling
            // a tag prefix at the end, hold back enough characters that a
            // future chunk can complete it.
            if allowPartialTagAtEnd {
                let safeTail = max(startTag.count, endTag.count) - 1
                if buffer.count > safeTail {
                    let splitAt = buffer.index(buffer.endIndex, offsetBy: -safeTail)
                    let safe = String(buffer[..<splitAt])
                    if !safe.isEmpty {
                        out.append(insideReasoning ? .reasoning(safe) : .content(safe))
                    }
                    buffer = String(buffer[splitAt...])
                }
            } else {
                // End-of-stream drain: emit everything, no holdback.
                if !buffer.isEmpty {
                    out.append(insideReasoning ? .reasoning(buffer) : .content(buffer))
                    buffer.removeAll(keepingCapacity: false)
                }
            }
            break
        }

        return out
    }
}

// MARK: - Capability-name resolution

extension ReasoningParser {
    /// Build a parser from a `JangCapabilities.reasoningParser` string.
    ///
    /// Accepts every name the JANG converter currently produces plus the
    /// canonical `think_xml`/`none` values. Unknown names → `nil` (caller
    /// should fall back to model-type heuristics or skip parsing).
    ///
    /// All `<think>...</think>`-style aliases collapse to a single
    /// `ReasoningParser` instance because the syntax is identical across
    /// Qwen 3.5/3.6, DeepSeek-R1, GLM 4.x, and Nemotron Cascade. Models
    /// that have no native reasoning tags (`mistral`, `gemma4`, `none`)
    /// return `nil` so callers know to skip parsing entirely.
    public static func fromCapabilityName(_ name: String?) -> ReasoningParser? {
        guard let name, !name.isEmpty else { return nil }
        switch name.lowercased() {
        case "think_xml", "qwen3", "qwen3_5", "qwen35", "qwen3_6", "qwen36",
            "deepseek_r1", "deepseek-r1", "deepseek", "glm", "glm4", "glm5",
            "nemotron", "nemotron_h", "minimax", "minimax_m2":
            return ReasoningParser()
        case "none", "off", "disabled", "mistral", "gemma", "gemma4":
            return nil
        default:
            return nil
        }
    }
}

// MARK: - Whole-string convenience

extension ReasoningParser {
    /// One-shot extraction for non-streaming callers — splits a complete
    /// model response into reasoning + visible content.
    ///
    /// - Parameters:
    ///   - text: The full model output.
    ///   - startTag: Override start tag (default `<think>`).
    ///   - endTag: Override end tag (default `</think>`).
    /// - Returns: `(reasoning: String, content: String)`. Empty strings
    ///   if the corresponding segment is absent.
    public static func split(
        _ text: String,
        startTag: String = "<think>",
        endTag: String = "</think>"
    ) -> (reasoning: String, content: String) {
        var parser = ReasoningParser(startTag: startTag, endTag: endTag)
        var segments = parser.feed(text)
        segments.append(contentsOf: parser.flush())
        var reasoning = ""
        var content = ""
        for s in segments {
            switch s {
            case .reasoning(let r): reasoning.append(r)
            case .content(let c): content.append(c)
            }
        }
        return (reasoning, content)
    }
}
