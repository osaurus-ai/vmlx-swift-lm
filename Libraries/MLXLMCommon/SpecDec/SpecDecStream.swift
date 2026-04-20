// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Streaming wrapper around SpecDecRuntimeLinear / SpecDecRuntimeDDTree
// that emits `Generation` events (.chunk + .info) exactly like the
// non-speculative `Evaluate.generate` path does. This is what osaurus
// integrates with — same stream contract, same event types.
//
// Iter 10 deliverable: osaurus can flip `GenerateParameters.draftStrategy`
// to `.dflash(...)` or `.ddtree(...)` and consume the resulting
// `AsyncStream<Generation>` without any other API changes.

import Foundation
import MLX

/// Helper that runs a SpecDec runtime and surfaces its per-round
/// commits as a streaming `AsyncStream<Generation>`.
///
/// The runtime runs on a background `Task`; each committed batch of
/// tokens is detokenized via `NaiveStreamingDetokenizer` and yielded as
/// `.chunk(String)`. A final `.info(GenerateCompletionInfo)` fires on
/// completion.
///
/// Tool-call and reasoning parsing are applied to the detokenized
/// stream using the same `ToolCallProcessor` + `ReasoningParser`
/// pipeline the non-speculative path uses (see
/// `Evaluate.TextToolTokenLoopHandler`). That keeps osaurus's chunk
/// contract byte-identical whether speculative decoding is on or off.
public enum SpecDecStream {

    /// Run a DFlash-linear generation and stream `Generation` events.
    ///
    /// - Parameters:
    ///   - args: the runtime args.
    ///   - tokenizer: used to detokenize committed tokens into chunks.
    ///   - toolCallFormat: tool-call wire format for the stream's
    ///     `.toolCall(ToolCall)` events. Pass `.json` default.
    ///   - reasoningParserName: optional JANG capability stamp for
    ///     `<think>`-style reasoning strip.
    /// - Returns: `AsyncStream<Generation>` — same event shape as
    ///   `Evaluate.generate(input:cache:parameters:context:)`.
    public static func streamDflashLinear(
        args: DFlashLinearArgs,
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat = .json,
        reasoningParserName: String? = nil
    ) -> AsyncStream<Generation> {
        AsyncStream<Generation> { continuation in
            Task {
                do {
                    let startTime = Date()
                    let promptTokenCount = args.inputIds.dim(1)
                    var detokenizer = NaiveStreamingDetokenizer(
                        tokenizer: tokenizer)
                    let toolCallProcessor = ToolCallProcessor(
                        format: toolCallFormat)
                    var reasoningParser = ReasoningParser
                        .fromCapabilityName(reasoningParserName)

                    let onCommitted: ([Int32]) -> Void = { batch in
                        pushBatch(
                            tokens: batch,
                            detokenizer: &detokenizer,
                            toolCallProcessor: toolCallProcessor,
                            reasoningParser: &reasoningParser,
                            continuation: continuation)
                    }

                    let result = try SpecDecRuntimeLinear.run(
                        args, onCommitted: onCommitted)

                    // Flush any buffered content in the reasoning parser
                    // + tool-call processor before finishing.
                    flush(
                        detokenizer: &detokenizer,
                        toolCallProcessor: toolCallProcessor,
                        reasoningParser: &reasoningParser,
                        continuation: continuation)

                    let elapsed = Date().timeIntervalSince(startTime)
                    let generatedCount = result.tokenIds.count - promptTokenCount
                    let info = GenerateCompletionInfo(
                        promptTokenCount: promptTokenCount,
                        generationTokenCount: max(0, generatedCount),
                        promptTime: 0,
                        generationTime: elapsed,
                        stopReason: .length)
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch {
                    // Terminate the stream on error — callers observe
                    // completion without an info event.
                    continuation.finish()
                }
            }
        }
    }

    /// Run a DDTree generation and stream `Generation` events. See
    /// ``streamDflashLinear(args:tokenizer:toolCallFormat:reasoningParserName:)``
    /// for the event contract.
    public static func streamDDTree(
        args: DDTreeArgs,
        tokenizer: any Tokenizer,
        toolCallFormat: ToolCallFormat = .json,
        reasoningParserName: String? = nil
    ) -> AsyncStream<Generation> {
        AsyncStream<Generation> { continuation in
            Task {
                do {
                    let startTime = Date()
                    let promptTokenCount = args.inputIds.dim(1)
                    var detokenizer = NaiveStreamingDetokenizer(
                        tokenizer: tokenizer)
                    let toolCallProcessor = ToolCallProcessor(
                        format: toolCallFormat)
                    var reasoningParser = ReasoningParser
                        .fromCapabilityName(reasoningParserName)

                    let onCommitted: ([Int32]) -> Void = { batch in
                        pushBatch(
                            tokens: batch,
                            detokenizer: &detokenizer,
                            toolCallProcessor: toolCallProcessor,
                            reasoningParser: &reasoningParser,
                            continuation: continuation)
                    }

                    let result = try SpecDecRuntimeDDTree.run(
                        args, onCommitted: onCommitted)

                    flush(
                        detokenizer: &detokenizer,
                        toolCallProcessor: toolCallProcessor,
                        reasoningParser: &reasoningParser,
                        continuation: continuation)

                    let elapsed = Date().timeIntervalSince(startTime)
                    let generatedCount = result.tokenIds.count - promptTokenCount
                    let info = GenerateCompletionInfo(
                        promptTokenCount: promptTokenCount,
                        generationTokenCount: max(0, generatedCount),
                        promptTime: 0,
                        generationTime: elapsed,
                        stopReason: .length)
                    continuation.yield(.info(info))
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Internals

    /// Feed one round's committed tokens through the detokenizer +
    /// reasoning parser + tool-call processor, yielding any produced
    /// events to the caller.
    private static func pushBatch(
        tokens: [Int32],
        detokenizer: inout NaiveStreamingDetokenizer,
        toolCallProcessor: ToolCallProcessor,
        reasoningParser: inout ReasoningParser?,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        for t in tokens {
            detokenizer.append(token: Int(t))
            guard let chunk = detokenizer.next() else { continue }

            // 1. Reasoning pass (if configured) — peels off <think>…
            //    segments; reasoning content is silently dropped to
            //    match upstream ml-explore/mlx-swift-lm `Generation`
            //    which has no `.reasoning(String)` case.
            let contentPieces: [String]
            if var parser = reasoningParser {
                var pieces: [String] = []
                for segment in parser.feed(chunk) {
                    if case .content(let c) = segment { pieces.append(c) }
                }
                reasoningParser = parser
                contentPieces = pieces
            } else {
                contentPieces = [chunk]
            }

            // 2. Tool-call pass — same contract as non-speculative path.
            for piece in contentPieces {
                if let visibleText = toolCallProcessor.processChunk(piece) {
                    continuation.yield(.chunk(visibleText))
                }
                if let call = toolCallProcessor.toolCalls.popLast() {
                    continuation.yield(.toolCall(call))
                }
            }
        }
    }

    /// End-of-stream flush — same idea as
    /// `Evaluate.TextToolTokenLoopHandler.onGenerationEnd`.
    private static func flush(
        detokenizer: inout NaiveStreamingDetokenizer,
        toolCallProcessor: ToolCallProcessor,
        reasoningParser: inout ReasoningParser?,
        continuation: AsyncStream<Generation>.Continuation
    ) {
        if var parser = reasoningParser {
            for segment in parser.flush() {
                if case .content(let c) = segment,
                    let visibleText = toolCallProcessor.processChunk(c)
                {
                    continuation.yield(.chunk(visibleText))
                }
                if let call = toolCallProcessor.toolCalls.popLast() {
                    continuation.yield(.toolCall(call))
                }
            }
            reasoningParser = parser
        }
        toolCallProcessor.processEOS()
        for call in toolCallProcessor.toolCalls {
            continuation.yield(.toolCall(call))
        }
    }
}
