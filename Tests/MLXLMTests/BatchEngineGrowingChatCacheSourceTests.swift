// Copyright © 2026 Osaurus AI. All rights reserved.

import Foundation
import Testing

@Suite("BatchEngine growing-chat cache source coverage")
struct BatchEngineGrowingChatCacheSourceTests {
    @Test("batch engine stores post-answer cache boundaries and keeps hybrid full-hit guard")
    func batchEngineStoresPostAnswerBoundaryForGrowingChat() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)
        let scheduler = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift",
            encoding: .utf8)

        #expect(scheduler.contains("var generatedTokenIds: [Int] = []"))
        #expect(source.contains("slot.generatedTokenIds.append(tokenID)"))
        #expect(source.contains(#"label: "post-answer""#))
        #expect(source.contains("promptTokens + slot.generatedTokenIds"))
        #expect(source.contains("let unsafePartial = !remaining.isEmpty && hasMediaContent"))
        #expect(source.contains("let unsafeFullHit = remaining.isEmpty && hasPathDependentLayer"))
        #expect(source.contains("layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache"))
        #expect(!source.contains("let unsafePartial = !remaining.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("token iterator mirrors post-answer cache boundary policy")
    func tokenIteratorStoresPostAnswerBoundaryForGrowingChat() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(source.contains("mutating func storeCacheAfterGeneration"))
        #expect(source.contains("generatedTokenIds.append(token)"))
        #expect(source.contains("let generatedBoundaryTokens = promptTokenIds + generatedTokenIds"))
        #expect(source.contains("includeGeneratedBoundary: stopReason == .stop && !handler.stopSequenceHit"))
        #expect(source.contains("let unsafePartial = !remainingTokens.isEmpty && hasMediaContent"))
        #expect(source.contains("let unsafeFullHit = remainingTokens.isEmpty && hasPathDependentLayer"))
        #expect(source.contains("layer is MambaCache || layer is ArraysCache || layer is ZayaCCACache"))
        #expect(!source.contains("let unsafePartial = !remainingTokens.isEmpty &&\n                        (hasMediaContent || hasSSMLayer)"))
    }

    @Test("token iterator materializes disk cache restores before prefill")
    func tokenIteratorMaterializesDiskRestoreBeforePrefill() throws {
        let source = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)

        #expect(source.contains("let diskRestored = restoreFromDiskArrays(diskArrays, into: self.cache)"))
        #expect(source.contains("MLX.eval(self.cache)"))
        #expect(source.contains("Cache \\(detail.rawValue) hit: restored \\(diskRestored) tokens from disk"))
    }

    @Test("MiniMax open-thinking prompts get decode-time close-token correction")
    func minimaxOpenThinkingGetsReasoningCloseBias() throws {
        let evaluate = try String(
            contentsOfFile: "Libraries/MLXLMCommon/Evaluate.swift",
            encoding: .utf8)
        let engine = try String(
            contentsOfFile: "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift",
            encoding: .utf8)

        #expect(evaluate.contains("public struct ReasoningCloseBiasConfig"))
        #expect(evaluate.contains("public struct ReasoningCloseBiasProcessor"))
        #expect(evaluate.contains("forceAfterTokens"))
        #expect(evaluate.contains("token.item(Int.self) == config.tokenID"))
        #expect(engine.contains("parametersWithAutomaticReasoningCloseBias"))
        #expect(evaluate.contains("name.contains(\"minimax\") || modelTypeName.contains(\"minimax\")"))
        #expect(evaluate.contains("promptTail.range(of: \"<think>\", options: .backwards)"))
        #expect(evaluate.contains("promptTail.range(of: \"</think>\", options: .backwards)"))
        #expect(evaluate.contains("_specialTokenID(\"</think>\", tokenizer: tokenizer)"))
        #expect(evaluate.contains("tokenizer.encode(text: token, addSpecialTokens: false)"))
        #expect(evaluate.contains("reasoningCloseBias active"))
    }
}
