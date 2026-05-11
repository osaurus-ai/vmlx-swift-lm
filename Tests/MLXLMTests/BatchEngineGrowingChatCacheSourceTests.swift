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
}
