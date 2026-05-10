import Foundation
import Testing

@Suite("ChatSession runtime wiring source coverage")
struct ChatSessionRuntimeWiringSourceCoverageTests {
    private static func source(_ relativePath: String) throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: repoRoot.appending(path: relativePath),
            encoding: .utf8)
    }

    @Test("streamMap passes container cache coordinator into TokenIterator")
    func streamMapPassesCacheCoordinatorIntoTokenIterator() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ChatSession.swift")
        #expect(source.contains("let cacheCoordinator = model.cacheCoordinator"))
        #expect(source.contains("cacheCoordinator: cacheCoordinator"))
        #expect(source.contains("let iterator = try TokenIterator("))
    }

    @Test("streamMap consumer termination cancels the producer task")
    func streamMapTerminationCancelsProducerTask() throws {
        let source = try Self.source("Libraries/MLXLMCommon/ChatSession.swift")
        #expect(source.contains("continuation.onTermination = { _ in"))
        #expect(source.contains("task.cancel()"))
        #expect(source.contains("await task.value"))
    }
}
