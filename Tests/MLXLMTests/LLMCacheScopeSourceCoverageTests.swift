import Foundation
import Testing

@Suite("LLM cache scope source coverage")
struct LLMCacheScopeSourceCoverageTests {
    @Test("LLM user input processor carries cacheScopeSalt from merged context")
    func llmUserInputProcessorCarriesCacheScopeSalt() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let factory = repoRoot.appending(path: "Libraries/MLXLLM/LLMModelFactory.swift")
        let source = try String(contentsOf: factory, encoding: .utf8)
        let processor = try #require(
            Self.extractStruct(named: "LLMUserInputProcessor", from: source),
            "LLMUserInputProcessor not found"
        )

        let missing = Self.lmInputCalls(in: processor.body)
            .filter { !$0.body.contains("cacheScopeSalt: cacheScopeSalt(from: additionalContext)") }
            .map(\.line)

        #expect(
            missing.isEmpty,
            "LLMUserInputProcessor LMInput returns missing cacheScopeSalt at relative lines: \(missing)"
        )
    }

    private static func extractStruct(named name: String, from source: String) -> (line: Int, body: String)? {
        guard let range = source.range(of: "struct \(name)") else { return nil }
        guard let brace = source[range.lowerBound...].firstIndex(of: "{") else { return nil }
        let line = source[..<range.lowerBound].reduce(1) { count, character in
            character == "\n" ? count + 1 : count
        }
        var depth = 0
        var index = brace
        while index < source.endIndex {
            let character = source[index]
            if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    let end = source.index(after: index)
                    return (line, String(source[range.lowerBound..<end]))
                }
            }
            index = source.index(after: index)
        }
        return nil
    }

    private static func lmInputCalls(in source: String) -> [(line: Int, body: String)] {
        var calls: [(line: Int, body: String)] = []
        var searchStart = source.startIndex

        while let range = source.range(of: "LMInput(", range: searchStart..<source.endIndex) {
            let line = source[..<range.lowerBound].reduce(1) { count, character in
                character == "\n" ? count + 1 : count
            }
            var index = range.lowerBound
            var depth = 0

            while index < source.endIndex {
                let character = source[index]
                if character == "(" {
                    depth += 1
                } else if character == ")" {
                    depth -= 1
                    if depth == 0 {
                        let end = source.index(after: index)
                        calls.append((line, String(source[range.lowerBound..<end])))
                        searchStart = end
                        break
                    }
                }
                index = source.index(after: index)
            }

            if index == source.endIndex {
                searchStart = source.endIndex
            }
        }

        return calls
    }
}
