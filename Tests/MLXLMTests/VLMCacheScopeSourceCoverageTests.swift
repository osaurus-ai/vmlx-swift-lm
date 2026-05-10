import Foundation
import Testing

@Suite("VLM cache scope source coverage")
struct VLMCacheScopeSourceCoverageTests {
    @Test("every VLM LMInput construction carries cacheScopeSalt")
    func everyVLMInputConstructionCarriesCacheScopeSalt() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let modelsRoot = repoRoot.appending(path: "Libraries/MLXVLM/Models")
        let fileManager = FileManager.default
        let urls = try #require(fileManager.enumerator(
            at: modelsRoot,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ))
        var failures: [String] = []

        for case let url as URL in urls {
            guard url.pathExtension == "swift" else { continue }
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }

            let source = try String(contentsOf: url, encoding: .utf8)
            for call in Self.lmInputCalls(in: source) where !call.body.contains("cacheScopeSalt:") {
                let relative = url.path.replacingOccurrences(of: repoRoot.path + "/", with: "")
                failures.append("\(relative):\(call.line)")
            }
        }

        #expect(
            failures.isEmpty,
            "VLM LMInput calls missing cacheScopeSalt: \(failures.sorted().joined(separator: ", "))"
        )
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
