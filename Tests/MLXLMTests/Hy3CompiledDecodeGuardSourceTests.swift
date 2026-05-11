import Foundation
import Testing

struct Hy3CompiledDecodeGuardSourceTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    @Test("BatchEngine does not promote Hy3/Hunyuan to compiled decode")
    func batchEngineCompileGuardPinsHy3() throws {
        let source = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift"))

        #expect(source.contains("private var compiledDecodeDeniedForModel: Bool"))
        #expect(source.contains("context.configuration.toolCallFormat == .hunyuan"))
        #expect(source.contains("!compiledDecodeDeniedForModel && !soloParameters.enableCompiledDecode"))
        #expect(source.contains("guard !compiledDecodeDeniedForModel else { return }"))
    }

    @Test("TokenIterator direct compiled decode also denies Hy3/Hunyuan")
    func tokenIteratorCompileGuardPinsHy3() throws {
        let source = try String(contentsOf: Self.repoRoot.appendingPathComponent(
            "Libraries/MLXLMCommon/Evaluate.swift"))

        #expect(source.contains("private static func compiledDecodeDenied(for model: any LanguageModel) -> Bool"))
        #expect(source.contains("typeName.contains(\"hy3\") || typeName.contains(\"hunyuan\")"))
        #expect(source.contains("parameters.enableCompiledDecode && !Self.compiledDecodeDenied(for: model)"))
    }
}
