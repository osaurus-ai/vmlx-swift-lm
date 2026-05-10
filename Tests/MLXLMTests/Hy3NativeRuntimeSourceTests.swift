import Foundation
import Testing

@Suite("Hy3 native runtime source contract")
struct Hy3NativeRuntimeSourceTests {
    private static let repoRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    private static let hy3SourceURL =
        repoRoot.appendingPathComponent("Libraries/MLXLLM/Models/Hy3.swift")
    private static let factorySourceURL =
        repoRoot.appendingPathComponent("Libraries/MLXLLM/LLMModelFactory.swift")

    @Test("Hy3.swift contains the native runtime pieces required for Phase B")
    func nativeRuntimePiecesExist() throws {
        let source = try String(contentsOf: Self.hy3SourceURL)

        #expect(source.contains("class Hy3Attention"))
        #expect(source.contains("class Hy3MoE"))
        #expect(source.contains("class Hy3DecoderLayer"))
        #expect(source.contains("class Hy3Model"))
        #expect(source.contains("TurboQuantSwitchGLU"))
        #expect(source.contains("shared_mlp") || source.contains("sharedExperts"))
        #expect(source.contains("expert_bias") || source.contains("expertBias"))
        #expect(source.contains("router.gate") || source.contains("gate.weight"))
        #expect(source.contains("JANGTQStreamingExperts"))
    }

    @Test("Hy3 sanitizer handles real JANGTQ2 bundle key families")
    func sanitizerHandlesJANGTQKeyFamilies() throws {
        let source = try String(contentsOf: Self.hy3SourceURL)

        #expect(source.contains("model.layers.\\(configuration.numHiddenLayers).")
            || source.contains("model.layers.\\(baseLayerCount)."))
        #expect(source.contains(".router.gate."))
        #expect(source.contains(".expert_bias"))
        #expect(source.contains(".shared_mlp."))
        #expect(source.contains(".switch_mlp."))
        #expect(source.contains("tq_packed"))
        #expect(source.contains("tq_norms"))
        #expect(source.contains("tq_bits"))
        #expect(source.contains("loadTimeMaterializedStacked")
            || source.contains("MLX.stacked"))
    }

    @Test("Factory no longer routes Hy3 through the unsupported recognition gate")
    func factoryDispatchesHy3Model() throws {
        let source = try String(contentsOf: Self.factorySourceURL)

        #expect(!source.contains("\"hy_v3\": dispatchHy3Unsupported"))
        #expect(!source.contains("private static func dispatchHy3Unsupported"))
        #expect(source.contains("\"hy_v3\": create(Hy3Configuration.self, Hy3Model.init)")
            || source.contains("\"hy_v3\": dispatchHy3"))
    }
}
