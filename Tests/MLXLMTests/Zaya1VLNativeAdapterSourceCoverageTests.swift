import Foundation
import Testing

@Suite("ZAYA1-VL native adapter source coverage")
struct Zaya1VLNativeAdapterSourceCoverageTests {
    @Test("Input embedding adapter keeps vision tower namespace and merge contract explicit")
    func inputEmbeddingAdapterSourceContract() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sourceURL = repo
            .appendingPathComponent("Libraries")
            .appendingPathComponent("MLXVLM")
            .appendingPathComponent("Models")
            .appendingPathComponent("Zaya1VL.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)

        #expect(source.contains("public final class Zaya1VLInputEmbeddingAdapter: Module"))
        #expect(source.contains("@ModuleInfo(key: \"vision_tower\")"))
        #expect(source.contains("Qwen25Vision.VisionModel"))
        #expect(source.contains("public func projectImageFeatures(pixelValues: MLXArray, frames: [THW])"))
        #expect(source.contains("public func mergeImageFeatures("))
        #expect(source.contains("Zaya1VLRuntimeSupport.mergeImageFeatures("))
        #expect(source.contains("imageMask: nil"))
        #expect(source.contains("image pixels and frame metadata must be provided together"))
    }

    @Test("Cache policy stays external-salt plus Zaya CCA, not fake paged KV")
    func cachePolicySourceContract() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let zayaVLSource = try String(
            contentsOf: repo
                .appendingPathComponent("Libraries")
                .appendingPathComponent("MLXVLM")
                .appendingPathComponent("Models")
                .appendingPathComponent("Zaya1VL.swift"),
            encoding: .utf8)
        let cacheHelpersSource = try String(
            contentsOf: repo
                .appendingPathComponent("Libraries")
                .appendingPathComponent("MLXLMCommon")
                .appendingPathComponent("Cache")
                .appendingPathComponent("CacheHelpers.swift"),
            encoding: .utf8)
        let mediaSaltSource = try String(
            contentsOf: repo
                .appendingPathComponent("Libraries")
                .appendingPathComponent("MLXLMCommon")
                .appendingPathComponent("Cache")
                .appendingPathComponent("MediaSalt.swift"),
            encoding: .utf8)

        #expect(!zayaVLSource.contains("final class Zaya1VLCache"))
        #expect(!zayaVLSource.contains("TurboQuantKVCache"))
        #expect(cacheHelpersSource.contains("layer is ZayaCCACache"))
        #expect(cacheHelpersSource.contains("cacheRequiresDiskBackedCoordinatorRestore"))
        #expect(mediaSaltSource.contains("public func computeCacheSalt(for input: LMInput) -> String?"))
        #expect(mediaSaltSource.contains("let media = computeMediaSalt(for: input)"))
        #expect(mediaSaltSource.contains("let scope = input.cacheScopeSalt"))
    }

    @Test("Vision LoRA adapter matches canonicalized two-linear tensor contract")
    func visionLoRAAdapterSourceContract() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repo
                .appendingPathComponent("Libraries")
                .appendingPathComponent("MLXVLM")
                .appendingPathComponent("Models")
                .appendingPathComponent("Zaya1VL.swift"),
            encoding: .utf8)

        #expect(source.contains("public final class Zaya1VLLowRankAdapter: Module"))
        #expect(source.contains("@ModuleInfo(key: \"down\")"))
        #expect(source.contains("@ModuleInfo(key: \"up\")"))
        #expect(source.contains("canonicalizeLowRankSequentialKey"))
        #expect(source.contains("of: \".\\(name).0.weight\""))
        #expect(source.contains("with: \".\\(name).down.weight\""))
        #expect(source.contains("of: \".\\(name).1.weight\""))
        #expect(source.contains("with: \".\\(name).up.weight\""))
        #expect(source.contains("Linear(inputDimensions, rank, bias: false)"))
        #expect(source.contains("Linear(rank, outputDimensions, bias: false)"))
        #expect(source.contains("Zaya1VLRuntimeSupport.applyImageMaskedAdd("))
    }

    @Test("Vision LoRA module namespaces match canonicalized attention and expert sidecars")
    func visionLoRAModuleNamespaceContract() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: repo
                .appendingPathComponent("Libraries")
                .appendingPathComponent("MLXVLM")
                .appendingPathComponent("Models")
                .appendingPathComponent("Zaya1VL.swift"),
            encoding: .utf8)

        #expect(source.contains("public final class Zaya1VLQKVLoRAAdapters: Module"))
        #expect(source.contains("@ModuleInfo(key: \"lora_linear_q\")"))
        #expect(source.contains("@ModuleInfo(key: \"lora_linear_k\")"))
        #expect(source.contains("@ModuleInfo(key: \"lora_val_proj1\")"))
        #expect(source.contains("@ModuleInfo(key: \"lora_val_proj2\")"))
        #expect(source.contains("public final class Zaya1VLAttentionLoRAAdapters: Module"))
        #expect(source.contains("@ModuleInfo(key: \"qkv\")"))
        #expect(source.contains("@ModuleInfo(key: \"lora_linear_o\")"))
        #expect(source.contains("public final class Zaya1VLExpertLoRAAdapters: Module"))
        #expect(source.contains("@ModuleInfo(key: \"lora_fc1\")"))
        #expect(source.contains("@ModuleInfo(key: \"lora_fc2\")"))
        #expect(source.contains("public final class Zaya1VLLocalExpertLoRAAdapters: Module"))
        #expect(source.contains("@ModuleInfo(key: \"expert_0\") private var expert0"))
        #expect(source.contains("@ModuleInfo(key: \"expert_15\") private var expert15"))
        #expect(source.contains("canonicalizeLocalExpertKey"))
        #expect(source.contains("with: \".local_experts.expert_\\(expertID).\")"))
        #expect(source.contains("public func adapter(for expertIndex: Int) throws"))
        #expect(source.contains("config.numAttentionHeads * config.headDim"))
        #expect(source.contains("config.numKeyValueHeads * config.headDim"))
        #expect(source.contains("config.ffnHiddenSize / 2"))
    }
}
