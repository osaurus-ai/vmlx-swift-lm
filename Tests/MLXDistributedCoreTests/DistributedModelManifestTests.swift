import XCTest
@testable import MLXDistributedCore

final class DistributedModelManifestTests: XCTestCase {
    func testManifestHashUsesOnlyKnownIdentityFiles() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("Laguna-XS.2-JANGTQ")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try write("""
        {"model_type":"laguna","architectures":["LagunaForCausalLM"],"num_hidden_layers":40,"hidden_size":2048,"num_attention_heads":48,"num_key_value_heads":8,"quantization":{"bits":8,"group_size":64}}
        """, to: model.appendingPathComponent("config.json"))
        try write("tokenizer-a", to: model.appendingPathComponent("tokenizer.json"))

        let first = try DistributedModelManifest.build(modelURL: model)
        try write("ignored local note", to: model.appendingPathComponent("README.local"))
        let withIgnoredFile = try DistributedModelManifest.build(modelURL: model)
        try write("template-b", to: model.appendingPathComponent("chat_template.jinja"))
        let changedTemplate = try DistributedModelManifest.build(modelURL: model)
        try write("tokenizer-b", to: model.appendingPathComponent("tokenizer.json"))
        let changed = try DistributedModelManifest.build(modelURL: model)

        XCTAssertEqual(first.bundleHash, withIgnoredFile.bundleHash)
        XCTAssertNotEqual(first.bundleHash, changedTemplate.bundleHash)
        XCTAssertNotEqual(first.bundleHash, changed.bundleHash)
        XCTAssertEqual(first.fullBundleHash.count, 64)
        XCTAssertEqual(first.identityMode, .identityFilesOnly)
        XCTAssertEqual(first.displayName, "Laguna-XS.2-JANGTQ")
        XCTAssertEqual(first.metadata.modelType, "laguna")
        XCTAssertEqual(first.metadata.architectures, ["LagunaForCausalLM"])
        XCTAssertEqual(first.metadata.layerCount, 40)
        XCTAssertEqual(first.metadata.hiddenSize, 2048)
        XCTAssertEqual(first.metadata.attentionHeads, 48)
        XCTAssertEqual(first.metadata.keyValueHeads, 8)
        XCTAssertEqual(first.metadata.quantizationBits, 8)
        XCTAssertEqual(first.metadata.quantizationGroupSize, 64)
        XCTAssertEqual(first.cacheClass, .standardKV)
        XCTAssertTrue(first.compatibleModes.contains(.replica))
    }

    func testManifestReadsJangSidecarAndHybridCacheHints() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("Ling-2.6-JANGTQ2")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try write("""
        {"model_type":"ling","architectures":["LingForCausalLM"],"num_hidden_layers":61,"ssm_state_size":16}
        """, to: model.appendingPathComponent("config.json"))
        try write("""
        {"format":"JANGTQ2","weight_format":"jangtq2"}
        """, to: model.appendingPathComponent("jang_config.json"))

        let manifest = try DistributedModelManifest.build(modelURL: model)

        XCTAssertTrue(manifest.metadata.hasJangConfig)
        XCTAssertEqual(manifest.metadata.weightFormat, "jangtq2")
        XCTAssertEqual(manifest.cacheClass, .hybridState)
        XCTAssertTrue(manifest.compatibilityWarnings.contains {
            $0.contains("state") || $0.contains("cache")
        })
    }

    func testUnknownConfigDoesNotAdvertiseReplicaCompatibility() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("Unknown")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try write("{}", to: model.appendingPathComponent("config.json"))

        let manifest = try DistributedModelManifest.build(modelURL: model)

        XCTAssertEqual(manifest.cacheClass, .unknown)
        XCTAssertFalse(manifest.compatibleModes.contains(.replica))
        XCTAssertTrue(manifest.compatibilityWarnings.contains {
            $0.contains("do not advertise replica")
        })
    }

    func testDiscoverFindsModelDirectoriesUnderRoots() throws {
        let root = try temporaryDirectory()
        let direct = root.appendingPathComponent("A")
        let nested = root.appendingPathComponent("org").appendingPathComponent("B")
        let nonModel = root.appendingPathComponent("notes")
        for url in [direct, nested, nonModel] {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        try write("{}", to: direct.appendingPathComponent("config.json"))
        try write("{}", to: nested.appendingPathComponent("config.json"))
        try write("not a model", to: nonModel.appendingPathComponent("README.md"))

        let discovered = try DistributedModelManifest.discover(roots: [root.path])
        let names = discovered.map(\.displayName)

        XCTAssertEqual(names, ["A", "B"])
    }

    func testDiscoverReportingKeepsValidModelsWhenOneConfigIsInvalid() throws {
        let root = try temporaryDirectory()
        let valid = root.appendingPathComponent("Valid")
        let invalid = root.appendingPathComponent("Invalid")
        try FileManager.default.createDirectory(at: valid, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: invalid, withIntermediateDirectories: true)
        try write("""
        {"model_type":"laguna","architectures":["LagunaForCausalLM"]}
        """, to: valid.appendingPathComponent("config.json"))
        try write("{", to: invalid.appendingPathComponent("config.json"))

        let result = DistributedModelManifest.discoverReporting(roots: [root.path])

        XCTAssertEqual(result.models.map(\.displayName), ["Valid"])
        XCTAssertEqual(result.errors.count, 1)
        XCTAssertTrue(result.errors[0].path.hasSuffix("/Invalid"))
    }

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("DistributedModelManifestTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        return url
    }

    private func write(_ text: String, to url: URL) throws {
        try text.data(using: .utf8)?.write(to: url)
    }
}
