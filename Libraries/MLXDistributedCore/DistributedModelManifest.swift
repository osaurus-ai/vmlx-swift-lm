import CryptoKit
import Foundation

/// Lightweight identity and compatibility summary for a local model bundle.
///
/// This intentionally hashes loader identity files, not the full weight payload.
/// It is a fast compatibility gate for peer discovery and planning; real
/// execution still has to run through the model loader or a current benchmark.
public struct DistributedModelManifest: Sendable, Codable, Equatable {
    public static let identityFileNames = [
        "config.json",
        "chat_template.jinja",
        "chat_template.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "jang_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "model.safetensors.index.json",
    ]

    public let path: String
    public let displayName: String
    public let fullBundleHash: String
    public let bundleHash: String
    public let identityMode: DistributedModelIdentityMode
    public let files: [DistributedModelManifestFile]
    public let metadata: DistributedModelMetadata
    public let cacheClass: DistributedModelCacheClass
    public let compatibleModes: Set<Mode>
    public let compatibilityWarnings: [String]

    public static func build(modelPath: String) throws -> DistributedModelManifest {
        try build(modelURL: URL(fileURLWithPath: modelPath))
    }

    public static func build(modelURL: URL) throws -> DistributedModelManifest {
        let files = try identityFiles(in: modelURL)
        guard !files.isEmpty else {
            throw DistributedModelManifestError.emptyIdentity(modelURL.path)
        }

        let metadata = try metadata(in: modelURL)
        let cacheClass = classifyCache(metadata: metadata)
        let digestInput = files
            .map { "\($0.relativePath) \($0.sha256)" }
            .joined(separator: "\n")
        let digest = SHA256.hash(data: Data(digestInput.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let modes: Set<Mode> = cacheClass == .unknown ? [] : [.replica]
        let warnings = compatibilityWarnings(metadata: metadata, cacheClass: cacheClass)

        return DistributedModelManifest(
            path: modelURL.path,
            displayName: modelURL.lastPathComponent,
            fullBundleHash: digest,
            bundleHash: String(digest.prefix(16)),
            identityMode: .identityFilesOnly,
            files: files,
            metadata: metadata,
            cacheClass: cacheClass,
            compatibleModes: modes,
            compatibilityWarnings: warnings)
    }

    public static func discover(roots: [String]) throws -> [DistributedModelManifest] {
        let result = discoverReporting(roots: roots)
        if let firstError = result.errors.first {
            throw DistributedModelManifestError.scanFailed(firstError.path, firstError.message)
        }
        return result.models
    }

    public static func discoverReporting(roots: [String]) -> DistributedModelManifestScanResult {
        let manager = FileManager.default
        var modelURLs: Set<URL> = []
        var errors: [DistributedModelManifestScanError] = []
        for root in roots {
            let rootURL = URL(fileURLWithPath: NSString(string: root).expandingTildeInPath)
            guard manager.fileExists(atPath: rootURL.path) else {
                errors.append(DistributedModelManifestScanError(
                    path: rootURL.path,
                    message: "root does not exist"))
                continue
            }
            if manager.fileExists(atPath: rootURL.appendingPathComponent("config.json").path) {
                modelURLs.insert(rootURL)
            }
            guard let enumerator = manager.enumerator(
                at: rootURL,
                includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else {
                continue
            }
            for case let fileURL as URL in enumerator where fileURL.lastPathComponent == "config.json" {
                let parent = fileURL.deletingLastPathComponent()
                if parent.path.contains("/.build/") || parent.path.contains("/.git/") {
                    continue
                }
                modelURLs.insert(parent)
            }
        }

        var models: [DistributedModelManifest] = []
        let sortedModelURLs = modelURLs.sorted {
            $0.path.localizedStandardCompare($1.path) == .orderedAscending
        }
        for url in sortedModelURLs {
            do {
                models.append(try build(modelURL: url))
            } catch {
                errors.append(DistributedModelManifestScanError(
                    path: url.path,
                    message: "\(error)"))
            }
        }
        return DistributedModelManifestScanResult(models: models, errors: errors)
    }

    private static func identityFiles(in modelURL: URL) throws -> [DistributedModelManifestFile] {
        var files: [DistributedModelManifestFile] = []
        for name in identityFileNames {
            let fileURL = modelURL.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: fileURL.path) else { continue }
            let data = try Data(contentsOf: fileURL)
            let sha = SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }
                .joined()
            files.append(DistributedModelManifestFile(relativePath: name, sha256: sha))
        }
        return files
    }

    private static func metadata(in modelURL: URL) throws -> DistributedModelMetadata {
        let config = try jsonDictionary(at: modelURL.appendingPathComponent("config.json")) ?? [:]
        let jangConfig = try jsonDictionary(at: modelURL.appendingPathComponent("jang_config.json"))
        let quantization = dictionary(config["quantization"])
            ?? dictionary(config["quantization_config"])
            ?? [:]
        let modelType = string(config["model_type"])
        let architectures = stringArray(config["architectures"])
        let hasIndex = FileManager.default.fileExists(
            atPath: modelURL.appendingPathComponent("model.safetensors.index.json").path)
        let weightFormat = string(jangConfig?["weight_format"])
            ?? string(jangConfig?["format"])
            ?? string(config["weight_format"])

        return DistributedModelMetadata(
            modelType: modelType,
            architectures: architectures,
            layerCount: int(config["num_hidden_layers"])
                ?? int(config["n_layers"])
                ?? int(config["num_layers"]),
            hiddenSize: int(config["hidden_size"])
                ?? int(config["n_embd"])
                ?? int(config["model_dim"]),
            attentionHeads: int(config["num_attention_heads"])
                ?? int(config["n_head"]),
            keyValueHeads: int(config["num_key_value_heads"])
                ?? int(config["n_kv_heads"]),
            quantizationBits: int(quantization["bits"])
                ?? int(quantization["bits_per_weight"]),
            quantizationGroupSize: int(quantization["group_size"])
                ?? int(quantization["q_group_size"]),
            hasJangConfig: jangConfig != nil,
            hasSafetensorsIndex: hasIndex,
            weightFormat: weightFormat,
            hasStateSpaceHints: hasStateSpaceHints(config: config, modelType: modelType, architectures: architectures))
    }

    private static func classifyCache(metadata: DistributedModelMetadata) -> DistributedModelCacheClass {
        let familyTokens = ([metadata.modelType] + metadata.architectures)
            .compactMap { $0?.lowercased() }
            .joined(separator: " ")
        if metadata.modelType == nil
            && metadata.architectures.isEmpty
            && metadata.layerCount == nil
            && metadata.hiddenSize == nil {
            return .unknown
        }
        if metadata.hasStateSpaceHints || familyTokens.contains("ling") || familyTokens.contains("mamba") {
            return .hybridState
        }
        if familyTokens.contains("vl")
            || familyTokens.contains("vision")
            || familyTokens.contains("omni")
            || familyTokens.contains("audio") {
            return .multimodal
        }
        return .standardKV
    }

    private static func compatibilityWarnings(
        metadata: DistributedModelMetadata,
        cacheClass: DistributedModelCacheClass
    ) -> [String] {
        switch cacheClass {
        case .standardKV:
            return []
        case .hybridState:
            return [
                "Hybrid state/cache model: request-level replica can be considered after manifest match; activation pipeline and tensor parallel modes need model-specific state handoff."
            ]
        case .multimodal:
            return [
                "Multimodal model: request-level replica can be considered after manifest match; media preprocessing and cache salting must be validated before pipeline or tensor modes."
            ]
        case .unknown:
            return [
                "Unknown model/cache topology: do not advertise replica, pipeline, or tensor modes without a real loader proof."
            ]
        }
    }

    private static func hasStateSpaceHints(
        config: [String: Any],
        modelType: String?,
        architectures: [String]
    ) -> Bool {
        let keys = [
            "ssm_state_size",
            "state_size",
            "conv_kernel",
            "time_step_rank",
            "mamba_d_state",
        ]
        if keys.contains(where: { config[$0] != nil }) {
            return true
        }
        let text = ([modelType] + architectures.map(Optional.some))
            .compactMap { $0?.lowercased() }
            .joined(separator: " ")
        return text.contains("ssm") || text.contains("mamba")
    }
}

public enum DistributedModelIdentityMode: String, Sendable, Codable, Equatable {
    case identityFilesOnly = "identity_files_only"
    case fullWeights = "full_weights"
}

public struct DistributedModelManifestFile: Sendable, Codable, Equatable {
    public let relativePath: String
    public let sha256: String

    public init(relativePath: String, sha256: String) {
        self.relativePath = relativePath
        self.sha256 = sha256
    }
}

public struct DistributedModelManifestScanResult: Sendable, Codable, Equatable {
    public let models: [DistributedModelManifest]
    public let errors: [DistributedModelManifestScanError]

    public init(models: [DistributedModelManifest], errors: [DistributedModelManifestScanError]) {
        self.models = models
        self.errors = errors
    }
}

public struct DistributedModelManifestScanError: Sendable, Codable, Equatable {
    public let path: String
    public let message: String

    public init(path: String, message: String) {
        self.path = path
        self.message = message
    }
}

public struct DistributedModelMetadata: Sendable, Codable, Equatable {
    public let modelType: String?
    public let architectures: [String]
    public let layerCount: Int?
    public let hiddenSize: Int?
    public let attentionHeads: Int?
    public let keyValueHeads: Int?
    public let quantizationBits: Int?
    public let quantizationGroupSize: Int?
    public let hasJangConfig: Bool
    public let hasSafetensorsIndex: Bool
    public let weightFormat: String?
    public let hasStateSpaceHints: Bool
}

public enum DistributedModelCacheClass: String, Sendable, Codable, Equatable {
    case standardKV = "standard_kv"
    case hybridState = "hybrid_state"
    case multimodal
    case unknown
}

public enum DistributedModelManifestError: Error, CustomStringConvertible {
    case emptyIdentity(String)
    case scanFailed(String, String)

    public var description: String {
        switch self {
        case .emptyIdentity(let path):
            return "model identity manifest had no known files: \(path)"
        case .scanFailed(let path, let message):
            return "model scan failed at \(path): \(message)"
        }
    }
}

private func jsonDictionary(at url: URL) throws -> [String: Any]? {
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    let data = try Data(contentsOf: url)
    guard !data.isEmpty else { return [:] }
    let object = try JSONSerialization.jsonObject(with: data)
    return object as? [String: Any]
}

private func dictionary(_ value: Any?) -> [String: Any]? {
    value as? [String: Any]
}

private func string(_ value: Any?) -> String? {
    value as? String
}

private func stringArray(_ value: Any?) -> [String] {
    (value as? [Any])?.compactMap { $0 as? String } ?? []
}

private func int(_ value: Any?) -> Int? {
    switch value {
    case let value as Int:
        return value
    case let value as Int64:
        return Int(value)
    case let value as Double:
        return Int(value)
    case let value as String:
        return Int(value)
    default:
        return nil
    }
}
