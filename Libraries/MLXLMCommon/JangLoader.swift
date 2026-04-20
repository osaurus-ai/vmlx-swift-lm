// Copyright © 2024-2026 Jinho Jang (eric@jangq.ai)
// JANG format support for mlx-swift-lm

import Foundation
import MLX

// MARK: - Config File Names

/// Primary JANG config file name.
public let jangConfigFileName = "jang_config.json"

/// Legacy config file names to search for (fallback only).
public let jangConfigFileNames = [
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json",
]

// MARK: - JANG Config Structs

/// Quantization settings from jang_config.json `quantization` block.
public struct JangQuantization: Sendable, Equatable {
    public let method: String
    public let profile: String
    public let targetBits: Float
    public let actualBits: Float
    public let blockSize: Int
    public let bitWidthsUsed: [Int]
    public let quantizationScheme: String
    public let quantizationBackend: String

    public init(
        method: String = "jang-importance",
        profile: String = "JANG_2S",
        targetBits: Float = 2.5,
        actualBits: Float = 2.85,
        blockSize: Int = 64,
        bitWidthsUsed: [Int] = [2, 4, 6],
        quantizationScheme: String = "asymmetric",
        quantizationBackend: String = "mx.quantize"
    ) {
        self.method = method
        self.profile = profile
        self.targetBits = targetBits
        self.actualBits = actualBits
        self.blockSize = blockSize
        self.bitWidthsUsed = bitWidthsUsed
        self.quantizationScheme = quantizationScheme
        self.quantizationBackend = quantizationBackend
    }
}

/// Source model info from jang_config.json `source_model` block.
public struct JangSourceModel: Sendable, Equatable {
    public let name: String
    public let org: String
    public let architecture: String
    public let dtype: String
    public let parameters: String

    public init(
        name: String = "",
        org: String = "",
        architecture: String = "",
        dtype: String = "bfloat16",
        parameters: String = "0"
    ) {
        self.name = name
        self.org = org
        self.architecture = architecture
        self.dtype = dtype
        self.parameters = parameters
    }

    public var parameterCount: Int { Int(parameters) ?? 0 }

    /// HuggingFace canonical repo id, e.g. `MiniMaxAI/MiniMax-M2.7`. Empty if
    /// either `org` or `name` is missing.
    public var huggingFaceRepoID: String {
        guard !org.isEmpty, !name.isEmpty else { return "" }
        return "\(org)/\(name)"
    }
}

/// Architecture info from jang_config.json `architecture` block.
public struct JangArchitecture: Sendable, Equatable {
    public let type: String
    public let attention: String
    public let hasVision: Bool
    public let hasSSM: Bool
    public let hasMoE: Bool

    public init(
        type: String = "transformer",
        attention: String = "gqa",
        hasVision: Bool = false,
        hasSSM: Bool = false,
        hasMoE: Bool = false
    ) {
        self.type = type
        self.attention = attention
        self.hasVision = hasVision
        self.hasSSM = hasSSM
        self.hasMoE = hasMoE
    }
}

/// Runtime info from jang_config.json `runtime` block.
public struct JangRuntime: Sendable, Equatable {
    public let totalWeightBytes: Int
    public let totalWeightGB: Float

    public init(totalWeightBytes: Int = 0, totalWeightGB: Float = 0) {
        self.totalWeightBytes = totalWeightBytes
        self.totalWeightGB = totalWeightGB
    }
}

/// Capability hints stamped into `jang_config.json` by the JANG converter.
///
/// Allows downstream consumers (osaurus, llm-tool, etc.) to pick the right
/// reasoning / tool-call parser without hard-coding per-model branching.
/// All fields are optional — missing values mean "unknown, fall back to
/// model-type heuristics."
///
/// Field naming is intentionally lenient: aliases produced by the JANG
/// converter (e.g. `tool_parser: "qwen"` instead of vmlx's canonical
/// `"xml_function"`) are normalized at consumption time by
/// `ToolCallFormat.fromCapabilityName(_:)` and
/// `ReasoningParser.fromCapabilityName(_:)`.
public struct JangCapabilities: Sendable {
    /// Reasoning-tag style. Known values: `qwen3`, `deepseek_r1`,
    /// `think_xml` (all → `<think>...</think>`); `mistral`, `gemma4`, `none`
    /// (no reasoning tags emitted). `nil` means unknown.
    public let reasoningParser: String?

    /// Tool-call format. Known values: `qwen`, `qwen3_coder` → `xml_function`;
    /// `minimax` → `minimax_m2`; `glm47`, `deepseek` → `glm4`; `nemotron` →
    /// `xml_function`; plus any canonical `ToolCallFormat` rawValue. `nil`
    /// means unknown.
    public let toolParser: String?

    /// Whether the model's chat template natively gates `<think>` blocks
    /// behind an `enable_thinking` flag. Consumers may flip this flag to
    /// suppress / require reasoning per request.
    public let thinkInTemplate: Bool?

    /// Whether the model is trained to emit tool calls.
    public let supportsTools: Bool?

    /// Whether the model is trained to emit reasoning blocks.
    public let supportsThinking: Bool?

    /// Family bucket for UI/registry grouping (e.g. `qwen3_5`, `gemma4`).
    public let family: String?

    /// `text` or `vision`. Hint for UI affordances; vmlx detects vision
    /// support from the model class itself.
    public let modality: String?

    /// `kv`, `hybrid`, or `mla`. Hint for cache/memory budgeting. vmlx
    /// engine selects the actual cache type from the model class — `mla`
    /// is currently a forward-looking hint (vmlx falls back to standard
    /// KV for MLA models).
    public let cacheType: String?

    /// Speculative-decoding strategy the JANG bundle ships alongside
    /// this target. Known values: `dflash`, `ddtree`, `autoregressive`,
    /// `none`. `nil` means the bundle does not ship a compatible
    /// drafter. Maps to ``DraftStrategy`` via
    /// ``ParserResolution/draftStrategy(capabilities:modelDirectory:)``.
    public let draftStrategy: String?

    /// Path to the drafter checkpoint, RELATIVE to `jang_config.json`.
    /// Typical value: `"drafter/"` (i.e. a subdirectory next to the
    /// target weights). `nil` when `draftStrategy` is absent or `none`.
    public let drafterPath: String?

    /// Branching budget for ``DraftStrategy/ddtree(drafterPath:branchingBudget:blockSize:)``.
    /// Paper recommends 32-64 for greedy, 16-24 for sampling. `nil`
    /// when `draftStrategy != "ddtree"`.
    public let branchingBudget: Int?

    /// Block size the drafter was trained with — must match
    /// `config.json["block_size"]` inside the drafter snapshot. When
    /// present, callers use this to satisfy
    /// ``DraftStrategy/dflash(drafterPath:blockSize:)`` etc.
    public let blockSize: Int?

    public init(
        reasoningParser: String? = nil,
        toolParser: String? = nil,
        thinkInTemplate: Bool? = nil,
        supportsTools: Bool? = nil,
        supportsThinking: Bool? = nil,
        family: String? = nil,
        modality: String? = nil,
        cacheType: String? = nil,
        draftStrategy: String? = nil,
        drafterPath: String? = nil,
        branchingBudget: Int? = nil,
        blockSize: Int? = nil
    ) {
        self.reasoningParser = reasoningParser
        self.toolParser = toolParser
        self.thinkInTemplate = thinkInTemplate
        self.supportsTools = supportsTools
        self.supportsThinking = supportsThinking
        self.family = family
        self.modality = modality
        self.cacheType = cacheType
        self.draftStrategy = draftStrategy
        self.drafterPath = drafterPath
        self.branchingBudget = branchingBudget
        self.blockSize = blockSize
    }

    /// Source of a parser resolution — used for telemetry and so callers
    /// can log `detection_source=jang_stamped` when the JANG capabilities
    /// stamp wins, vs `detection_source=model_type_heuristic` when the
    /// loader had to fall back.
    public enum ResolutionSource: String, Sendable {
        /// Resolved from `jang_config.json["capabilities"]`.
        case jangStamped = "jang_stamped"
        /// Resolved from `config.json["model_type"]` heuristic (no stamp,
        /// or stamp value was unrecognised).
        case modelTypeHeuristic = "model_type_heuristic"
        /// Neither stamp nor heuristic resolved a parser.
        case none = "none"
    }
}

/// Convenience facade for resolving parsers with explicit precedence.
///
/// Precedence (per vmlx-swift-lm production contract — matches the
/// Tier-1/Tier-2 split osaurus's engine uses):
/// 1. **JANG stamp wins** when present and value resolves.
/// 2. Otherwise fall back to `model_type` heuristic
///    (`ToolCallFormat.infer(from:)`).
/// 3. Otherwise `nil` (caller can render raw).
///
/// Designed so consumers can call this once and log a single
/// `detection_source=` value for diagnostics.
public enum ParserResolution {

    /// Resolve a `ReasoningParser` for a model.
    ///
    /// - Parameters:
    ///   - capabilities: the `JangCapabilities` block from `jang_config.json`
    ///     (pass `nil` for non-JANG models).
    ///   - modelType: the `model_type` field from `config.json` — used as
    ///     a heuristic fallback when no stamp is present.
    /// - Returns: a parser instance and the source it came from. The
    ///   parser is `nil` for models that don't emit reasoning (Mistral 4,
    ///   Gemma 4) — callers should skip parsing and stream raw.
    public static func reasoning(
        capabilities: JangCapabilities?,
        modelType: String?
    ) -> (parser: ReasoningParser?, source: JangCapabilities.ResolutionSource) {
        if let cap = capabilities, cap.reasoningParser != nil {
            // Stamped — honour exactly. `nil` is a valid stamp meaning
            // "this model emits no reasoning".
            return (
                ReasoningParser.fromCapabilityName(cap.reasoningParser),
                .jangStamped
            )
        }
        // Heuristic by model_type. We treat any model_type that hasn't
        // been listed as `mistral`/`gemma`/`gemma4` as potentially
        // reasoning-capable and return a default `<think>...</think>`
        // parser. This matches Qwen 3.5 / 3.6 / DeepSeek / GLM / Nemotron
        // behaviour. False positives are harmless — the parser only
        // splits when it sees actual `<think>` markers in the stream.
        guard let modelType, !modelType.isEmpty else {
            return (nil, .none)
        }
        let t = modelType.lowercased()
        if t.hasPrefix("mistral") || t.hasPrefix("gemma") {
            return (nil, .modelTypeHeuristic)
        }
        return (ReasoningParser(), .modelTypeHeuristic)
    }

    /// Resolve a `ToolCallFormat` for a model.
    ///
    /// - Parameters:
    ///   - capabilities: stamped capabilities, or `nil`.
    ///   - modelType: `model_type` from `config.json` for heuristic fallback.
    public static func toolCall(
        capabilities: JangCapabilities?,
        modelType: String?
    ) -> (format: ToolCallFormat?, source: JangCapabilities.ResolutionSource) {
        if let cap = capabilities,
            let stamped = ToolCallFormat.fromCapabilityName(cap.toolParser)
        {
            return (stamped, .jangStamped)
        }
        if let modelType, let inferred = ToolCallFormat.infer(from: modelType) {
            return (inferred, .modelTypeHeuristic)
        }
        return (nil, .none)
    }

    /// Resolve a ``DraftStrategy`` from JANG capability stamp.
    ///
    /// Maps `capabilities.draft_strategy` + `capabilities.drafter_path`
    /// + `capabilities.branching_budget` + `capabilities.block_size` into
    /// a concrete `DraftStrategy` enum. The drafter path is resolved
    /// relative to `modelDirectory` (the snapshot root containing
    /// `jang_config.json`) — JANG bundles ship drafters co-located.
    ///
    /// Returns `nil` when:
    /// - `capabilities` is nil.
    /// - `draftStrategy` is nil, `"none"`, or unrecognised.
    /// - `drafterPath` is nil (strategy requires one but bundle
    ///   doesn't ship it).
    /// - `blockSize` is nil (required for both `.dflash` + `.ddtree`).
    ///
    /// - Parameters:
    ///   - capabilities: the `JangCapabilities` block from
    ///     `jang_config.json`.
    ///   - modelDirectory: the snapshot root. `capabilities.drafter_path`
    ///     is appended to this.
    public static func draftStrategy(
        capabilities: JangCapabilities?,
        modelDirectory: URL
    ) -> (strategy: DraftStrategy?, source: JangCapabilities.ResolutionSource) {
        guard let cap = capabilities,
            let name = cap.draftStrategy?.lowercased(),
            name != "none",
            let relativePath = cap.drafterPath,
            let blockSize = cap.blockSize
        else {
            return (nil, .none)
        }
        let drafterURL = modelDirectory
            .appendingPathComponent(relativePath, isDirectory: true)
            .resolvingSymlinksInPath()
        switch name {
        case "dflash":
            return (
                .dflash(drafterPath: drafterURL, blockSize: blockSize),
                .jangStamped
            )
        case "ddtree":
            let budget = cap.branchingBudget ?? 32
            return (
                .ddtree(
                    drafterPath: drafterURL,
                    branchingBudget: budget,
                    blockSize: blockSize),
                .jangStamped
            )
        default:
            return (nil, .none)
        }
    }
}

/// Parsed JANG model configuration from jang_config.json.
public struct JangConfig: Sendable {
    public let format: String
    public let formatVersion: String
    public var isV2: Bool { formatVersion.hasPrefix("2") }
    public let quantization: JangQuantization
    public let sourceModel: JangSourceModel
    public let architecture: JangArchitecture
    public let runtime: JangRuntime

    /// Optional capability stamp from the JANG converter. `nil` for
    /// pre-stamp models — consumers should fall back to model-type
    /// heuristics.
    public let capabilities: JangCapabilities?

    public init(
        format: String = "jang",
        formatVersion: String = "2.0",
        quantization: JangQuantization = JangQuantization(),
        sourceModel: JangSourceModel = JangSourceModel(),
        architecture: JangArchitecture = JangArchitecture(),
        runtime: JangRuntime = JangRuntime(),
        capabilities: JangCapabilities? = nil
    ) {
        self.format = format
        self.formatVersion = formatVersion
        self.quantization = quantization
        self.sourceModel = sourceModel
        self.architecture = architecture
        self.runtime = runtime
        self.capabilities = capabilities
    }
}

// MARK: - JANG Loader

/// JANG model loader — detects, parses config, and infers per-layer quantization.
public struct JangLoader: Sendable {

    /// Check if a model directory contains a JANG model.
    public static func isJangModel(at path: URL) -> Bool {
        findConfigPath(at: path) != nil
    }

    /// Find the JANG config file in a model directory.
    public static func findConfigPath(at modelPath: URL) -> URL? {
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        // .jangspec bundles built before the Plan 6 builder update only place
        // jang_config.json under target/. Fall back to the bundle layout so
        // those still load without rebuilding the bundle.
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent("target")
                .appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        return nil
    }

    /// Resolve the directory that holds tokenizer files for a given model.
    ///
    /// The HuggingFace tokenizer loader (`AutoTokenizer.from(modelFolder:)`)
    /// expects `tokenizer.json` and/or `tokenizer_config.json` (plus optionally
    /// `chat_template.jinja`) in the directory it is pointed at. Most JANG /
    /// JANGTQ bundles ship **weights-only** — the snapshot directory contains
    /// `model.safetensors`, `config.json`, `jang_config.json` (and sometimes
    /// `jangtq_runtime.safetensors`) but no tokenizer files. Users are
    /// expected to re-use the tokenizer from the source model declared in
    /// `jang_config.json["source_model"]`.
    ///
    /// This helper implements that fallback for local-directory loads:
    ///
    /// 1. If `modelDirectory` itself has `tokenizer_config.json` or
    ///    `tokenizer.json` → return it unchanged (standard path).
    /// 2. Else if `modelDirectory` has `jang_config.json` with a populated
    ///    `source_model.org` + `source_model.name` → look up the HuggingFace
    ///    cache directory for that repo (`~/.cache/huggingface/hub/models--<org>--<name>`)
    ///    and return the first snapshot that has tokenizer files.
    /// 3. Else → return `modelDirectory` unchanged. The tokenizer loader will
    ///    surface its own error, which is the same behaviour as before this
    ///    helper existed.
    ///
    /// The fallback path **does not** perform network downloads. It only
    /// finds a tokenizer that has already been cached by `Downloader`. If the
    /// source model isn't cached, the returned URL still won't have
    /// tokenizer files and the loader will fail with a clear "no tokenizer"
    /// error — which is the signal for callers to `.download(id:)` the source
    /// repo first.
    ///
    /// - Parameters:
    ///   - modelDirectory: Directory of the model being loaded.
    ///   - huggingFaceCacheRoot: Override for the HF cache root. Defaults to
    ///     `~/.cache/huggingface/hub`. Exposed for unit tests.
    ///   - fileManager: File-manager used for probe. Exposed for unit tests.
    /// - Returns: A directory that should be passed to the tokenizer loader.
    public static func resolveTokenizerDirectory(
        for modelDirectory: URL,
        huggingFaceCacheRoot: URL? = nil,
        fileManager: FileManager = .default
    ) -> URL {
        if hasTokenizerFiles(at: modelDirectory, fileManager: fileManager) {
            return modelDirectory
        }
        guard isJangModel(at: modelDirectory) else { return modelDirectory }

        // Read source_model from jang_config.json. Any parse failure or
        // missing org/name → caller gets the default (unchanged) path.
        let config: JangConfig
        do {
            config = try loadConfig(at: modelDirectory)
        } catch {
            return modelDirectory
        }
        let repo = config.sourceModel.huggingFaceRepoID
        guard !repo.isEmpty else { return modelDirectory }

        let cacheRoot = huggingFaceCacheRoot ?? defaultHuggingFaceCacheRoot()
        let cacheDirName = "models--\(config.sourceModel.org)--\(config.sourceModel.name)"
        let snapshotsRoot = cacheRoot
            .appendingPathComponent(cacheDirName)
            .appendingPathComponent("snapshots")

        guard let entries = try? fileManager.contentsOfDirectory(
            at: snapshotsRoot,
            includingPropertiesForKeys: nil
        ) else {
            return modelDirectory
        }

        // First snapshot directory that actually has tokenizer files wins.
        // HuggingFace snapshots are immutable per revision, so any of them
        // with the files is equally good; the presence check is what matters.
        for snapshot in entries where hasTokenizerFiles(at: snapshot, fileManager: fileManager) {
            return snapshot
        }
        return modelDirectory
    }

    /// Check whether a directory already has the files that the HuggingFace
    /// tokenizer loader needs. Used by `resolveTokenizerDirectory(for:)`.
    public static func hasTokenizerFiles(
        at directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let url = directory.appendingPathComponent(name)
            if fileManager.fileExists(atPath: url.path) { return true }
        }
        return false
    }

    // MARK: - tokenizer_class substitution

    /// swift-transformers 0.1.21's `knownTokenizers` doesn't include
    /// `TokenizersBackend` (used by some mlx-community snapshots like
    /// `mlx-community/Qwen3.5-VL-9B-8bit`) — loads throw
    /// `TokenizerError.unsupportedTokenizer("TokenizersBackend")`. This
    /// set lists all classes we know swift-transformers accepts. Callers
    /// that need different substitutions can override via env var
    /// `VMLX_TOKENIZER_CLASS_OVERRIDE=<target>`.
    public static let knownSupportedTokenizerClasses: Set<String> = [
        "CodeGenTokenizer", "CodeLlamaTokenizer", "FalconTokenizer",
        "GemmaTokenizer", "GPT2Tokenizer", "LlamaTokenizer", "T5Tokenizer",
        "WhisperTokenizer", "CohereTokenizer", "Qwen2Tokenizer",
        "PreTrainedTokenizer",
    ]

    /// Substitution map: when `tokenizer_class` is a key in this map
    /// and no env override is set, rewrite to the value. Tuned from
    /// real-world snapshots: `TokenizersBackend` on Qwen-family VL
    /// models is functionally `Qwen2Tokenizer`.
    public static let defaultTokenizerClassSubstitutions: [String: String] = [
        "TokenizersBackend": "Qwen2Tokenizer",
    ]

    /// Like `resolveTokenizerDirectory(for:)` but also fixes
    /// `tokenizer_class` in `tokenizer_config.json` to an entry that
    /// swift-transformers 0.1.21 knows. If the class is already known,
    /// returns the input directory unchanged. If unknown and no
    /// substitute is available, returns unchanged (let the loader
    /// surface the clear error).
    ///
    /// When a substitution is required, writes a shim directory into
    /// `<tmp>/vmlx-tokenizer-shim-<uuid>/` containing the rewritten
    /// `tokenizer_config.json` plus symlinks to every other tokenizer
    /// file (tokenizer.json, chat_template.jinja, etc.). The caller
    /// should clean up the shim dir when done, but since they live in
    /// the OS temp dir the OS sweeps them eventually.
    ///
    /// Order of operations for a full load:
    ///
    /// 1. Caller has a model directory (maybe JANG, maybe not).
    /// 2. `resolveTokenizerDirectory(for:)` redirects weights-only JANG
    ///    bundles to their source-model snapshot.
    /// 3. `resolveTokenizerClassSubstitution(for:)` (this function)
    ///    rewrites `tokenizer_class` if it's unsupported.
    /// 4. The returned URL is passed to
    ///    `AutoTokenizer.from(modelFolder:)`.
    public static func resolveTokenizerClassSubstitution(
        for directory: URL,
        overrideClass: String? = nil,
        fileManager: FileManager = .default
    ) -> URL {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard fileManager.fileExists(atPath: configURL.path) else {
            return directory  // nothing to rewrite; downstream loader errors
        }
        guard let data = try? Data(contentsOf: configURL),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return directory
        }

        let currentClass = json["tokenizer_class"] as? String ?? ""
        let trimmedCurrent = currentClass.replacingOccurrences(of: "Fast", with: "")

        // Decide the target class.
        let target: String
        let envOverride = overrideClass
            ?? ProcessInfo.processInfo.environment["VMLX_TOKENIZER_CLASS_OVERRIDE"]
        if let envOverride, !envOverride.isEmpty {
            target = envOverride
        } else if knownSupportedTokenizerClasses.contains(trimmedCurrent) {
            return directory  // already supported
        } else if let mapped = defaultTokenizerClassSubstitutions[currentClass]
                            ?? defaultTokenizerClassSubstitutions[trimmedCurrent] {
            target = mapped
        } else {
            return directory  // unknown class, no known substitute
        }

        // If nothing to change, skip.
        if target == currentClass { return directory }

        json["tokenizer_class"] = target

        // Write to a shim dir next to the original.
        let shimDir = fileManager.temporaryDirectory.appendingPathComponent(
            "vmlx-tokenizer-shim-\(UUID().uuidString)")
        do {
            try fileManager.createDirectory(
                at: shimDir, withIntermediateDirectories: true)
            let rewritten = try JSONSerialization.data(
                withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            try rewritten.write(
                to: shimDir.appendingPathComponent("tokenizer_config.json"))
            // Symlink all OTHER files — tokenizer.json especially is often
            // large and we don't want to duplicate it.
            let entries = (try? fileManager.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)) ?? []
            for entry in entries where entry.lastPathComponent != "tokenizer_config.json" {
                let dest = shimDir.appendingPathComponent(entry.lastPathComponent)
                // Some tokenizer caches already contain symlinks — follow them
                // so our shim links to the actual file, not another link.
                let real = (try? fileManager.destinationOfSymbolicLink(atPath: entry.path))
                    .flatMap { relative in
                        URL(fileURLWithPath: relative, relativeTo: entry.deletingLastPathComponent())
                            .standardizedFileURL
                    } ?? entry
                try? fileManager.createSymbolicLink(at: dest, withDestinationURL: real)
            }
            return shimDir
        } catch {
            return directory
        }
    }

    /// Default HuggingFace hub cache root. Honours `HF_HOME` and `HF_HUB_CACHE`
    /// environment variables, otherwise falls back to `~/.cache/huggingface/hub`
    /// — matching the Python `huggingface_hub` resolution order.
    public static func defaultHuggingFaceCacheRoot() -> URL {
        let env = ProcessInfo.processInfo.environment
        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            return URL(fileURLWithPath: hubCache)
        }
        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
    }

    /// Load and parse the JANG config from a model directory.
    public static func loadConfig(at modelPath: URL) throws -> JangConfig {
        guard let configURL = findConfigPath(at: modelPath) else {
            throw JangLoaderError.configNotFound(modelPath.path)
        }

        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw JangLoaderError.invalidConfig("Failed to parse JSON")
        }

        return try parseConfig(from: json)
    }

    /// Parse a JangConfig from a raw JSON dictionary.
    public static func parseConfig(from json: [String: Any]) throws -> JangConfig {
        let format = json["format"] as? String ?? "jang"
        let formatVersion = json["format_version"] as? String ?? "2.0"

        let quantization: JangQuantization
        if let qDict = json["quantization"] as? [String: Any] {
            quantization = JangQuantization(
                method: qDict["method"] as? String ?? "jang-importance",
                profile: qDict["profile"] as? String ?? "JANG_2S",
                targetBits: floatValue(qDict["target_bits"]) ?? 2.5,
                actualBits: floatValue(qDict["actual_bits"]) ?? 2.5,
                blockSize: qDict["block_size"] as? Int ?? 64,
                bitWidthsUsed: qDict["bit_widths_used"] as? [Int] ?? [],
                quantizationScheme: qDict["quantization_scheme"] as? String ?? "asymmetric",
                quantizationBackend: qDict["quantization_backend"] as? String ?? "mx.quantize"
            )
        } else {
            quantization = JangQuantization()
        }

        let sourceModel: JangSourceModel
        if let smDict = json["source_model"] as? [String: Any] {
            let params: String
            if let s = smDict["parameters"] as? String {
                params = s
            } else if let n = smDict["parameters"] as? Int {
                params = String(n)
            } else {
                params = "0"
            }
            sourceModel = JangSourceModel(
                name: smDict["name"] as? String ?? "",
                org: smDict["org"] as? String ?? "",
                architecture: smDict["architecture"] as? String ?? "",
                dtype: smDict["dtype"] as? String ?? "bfloat16",
                parameters: params
            )
        } else {
            sourceModel = JangSourceModel()
        }

        let architecture: JangArchitecture
        if let aDict = json["architecture"] as? [String: Any] {
            architecture = JangArchitecture(
                type: aDict["type"] as? String ?? "transformer",
                attention: aDict["attention"] as? String ?? "gqa",
                hasVision: aDict["has_vision"] as? Bool ?? false,
                hasSSM: aDict["has_ssm"] as? Bool ?? false,
                hasMoE: aDict["has_moe"] as? Bool ?? false
            )
        } else {
            architecture = JangArchitecture()
        }

        let runtime: JangRuntime
        if let rDict = json["runtime"] as? [String: Any] {
            runtime = JangRuntime(
                totalWeightBytes: rDict["total_weight_bytes"] as? Int ?? 0,
                totalWeightGB: floatValue(rDict["total_weight_gb"]) ?? 0
            )
        } else {
            runtime = JangRuntime()
        }

        let capabilities: JangCapabilities?
        if let cDict = json["capabilities"] as? [String: Any] {
            capabilities = JangCapabilities(
                reasoningParser: cDict["reasoning_parser"] as? String,
                toolParser: cDict["tool_parser"] as? String,
                thinkInTemplate: cDict["think_in_template"] as? Bool,
                supportsTools: cDict["supports_tools"] as? Bool,
                supportsThinking: cDict["supports_thinking"] as? Bool,
                family: cDict["family"] as? String,
                modality: cDict["modality"] as? String,
                cacheType: cDict["cache_type"] as? String,
                draftStrategy: cDict["draft_strategy"] as? String,
                drafterPath: cDict["drafter_path"] as? String,
                branchingBudget: cDict["branching_budget"] as? Int,
                blockSize: cDict["block_size"] as? Int
            )
        } else {
            capabilities = nil
        }

        return JangConfig(
            format: format,
            formatVersion: formatVersion,
            quantization: quantization,
            sourceModel: sourceModel,
            architecture: architecture,
            runtime: runtime,
            capabilities: capabilities
        )
    }

    // MARK: - Per-Layer Bit Width Inference

    /// Infer per-layer quantization from loaded JANG weights.
    ///
    /// JANG v2 stores different tensors at different bit widths. The bit width is
    /// inferred from tensor shapes: `actual_bits = (weight.shape[-1] * 32) / (scales.shape[-1] * group_size)`
    ///
    /// Returns a `BaseConfiguration.PerLayerQuantization` that the existing
    /// `loadWeights()` quantization path can use directly.
    public static func inferPerLayerQuantization(
        weights: [String: MLXArray],
        jangConfig: JangConfig
    ) -> BaseConfiguration.PerLayerQuantization {
        let groupSize = jangConfig.quantization.blockSize
        var perLayer = [String: BaseConfiguration.QuantizationOption]()

        // Find the default (most common) bit width from jang_config
        let defaultBits = jangConfig.quantization.bitWidthsUsed.min() ?? 4

        // Group weight keys by their base path (strip .weight/.scales/.biases suffix)
        var quantizedLayers = Set<String>()
        for key in weights.keys {
            if key.hasSuffix(".scales") {
                let basePath = String(key.dropLast(".scales".count))
                quantizedLayers.insert(basePath)
            }
        }

        // For each quantized layer, infer the actual bit width.
        // First try with the JANG block_size as group_size. If that doesn't produce
        // a valid integer bit width, fall back to inferring both bits and group_size
        // from shapes (handles gates with different group_size like 64 vs 128).
        for basePath in quantizedLayers {
            guard let weightArray = weights[basePath + ".weight"],
                let scalesArray = weights[basePath + ".scales"]
            else {
                continue
            }

            // Try with known group_size first; pass bitWidthsUsed so the
            // fallback can disambiguate layers whose group_size differs
            // from the body (e.g., MoE gates).
            let (bits, inferredGroupSize) = inferBitWidthAndGroupSize(
                weight: weightArray, scales: scalesArray,
                knownGroupSize: groupSize,
                bitWidthsUsed: jangConfig.quantization.bitWidthsUsed)

            if bits != defaultBits || inferredGroupSize != groupSize {
                perLayer[basePath] = .quantize(
                    BaseConfiguration.Quantization(groupSize: inferredGroupSize, bits: bits))
            }
        }

        // Layers without .scales are unquantized (norms, biases) — they don't need entries
        // The default quantization covers all layers not in perLayer

        return BaseConfiguration.PerLayerQuantization(
            quantization: BaseConfiguration.Quantization(groupSize: groupSize, bits: defaultBits),
            perLayerQuantization: perLayer
        )
    }

    /// Infer bit width from weight and scales tensor shapes using a fixed group size.
    public static func inferBitWidth(
        weight: MLXArray, scales: MLXArray, groupSize: Int
    ) -> Int {
        inferBitWidthAndGroupSize(weight: weight, scales: scales, knownGroupSize: groupSize).bits
    }

    /// Infer BOTH bit width and group size from weight and scales tensor shapes.
    ///
    /// A JANG quantized tensor has:
    ///   weight.shape[-1] = (in_dim * bits) / 32   (packed into uint32)
    ///   scales.shape[-1] = in_dim / groupSize     (one scale per group per row)
    ///
    /// From these two equations:
    ///   in_dim = scales.shape[-1] * groupSize
    ///   bits   = weight.shape[-1] * 32 / in_dim
    ///
    /// With knownGroupSize this is a direct calculation. Without it, the answer
    /// is not unique from shapes alone — multiple (bits, groupSize) pairs can
    /// produce the same packed shape. In that case we require the provided
    /// `bitWidthsUsed` from the JANG config to disambiguate, preferring
    /// higher bits first (JANG CRITICAL tier uses the highest bits).
    public static func inferBitWidthAndGroupSize(
        weight: MLXArray, scales: MLXArray, knownGroupSize: Int? = nil,
        bitWidthsUsed: [Int] = []
    ) -> (bits: Int, groupSize: Int) {
        let packedDim = weight.shape.last ?? 0
        let numGroups = scales.shape.last ?? 1

        guard packedDim > 0 && numGroups > 0 else { return (4, knownGroupSize ?? 64) }

        let validBits = [2, 3, 4, 5, 6, 8]

        // Primary path: knownGroupSize gives an unambiguous answer.
        // bits must divide (packedDim * 32) exactly.
        if let gs = knownGroupSize, gs > 0 {
            let inDim = numGroups * gs
            if inDim > 0 && (packedDim * 32) % inDim == 0 {
                let bits = (packedDim * 32) / inDim
                if validBits.contains(bits) {
                    return (bits, gs)
                }
            }
        }

        // Fallback: the provided knownGroupSize is wrong for this tensor (e.g.,
        // MoE gates use a different group_size than body layers). Search for
        // (bits, groupSize) pairs that produce an exact round-trip. Prefer
        // higher bits first to favor CRITICAL tier layers (which JANG stores
        // at the highest available bit width).
        //
        // For any candidate `bits`, the implied group size must be an integer:
        //   in_dim = (packedDim * 32) / bits    (must be exact)
        //   gs     = in_dim / numGroups         (must be exact)
        let candidates = bitWidthsUsed.isEmpty
            ? validBits.sorted(by: >)  // [8, 6, 5, 4, 3, 2]
            : bitWidthsUsed.sorted(by: >)
        for bits in candidates {
            guard bits > 0, (packedDim * 32) % bits == 0 else { continue }
            let inDim = (packedDim * 32) / bits
            guard inDim > 0, inDim % numGroups == 0 else { continue }
            let gs = inDim / numGroups
            return (bits, gs)
        }

        return (4, knownGroupSize ?? 64)
    }

    // MARK: - MoE Gate Dequantization

    /// Dequantize MoE gate/router weights from quantized uint32 to float.
    ///
    /// JANG quantizes MoE gate weights at CRITICAL tier (highest available bits)
    /// for routing precision, but the model expects them as plain float Linear
    /// (not QuantizedLinear). This function detects gate weights that have
    /// .scales/.biases companions and dequantizes them in-place.
    ///
    /// Gate patterns matched:
    /// - `.gate.weight` (not `.gate_proj.weight`) — Nemotron, MiniMax
    /// - `.mlp.gate.weight` — Qwen3.5 MoE, general MoE
    /// - `.mixer.gate.weight` — Nemotron-H
    /// - `.router.proj.weight` — Gemma4 (already handled separately)
    public static func dequantizeMoEGates(
        weights: inout [String: MLXArray], groupSize: Int, bitWidthsUsed: [Int] = []
    ) {
        // Find gate weight keys that have .scales companion (meaning they're quantized)
        var gateBasePaths = Set<String>()

        for key in weights.keys {
            // Match gate patterns but NOT gate_proj (which is an expert MLP weight)
            if key.hasSuffix(".gate.scales") && !key.contains("gate_proj") && !key.contains("gate_up") {
                let basePath = String(key.dropLast(".scales".count))
                gateBasePaths.insert(basePath)
            }
            // Also match shared_expert_gate (Qwen3.5 MoE)
            if key.hasSuffix(".shared_expert_gate.scales") {
                let basePath = String(key.dropLast(".scales".count))
                gateBasePaths.insert(basePath)
            }
        }

        for basePath in gateBasePaths {
            guard let gateWeight = weights[basePath + ".weight"],
                let gateScales = weights[basePath + ".scales"]
            else { continue }

            let gateBiases = weights[basePath + ".biases"]

            let packedDim = gateWeight.shape.last ?? 0
            let numGroups = gateScales.shape.last ?? 1

            // Infer bits using bitWidthsUsed (highest first — gates are CRITICAL tier).
            // Shape-only inference is ambiguous: multiple (bits, gs) produce the same
            // packed shapes. Using the known bit widths resolves the ambiguity.
            // For each candidate bits, compute inDim = packedDim * 32 / bits and check
            // that numGroups divides it evenly.
            var inferredBits = 4
            var inferredGroupSize = groupSize

            let candidates = bitWidthsUsed.isEmpty
                ? [8, 6, 5, 4, 3, 2]
                : bitWidthsUsed.sorted(by: >)

            for bits in candidates {
                guard bits > 0 && (packedDim * 32) % bits == 0 else { continue }
                let inDim = (packedDim * 32) / bits
                guard numGroups > 0 && inDim % numGroups == 0 else { continue }
                let gs = inDim / numGroups
                // Verify round-trip: packing inDim at this bits produces packedDim
                if (inDim * bits + 31) / 32 == packedDim || inDim * bits / 32 == packedDim {
                    inferredBits = bits
                    inferredGroupSize = gs
                    break
                }
            }

            // Dequantize to float32 for routing precision
            let dequantized = MLX.dequantized(
                gateWeight, scales: gateScales, biases: gateBiases,
                groupSize: inferredGroupSize, bits: inferredBits)

            // Replace quantized gate with float version, remove scales/biases
            weights[basePath + ".weight"] = dequantized.asType(.float32)
            weights.removeValue(forKey: basePath + ".scales")
            weights.removeValue(forKey: basePath + ".biases")
        }
    }

    // MARK: - V1 Format Support

    /// Check if a model directory contains v1 format JANG weights.
    public static func hasV1Weights(at modelPath: URL) -> Bool {
        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: modelPath, includingPropertiesForKeys: nil)
        else { return false }
        return files.contains {
            $0.pathExtension == "safetensors" && $0.lastPathComponent.contains(".jang.")
        }
    }

    /// Load JANG v1 format weights (legacy uint8 → uint32 repacking).
    public static func loadV1Weights(at modelPath: URL) throws -> [String: MLXArray] {
        let fm = FileManager.default
        let files =
            try fm.contentsOfDirectory(at: modelPath, includingPropertiesForKeys: nil)
            .filter {
                $0.pathExtension == "safetensors" && $0.lastPathComponent.contains(".jang.")
            }

        guard !files.isEmpty else {
            throw JangLoaderError.loadFailed(
                "No .jang.safetensors files found at \(modelPath.path)")
        }

        var allWeights: [String: MLXArray] = [:]
        for file in files {
            let (weights, _) = try loadArraysAndMetadata(url: file)
            for (key, array) in weights {
                if array.dtype == .uint8 {
                    allWeights[key] = repackUint8ToUint32(array)
                } else {
                    allWeights[key] = array
                }
            }
        }
        return allWeights
    }

    /// Repack a uint8 array to uint32 by packing groups of 4 bytes (little-endian).
    private static func repackUint8ToUint32(_ array: MLXArray) -> MLXArray {
        let shape = array.shape
        let lastDim = shape.last ?? 0
        guard lastDim % 4 == 0 else { return array.asType(.uint32) }

        var newShape = shape
        newShape[newShape.count - 1] = lastDim / 4
        newShape.append(4)

        let reshaped = array.reshaped(newShape)
        let b0 = reshaped[0..., 0].asType(.uint32)
        let b1 = reshaped[0..., 1].asType(.uint32) << 8
        let b2 = reshaped[0..., 2].asType(.uint32) << 16
        let b3 = reshaped[0..., 3].asType(.uint32) << 24
        return b0 | b1 | b2 | b3
    }

    // MARK: - Helpers

    private static func floatValue(_ value: Any?) -> Float? {
        if let d = value as? Double { return Float(d) }
        if let f = value as? Float { return f }
        if let i = value as? Int { return Float(i) }
        return nil
    }
}

// MARK: - Errors

public enum JangLoaderError: Error, LocalizedError, Sendable {
    case configNotFound(String)
    case invalidConfig(String)
    case unsupportedVersion(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let path): return "JANG config not found at: \(path)"
        case .invalidConfig(let msg): return "Invalid JANG config: \(msg)"
        case .unsupportedVersion(let ver): return "Unsupported JANG version: \(ver)"
        case .loadFailed(let msg): return "JANG load failed: \(msg)"
        }
    }
}
