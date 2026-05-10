import Foundation
import CoreGraphics
import CoreImage
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Lightweight configuration contract for `model_type = zaya1_vl`.
///
/// This intentionally does not instantiate a fake runtime. ZAYA1-VL needs a
/// native adapter that combines the Zaya CCA/MoE decoder, Qwen2.5-VL image
/// preprocessing/vision tower semantics, and vision-token-gated LoRA. Until
/// that adapter exists, the VLM factory registers the type only to provide a
/// precise recognition gate instead of falling through as an unknown model or
/// silently routing through the text-only `zaya` path.
public struct Zaya1VLConfiguration: Codable, Sendable {
    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String?
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullattBlockIndexes: [Int]
        public let tokensPerSecond: Int
        public let inChannels: Int?
        public let layerNormEps: Float
        public let skipVision: Bool
        public let hiddenAct: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialPatchSize = "spatial_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size"
            case fullattBlockIndexes = "fullatt_block_indexes"
            case tokensPerSecond = "tokens_per_second"
            case inChannels = "in_chans"
            case layerNormEps = "layer_norm_eps"
            case skipVision = "skip_vision"
            case hiddenAct = "hidden_act"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
            self.depth = try container.decodeIfPresent(Int.self, forKey: .depth) ?? 32
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3420
            self.outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
            self.numHeads = try container.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
            self.patchSize =
                try container.decodeIfPresent(Int.self, forKey: .patchSize)
                ?? container.decodeIfPresent(Int.self, forKey: .spatialPatchSize)
                ?? 14
            self.spatialPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .spatialPatchSize) ?? patchSize
            self.spatialMergeSize =
                try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            self.temporalPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            self.windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 112
            self.fullattBlockIndexes =
                try container.decodeIfPresent([Int].self, forKey: .fullattBlockIndexes)
                ?? [7, 15, 23, 31]
            self.tokensPerSecond =
                try container.decodeIfPresent(Int.self, forKey: .tokensPerSecond) ?? 4
            self.inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels)
            self.layerNormEps =
                try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            self.skipVision = try container.decodeIfPresent(Bool.self, forKey: .skipVision) ?? false
            self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encodeIfPresent(modelType, forKey: .modelType)
            try container.encode(depth, forKey: .depth)
            try container.encode(hiddenSize, forKey: .hiddenSize)
            try container.encode(intermediateSize, forKey: .intermediateSize)
            try container.encode(outHiddenSize, forKey: .outHiddenSize)
            try container.encode(numHeads, forKey: .numHeads)
            try container.encode(patchSize, forKey: .patchSize)
            try container.encode(spatialPatchSize, forKey: .spatialPatchSize)
            try container.encode(spatialMergeSize, forKey: .spatialMergeSize)
            try container.encode(temporalPatchSize, forKey: .temporalPatchSize)
            try container.encode(windowSize, forKey: .windowSize)
            try container.encode(fullattBlockIndexes, forKey: .fullattBlockIndexes)
            try container.encode(tokensPerSecond, forKey: .tokensPerSecond)
            try container.encodeIfPresent(inChannels, forKey: .inChannels)
            try container.encode(layerNormEps, forKey: .layerNormEps)
            try container.encode(skipVision, forKey: .skipVision)
            try container.encode(hiddenAct, forKey: .hiddenAct)
        }
    }

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let numQueryGroups: Int
    public let maxPositionEmbeddings: Int
    public let rotaryBase: Int
    public let ropePct: Float
    public let ffnHiddenSize: Int
    public let zayaMLPExpansion: Int
    public let zayaExpertLayout: String
    public let normEpsilon: Float
    public let clampTemp: Bool
    public let projectorHiddenAct: String
    public let numExperts: Int
    public let moeRouterTopk: Int
    public let cca: Bool
    public let zayaUseEDA: Bool
    public let zayaUseMOD: Bool
    public let scaleResidualMerge: Bool
    public let residualInFP32: Bool
    public let tieWordEmbeddings: Bool
    public let visionLora: Bool
    public let visionLoraRankAttn: Int?
    public let visionLoraRankMLP: Int?
    public let vocabSize: Int
    public let imageTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int
    public let weightFormat: String?
    public let visionConfiguration: VisionConfiguration

    private let mxtqBits: Int?
    public var routedExpertBits: Int? { mxtqBits }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numQueryGroups = "num_query_groups"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rotaryBase = "rotary_base"
        case ropePct = "rope_pct"
        case ffnHiddenSize = "ffn_hidden_size"
        case zayaMLPExpansion = "zaya_mlp_expansion"
        case zayaExpertLayout = "zaya_expert_layout"
        case normEpsilon = "norm_epsilon"
        case clampTemp = "clamp_temp"
        case projectorHiddenAct = "projector_hidden_act"
        case numExperts = "num_experts"
        case moeRouterTopk = "moe_router_topk"
        case cca
        case zayaUseEDA = "zaya_use_eda"
        case zayaUseMOD = "zaya_use_mod"
        case scaleResidualMerge = "scale_residual_merge"
        case residualInFP32 = "residual_in_fp32"
        case tieWordEmbeddings = "tie_word_embeddings"
        case visionLora = "vision_lora"
        case visionLoraRankAttn = "vision_lora_rank_attn"
        case visionLoraRankMLP = "vision_lora_rank_mlp"
        case vocabSize = "vocab_size"
        case imageTokenId = "image_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        self.numQueryGroups =
            try container.decodeIfPresent(Int.self, forKey: .numQueryGroups) ?? 2
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.rotaryBase = try container.decodeIfPresent(Int.self, forKey: .rotaryBase) ?? 1_000_000
        self.ropePct = try container.decodeIfPresent(Float.self, forKey: .ropePct) ?? 0.5
        self.ffnHiddenSize = try container.decodeIfPresent(Int.self, forKey: .ffnHiddenSize) ?? 4096
        self.zayaMLPExpansion =
            try container.decodeIfPresent(Int.self, forKey: .zayaMLPExpansion) ?? 256
        self.zayaExpertLayout =
            try container.decodeIfPresent(String.self, forKey: .zayaExpertLayout)
            ?? "split_switch_mlp"
        self.normEpsilon = try container.decodeIfPresent(Float.self, forKey: .normEpsilon) ?? 1e-5
        self.clampTemp = try container.decodeIfPresent(Bool.self, forKey: .clampTemp) ?? false
        self.projectorHiddenAct =
            try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 16
        self.moeRouterTopk = try container.decodeIfPresent(Int.self, forKey: .moeRouterTopk) ?? 1
        self.cca = try container.decodeIfPresent(Bool.self, forKey: .cca) ?? true
        self.zayaUseEDA = try container.decodeIfPresent(Bool.self, forKey: .zayaUseEDA) ?? true
        self.zayaUseMOD = try container.decodeIfPresent(Bool.self, forKey: .zayaUseMOD) ?? true
        self.scaleResidualMerge =
            try container.decodeIfPresent(Bool.self, forKey: .scaleResidualMerge) ?? true
        self.residualInFP32 =
            try container.decodeIfPresent(Bool.self, forKey: .residualInFP32) ?? false
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.visionLora = try container.decodeIfPresent(Bool.self, forKey: .visionLora) ?? false
        self.visionLoraRankAttn =
            try container.decodeIfPresent(Int.self, forKey: .visionLoraRankAttn)
        self.visionLoraRankMLP =
            try container.decodeIfPresent(Int.self, forKey: .visionLoraRankMLP)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
        self.visionStartTokenId = try container.decode(Int.self, forKey: .visionStartTokenId)
        self.visionEndTokenId = try container.decode(Int.self, forKey: .visionEndTokenId)
        self.weightFormat = try container.decodeIfPresent(String.self, forKey: .weightFormat)
        self.mxtqBits = try Self.decodeRoutedExpertBits(from: container)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(architectures, forKey: .architectures)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(numQueryGroups, forKey: .numQueryGroups)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(rotaryBase, forKey: .rotaryBase)
        try container.encode(ropePct, forKey: .ropePct)
        try container.encode(ffnHiddenSize, forKey: .ffnHiddenSize)
        try container.encode(zayaMLPExpansion, forKey: .zayaMLPExpansion)
        try container.encode(zayaExpertLayout, forKey: .zayaExpertLayout)
        try container.encode(normEpsilon, forKey: .normEpsilon)
        try container.encode(clampTemp, forKey: .clampTemp)
        try container.encode(projectorHiddenAct, forKey: .projectorHiddenAct)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(moeRouterTopk, forKey: .moeRouterTopk)
        try container.encode(cca, forKey: .cca)
        try container.encode(zayaUseEDA, forKey: .zayaUseEDA)
        try container.encode(zayaUseMOD, forKey: .zayaUseMOD)
        try container.encode(scaleResidualMerge, forKey: .scaleResidualMerge)
        try container.encode(residualInFP32, forKey: .residualInFP32)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(visionLora, forKey: .visionLora)
        try container.encodeIfPresent(visionLoraRankAttn, forKey: .visionLoraRankAttn)
        try container.encodeIfPresent(visionLoraRankMLP, forKey: .visionLoraRankMLP)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(imageTokenId, forKey: .imageTokenId)
        try container.encode(visionStartTokenId, forKey: .visionStartTokenId)
        try container.encode(visionEndTokenId, forKey: .visionEndTokenId)
        try container.encodeIfPresent(weightFormat, forKey: .weightFormat)
        try container.encodeIfPresent(mxtqBits, forKey: .mxtqBits)
        try container.encode(visionConfiguration, forKey: .visionConfiguration)
    }

    private static func decodeRoutedExpertBits(
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> Int? {
        if let value = try? container.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            return value
        }
        guard container.contains(.mxtqBits) else {
            return nil
        }
        let dict = try container.decode([String: Int].self, forKey: .mxtqBits)
        return dict["routed_expert"] ?? dict["experts"] ?? dict.values.min()
    }

    public func makeQwen25VisionConfiguration() throws -> Qwen25VLConfiguration.VisionConfiguration {
        let data = try JSONEncoder().encode(visionConfiguration)
        return try JSONDecoder.json5().decode(
            Qwen25VLConfiguration.VisionConfiguration.self, from: data)
    }

    /// Build the shared Zaya text-primitive configuration used by the eventual
    /// ZAYA1-VL 40-block trunk.
    ///
    /// This is not a route through text-only `ZayaModel`: ZAYA1-VL layers run
    /// CCA attention and MoE sequentially in each of 40 blocks, while text-only
    /// ZAYA alternates those sublayers across 80 layers. The bridge exists only
    /// to keep dimensions, RoPE, routing, quantization bits, and cache sizing
    /// identical when the native VL block classes instantiate the shared
    /// attention/MoE primitives.
    public func makeZayaTextConfiguration() -> ZayaTextConfiguration {
        var text = ZayaTextConfiguration()
        text.modelType = modelType
        text.hiddenSize = hiddenSize
        text.numHiddenLayers = numHiddenLayers
        text.numAttentionHeads = numAttentionHeads
        text.numKeyValueHeads = numKeyValueHeads
        text.numQueryGroups = numQueryGroups
        text.ccaNumQHeads = numAttentionHeads
        text.kvChannels = headDim
        text.numExperts = numExperts
        text.moeRouterTopk = moeRouterTopk
        text.maxPositionEmbeddings = maxPositionEmbeddings
        text.ropeTheta = Float(rotaryBase)
        text.partialRotaryFactor = ropePct
        text.vocabSize = vocabSize
        text.normEpsilon = normEpsilon
        text.ffnHiddenSize = ffnHiddenSize
        text.tieWordEmbeddings = tieWordEmbeddings
        text.scaleResidualMerge = scaleResidualMerge
        text.residualInFP32 = residualInFP32
        text.weightFormat = weightFormat
        text.mxtqBits = routedExpertBits
        text.zayaExpertLayout = zayaExpertLayout
        return text
    }
}

public enum Zaya1VLRuntimeSupport {
    public struct ImageMergeResult {
        public let embeddings: MLXArray
        public let imageMask: MLXArray
    }

    /// Apply a LoRA/additive branch only at image-token positions.
    ///
    /// ZAYA1-VL gates several text-trunk LoRA additions by the image-token
    /// mask. Keep the mask broadcast logic in one place so attention and MoE
    /// ports do not each hand-roll slightly different shape rules.
    public static func applyImageMaskedAdd(
        base: MLXArray,
        addon: MLXArray,
        imageMask: MLXArray?
    ) throws -> MLXArray {
        guard let imageMask else {
            return base
        }
        guard base.shape == addon.shape else {
            throw VLMError.processing(
                "ZAYA1-VL LoRA addon shape \(addon.shape) does not match base shape \(base.shape)")
        }

        var mask = imageMask
        if mask.ndim == 1 && base.ndim >= 3 && mask.dim(0) == base.dim(1) {
            mask = mask.expandedDimensions(axis: 0)
        }
        while mask.ndim < base.ndim {
            mask = mask.expandedDimensions(axis: -1)
        }
        guard mask.ndim == base.ndim else {
            throw VLMError.processing(
                "ZAYA1-VL image mask shape \(imageMask.shape) is incompatible with base shape \(base.shape)")
        }
        for axis in 0 ..< base.ndim where mask.dim(axis) != 1 && mask.dim(axis) != base.dim(axis) {
            throw VLMError.processing(
                "ZAYA1-VL image mask shape \(imageMask.shape) is incompatible with base shape \(base.shape)")
        }

        let broadcastMask = broadcast(mask, to: base.shape).asType(base.dtype)
        return base + addon.asType(base.dtype) * broadcastMask
    }

    /// Replace text embeddings at ZAYA image-token positions with projected
    /// image features, and return the exact token mask needed by image-gated
    /// LoRA in the ZAYA1-VL text trunk.
    public static func mergeImageFeatures(
        inputIds: MLXArray,
        inputEmbeds: MLXArray,
        imageFeatures: MLXArray,
        imageTokenId: Int
    ) throws -> ImageMergeResult {
        let imageMask = inputIds .== MLXArray(imageTokenId)
        let imageTokenCount = imageMask.sum().item(Int.self)
        guard imageTokenCount == imageFeatures.dim(0) else {
            throw VLMError.processing(
                "ZAYA1-VL image token count (\(imageTokenCount)) does not match image features (\(imageFeatures.dim(0)))")
        }

        var maskForEmbeds = imageMask
        if inputIds.ndim == 1 && inputEmbeds.ndim == 3 {
            maskForEmbeds = imageMask.expandedDimensions(axis: 0)
        }
        var maskExpanded = maskForEmbeds.expandedDimensions(axis: -1)
        maskExpanded = broadcast(maskExpanded, to: inputEmbeds.shape)

        guard maskExpanded.sum().item(Int.self) == imageFeatures.size else {
            throw VLMError.processing(
                "ZAYA1-VL image feature width does not match text embedding width")
        }

        let indices = nonZero(maskExpanded.flattened().asType(.bool))
        let result = inputEmbeds.flattened()
        if !indices.isEmpty {
            let flattenedFeatures = imageFeatures.asType(inputEmbeds.dtype).flattened()
            guard indices.count == flattenedFeatures.size else {
                throw VLMError.processing(
                    "ZAYA1-VL image feature count does not match expanded image-token mask")
            }
            result[MLXArray(indices.map { UInt32($0) })] = flattenedFeatures
        }

        return ImageMergeResult(
            embeddings: result.reshaped(inputEmbeds.shape),
            imageMask: imageMask)
    }

    private static func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }
}

/// Weight-key normalization for native ZAYA1-VL.
///
/// The shipped bundles mirror the HF module tree:
/// `model.layers.N.attn.*`, `model.layers.N.mlp.*`, and a converter-side
/// JANGTQ sidecar path at `model.layers.N.zaya_block.experts.switch_mlp.*`.
/// The Swift native trunk uses the explicit `mlp.zaya_block` ownership path
/// for every MoE module, so loading must canonicalize those keys before
/// MLXNN's module update walk. Keep this separate from factory dispatch so
/// the contract is testable before we enable full model loading.
public enum Zaya1VLWeightSanitizer {
    public static func sanitize(
        weights input: [String: MLXArray],
        configuration: Zaya1VLConfiguration
    ) -> [String: MLXArray] {
        var weights = input

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }

        for key in Array(weights.keys) where key.hasSuffix(".tq_bits") {
            weights[key] = nil
        }

        for key in Array(weights.keys) where key.contains(".conv_qk.")
            && key.hasSuffix(".weight")
        {
            guard let value = weights[key], value.ndim == 3 else { continue }
            if value.dim(-1) == 2 {
                weights[key] = value.movedAxis(source: 2, destination: 1)
            }
        }

        for layer in 0 ..< configuration.numHiddenLayers {
            let from = "model.layers.\(layer).zaya_block.experts.switch_mlp."
            let to = "model.layers.\(layer).mlp.zaya_block.experts.switch_mlp."
            for key in Array(weights.keys) where key.hasPrefix(from) {
                let suffix = String(key.dropFirst(from.count))
                weights[to + suffix] = weights[key]
                weights[key] = nil
            }

            fillResidualScaleDefaults(
                in: &weights,
                prefix: "model.layers.\(layer).attn.res_scale")
            fillResidualScaleDefaults(
                in: &weights,
                prefix: "model.layers.\(layer).mlp.res_scale")
        }

        compressRouterMLPSequentialIndices(in: &weights, layers: configuration.numHiddenLayers)

        let routerHidden = configuration.zayaMLPExpansion
        for layer in 0 ..< configuration.numHiddenLayers {
            let key = "model.layers.\(layer).mlp.zaya_block.router.router_states_scale"
            if weights[key] == nil {
                weights[key] = MLXArray.ones([routerHidden], dtype: .float32)
            }

            let finalBias = "model.layers.\(layer).mlp.zaya_block.router.router_mlp.2.bias"
            if weights[finalBias] == nil {
                weights[finalBias] = MLXArray.zeros(
                    [configuration.numExperts + 1], dtype: .float32)
            }
        }

        return weights
    }

    private static func fillResidualScaleDefaults(
        in weights: inout [String: MLXArray],
        prefix: String
    ) {
        if weights["\(prefix).residual_scale"] == nil {
            weights["\(prefix).residual_scale"] = MLXArray.ones([1], dtype: .float32)
        }
        if weights["\(prefix).residual_bias"] == nil {
            weights["\(prefix).residual_bias"] = MLXArray.zeros([1], dtype: .float32)
        }
    }

    private static func compressRouterMLPSequentialIndices(
        in weights: inout [String: MLXArray],
        layers: Int
    ) {
        for layer in 0 ..< layers {
            let prefix = "model.layers.\(layer).mlp.zaya_block.router.router_mlp."
            let renames: [(String, String)] = [
                ("\(prefix)2.", "\(prefix)1."),
                ("\(prefix)4.", "\(prefix)2."),
            ]
            var rewrites: [(String, MLXArray)] = []
            var deletes: [String] = []

            for key in Array(weights.keys) {
                guard let value = weights[key] else { continue }
                for (from, to) in renames where key.hasPrefix(from) {
                    rewrites.append((to + String(key.dropFirst(from.count)), value))
                    deletes.append(key)
                    break
                }
            }
            for key in deletes {
                weights[key] = nil
            }
            for (key, value) in rewrites {
                weights[key] = value
            }
        }
    }
}

/// Two-linear LoRA adapter matching the shipped ZAYA1-VL tensor layout.
///
/// Bundle keys use `.0.weight` for the input-to-rank projection and
/// `.1.weight` for the rank-to-output projection, for example:
/// `lora_linear_q.0.weight` is `[rank=8, hidden=2048]` and
/// `lora_linear_q.1.weight` is `[out=1024, rank=8]`.
public final class Zaya1VLLowRankAdapter: Module {
    @ModuleInfo(key: "0") private var down: Linear
    @ModuleInfo(key: "1") private var up: Linear

    public init(inputDimensions: Int, rank: Int, outputDimensions: Int) {
        self._down.wrappedValue = Linear(inputDimensions, rank, bias: false)
        self._up.wrappedValue = Linear(rank, outputDimensions, bias: false)
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        up(down(input))
    }

    public func callAsFunction(
        _ input: MLXArray,
        base: MLXArray,
        imageMask: MLXArray?
    ) throws -> MLXArray {
        try Zaya1VLRuntimeSupport.applyImageMaskedAdd(
            base: base,
            addon: self(input),
            imageMask: imageMask)
    }
}

/// Q/K/V LoRA adapters under `attn.self_attn.qkv.*`.
public final class Zaya1VLQKVLoRAAdapters: Module {
    @ModuleInfo(key: "lora_linear_q") private var queryAdapter: Zaya1VLLowRankAdapter
    @ModuleInfo(key: "lora_linear_k") private var keyAdapter: Zaya1VLLowRankAdapter
    @ModuleInfo(key: "lora_val_proj1") private var value1Adapter: Zaya1VLLowRankAdapter
    @ModuleInfo(key: "lora_val_proj2") private var value2Adapter: Zaya1VLLowRankAdapter

    public init(_ config: Zaya1VLConfiguration) {
        let rank = config.visionLoraRankAttn ?? 8
        let queryDimensions = config.numAttentionHeads * config.headDim
        let keyValueDimensions = config.numKeyValueHeads * config.headDim
        let splitValueDimensions = keyValueDimensions / 2
        self._queryAdapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: config.hiddenSize,
            rank: rank,
            outputDimensions: queryDimensions)
        self._keyAdapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: config.hiddenSize,
            rank: rank,
            outputDimensions: keyValueDimensions)
        self._value1Adapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: config.hiddenSize,
            rank: rank,
            outputDimensions: splitValueDimensions)
        self._value2Adapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: config.hiddenSize,
            rank: rank,
            outputDimensions: splitValueDimensions)
        super.init()
    }

    public func query(_ input: MLXArray, base: MLXArray, imageMask: MLXArray?) throws -> MLXArray {
        try queryAdapter(input, base: base, imageMask: imageMask)
    }

    public func key(_ input: MLXArray, base: MLXArray, imageMask: MLXArray?) throws -> MLXArray {
        try keyAdapter(input, base: base, imageMask: imageMask)
    }

    public func value1(_ input: MLXArray, base: MLXArray, imageMask: MLXArray?) throws -> MLXArray {
        try value1Adapter(input, base: base, imageMask: imageMask)
    }

    public func value2(_ input: MLXArray, base: MLXArray, imageMask: MLXArray?) throws -> MLXArray {
        try value2Adapter(input, base: base, imageMask: imageMask)
    }
}

/// Attention LoRA adapters under `attn.self_attn.*`.
public final class Zaya1VLAttentionLoRAAdapters: Module {
    @ModuleInfo(key: "qkv") public var qkv: Zaya1VLQKVLoRAAdapters
    @ModuleInfo(key: "lora_linear_o") private var outputAdapter: Zaya1VLLowRankAdapter

    public init(_ config: Zaya1VLConfiguration) {
        let rank = config.visionLoraRankAttn ?? 8
        let outputInputDimensions = config.numAttentionHeads * config.headDim
        self._qkv.wrappedValue = Zaya1VLQKVLoRAAdapters(config)
        self._outputAdapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: outputInputDimensions,
            rank: rank,
            outputDimensions: config.hiddenSize)
        super.init()
    }

    public func output(
        _ input: MLXArray,
        base: MLXArray,
        imageMask: MLXArray?
    ) throws -> MLXArray {
        try outputAdapter(input, base: base, imageMask: imageMask)
    }
}

/// Per-expert MLP LoRA adapters under `local_experts.{E}.*`.
public final class Zaya1VLExpertLoRAAdapters: Module {
    @ModuleInfo(key: "lora_fc1") private var fc1Adapter: Zaya1VLLowRankAdapter
    @ModuleInfo(key: "lora_fc2") private var fc2Adapter: Zaya1VLLowRankAdapter

    public init(_ config: Zaya1VLConfiguration) {
        let rank = config.visionLoraRankMLP ?? 32
        let fc2InputDimensions = config.ffnHiddenSize / 2
        self._fc1Adapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: config.hiddenSize,
            rank: rank,
            outputDimensions: config.ffnHiddenSize)
        self._fc2Adapter.wrappedValue = Zaya1VLLowRankAdapter(
            inputDimensions: fc2InputDimensions,
            rank: rank,
            outputDimensions: config.hiddenSize)
        super.init()
    }

    public func fc1(_ input: MLXArray, base: MLXArray, imageMask: MLXArray?) throws -> MLXArray {
        try fc1Adapter(input, base: base, imageMask: imageMask)
    }

    public func fc2(_ input: MLXArray, base: MLXArray, imageMask: MLXArray?) throws -> MLXArray {
        try fc2Adapter(input, base: base, imageMask: imageMask)
    }
}

/// Container for `mlp.zaya_block.experts.local_experts.{0...15}` LoRA sidecars.
public final class Zaya1VLLocalExpertLoRAAdapters: Module {
    @ModuleInfo(key: "0") private var expert0: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "1") private var expert1: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "2") private var expert2: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "3") private var expert3: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "4") private var expert4: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "5") private var expert5: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "6") private var expert6: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "7") private var expert7: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "8") private var expert8: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "9") private var expert9: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "10") private var expert10: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "11") private var expert11: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "12") private var expert12: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "13") private var expert13: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "14") private var expert14: Zaya1VLExpertLoRAAdapters
    @ModuleInfo(key: "15") private var expert15: Zaya1VLExpertLoRAAdapters

    public init(_ config: Zaya1VLConfiguration) {
        self._expert0.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert1.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert2.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert3.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert4.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert5.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert6.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert7.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert8.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert9.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert10.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert11.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert12.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert13.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert14.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        self._expert15.wrappedValue = Zaya1VLExpertLoRAAdapters(config)
        super.init()
    }

    public func adapter(for expertIndex: Int) throws -> Zaya1VLExpertLoRAAdapters {
        switch expertIndex {
        case 0: expert0
        case 1: expert1
        case 2: expert2
        case 3: expert3
        case 4: expert4
        case 5: expert5
        case 6: expert6
        case 7: expert7
        case 8: expert8
        case 9: expert9
        case 10: expert10
        case 11: expert11
        case 12: expert12
        case 13: expert13
        case 14: expert14
        case 15: expert15
        default:
            throw VLMError.processing("ZAYA1-VL expert index \(expertIndex) is out of range")
        }
    }
}

/// Native image-feature bridge for ZAYA1-VL.
///
/// This is the first executable piece of the native adapter: it keeps the
/// shipped `vision_tower.*` tensor namespace, reuses the Qwen2.5-VL ViT
/// implementation, and performs the ZAYA-specific image-token replacement.
/// The 40-layer CCA/MoE text trunk is still a separate missing component.
public final class Zaya1VLInputEmbeddingAdapter: Module {
    public struct Result {
        public let embeddings: MLXArray
        public let imageMask: MLXArray?
    }

    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen25Vision.VisionModel
    private let imageTokenId: Int

    public init(_ config: Zaya1VLConfiguration) throws {
        self.imageTokenId = config.imageTokenId
        self._visionModel.wrappedValue = Qwen25Vision.VisionModel(
            try config.makeQwen25VisionConfiguration())
        super.init()
    }

    public func projectImageFeatures(pixelValues: MLXArray, frames: [THW]) -> MLXArray {
        visionModel(pixelValues, frames: frames)
    }

    public func mergeImageFeatures(
        inputIds: MLXArray,
        inputEmbeds: MLXArray,
        pixelValues: MLXArray?,
        frames: [THW]?
    ) throws -> Result {
        switch (pixelValues, frames) {
        case (nil, nil):
            return Result(embeddings: inputEmbeds, imageMask: nil)
        case let (pixelValues?, frames?):
            let imageFeatures = projectImageFeatures(pixelValues: pixelValues, frames: frames)
            let merged = try Zaya1VLRuntimeSupport.mergeImageFeatures(
                inputIds: inputIds,
                inputEmbeds: inputEmbeds,
                imageFeatures: imageFeatures,
                imageTokenId: imageTokenId)
            return Result(embeddings: merged.embeddings, imageMask: merged.imageMask)
        default:
            throw VLMError.processing(
                "ZAYA1-VL image pixels and frame metadata must be provided together")
        }
    }
}

/// ZAYA1-VL user-input processor.
///
/// ZAYA1-VL uses the Qwen2.5-VL image geometry and normalization contract,
/// but its chat template emits `<image>` inside the vision delimiters rather
/// than Qwen's `<|image_pad|>` placeholder. Keep this as a separate processor
/// so image-token expansion matches the shipped ZAYA1-VL templates.
public struct Zaya1VLProcessor: UserInputProcessor {
    private let config: Qwen25VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen25VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }
        let (extentH, extentW) = try QwenVL.intExtent(images[0].extent.size)
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: extentH, width: extentW,
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels, maxPixels: config.size.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages =
            images
            .map { MediaProcessing.inSRGBToneCurveSpace($0) }
            .map { MediaProcessing.resampleBicubic($0, to: resizedSize) }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple)
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: processedImages, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.processing("ZAYA1-VL video input is not implemented")
        }

        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        if input.images.isEmpty {
            return LMInput(
                tokens: MLXArray(promptTokens),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
        }

        let imagePixelsAndFrames = try input.images.map {
            try preprocess(images: [$0.asCIImage()], processing: input.processing)
        }
        let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
        let processedImage = LMInput.ProcessedImage(
            pixels: imagePixelsConcatenated, frames: imagePixelsAndFrames.map { $0.1 })

        if let imageFrames = processedImage.frames {
            promptTokens = try QwenVL.replacePaddingTokens(
                in: promptTokens, frames: imageFrames, paddingToken: "<image>",
                mergeSize: config.mergeSize, tokenizer: tokenizer)
        }

        return LMInput(
            text: LMInput.Text(tokens: MLXArray(promptTokens)),
            image: processedImage,
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext))
    }
}
