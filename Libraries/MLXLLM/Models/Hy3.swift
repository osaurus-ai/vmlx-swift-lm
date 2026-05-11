// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Native runtime + configuration contract for Tencent Hunyuan v3
// (`model_type = hy_v3`, `architectures = ["HYV3ForCausalLM"]`).
//
// Runtime shape: dense causal GQA + Q/K-norm before RoPE, layer-0 dense
// FFN, layers 1...79 sparse sigmoid+expert-bias top-k MoE, always-on
// shared expert, JANGTQ routed experts, and MTP layer preserved in the
// bundle but disabled for plain autoregressive decode.
//
// Source-of-truth: `~/jang/docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md`
// and `~/jang/jang-tools/examples/hy3/swift_runtime/Hy3JANGTQRuntimeSkeleton.swift`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Configuration contract for `model_type = hy_v3` (Tencent Hunyuan v3).
///
/// Matches the Hy3-preview source `config.json` (verified 2026-05-09 against
/// `~/models/Tencent/Hy3-preview/config.json`). Inference-relevant
/// fields are required; training-only fields are omitted.
public struct Hy3Configuration: Codable, Sendable {
    public struct RopeParameters: Codable, Sendable {
        public let ropeTheta: Double
        public let ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let expertHiddenDim: Int
    public let firstKDenseReplace: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let numSharedExperts: Int
    public let qkNorm: Bool
    public let rmsNormEps: Float
    public let ropeParameters: RopeParameters
    public let maxPositionEmbeddings: Int
    public let routeNorm: Bool
    public let routerScalingFactor: Double
    public let moeRouterEnableExpertBias: Bool
    public let moeRouterUseSigmoid: Bool
    public let tieWordEmbeddings: Bool
    public let vocabSize: Int
    public let hiddenAct: String?
    public let numNextnPredictLayers: Int?
    public let weightFormat: String?
    public let mxtqSeed: Int

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
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case expertHiddenDim = "expert_hidden_dim"
        case firstKDenseReplace = "first_k_dense_replace"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case qkNorm = "qk_norm"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case routeNorm = "route_norm"
        case routerScalingFactor = "router_scaling_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case moeRouterUseSigmoid = "moe_router_use_sigmoid"
        case tieWordEmbeddings = "tie_word_embeddings"
        case vocabSize = "vocab_size"
        case hiddenAct = "hidden_act"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case weightFormat = "weight_format"
        case mxtqSeed = "mxtq_seed"
        case mxtqBits = "mxtq_bits"
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
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1536
        self.expertHiddenDim =
            try container.decodeIfPresent(Int.self, forKey: .expertHiddenDim) ?? 1536
        self.firstKDenseReplace =
            try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerTok = RuntimeMoETopKOverride.effectiveTopK(
            currentTopK: try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8,
            modelType: modelType,
            field: CodingKeys.numExpertsPerTok.rawValue)
        self.numSharedExperts =
            try container.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        self.qkNorm = try container.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? true
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeParameters = try container.decode(RopeParameters.self, forKey: .ropeParameters)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.routeNorm = try container.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true
        self.routerScalingFactor =
            try container.decodeIfPresent(Double.self, forKey: .routerScalingFactor) ?? 2.826
        self.moeRouterEnableExpertBias =
            try container.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? true
        self.moeRouterUseSigmoid =
            try container.decodeIfPresent(Bool.self, forKey: .moeRouterUseSigmoid) ?? true
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct)
        self.numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers)
        self.weightFormat = try container.decodeIfPresent(String.self, forKey: .weightFormat)
        self.mxtqSeed = try container.decodeIfPresent(Int.self, forKey: .mxtqSeed) ?? 42
        self.mxtqBits = try Self.decodeRoutedExpertBits(from: container)
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
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try container.encode(expertHiddenDim, forKey: .expertHiddenDim)
        try container.encode(firstKDenseReplace, forKey: .firstKDenseReplace)
        try container.encode(numExperts, forKey: .numExperts)
        try container.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try container.encode(numSharedExperts, forKey: .numSharedExperts)
        try container.encode(qkNorm, forKey: .qkNorm)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeParameters, forKey: .ropeParameters)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(routeNorm, forKey: .routeNorm)
        try container.encode(routerScalingFactor, forKey: .routerScalingFactor)
        try container.encode(moeRouterEnableExpertBias, forKey: .moeRouterEnableExpertBias)
        try container.encode(moeRouterUseSigmoid, forKey: .moeRouterUseSigmoid)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encodeIfPresent(hiddenAct, forKey: .hiddenAct)
        try container.encodeIfPresent(numNextnPredictLayers, forKey: .numNextnPredictLayers)
        try container.encodeIfPresent(weightFormat, forKey: .weightFormat)
        try container.encode(mxtqSeed, forKey: .mxtqSeed)
        try container.encodeIfPresent(mxtqBits, forKey: .mxtqBits)
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

    public var numKeyValueGroups: Int {
        numAttentionHeads / numKeyValueHeads
    }
}

// MARK: - Attention

class Hy3Attention: Module {
    let configuration: Hy3Configuration
    let scale: Float
    let qOutDim: Int
    let kvOutDim: Int

    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ configuration: Hy3Configuration) {
        self.configuration = configuration
        let dim = configuration.hiddenSize
        let heads = configuration.numAttentionHeads
        let kvHeads = configuration.numKeyValueHeads
        let headDim = configuration.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.qOutDim = heads * headDim
        self.kvOutDim = kvHeads * headDim

        _wqkv.wrappedValue = Linear(dim, qOutDim + 2 * kvOutDim, bias: false)
        _wo.wrappedValue = Linear(qOutDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: configuration.rmsNormEps)
        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: Float(configuration.ropeParameters.ropeTheta),
            scale: 1)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = wqkv(x)
        var queries = qkv[.ellipsis, 0 ..< qOutDim]
        var keys = qkv[.ellipsis, qOutDim ..< (qOutDim + kvOutDim)]
        var values = qkv[.ellipsis, (qOutDim + kvOutDim) ..< (qOutDim + 2 * kvOutDim)]

        queries = qNorm(queries.reshaped(B, L, configuration.numAttentionHeads, -1))
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, configuration.numKeyValueHeads, -1))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, configuration.numKeyValueHeads, -1)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
        return wo(output)
    }
}

// MARK: - MLP / MoE

class Hy3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Hy3MoEGate: Module {
    let topK: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ configuration: Hy3Configuration) {
        self.topK = configuration.numExpertsPerTok
        self.routedScalingFactor = Float(configuration.routerScalingFactor)
        self.normTopkProb = configuration.routeNorm
        _weight.wrappedValue = MLXArray.zeros(
            [configuration.numExperts, configuration.hiddenSize],
            dtype: .float32)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros(
            [configuration.numExperts],
            dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = matmul(x.asType(.float32), weight.transposed())
        return groupExpertSelect(
            gates: gates,
            eSCB: eScoreCorrectionBias,
            topK: topK,
            nGroup: 1,
            topkGroup: 1,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb)
    }
}

class Hy3MoE: Module, UnaryLayer {
    let layerIdx: Int
    let sharedExpertCount: Int

    @ModuleInfo(key: "gate") var gate: Hy3MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Hy3MLP

    init(_ configuration: Hy3Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.sharedExpertCount = configuration.numSharedExperts
        _gate.wrappedValue = Hy3MoEGate(configuration)
        let bits = configuration.routedExpertBits ?? 2
        if JANGTQStreamingExperts.isEnabled {
            _switchMLP.wrappedValue = StreamingTurboQuantSwitchGLU(
                inputDims: configuration.hiddenSize,
                hiddenDims: configuration.moeIntermediateSize,
                numExperts: configuration.numExperts,
                gateUpBits: bits,
                downBits: bits,
                seed: configuration.mxtqSeed,
                layerIdx: layerIdx)
        } else {
            _switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: configuration.hiddenSize,
                hiddenDims: configuration.moeIntermediateSize,
                numExperts: configuration.numExperts,
                bits: bits,
                seed: configuration.mxtqSeed)
        }
        _sharedExperts.wrappedValue = Hy3MLP(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.moeIntermediateSize * configuration.numSharedExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: indices)

        let routed: MLXArray
        if let streaming = switchMLP as? StreamingTurboQuantSwitchGLU {
            routed = streaming.reduced(x, indices: indices, scores: scores)
        } else {
            let y = switchMLP(x, indices)
            routed = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        }
        return routed + sharedExperts(x)
    }
}

// MARK: - Logit head

func hy3LMHead(_ hidden: MLXArray, _ lmHead: Linear) -> MLXArray {
    if lmHead is QuantizedLinear {
        return lmHead(hidden)
    }

    let hiddenFP32 = hidden.asType(.float32)
    let weightFP32 = lmHead.weight.asType(.float32)
    var logits = matmul(hiddenFP32, weightFP32.transposed())
    if let bias = lmHead.bias {
        logits = logits + bias.asType(.float32)
    }
    return logits
}

// MARK: - Decoder / model

class Hy3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Hy3Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ configuration: Hy3Configuration, layerIdx: Int) {
        _selfAttn.wrappedValue = Hy3Attention(configuration)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps)

        if layerIdx >= configuration.firstKDenseReplace {
            mlp = Hy3MoE(configuration, layerIdx: layerIdx)
        } else {
            mlp = Hy3MLP(
                dimensions: configuration.hiddenSize,
                hiddenDimensions: configuration.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Hy3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Hy3DecoderLayer]
    let norm: RMSNorm
    let configuration: Hy3Configuration

    init(_ configuration: Hy3Configuration) {
        self.configuration = configuration
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize)
        self.layers = (0 ..< configuration.numHiddenLayers).map {
            Hy3DecoderLayer(configuration, layerIdx: $0)
        }
        self.norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class Hy3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: Hy3ModelInner
    let configuration: Hy3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: Hy3Configuration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabSize
        self.kvHeads = (0 ..< configuration.numHiddenLayers).map {
            _ in configuration.numKeyValueHeads
        }
        self.model = Hy3ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabSize,
                bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return hy3LMHead(out, lmHead)
        }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        let mtpPrefix = "model.layers.\(configuration.numHiddenLayers)."
        for key in Array(sanitized.keys) where key.hasPrefix(mtpPrefix) {
            sanitized[key] = nil
        }

        for key in Array(sanitized.keys)
            where key.hasSuffix(".tq_bits") || key.contains("rotary_emb.inv_freq")
        {
            sanitized[key] = nil
        }

        for layer in 0 ..< configuration.numHiddenLayers {
            let prefix = "model.layers.\(layer).self_attn"
            let qKey = "\(prefix).q_proj"
            let kKey = "\(prefix).k_proj"
            let vKey = "\(prefix).v_proj"
            let fusedKey = "\(prefix).qkv_proj"

            guard let qW = sanitized["\(qKey).weight"],
                  let kW = sanitized["\(kKey).weight"],
                  let vW = sanitized["\(vKey).weight"]
            else { continue }

            let qPacked = qW.dim(qW.ndim - 1)
            let kPacked = kW.dim(kW.ndim - 1)
            let vPacked = vW.dim(vW.ndim - 1)
            if qPacked != kPacked || kPacked != vPacked {
                fatalError(
                    """
                    [Hy3 sanitize] layer \(layer) self_attn has mismatched \
                    bit widths across q/k/v projections (q packed_in=\(qPacked), \
                    k=\(kPacked), v=\(vPacked)). QKV fusion requires identical \
                    bit widths.
                    """
                )
            }

            sanitized["\(fusedKey).weight"] = concatenated([qW, kW, vW], axis: 0)
            sanitized.removeValue(forKey: "\(qKey).weight")
            sanitized.removeValue(forKey: "\(kKey).weight")
            sanitized.removeValue(forKey: "\(vKey).weight")

            for suffix in ["scales", "biases", "bias"] {
                let qS = sanitized["\(qKey).\(suffix)"]
                let kS = sanitized["\(kKey).\(suffix)"]
                let vS = sanitized["\(vKey).\(suffix)"]
                guard let qS, let kS, let vS else { continue }
                sanitized["\(fusedKey).\(suffix)"] = concatenated([qS, kS, vS], axis: 0)
                sanitized.removeValue(forKey: "\(qKey).\(suffix)")
                sanitized.removeValue(forKey: "\(kKey).\(suffix)")
                sanitized.removeValue(forKey: "\(vKey).\(suffix)")
            }
        }

        for layer in configuration.firstKDenseReplace ..< configuration.numHiddenLayers {
            let prefix = "model.layers.\(layer).mlp"

            for suffix in ["weight", "scales", "biases"] {
                let src = "\(prefix).router.gate.\(suffix)"
                let dst = "\(prefix).gate.\(suffix)"
                if let value = sanitized.removeValue(forKey: src) {
                    sanitized[dst] = value
                }
            }

            if let expertBias = sanitized.removeValue(forKey: "\(prefix).expert_bias") {
                sanitized["\(prefix).gate.e_score_correction_bias"] = expertBias
            }

            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let src = "\(prefix).shared_mlp.\(projection).\(suffix)"
                    let dst = "\(prefix).shared_experts.\(projection).\(suffix)"
                    if let value = sanitized.removeValue(forKey: src) {
                        sanitized[dst] = value
                    }
                }
            }

            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["tq_packed", "tq_norms"] {
                    let first = "\(prefix).experts.0.\(projection).\(suffix)"
                    guard sanitized[first] != nil else { continue }
                    if JANGTQStreamingExperts.isEnabled {
                        for expert in 0 ..< configuration.numExperts {
                            sanitized["\(prefix).experts.\(expert).\(projection).\(suffix)"] = nil
                        }
                        continue
                    }
                    let stacked = (0 ..< configuration.numExperts).map { expert in
                        sanitized.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)")!
                    }
                    sanitized["\(prefix).switch_mlp.\(projection).\(suffix)"] =
                        loadTimeMaterializedStacked(stacked)
                }
            }
        }

        return sanitized
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
