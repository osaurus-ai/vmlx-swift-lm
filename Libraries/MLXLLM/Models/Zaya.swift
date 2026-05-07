// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// ZAYA1-8B port — single model class, three MoE backends.
//
// Architecture from /Users/eric/jang/research/ZAYA1-8B-RUNTIME-PREP-2026-05-06.md:
//   - 80 decoder layers, alternating: even = CCA-attention, odd = MoE
//   - Hidden 2048, 16 query heads, 2 KV heads, head_dim 128, cca_num_q_heads 8
//   - CCA attention: linear_q (→1024), linear_k (→256), val_proj1+val_proj2
//     (concat → 256), conv_qk(2 layers, kernel 2), o_proj (1024 → 2048)
//   - CCA state per attention layer (FLOAT32): conv_state[B,1280,2], prev_hs[B,2048]
//   - MoE: 16 experts top-1, router MLP (256 hidden), MOD skip route (17th logit)
//   - Tied embeddings, rope_theta=5_000_000, partial_rotary_factor=0.5

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct ZayaTextConfiguration: Codable, Sendable {
    public var modelType: String = "zaya"
    public var hiddenSize: Int = 2048
    public var numHiddenLayers: Int = 80
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 2
    public var numQueryGroups: Int = 2
    public var ccaNumQHeads: Int = 8
    public var kvChannels: Int = 128            // head_dim
    public var numExperts: Int = 16
    public var moeRouterTopk: Int = 1
    public var maxPositionEmbeddings: Int = 131_072
    public var ropeTheta: Float = 5_000_000
    public var partialRotaryFactor: Float = 0.5
    public var vocabSize: Int = 262_272
    public var normEpsilon: Float = 1e-6
    public var ffnHiddenSize: Int = 2048
    public var tieWordEmbeddings: Bool = true

    // JANGTQ / quantization knobs (filled by factory after jang_config merge).
    public var weightFormat: String?
    public var mxtqBits: Int?
    public var mxtqGateUpBits: Int?
    public var mxtqDownBits: Int?
    public var mxtqSeed: Int?
    public var zayaExpertLayout: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numQueryGroups = "num_query_groups"
        case ccaNumQHeads = "cca_num_q_heads"
        case kvChannels = "kv_channels"
        case numExperts = "num_experts"
        case moeRouterTopk = "moe_router_topk"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case vocabSize = "vocab_size"
        case normEpsilon = "norm_epsilon"
        case ffnHiddenSize = "ffn_hidden_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case weightFormat = "weight_format"
        case mxtqBits = "mxtq_bits"
        case mxtqGateUpBits = "mxtq_gate_up_bits"
        case mxtqDownBits = "mxtq_down_bits"
        case mxtqSeed = "mxtq_seed"
        case zayaExpertLayout = "zaya_expert_layout"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Scalar fields with defaults — decode-if-present to keep tolerant
        // of partial configs (test fixtures, future variants).
        if let v = try c.decodeIfPresent(String.self, forKey: .modelType) { self.modelType = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) { self.hiddenSize = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) { self.numHiddenLayers = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) { self.numAttentionHeads = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) { self.numKeyValueHeads = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numQueryGroups) { self.numQueryGroups = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .ccaNumQHeads) { self.ccaNumQHeads = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .kvChannels) { self.kvChannels = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .numExperts) { self.numExperts = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .moeRouterTopk) { self.moeRouterTopk = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) { self.maxPositionEmbeddings = v }
        if let v = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) { self.ropeTheta = v }
        if let v = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) { self.partialRotaryFactor = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .vocabSize) { self.vocabSize = v }
        if let v = try c.decodeIfPresent(Float.self, forKey: .normEpsilon) { self.normEpsilon = v }
        if let v = try c.decodeIfPresent(Int.self, forKey: .ffnHiddenSize) { self.ffnHiddenSize = v }
        if let v = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) { self.tieWordEmbeddings = v }

        self.weightFormat = try c.decodeIfPresent(String.self, forKey: .weightFormat)
        // mxtqBits cascade — accept flat int or per-role dict (factory pre-merges
        // the nested layout into mxtq_gate_up_bits / mxtq_down_bits / mxtq_bits).
        if let flat = try? c.decodeIfPresent(Int.self, forKey: .mxtqBits) {
            self.mxtqBits = flat
        } else if let dict = try? c.decodeIfPresent([String: Int].self, forKey: .mxtqBits) {
            self.mxtqBits = dict["routed_expert"] ?? dict.values.first
        }
        self.mxtqGateUpBits = try c.decodeIfPresent(Int.self, forKey: .mxtqGateUpBits)
        self.mxtqDownBits = try c.decodeIfPresent(Int.self, forKey: .mxtqDownBits)
        self.mxtqSeed = try c.decodeIfPresent(Int.self, forKey: .mxtqSeed)
        self.zayaExpertLayout = try c.decodeIfPresent(String.self, forKey: .zayaExpertLayout)
    }
}

public struct ZayaConfiguration: Codable, Sendable {
    public var modelType: String = "zaya"
    public var textConfig: ZayaTextConfiguration = ZayaTextConfiguration()

    public init() {}

    public init(from decoder: Decoder) throws {
        // ZAYA configs ship flat — no text_config wrapper. Decode the same
        // payload into both modelType and textConfig.
        let single = try ZayaTextConfiguration(from: decoder)
        self.modelType = single.modelType
        self.textConfig = single
    }

    public func encode(to encoder: Encoder) throws {
        try textConfig.encode(to: encoder)
    }
}

/// Picks the MoE backend at module-init time.
public enum ZayaMoEContext: Sendable, Equatable {
    /// JANGTQ2 / JANGTQ4 / JANGTQ_K — codebook-quantized routed experts.
    case jangtq(gateUpBits: Int, downBits: Int, seed: Int)
    /// MXFP4 — affine-4 routed experts.
    case affine(bits: Int, groupSize: Int)
    /// Base BF16 — bf16 routed experts (stack-at-load from per-expert keys).
    case bf16
}

// MARK: - Residual scale block

/// Per-layer residual merge: out = (hidden_states_scale * h + hidden_states_bias)
///                              + (residual_scale * x + residual_bias)
/// Layer 0 has only the (hidden_states_*) pair — `residual_*` is nil there.
final class ZayaResScale: Module {
    @ModuleInfo(key: "hidden_states_scale") var hiddenScale: MLXArray
    @ModuleInfo(key: "hidden_states_bias") var hiddenBias: MLXArray
    @ModuleInfo(key: "residual_scale") var residualScale: MLXArray?
    @ModuleInfo(key: "residual_bias") var residualBias: MLXArray?

    override init() {
        self._hiddenScale.wrappedValue = MLXArray.ones([1])
        self._hiddenBias.wrappedValue = MLXArray.zeros([1])
        self._residualScale.wrappedValue = MLXArray.ones([1])
        self._residualBias.wrappedValue = MLXArray.zeros([1])
        super.init()
    }

    func merge(_ h: MLXArray, residual x: MLXArray) -> MLXArray {
        let scaledH = hiddenScale * h + hiddenBias
        if let rs = residualScale, let rb = residualBias {
            return scaledH + (rs * x + rb)
        }
        return scaledH + x
    }
}

// MARK: - CCA-attention QKV block

final class ZayaCCAQKV: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "val_proj1") var valProj1: Linear
    @ModuleInfo(key: "val_proj2") var valProj2: Linear
    @ModuleInfo(key: "conv_qk") var convQK: [Conv1d]
    @ModuleInfo(key: "temp") var temp: MLXArray

    let qDim: Int
    let kDim: Int
    let headDim: Int
    let convChannels: Int

    init(_ cfg: ZayaTextConfiguration) {
        let H = cfg.hiddenSize
        let qDim = cfg.ccaNumQHeads * cfg.kvChannels
        let kDim = cfg.numQueryGroups * cfg.kvChannels
        self.qDim = qDim
        self.kDim = kDim
        self.headDim = cfg.kvChannels
        self.convChannels = qDim + kDim

        self._linearQ.wrappedValue = Linear(H, qDim, bias: false)
        self._linearK.wrappedValue = Linear(H, kDim, bias: false)
        self._valProj1.wrappedValue = Linear(H, cfg.kvChannels, bias: false)
        self._valProj2.wrappedValue = Linear(H, cfg.kvChannels, bias: false)
        // Two causal Conv1d in series.
        // Layer 0: kernel 2, in_channels=1, out_channels=convChannels (per bundle: [1280,1,2]).
        // Layer 1: kernel 2, in_channels=headDim (128), out_channels=convChannels (per bundle: [1280,128,2]).
        self._convQK.wrappedValue = [
            Conv1d(inputChannels: 1, outputChannels: convChannels, kernelSize: 2, bias: true),
            Conv1d(inputChannels: cfg.kvChannels, outputChannels: convChannels, kernelSize: 2, bias: true),
        ]
        self._temp.wrappedValue = MLXArray.ones([2])
        super.init()
    }
}

// MARK: - CCA-attention layer

final class ZayaCCAAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: ZayaCCAQKV
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let qHeads: Int
    let kvHeads: Int
    let headDim: Int
    let qDim: Int
    let kDim: Int
    let convChannels: Int
    let ropeDim: Int
    let ropeTheta: Float
    let scale: Float
    let hiddenSize: Int
    let rope: RoPE

    init(_ cfg: ZayaTextConfiguration) {
        self.qHeads = cfg.ccaNumQHeads
        self.kvHeads = cfg.numQueryGroups
        self.headDim = cfg.kvChannels
        self.qDim = qHeads * headDim
        self.kDim = kvHeads * headDim
        self.convChannels = qDim + kDim
        self.ropeDim = Int((Float(headDim) * cfg.partialRotaryFactor).rounded(.toNearestOrEven))
        self.ropeTheta = cfg.ropeTheta
        self.scale = 1.0 / Float(headDim).squareRoot()
        self.hiddenSize = cfg.hiddenSize

        self._qkv.wrappedValue = ZayaCCAQKV(cfg)
        self._oProj.wrappedValue = Linear(qDim, cfg.hiddenSize, bias: false)
        self.rope = RoPE(dimensions: ropeDim, traditional: false, base: ropeTheta)
        super.init()
    }

    /// Per the runtime contract:
    /// - q = linear_q(norm_x).reshape([B,T,8,128]).transpose([B,8,T,128])
    /// - k = linear_k(norm_x).reshape([B,T,2,128]).transpose([B,2,T,128])
    /// - v = concat(val_proj1, val_proj2, axis=-1).reshape(...) → [B,2,T,128]
    /// - Apply RoPE (partial_rotary_factor=0.5) on q,k.
    /// - Concat q,k along feature, transpose to [B, 1280, T].
    /// - Prepend prior conv_state [B,1280,2] → [B,1280,T+2].
    /// - conv_qk[0] (kernel 2, in_channels=1, depthwise-style on each channel) → [B,1280,T+1].
    ///   Apply silu, then conv_qk[1] (kernel 2, in_channels=128) → [B,1280,T].
    /// - Update conv_state ← out[:, :, -2:].
    /// - Split out → q (first 1024) and k (last 256), reshape back to [B,H,T,D].
    /// - KV cache update; repeat KV by 4 for 8 query heads.
    /// - SDPA(q,k,v) → [B,8,T,128]. Reshape and apply temp[0].
    /// - o_proj → [B,T,2048]. Residual via res_scale (caller wraps).
    /// - Update prev_hs ← out[:, -1, :].
    func callAsFunction(
        _ x: MLXArray,
        cache: ZayaCCACache?,
        batchCache: BatchZayaCCACache?
    ) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Q/K/V projections.
        let q0 = qkv.linearQ(x).reshaped([B, T, qHeads, headDim]).transposed(0, 2, 1, 3)  // [B,8,T,128]
        let k0 = qkv.linearK(x).reshaped([B, T, kvHeads, headDim]).transposed(0, 2, 1, 3)  // [B,2,T,128]
        let v = concatenated([qkv.valProj1(x), qkv.valProj2(x)], axis: -1)
                  .reshaped([B, T, kvHeads, headDim])
                  .transposed(0, 2, 1, 3)                                                    // [B,2,T,128]

        // RoPE on partial dims.
        let offset = cache?.offset ?? batchCache?.offset ?? 0
        let q1 = rope(q0, offset: offset)
        let k1 = rope(k0, offset: offset)

        // Concat q,k along feature dim, then transpose to [B, C, T] for conv.
        let qSeq = q1.transposed(0, 2, 1, 3).reshaped([B, T, qDim])
        let kSeq = k1.transposed(0, 2, 1, 3).reshaped([B, T, kDim])
        let qkFeat = concatenated([qSeq, kSeq], axis: -1).transposed(0, 2, 1)  // [B, 1280, T]

        // Read prior conv state (gathered for batched B>1).
        let priorConv: MLXArray
        if let bc = batchCache {
            priorConv = bc.gatherCCA().conv
        } else if let c = cache {
            priorConv = c.readCCA().conv
        } else {
            priorConv = MLXArray.zeros([B, convChannels, 2], dtype: .float32)
        }
        let qkAug = concatenated([priorConv.asType(qkFeat.dtype), qkFeat], axis: -1)  // [B,1280,T+2]

        // conv_qk[0]: kernel size 2, channels 1 → 1280 (per-channel). Treat as channel-mixing
        // convolution. MLX Conv1d expects [N, L, C_in], so transpose, conv, transpose back.
        let c0in = qkAug.transposed(0, 2, 1)                                          // [B, T+2, 1280]
        let c0out = qkv.convQK[0](c0in)                                                // [B, T+1, 1280]
        let c0act = silu(c0out)
        let c1out = qkv.convQK[1](c0act)                                               // [B, T,   1280]
        let qkPostConv = c1out.transposed(0, 2, 1)                                     // [B, 1280, T]

        // New conv state = last 2 timesteps of conv output. Pad with zeros if T<2.
        let lastN = qkPostConv.dim(-1)
        let newConv: MLXArray = {
            if lastN >= 2 {
                return qkPostConv[0..., 0..., (lastN - 2)..<lastN].asType(.float32)
            }
            // T < 2: pad zeros on the left.
            let pad = MLXArray.zeros([B, convChannels, 2 - lastN], dtype: qkPostConv.dtype)
            return concatenated([pad, qkPostConv], axis: -1).asType(.float32)
        }()

        // Split conv output back into q, k.
        let qkPostFeat = qkPostConv.transposed(0, 2, 1)                                // [B, T, 1280]
        let qOut = qkPostFeat[0..., 0..., 0..<qDim]
                     .reshaped([B, T, qHeads, headDim]).transposed(0, 2, 1, 3)         // [B,8,T,128]
        let kOut = qkPostFeat[0..., 0..., qDim..<(qDim + kDim)]
                     .reshaped([B, T, kvHeads, headDim]).transposed(0, 2, 1, 3)        // [B,2,T,128]

        // KV cache update (per-slot in batched mode via BatchZayaCCACache).
        var (kFull, vFull) = (kOut, v)
        if let bc = batchCache {
            (kFull, vFull) = bc.update(keys: kOut, values: v)
        } else if let c = cache {
            (kFull, vFull) = c.update(keys: kOut, values: v)
        }

        // Repeat KV to match query heads.
        let repeatN = qHeads / kvHeads
        let kRep = repeated(kFull, count: repeatN, axis: 1)
        let vRep = repeated(vFull, count: repeatN, axis: 1)

        // Mask.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let bc = batchCache {
            mask = bc.makeMask(n: T, windowSize: nil, returnArray: false)
        } else if let c = cache {
            mask = c.makeMask(n: T, windowSize: nil, returnArray: false)
        } else if T > 1 {
            mask = .causal
        } else {
            mask = .none
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: qOut, keys: kRep, values: vRep, scale: scale, mask: mask)
        let attnFlat = attn.transposed(0, 2, 1, 3).reshaped([B, T, qDim])
        let scaled = attnFlat * qkv.temp[0]
        let out = oProj(scaled)

        // New prev_hs = last token's output (post-o_proj, pre-residual). Use x's
        // hidden representation to get [B, hiddenSize] in float32.
        let newPrev = out[0..., (T - 1)..<T, 0...]
                        .reshaped([B, hiddenSize])
                        .asType(.float32)

        if let bc = batchCache {
            bc.scatterCCA(conv: newConv, prev: newPrev)
        } else if let c = cache {
            c.writeCCA(conv: newConv, prev: newPrev)
        }
        return out
    }
}

// MARK: - MoE layer

final class ZayaRouter: Module {
    @ModuleInfo(key: "rmsnorm_eda") var edaNorm: RMSNorm
    @ModuleInfo(key: "router_mlp") var routerMLP: [Linear]   // sanitize compresses 0/2/4 → 0/1/2
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "balancing_biases") var balancingBiases: MLXArray

    let numExperts: Int

    init(_ cfg: ZayaTextConfiguration) {
        self.numExperts = cfg.numExperts
        let H = cfg.hiddenSize
        let R = 256  // router hidden dim (canonical to ZAYA1; not in config)
        self._edaNorm.wrappedValue = RMSNorm(dimensions: H, eps: cfg.normEpsilon)
        self._routerMLP.wrappedValue = [
            Linear(H, R, bias: true),
            Linear(R, R, bias: true),
            Linear(R, cfg.numExperts + 1, bias: false),
        ]
        self._downProj.wrappedValue = Linear(H, H, bias: true)
        self._balancingBiases.wrappedValue = MLXArray.zeros([cfg.numExperts])
        super.init()
    }

    /// Returns: (expertIndices [B*T], expertWeights [B*T], routerSkipDelta [B*T, H])
    /// where the skip delta is the auxiliary down_proj contribution that gets
    /// added to the expert output.
    func route(_ x: MLXArray) -> (idx: MLXArray, weights: MLXArray, aux: MLXArray) {
        let normed = edaNorm(x)
        let r0 = relu(routerMLP[0](normed))
        let r1 = relu(routerMLP[1](r0))
        let logits = routerMLP[2](r1)                                     // [B*T, E+1]
        // Take the first E logits and add balancing biases. The 17th
        // (MOD skip) route is currently not consumed in v1 — top-1 over
        // experts only. vLLM reference does the same.
        let routedLogits = logits[0..., 0..<numExperts] + balancingBiases
        let probs = softmax(routedLogits, axis: -1)                       // [B*T, E]
        let idx = argMax(probs, axis: -1)                                 // [B*T]
        let weights = takeAlong(probs, idx.expandedDimensions(axis: -1), axis: -1)
                        .squeezed(axis: -1)                               // [B*T]
        let aux = downProj(normed)                                        // [B*T, H]
        return (idx, weights, aux)
    }
}

/// Polymorphic switch primitive — JANGTQ uses TurboQuantSwitchGLU,
/// MXFP4/BF16 use the standard SwitchGLU.
public protocol ZayaSwitchPrimitive: Module {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray
}
extension SwitchGLU: ZayaSwitchPrimitive {}
extension TurboQuantSwitchGLU: ZayaSwitchPrimitive {}

final class ZayaExperts: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: ZayaSwitchPrimitive

    init(_ cfg: ZayaTextConfiguration, context: ZayaMoEContext?) {
        let H = cfg.hiddenSize
        let I = cfg.ffnHiddenSize
        let E = cfg.numExperts
        switch context {
        case .some(.jangtq(let gateUp, let down, let seed)):
            self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
                inputDims: H, hiddenDims: I, numExperts: E,
                gateUpBits: gateUp, downBits: down, seed: seed)
        case .some(.affine), .some(.bf16), nil:
            self._switchMLP.wrappedValue = SwitchGLU(
                inputDims: H, hiddenDims: I, numExperts: E)
        }
        super.init()
    }
}

final class ZayaMoEBlock: Module {
    @ModuleInfo(key: "router") var router: ZayaRouter
    @ModuleInfo(key: "experts") var experts: ZayaExperts

    let hiddenSize: Int

    init(_ cfg: ZayaTextConfiguration, context: ZayaMoEContext?) {
        self.hiddenSize = cfg.hiddenSize
        self._router.wrappedValue = ZayaRouter(cfg)
        self._experts.wrappedValue = ZayaExperts(cfg, context: context)
        super.init()
    }

    /// Given normed input `nx` (the input_norm output of the layer),
    /// runs the router + experts and returns the additive contribution
    /// (expert output + auxiliary skip). The decoder layer is responsible
    /// for residual blending via res_scale.
    func callAsFunction(_ nx: MLXArray) -> MLXArray {
        let B = nx.dim(0)
        let T = nx.dim(1)
        let xFlat = nx.reshaped([B * T, hiddenSize])

        let (idx, weights, aux) = router.route(xFlat)            // idx [B*T], weights [B*T], aux [B*T,H]
        // SwitchGLU / TurboQuantSwitchGLU expect (x, indices).
        let xIn = xFlat.reshaped([B, T, hiddenSize])
        let idx2D = idx.reshaped([B, T, 1])                       // [B,T,K=1]
        let expertOut = experts.switchMLP(xIn, idx2D)             // [B,T,K,H] or [B,T,H]
        let expertFlat: MLXArray
        if expertOut.ndim == 4 {
            // Sum across the K=1 axis.
            expertFlat = expertOut.sum(axis: 2).reshaped([B * T, hiddenSize])
        } else {
            expertFlat = expertOut.reshaped([B * T, hiddenSize])
        }
        let weighted = expertFlat * weights.reshaped([B * T, 1]).asType(expertFlat.dtype)
        let combined = weighted + aux
        return combined.reshaped([B, T, hiddenSize])
    }
}

// MARK: - Decoder layer

final class ZayaDecoderLayer: Module {
    @ModuleInfo(key: "input_norm") var inputNorm: RMSNorm
    @ModuleInfo(key: "res_scale") var resScale: ZayaResScale
    @ModuleInfo(key: "self_attn") var selfAttn: ZayaCCAAttention?
    @ModuleInfo(key: "zaya_block") var zayaBlock: ZayaMoEBlock?

    let isAttention: Bool

    init(_ cfg: ZayaTextConfiguration, layerIdx: Int, context: ZayaMoEContext?) {
        self.isAttention = (layerIdx % 2 == 0)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.normEpsilon)
        self._resScale.wrappedValue = ZayaResScale()
        if isAttention {
            self._selfAttn.wrappedValue = ZayaCCAAttention(cfg)
        } else {
            self._zayaBlock.wrappedValue = ZayaMoEBlock(cfg, context: context)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let nx = inputNorm(x)
        let h: MLXArray
        if isAttention {
            // Pass either the per-slot ZayaCCACache or the BatchZayaCCACache wrapper.
            h = selfAttn!(nx,
                cache: cache as? ZayaCCACache,
                batchCache: cache as? BatchZayaCCACache)
        } else {
            h = zayaBlock!(nx)
        }
        return resScale.merge(h, residual: x)
    }
}

// MARK: - Inner trunk

public final class ZayaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [ZayaDecoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ModuleInfo(key: "res_scale") var resScale: ZayaResScale

    let numHiddenLayers: Int

    init(_ cfg: ZayaTextConfiguration, context: ZayaMoEContext?) {
        self.numHiddenLayers = cfg.numHiddenLayers
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self._layers.wrappedValue = (0 ..< cfg.numHiddenLayers).map { l in
            ZayaDecoderLayer(cfg, layerIdx: l, context: context)
        }
        self._finalNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.normEpsilon)
        self._resScale.wrappedValue = ZayaResScale()
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let embed = embedTokens(inputs)
        var h = embed
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: cache?[i])
        }
        // Top-level res_scale wraps around the embed→trunk path.
        h = resScale.merge(h, residual: embed)
        return finalNorm(h)
    }
}

// MARK: - Top-level model

public final class ZayaModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") public var model: ZayaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: ZayaConfiguration
    public let context: ZayaMoEContext?
    public let kvHeads: [Int]
    public var vocabularySize: Int { configuration.textConfig.vocabSize }

    public init(_ configuration: ZayaConfiguration, moe context: ZayaMoEContext?) {
        self.configuration = configuration
        self.context = context
        let cfg = configuration.textConfig
        self._model.wrappedValue = ZayaModelInner(cfg, context: context)
        if !cfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabSize, bias: false)
        }
        self.kvHeads = (0 ..< cfg.numHiddenLayers).map { _ in cfg.numQueryGroups }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(h)
        }
        return model.embedTokens.asLinear(h)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let cfg = configuration.textConfig
        let convChannels = cfg.ccaNumQHeads * cfg.kvChannels + cfg.numQueryGroups * cfg.kvChannels
        return (0 ..< cfg.numHiddenLayers).map { l in
            if l % 2 == 0 {
                return ZayaCCACache(
                    batchSize: 1,
                    convChannels: convChannels,
                    hiddenSize: cfg.hiddenSize)
            } else {
                // No-op stub for MoE layers — the layer's forward never
                // touches its slot, but the engine indexes per-decoder-layer.
                return KVCacheSimple()
            }
        }
    }

    /// Rewrite bundle keys to the module's hierarchy and stack BF16 per-expert
    /// weights when present. JANGTQ + MXFP4 already ship pre-stacked.
    public func sanitize(weights w: [String: MLXArray]) -> [String: MLXArray] {
        var weights = w

        // Tied embeddings — drop lm_head.weight so MLX doesn't try to bind it
        // to a non-existent module (lmHead is nil under tieWordEmbeddings=true).
        if configuration.textConfig.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }

        // Strip per-tensor .tq_bits hints — metadata, not module parameters.
        for k in Array(weights.keys) where k.hasSuffix(".tq_bits") {
            weights[k] = nil
        }

        // Compress router_mlp.{0,2,4} → router_mlp.{0,1,2}. The bundle ships
        // a Sequential(Linear, ReLU, Linear, ReLU, Linear); the Swift
        // module declares 3 Linears in a [Linear] array with explicit ReLU.
        for k in Array(weights.keys) {
            guard k.contains(".router_mlp.") else { continue }
            // Match exactly ".router_mlp.0." / ".router_mlp.2." / ".router_mlp.4."
            let renames: [(String, String)] = [
                (".router_mlp.0.", ".router_mlp.0."),    // unchanged
                (".router_mlp.2.", ".router_mlp.1."),
                (".router_mlp.4.", ".router_mlp.2."),
            ]
            for (from, to) in renames {
                if k.contains(from), from != to {
                    let newKey = k.replacingOccurrences(of: from, with: to)
                    weights[newKey] = weights[k]
                    weights[k] = nil
                    break
                }
            }
        }

        // Per-MoE-layer layout sniff.
        let cfg = configuration.textConfig
        let H = cfg.hiddenSize
        let E = cfg.numExperts
        for layer in stride(from: 1, to: cfg.numHiddenLayers, by: 2) {
            let prefix = "model.layers.\(layer).zaya_block.experts"
            let stackedTQProbe = "\(prefix).switch_mlp.gate_proj.tq_packed"
            let stackedAffineProbe = "\(prefix).switch_mlp.gate_proj.weight"
            let perExpertProbe = "\(prefix).local_experts.0.linear_fc1.weight"

            if weights[stackedTQProbe] != nil { continue }
            if weights[stackedAffineProbe] != nil { continue }

            guard weights[perExpertProbe] != nil else { continue }

            // Stack per-expert BF16 weights.
            // linear_fc1 = [2H, H], split rows: [:H] = gate_proj, [H:] = up_proj.
            // linear_fc2 = [H, H] = down_proj.
            var gates: [MLXArray] = []
            var ups: [MLXArray] = []
            var downs: [MLXArray] = []
            gates.reserveCapacity(E); ups.reserveCapacity(E); downs.reserveCapacity(E)
            for e in 0 ..< E {
                let fc1Key = "\(prefix).local_experts.\(e).linear_fc1.weight"
                let fc2Key = "\(prefix).local_experts.\(e).linear_fc2.weight"
                guard let fc1 = weights.removeValue(forKey: fc1Key),
                      let fc2 = weights.removeValue(forKey: fc2Key) else { continue }
                gates.append(fc1[0..<H, 0...])
                ups.append(fc1[H..<(2 * H), 0...])
                downs.append(fc2)
            }
            weights["\(prefix).switch_mlp.gate_proj.weight"] = loadTimeMaterializedStacked(gates)
            weights["\(prefix).switch_mlp.up_proj.weight"] = loadTimeMaterializedStacked(ups)
            weights["\(prefix).switch_mlp.down_proj.weight"] = loadTimeMaterializedStacked(downs)
        }

        return weights
    }
}

extension ZayaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
