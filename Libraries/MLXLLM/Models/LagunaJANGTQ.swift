//
// LagunaJANGTQ — JANGTQ-quantized variant of LagunaModel.
//
// Drop-in replacement when a Laguna bundle ships `weight_format=mxtq`.
// Architecture is identical to LagunaModel — only difference is every
// dense `Linear` (attention Q/K/V/O/G + MLP gate/up/down + MoE expert
// gate/up/down + shared-expert gate/up/down) is replaced with
// `JANGTQDenseLinear` so the safetensors' `.tq_packed` + `.tq_norms`
// keys feed the codebook kernels.
//
// `embed_tokens` and `lm_head` stay full-precision (matching the
// `mxtq_bits.embed_lm_head=passthrough_fp16` profile shipped in
// jang_tools/laguna). The router `gate` (small Linear:
// hidden_size → num_experts) and `e_score_correction_bias` also stay
// vanilla — those are tiny and aren't worth quantizing.
//
// Same per-layer mixed cache topology as LagunaModel
// (RotatingKVCache for SWA layers, KVCacheSimple for full layers).
// Same sanitize() remap.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - JANGTQ Attention

internal final class LagunaJANGTQAttention: Module {
    let layerIndex: Int
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let kvGroups: Int
    let scale: Float
    let layerType: String
    let ropeBase: Float
    let partial: Float
    let ropeDim: Int

    @ModuleInfo(key: "q_proj") var wq: JANGTQDenseLinear
    @ModuleInfo(key: "k_proj") var wk: JANGTQDenseLinear
    @ModuleInfo(key: "v_proj") var wv: JANGTQDenseLinear
    @ModuleInfo(key: "o_proj") var wo: JANGTQDenseLinear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var gProj: JANGTQDenseLinear?

    let ropeFull: RoPE
    let ropePartial: RoPE?

    init(_ cfg: LagunaConfiguration, layerIndex: Int, bits: Int, seed: Int) {
        self.layerIndex = layerIndex
        self.nHeads = cfg.numAttentionHeadsPerLayer[layerIndex]
        self.nKVHeads = cfg.numKeyValueHeads
        self.headDim = cfg.headDim
        self.kvGroups = nHeads / nKVHeads
        self.scale = pow(Float(headDim), -0.5)
        self.layerType = cfg.layerTypes[layerIndex]
        let (theta, partial) = cfg.ropeFor(layerType: layerType)
        self.ropeBase = theta
        self.partial = partial
        self.ropeDim = Int(Float(headDim) * partial)

        let h = cfg.hiddenSize
        self._wq.wrappedValue = JANGTQDenseLinear(
            inFeatures: h, outFeatures: nHeads * headDim,
            bits: bits, seed: seed, bias: cfg.attentionBias)
        self._wk.wrappedValue = JANGTQDenseLinear(
            inFeatures: h, outFeatures: nKVHeads * headDim,
            bits: bits, seed: seed, bias: cfg.attentionBias)
        self._wv.wrappedValue = JANGTQDenseLinear(
            inFeatures: h, outFeatures: nKVHeads * headDim,
            bits: bits, seed: seed, bias: cfg.attentionBias)
        self._wo.wrappedValue = JANGTQDenseLinear(
            inFeatures: nHeads * headDim, outFeatures: h,
            bits: bits, seed: seed, bias: cfg.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        if cfg.gating {
            self._gProj.wrappedValue = JANGTQDenseLinear(
                inFeatures: h, outFeatures: nHeads,
                bits: bits, seed: seed, bias: false)
        }

        self.ropeFull = RoPE(dimensions: headDim, traditional: false, base: ropeBase)
        if ropeDim != headDim {
            self.ropePartial = RoPE(dimensions: ropeDim, traditional: false, base: ropeBase)
        } else {
            self.ropePartial = nil
        }
    }

    private func applyPartialRope(_ t: MLXArray, offset: Int) -> MLXArray {
        if let ropePartial {
            let rot = t[.ellipsis, ..<ropeDim]
            let keep = t[.ellipsis, ropeDim...]
            let rotated = ropePartial(rot, offset: offset)
            return MLX.concatenated([rotated, keep], axis: -1)
        }
        return ropeFull(t, offset: offset)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, T, _) = (x.dim(0), x.dim(1), x.dim(2))
        var q = wq(x).reshaped(B, T, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(B, T, nKVHeads, headDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        k = kNorm(k)
        q = applyPartialRope(q, offset: cache?.offset ?? 0)
        k = applyPartialRope(k, offset: cache?.offset ?? 0)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, T, nHeads * headDim)

        var result = out
        if let g = gProj {
            let gate = sigmoid(g(x))
            let gated = result.reshaped(B, T, nHeads, headDim)
                * gate.expandedDimensions(axis: -1)
            result = gated.reshaped(B, T, nHeads * headDim)
        }
        return wo(result)
    }
}

// MARK: - JANGTQ Dense MLP (layer 0)

internal final class LagunaJANGTQDenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: JANGTQDenseLinear
    @ModuleInfo(key: "up_proj") var up: JANGTQDenseLinear
    @ModuleInfo(key: "down_proj") var down: JANGTQDenseLinear

    init(hidden: Int, intermediate: Int, bits: Int, seed: Int) {
        self._gate.wrappedValue = JANGTQDenseLinear(
            inFeatures: hidden, outFeatures: intermediate,
            bits: bits, seed: seed, bias: false)
        self._up.wrappedValue = JANGTQDenseLinear(
            inFeatures: hidden, outFeatures: intermediate,
            bits: bits, seed: seed, bias: false)
        self._down.wrappedValue = JANGTQDenseLinear(
            inFeatures: intermediate, outFeatures: hidden,
            bits: bits, seed: seed, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - JANGTQ MoE Block

internal final class LagunaJANGTQMoE: Module, UnaryLayer {
    let cfg: LagunaConfiguration

    /// Router stays vanilla (small Linear: hidden_size → num_experts;
    /// not worth JANGTQ-quantizing).
    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    // 2026-05-01: parity with LagunaModel — Module-array fields need
    // @ModuleInfo + var so the parameter loader can write per-expert
    // weights. Without this, weight load fails with 'Unable to set
    // layers.N.mlp.experts on LagunaJANGTQModel.LagunaJANGTQLayer.LagunaJANGTQMoE'.
    @ModuleInfo(key: "experts") var experts: [LagunaJANGTQDenseMLP]
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaJANGTQDenseMLP

    init(_ cfg: LagunaConfiguration, bits: Int, seed: Int) {
        self.cfg = cfg
        self._gate.wrappedValue = Linear(cfg.hiddenSize, cfg.numExperts, bias: false)
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros([cfg.numExperts])
        self._experts.wrappedValue = (0..<cfg.numExperts).map { _ in
            LagunaJANGTQDenseMLP(
                hidden: cfg.hiddenSize, intermediate: cfg.moeIntermediateSize,
                bits: bits, seed: seed)
        }
        self._sharedExpert.wrappedValue = LagunaJANGTQDenseMLP(
            hidden: cfg.hiddenSize, intermediate: cfg.sharedExpertIntermediateSize,
            bits: bits, seed: seed)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped(B * T, H)
        let logits = gate(flat).asType(.float32)
        let scores = sigmoid(logits + eScoreCorrectionBias.asType(.float32))

        let topK = cfg.numExpertsPerTok
        let sortedIdx = MLX.argSort(-scores, axis: 1)
        let topkIdx = sortedIdx[0..., ..<topK]
        var topkW = MLX.takeAlong(scores, topkIdx, axis: 1)
        let normSum = topkW.sum(axis: 1, keepDims: true) + 1e-20
        topkW = (topkW / normSum) * cfg.moeRoutedScalingFactor

        var out = MLXArray.zeros(flat.shape, dtype: topkW.dtype)
        for e in 0..<cfg.numExperts {
            let mask = (topkIdx .== MLXArray(Int32(e)))
            let zeros = MLXArray.zeros(topkW.shape, dtype: topkW.dtype)
            let weighted = MLX.which(mask, topkW, zeros)
            let perRowWeight = weighted.sum(axis: 1, keepDims: false)
            let totalWeight = perRowWeight.sum().item(Float.self)
            if totalWeight == 0 { continue }
            let y = experts[e](flat)
            out = out + y * perRowWeight.expandedDimensions(axis: -1).asType(out.dtype)
        }
        out = out + sharedExpert(flat)
        return out.reshaped(B, T, H).asType(x.dtype)
    }
}

// MARK: - JANGTQ Transformer Block

internal final class LagunaJANGTQLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: LagunaJANGTQAttention
    let mlp: UnaryLayer
    let layerType: String
    let useSliding: Bool

    init(_ cfg: LagunaConfiguration, layerIndex: Int, bits: Int, seed: Int) {
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._attention.wrappedValue = LagunaJANGTQAttention(
            cfg, layerIndex: layerIndex, bits: bits, seed: seed)
        if cfg.mlpLayerTypes[layerIndex] == "dense" {
            self.mlp = LagunaJANGTQDenseMLP(
                hidden: cfg.hiddenSize, intermediate: cfg.intermediateSize,
                bits: bits, seed: seed)
        } else {
            self.mlp = LagunaJANGTQMoE(cfg, bits: bits, seed: seed)
        }
        self.layerType = cfg.layerTypes[layerIndex]
        self.useSliding = layerType == "sliding_attention"
    }

    func callAsFunction(
        _ x: MLXArray,
        fullMask: MLXFast.ScaledDotProductAttentionMaskMode,
        swaMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let mask = useSliding ? swaMask : fullMask
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - JANGTQ Top-Level Model

public class LagunaJANGTQModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [LagunaJANGTQLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    fileprivate let cfg: LagunaConfiguration
    fileprivate let faIdx: Int
    fileprivate let swaIdx: Int?

    public init(_ cfg: LagunaConfiguration, bits: Int = 2, seed: Int = 42) {
        self.cfg = cfg
        self.vocabularySize = cfg.vocabularySize
        self.kvHeads = (0..<cfg.numHiddenLayers).map { _ in cfg.numKeyValueHeads }

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.vocabularySize, dimensions: cfg.hiddenSize)
        self.layers = (0..<cfg.numHiddenLayers).map {
            LagunaJANGTQLayer(cfg, layerIndex: $0, bits: bits, seed: seed)
        }
        self.norm = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        if !cfg.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                cfg.hiddenSize, cfg.vocabularySize, bias: false)
        }
        self.faIdx = cfg.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.swaIdx = cfg.layerTypes.firstIndex(of: "sliding_attention")
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let cacheArr = cache ?? []

        let fullMask = createAttentionMask(h: h, cache: cacheArr.isEmpty ? nil : cacheArr[faIdx])
        let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let swaIdx, !cacheArr.isEmpty {
            swaMask = createAttentionMask(
                h: h, cache: cacheArr[swaIdx], windowSize: cfg.slidingWindow)
        } else {
            swaMask = .none
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, fullMask: fullMask, swaMask: swaMask,
                cache: cacheArr.isEmpty ? nil : cacheArr[i])
        }

        var out = norm(h)
        if cfg.tieWordEmbeddings {
            out = embedTokens.asLinear(out)
        } else if let lmHead {
            out = lmHead(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        cfg.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                return RotatingKVCache(maxSize: cfg.slidingWindow)
            } else if let maxKVSize = parameters?.maxKVSize {
                return RotatingKVCache(maxSize: maxKVSize, keep: 4)
            } else {
                return KVCacheSimple()
            }
        }
    }

    /// Same key remap as `LagunaModel.sanitize` plus standard JANGTQ
    /// audit drops (rotary_emb.inv_freq, .tq_bits scalars, tied lm_head).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("model.") {
                k = String(k.dropFirst("model.".count))
            }
            k = k.replacingOccurrences(
                of: ".mlp.experts.e_score_correction_bias",
                with: ".mlp.e_score_correction_bias"
            )
            if k.contains("self_attn.rotary_emb.inv_freq") { continue }
            if k.hasSuffix(".tq_bits") { continue }
            if cfg.tieWordEmbeddings && k == "lm_head.weight" { continue }
            out[k] = value
        }
        return out
    }
}

extension LagunaJANGTQModel: LoRAModel {
    public var loraLayers: [Module] { [] }
}
