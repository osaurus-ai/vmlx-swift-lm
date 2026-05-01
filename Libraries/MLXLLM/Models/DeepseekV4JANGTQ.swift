// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// JANGTQ (TurboQuant codebook) variant of DeepseekV4.
//
// Swaps the per-expert routed `SwitchGLU` for `TurboQuantSwitchGLU` so
// the codebook Metal kernels run instead of affine `gather_qmm`.
// Everything else — MLA attention (with sinks + inverse RoPE + grouped
// O), mHC HyperConnection, MoE gate (sqrtsoftplus + hash routing),
// HyperHead reduce, norms, embed, lm_head, shared experts — is
// identical to `DeepseekV4Model` and reuses the same forward logic.
//
// Target bundles (from research/DSV4-RUNTIME-ARCHITECTURE.md §5):
//   - DSV4-Flash JANGTQ2 (74 GB, smallest) — routed 2-bit MXTQ, non-
//     routed 8-bit affine g=32
//   - DSV4-Flash JANGTQ4 (173 GB) — routed 4-bit affine g=32, non-
//     routed 8-bit affine g=32
//
// NOTE: `TurboQuantSwitchGLU` fuses gate+up+SwiGLU into a single Metal
// kernel (`fusedGateUpSwiGLU`) that does NOT currently apply the
// `swiglu_limit=10` clamp the Python reference uses on routed experts.
// Shared experts (dense path) still apply the clamp via `DeepseekV4MLP`.
// This is an acceptable approximation for Phase 1b runtime testing;
// the MXTQ codebook's natural boundedness provides regularization.
// Adding clamp support to the fused kernel is tracked for Phase 2 if
// coherence checks fail.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - JANGTQ MoE (routed via TurboQuantSwitchGLU)

final class DeepseekV4MoEJANGTQ: Module, UnaryLayer {
    let config: DeepseekV4Configuration
    let topK: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: TurboQuantSwitchGLU
    var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP
    var currentInputIds: MLXArray? = nil

    init(config: DeepseekV4Configuration, layerIdx: Int, mxtqBits: Int, mxtqSeed: Int) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self._switchMLP.wrappedValue = TurboQuantSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            bits: mxtqBits,
            seed: mxtqSeed)
        self.gate = DeepseekV4MoEGate(config: config, layerIdx: layerIdx)
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: config.swigluLimit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: currentInputIds)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        y = y + sharedExperts(x)
        return y
    }
}

// MARK: - JANGTQ decoder layer (same wrapping as DSV4, JANGTQ MoE swap)

final class DeepseekV4DecoderLayerJANGTQ: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV4Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV4MoEJANGTQ
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    init(config: DeepseekV4Configuration, layerIdx: Int, mxtqBits: Int, mxtqSeed: Int) {
        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DeepseekV4MoEJANGTQ(
            config: config, layerIdx: layerIdx, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
    }

    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray?
    ) -> MLXArray {
        let residualA = h
        let (xA, postA, combA) = attnHC.collapse(h)
        let attnOut = selfAttn(inputLayerNorm(xA), mask: mask, cache: cache)
        let hA = attnHC.expand(blockOut: attnOut, residual: residualA, post: postA, comb: combA)
        let residualF = hA
        let (xF, postF, combF) = ffnHC.collapse(hA)
        mlp.currentInputIds = inputIds
        let ffnOut = mlp(postAttentionLayerNorm(xF))
        mlp.currentInputIds = nil
        return ffnHC.expand(blockOut: ffnOut, residual: residualF, post: postF, comb: combF)
    }
}

// MARK: - JANGTQ inner + outer model

public final class DeepseekV4ModelInnerJANGTQ: Module {
    let config: DeepseekV4Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4DecoderLayerJANGTQ]
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration, mxtqBits: Int, mxtqSeed: Int) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map {
            DeepseekV4DecoderLayerJANGTQ(
                config: config, layerIdx: $0, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
        }
        self._hcHead.wrappedValue = DeepseekV4HyperHead(config: config)
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        h = h.expandedDimensions(axis: -2)
        h = repeated(h, count: config.hcMult, axis: -2)
        let hFlat2 = h.reshaped(h.dim(0), h.dim(1), -1)
        let mask = createAttentionMask(h: hFlat2, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], inputIds: inputs)
        }
        return norm(hcHead.reduce(h))
    }
}

public final class DeepseekV4JANGTQModel:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel
{
    public var kvHeads: [Int]
    let config: DeepseekV4Configuration
    public var model: DeepseekV4ModelInnerJANGTQ
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV4Configuration, mxtqBits: Int = 2, mxtqSeed: Int = 42) {
        self.config = config
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = DeepseekV4ModelInnerJANGTQ(
            config: config, mxtqBits: mxtqBits, mxtqSeed: mxtqSeed)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Mirrors `DeepseekV4Model.newCache`. **DEFAULT ON (2026-05-01)**:
    /// `DeepseekV4Cache` on compress_ratio>0 layers so the
    /// compressor/indexer state persists across decode steps. MMLU
    /// 200q at LC=1 is 81.5 % vs 74.5 % at LC=0 (apples-to-apples,
    /// same bundle/extractor; jang research commit 53f12a9). Set
    /// `DSV4_LONG_CTX=0` to opt back into the legacy
    /// RotatingKVCache(maxSize=sliding_window) path explicitly.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let env = ProcessInfo.processInfo.environment
        // Default ON; explicit `DSV4_LONG_CTX=0` opts out.
        let longCtxEnabled = (env["DSV4_LONG_CTX"] ?? "1") != "0"

        // Mirror DeepseekV4Model.newCache. Three modes:
        //   - "sliding" (default): RotatingKVCache, 128-token window.
        //   - "full":    KVCacheSimple, all tokens kept.
        //   - "tq":      KVCacheSimple now; BatchQuantize.maybeCompress
        //                promotes to TurboQuantKVCache once offset > min.
        //                Caller must also set kvMode=.turboQuant in
        //                GenerateParameters for the promotion to fire.
        let envMode = env["DSV4_KV_MODE"]?.lowercased()
        let callerWantsTQ: Bool = {
            guard let p = parameters else { return false }
            if case .turboQuant = p.kvMode { return true }
            return false
        }()
        let mode: String = envMode ?? (callerWantsTQ ? "tq" : "sliding")

        return (0..<config.numHiddenLayers).map { layerIdx in
            switch mode {
            case "full", "tq":
                return KVCacheSimple()
            default:
                if longCtxEnabled {
                    let cr =
                        config.compressRatios.count > layerIdx
                        ? config.compressRatios[layerIdx] : 0
                    if cr > 0 {
                        return DeepseekV4Cache(slidingWindow: config.slidingWindow)
                    }
                }
                return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Reuse DeepseekV4Model's sanitize — the weight naming contract
    /// is identical for JANGTQ bundles. The TurboQuantSwitchLinear
    /// infrastructure handles the MXTQ codebook indirection at load
    /// time via its own key renaming in `TurboQuantSwitchLinear.keyMap`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Delegate to the affine variant's sanitize — same remap rules.
        let stub = DeepseekV4Model(config)
        return stub.sanitize(weights: weights)
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
