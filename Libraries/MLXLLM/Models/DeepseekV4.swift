// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// DeepSeek-V4 (DSV4-Flash / DSV4-Pro) — full model forward.
//
// Reference:
//   - jang/research/DSV4-RUNTIME-ARCHITECTURE.md §1-14
//   - jang/research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md §1 (all 13 bug fixes)
//   - jang-tools/jang_tools/dsv4_prune/mlx_model.py (1128 LOC Python ref)
//
// Architecture vs DSV3 (all new, all non-negotiable):
//   • mHC residual stream (hc_mult=4 parallel copies, collapse/expand
//     per block using a Sinkhorn-normalized mixing matrix)
//   • MLA with head_dim=512, num_kv_heads=1 (single latent KV head
//     broadcast to all 64 Q heads via GQA), RoPE only on last
//     qk_rope_head_dim=64 dims
//   • Learned per-head `attn_sink` logit prepended pre-softmax
//   • Inverse RoPE on attention OUTPUT (strips positional info before
//     residual add-back)
//   • Grouped low-rank O projection: `bsgd,grd→bsgr` einsum with
//     o_groups=8, o_lora_rank=1024, then wo_b to hidden_size
//   • MoE routing via sqrtsoftplus instead of softmax
//   • Hash routing for first num_hash_layers=3 layers (tid2eid lookup)
//   • DSV4 SwiGLU with swiglu_limit=10.0 (clamp gate + up)
//   • Per-layer rope_theta: 10000 for compress_ratio=0 (no YaRN),
//     160000 for compress_ratio>0 (with YaRN)
//   • HyperHead reduce at the top of the model (mHC copies → hidden)
//
// Compressor + Indexer (for long-context attention with compress_ratio>0)
// are NOT wired in Phase 1b. Short prompts (L < sliding_window=128) run
// the bypass path correctly; long prompts degrade to sliding-window-only
// attention until Phase 2. Their weights are dropped in sanitize().

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - RoPE

/// DSV4 RoPE: YaRN scaling with `high = min(..., dim-1)` clamp (bug #10).
/// Per-layer theta — the layer chooses between `rope_theta=10000` (no
/// YaRN when compress_ratio=0) and `compress_rope_theta=160000` (with
/// YaRN scaling when compress_ratio>0).
class DeepseekV4RoPE: Module {
    let dim: Int
    let base: Float
    let factor: Float
    let origMaxPos: Int
    let betaFast: Float
    let betaSlow: Float
    // Precomputed half-dim inv-freq table.
    let invFreq: MLXArray

    init(
        dim: Int,
        base: Float,
        factor: Float = 1.0,
        origMaxPos: Int = 65536,
        betaFast: Float = 32,
        betaSlow: Float = 1
    ) {
        self.dim = dim
        self.base = base
        self.factor = factor
        self.origMaxPos = origMaxPos
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.invFreq = DeepseekV4Math.yarnInvFreq(
            dim: dim, base: base, maxPos: 0,
            origMaxPos: origMaxPos, factor: factor,
            betaFast: betaFast, betaSlow: betaSlow)
    }

    /// Compute cos/sin tables for positions `[offset, offset+L)`.
    /// Returned shape: `(L, dim/2)`.
    func cosSin(offset: Int, length: Int) -> (cos: MLXArray, sin: MLXArray) {
        let positions = MLXArray(Int32(offset)..<Int32(offset + length)).asType(.float32)
        // positions: (L,), invFreq: (dim/2,) → angles: (L, dim/2)
        let angles = positions.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
        return (cos: cos(angles), sin: sin(angles))
    }
}

// MARK: - Attention (MLA with sinks + inverse RoPE + grouped O)

class DeepseekV4Attention: Module {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let qLoraRank: Int
    let oGroups: Int
    let oLoraRank: Int
    /// Per-layer compress_ratio ∈ {0, 4, 128}. 0 = no compressor, plain
    /// sliding-window attention. 4 or 128 = Compressor (+ Indexer at 4)
    /// augments local KV with pooled global context.
    let compressRatio: Int
    let scale: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    // wo_a operates on PER-GROUP features (numHeads*headDim // oGroups),
    // mapping them to oGroups*oLoraRank via einsum bsgd,grd→bsgr.
    // Python: Linear(n_heads*head_dim // o_groups, o_groups*o_lora_rank).
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear
    /// q_norm is on `q_lora_rank` (1024), NOT head_dim. Applied BEFORE wq_b.
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    /// Shape (num_heads,) — one learned sink logit per head.
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray

    let rope: DeepseekV4RoPE

    // Compressor + Indexer (instantiated only when compressRatio > 0).
    // Swift can't have conditionally-present @ModuleInfo properties
    // cleanly, so we instantiate always and null the pooled path inside
    // forward when compressRatio == 0.
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.ropeDim = config.qkRopeHeadDim
        self.qLoraRank = config.qLoraRank
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = 1.0 / sqrt(Float(headDim))

        // Resolve per-layer compress_ratio. If config.compressRatios is
        // populated use it directly; otherwise fall back to the default
        // DSV4-Flash pattern (layer 0 and last → 0; middle: odd → 4,
        // even → 128 per layer index after accounting for layer 0).
        if !config.compressRatios.isEmpty && layerIdx < config.compressRatios.count {
            self.compressRatio = config.compressRatios[layerIdx]
        } else {
            let n = config.numHiddenLayers
            if layerIdx == 0 || layerIdx == n - 1 {
                self.compressRatio = 0
            } else {
                let i = layerIdx - 1
                self.compressRatio = (i % 2 == 1) ? 4 : 128
            }
        }

        self._wqA.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: false)
        self._wqB.wrappedValue = Linear(qLoraRank, numHeads * headDim, bias: false)
        self._wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        // wo_a: per-group features (n_heads*head_dim // o_groups) →
        // o_groups * o_lora_rank. For DSV4-Flash: 4096 → 8192.
        self._woA.wrappedValue = Linear(
            numHeads * headDim / oGroups, oGroups * oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(
            oGroups * oLoraRank, config.hiddenSize, bias: false)
        // q_norm operates on q_lora_rank (1024), not head_dim.
        self._qNorm.wrappedValue = RMSNorm(
            dimensions: qLoraRank, eps: config.rmsNormEps)
        self._kvNorm.wrappedValue = RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._attnSink.wrappedValue = zeros([numHeads])

        // RoPE: compressRatio>0 → compress_rope_theta (160000) + YaRN.
        // compressRatio==0 → rope_theta (10000), NO YaRN.
        let theta =
            compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta
        let factor: Float =
            compressRatio > 0
            ? Float((config.ropeScaling?["factor"]?.asFloat()) ?? 16.0)
            : 1.0
        let origMax =
            Int(
                (config.ropeScaling?["original_max_position_embeddings"]?.asInt()) ?? 65536)
        self.rope = DeepseekV4RoPE(
            dim: ropeDim, base: theta, factor: factor,
            origMaxPos: origMax, betaFast: 32, betaSlow: 1)

        // Compressor + Indexer are attached ONLY on layers with a
        // non-zero compress_ratio — matches bundle weight keys.
        if compressRatio > 0 {
            self._compressor.wrappedValue = DeepseekV4Compressor(
                config: config, compressRatio: compressRatio, headDim: headDim)
            if compressRatio == 4 {
                self._indexer.wrappedValue = DeepseekV4Indexer(
                    config: config, compressRatio: compressRatio)
            }
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let offset = cache?.offset ?? 0

        // --- Q projection ---
        // wq_a(x): (B, L, qLoraRank) → q_norm on qLoraRank → wq_b:
        // (B, L, numHeads*headDim). Keep the post-qnorm residual — the
        // Indexer uses it as its own Q source.
        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual)
        q = q.reshaped(B, L, numHeads, headDim)
        // Per-head unit-weight RMSNorm via the fused `MLXFast.rmsNorm`
        // kernel (1 dispatch vs 3 ops for the manual rsqrt path).
        // Mirrors Python `mx.fast.rms_norm(q, weight=_get_q_norm_ones(...),
        // eps=...)` — reuses a `(headDim, dtype)`-cached ones tensor so
        // the 64 heads × 43 layers don't reallocate per token.
        q = MLXFast.rmsNorm(
            q,
            weight: DeepseekV4Math.qNormOnes(headDim: headDim, dtype: q.dtype),
            eps: config.rmsNormEps)
        q = q.transposed(0, 2, 1, 3)

        // --- KV projection (single latent head) ---
        var kv = kvNorm(wkv(x))
        kv = kv.reshaped(B, L, 1, headDim).transposed(0, 2, 1, 3)

        // --- Partial RoPE on last ropeDim dims of Q and K ---
        let (cosT, sinT) = rope.cosSin(offset: offset, length: L)
        let cosQ = cosT.expandedDimensions(axes: [0, 1])
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(q, cos: cosQ, sin: sinQ, ropeDim: ropeDim)
        kv = DeepseekV4Math.applyPartialRoPE(kv, cos: cosQ, sin: sinQ, ropeDim: ropeDim)

        // --- Cache update (sliding-window local) ---
        var keys = kv
        if let cache = cache {
            (keys, _) = cache.update(keys: kv, values: kv)
        }
        var fullKV = keys
        let windowLen = fullKV.dim(2)

        // --- Compressor + Indexer global context (compressRatio > 0 layers) ---
        //
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // Two paths now distinguished by query length:
        //
        //   * decode (L == 1): build `(B, 1, k, D)` of selected pool
        //     rows for the single query (or the whole pool if no topk
        //     gating fires) and concat onto `full_kv`. No mask needed —
        //     the only query is causally OK against every selected row
        //     because the indexer enforces `(k_idx + 1) * ratio <= q + 1`
        //     in scoring, and the compressor only emits pool rows whose
        //     summarized window has fully ended.
        //
        //   * prefill (L > 1): keep the pool flat at `(B, 1, P, D)` and
        //     build a 2-segment mask `[window_visibility | comp_visibility]`
        //     so each query sees only the local window keys it should
        //     AND only the pool rows whose summarized window ended at
        //     or before that query's position. ANDed with the indexer's
        //     selection mask on `cr=4` layers. The previous implementation
        //     padded the mask with all-ones, allowing query `q` to see
        //     pool rows summarizing tokens with positions > q, AND
        //     gathered `(B, 1, L*k, D)` — leaking query `i`'s selected
        //     rows into query `j`'s attention.
        var dsv4PrefillMask: MLXArray? = nil
        var poolEntries: Int = 0
        if compressRatio > 0 {
            let v4Cache = cache as? DeepseekV4Cache
            if v4Cache != nil || L >= compressRatio {
                if let comp = compressor {
                    var pooled = comp(x, rope: rope, v4Cache: v4Cache, startPos: offset)
                    // pooled shape: (B, W, headDim) where W = pooled count.
                    let W = pooled.dim(1)
                    if W > 0 {
                        var topK: MLXArray? = nil
                        if compressRatio == 4, let idx = indexer {
                            topK = idx(
                                x, qResidual: qResidual, rope: rope,
                                positionRope: rope, v4Cache: v4Cache, startPos: offset)
                        }

                        if L == 1 {
                            // DECODE FAST PATH — gather only the topk
                            // rows for the single query (or all rows
                            // when topk == nil / W <= topK), shape
                            // `(B, 1, k, D)`.
                            if let tk = topK {
                                let k = tk.dim(-1)
                                // pooled: (B, W, D) → (B, 1, 1, W, D)
                                let expanded = pooled.expandedDimensions(axes: [1, 2])
                                let pooledBroad = broadcast(
                                    expanded, to: [B, 1, L, W, headDim])
                                // tk: (B, L=1, k) → (B, 1, L, k, 1)
                                let idxExp = tk.expandedDimensions(axes: [1, 4])
                                let idxBroad = broadcast(
                                    idxExp, to: [B, 1, L, k, headDim])
                                let gathered = takeAlong(
                                    pooledBroad, idxBroad, axis: 3)
                                // (B, 1, k, D)
                                pooled = gathered.reshaped(B, 1, k, headDim)
                            } else {
                                pooled = pooled.expandedDimensions(axis: 1)
                            }
                        } else {
                            // PREFILL PATH — flat pool, mask carries
                            // visibility.
                            pooled = pooled.expandedDimensions(axis: 1)
                            // local sliding-window visibility (B,1,L,windowLen)
                            var localMask = DeepseekV4Math.buildWindowMask(
                                batch: B, queryLen: L, offset: offset,
                                window: config.slidingWindow,
                                windowLen: windowLen)
                            // compressed-pool causal visibility (B,1,L,W)
                            var compMask = DeepseekV4Math.compressedVisibility(
                                batch: B, queryLen: L, offset: offset,
                                compressedLen: W, ratio: compressRatio)
                            if let tk = topK {
                                let sel = DeepseekV4Math.indexerSelectionMask(
                                    topk: tk, compressedLen: W)
                                compMask = MLX.logicalAnd(compMask, sel)
                            }
                            // Pre-broadcast both halves to the same query
                            // dim (already done by helpers); concat along
                            // last axis.
                            _ = localMask
                            dsv4PrefillMask = concatenated(
                                [localMask, compMask], axis: -1)
                        }

                        if pooled.dim(2) > 0 {
                            poolEntries = pooled.dim(2)
                            fullKV = concatenated([fullKV, pooled], axis: 2)
                        }
                    }
                }
            }
        }

        // --- Resolve final attention mask ---
        // Three cases:
        //   (a) DSV4-built prefill mask present → use it directly.
        //   (b) Caller-provided array mask → trim/pad to `fullKV.dim(2)`
        //       (legacy code path, also triggered for `cr == 0` SWA-only
        //        layers that bypass DSV4 mask construction).
        //   (c) Bool-causal sentinel from `createAttentionMask` → leave it
        //       alone; SDPA will compute the causal mask itself.
        var adjustedMask = mask
        if let dsv4 = dsv4PrefillMask {
            adjustedMask = .array(dsv4)
        } else if case .array(let maskArr) = mask,
            poolEntries > 0
        {
            // Decode path: extend the mask with all-ones for the pool
            // entries (every selected row is causally valid for the
            // single query — see above).
            let padShape =
                Array(maskArr.shape.dropLast()) + [fullKV.dim(2) - maskArr.dim(-1)]
            let pad = MLXArray.ones(padShape, dtype: maskArr.dtype)
            adjustedMask = .array(concatenated([maskArr, pad], axis: -1))
        } else if case .array(let maskArr) = mask,
            fullKV.dim(2) != maskArr.dim(-1)
        {
            // Defensive: align array mask to actual key length.
            if maskArr.dim(-1) > fullKV.dim(2) {
                let trimmed = maskArr[.ellipsis, (-fullKV.dim(2))...]
                adjustedMask = .array(trimmed)
            } else {
                let padShape =
                    Array(maskArr.shape.dropLast()) + [fullKV.dim(2) - maskArr.dim(-1)]
                let pad = MLXArray.zeros(padShape, dtype: maskArr.dtype)
                adjustedMask = .array(concatenated([maskArr, pad], axis: -1))
            }
        }

        // --- SDPA with attention sinks (fp32 accum for head_dim=512) ---
        var output = MLXFast.scaledDotProductAttention(
            queries: q, keys: fullKV, values: fullKV,
            scale: scale, mask: adjustedMask,
            sinks: config.useAttnSink ? attnSink.asType(q.dtype) : nil)
        // output shape: (B, numHeads, L, headDim)

        // --- Inverse RoPE on the output's head-major layout ---
        let cosI = cosT.expandedDimensions(axes: [0, 1])
        let sinI = sinT.expandedDimensions(axes: [0, 1])
        output = DeepseekV4Math.applyPartialRoPE(
            output, cos: cosI, sin: sinI, ropeDim: ropeDim, inverse: true)
        output = output.transposed(0, 2, 1, 3)  // (B, L, numHeads, headDim)
            .reshaped(B, L, numHeads * headDim)

        // --- Grouped low-rank O projection ---
        // Reshape to (B, L, oGroups, groupFeat) then per-group matmul
        // through `wo_a`, producing (B, L, oGroups, oLoraRank) → concat
        // groups → wo_b. Mirrors Python `_grouped_output_projection`
        // (mlx_model.py:700) — separate dispatch for QuantizedLinear vs
        // plain Linear because the quantized packed weight cannot be
        // reshaped element-wise.
        let groupFeat = (numHeads * headDim) / oGroups
        let oReshape = output.reshaped(B, L, oGroups, groupFeat)
        let oA: MLXArray
        if let qwo = woA as? QuantizedLinear {
            // Python ref:
            //   out = out.transpose(2, 0, 1, 3)               # (oGroups, B, L, gf)
            //   weight  = wo_a.weight.reshape(oGroups, oLoraRank, -1)[:, None]
            //   scales  = wo_a.scales.reshape(oGroups, oLoraRank, -1)[:, None]
            //   biases  = wo_a.biases.reshape(oGroups, oLoraRank, -1)[:, None]
            //   out = mx.quantized_matmul(out, weight, scales, biases,
            //                              transpose=True, group_size, bits, mode)
            //   out = out.transpose(1, 2, 0, 3).reshape(B, L, oGroups*oLoraRank)
            let xT = oReshape.transposed(2, 0, 1, 3)
            // Each per-group weight slab keeps its original packed-in
            // dim (last axis) — `-1` lets MLX work out 1024 → 128 for
            // 8-bit g=32 packing.
            let wPacked = qwo.weight.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let wScales = qwo.scales.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let wBiases = qwo.biases?.reshaped(oGroups, oLoraRank, -1)
                .expandedDimensions(axis: 1)
            let outQ = MLX.quantizedMatmul(
                xT, wPacked, scales: wScales, biases: wBiases,
                transpose: true, groupSize: qwo.groupSize, bits: qwo.bits)
            oA = outQ.transposed(1, 2, 0, 3).reshaped(
                B, L, oGroups * oLoraRank)
        } else {
            // Non-quantized path: keep the einsum.
            // wo_a.weight has shape (oGroups*oLoraRank, groupFeat) per
            // MLX Linear convention (out, in).
            let woaW = woA.weight.reshaped(oGroups, oLoraRank, groupFeat)
            oA = einsum("bsgd,grd->bsgr", oReshape, woaW)
                .reshaped(B, L, oGroups * oLoraRank)
        }
        return woB(oA)
    }
}

// MARK: - MoE gate (sqrtsoftplus + hash routing)

class DeepseekV4MoEGate: Module {
    let config: DeepseekV4Configuration
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let isHashLayer: Bool
    /// Gate projection weight: (nRoutedExperts, hiddenSize). Stored as a
    /// raw parameter (loaded via sanitize) rather than a Linear to allow
    /// the matmul to run in fp32 per the authoritative reference.
    @ParameterInfo(key: "weight") var weight: MLXArray
    /// Optional noaux bias added to scores for selection only. When
    /// absent the bias term is skipped.
    @ParameterInfo(key: "bias") var bias: MLXArray
    /// Hash routing lookup table (token_id → expert_id), shape (vocab,).
    /// Only populated for hash layers.
    @ParameterInfo(key: "tid2eid") var tid2eid: MLXArray

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.isHashLayer = config.isHashLayer(layerIdx)
        self._weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        self._bias.wrappedValue = zeros([nRoutedExperts])
        // Hash routing table: bundle ships (vocab, topK) — already
        // pre-stamped with which `topK` experts each token id should
        // route to, so the gate just gathers without computing scores.
        // Non-hash layers don't have this tensor, so we still allocate
        // a placeholder slot.
        self._tid2eid.wrappedValue =
            zeros([isHashLayer ? config.vocabSize : 1, isHashLayer ? topK : 1])
    }

    /// Returns (indices, weights) where indices has shape (B, L, topK)
    /// and weights has shape (B, L, topK).
    ///
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    /// Hash layers now match the Python `Gate.__call__` reference
    /// (`jang_tools.dsv4.mlx_model.Gate.__call__`) — they gather the
    /// PER-TOKEN gate scores at the hash-selected expert ids instead
    /// of returning a synthetic uniform `routedScalingFactor / topK`.
    /// Without this fix every hash-routed layer collapsed all six
    /// selected experts to the same weight, throwing away the
    /// information the gate matmul + sqrtsoftplus produced and
    /// flattening the routing geometry the model was trained with.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> (MLXArray, MLXArray) {
        // Compute the gate logits in fp32 even on hash layers — the
        // hash path needs them to score the (deterministic) selected
        // experts.
        let xF32 = x.asType(.float32)
        let wF32 = weight.asType(.float32)
        let logits = xF32.matmul(wF32.transposed())
        let scores = DeepseekV4Math.sqrtSoftplus(logits)

        if isHashLayer, let ids = inputIds {
            // Hash routing: tid2eid is (vocab, topK) — pre-stamped at
            // convert time with which topK experts each token id
            // routes to. `tid2eid[ids]` for ids shape (B, L) returns
            // (B, L, topK) directly via fancy index.
            let indices = tid2eid[ids].asType(.int32)  // (B, L, topK)
            // Gate the experts using their actual sqrtsoftplus score
            // (mirror Python `mx.take_along_axis(scores, inds, axis=-1)`).
            var weights = takeAlong(scores, indices, axis: -1)
            if normTopkProb {
                let denom = weights.sum(axis: -1, keepDims: true) + 1e-20
                weights = weights / denom
            }
            weights = weights * routedScalingFactor
            return (indices.asType(.uint32), weights)
        }

        // Non-hash: standard sqrtsoftplus + noaux-biased top-k.
        let (indices, weights) = DeepseekV4Math.sqrtSoftplusSelect(
            scores: scores,
            noauxBias: bias,  // zeros-initialized — effectively no bias unless loaded
            k: topK,
            normalize: normTopkProb,
            scalingFactor: routedScalingFactor
        )
        return (indices.asType(.uint32), weights)
    }
}

// MARK: - MoE (SwitchGLU routed + shared expert)

class DeepseekV4MoE: Module, UnaryLayer {
    let config: DeepseekV4Configuration
    let layerIdx: Int
    let topK: Int
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP
    /// Hack to thread the input token ids down into the gate when this
    /// layer is hash-routed. Set by the outer model before each layer
    /// call when hash routing applies.
    var currentInputIds: MLXArray? = nil

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.topK = config.numExpertsPerTok
        let limit = config.swigluLimit
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // Symmetric DSV4 limited-SwiGLU — `silu(min(gate, limit)) *
        // clip(up, -limit, +limit)`. We pass a 2-arg `glue` closure to
        // SwitchGLU (instead of a 1-arg `activation`) so BOTH gate and
        // up get clamped before the multiply. The Python reference
        // (`jang_tools.dsv4.mlx_model._dsv4_swiglu`) also runs the
        // multiply in fp32 before casting back to gate.dtype to avoid
        // per-layer precision drift across the 43 MoE layers; we mirror
        // that here.
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: MLXNN.silu,
            glue: { gate, up in
                DeepseekV4Math.dsv4SwiGLU(gate: gate, up: up, limit: limit)
            })
        self.gate = DeepseekV4MoEGate(config: config, layerIdx: layerIdx)
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: limit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: currentInputIds)
        JangPressCanonicalExpertAdvisor.shared.observe(layer: layerIdx, indices: indices)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        y = y + sharedExperts(x)
        return y
    }
}

// MARK: - Dense MLP (shared expert) with DSV4 SwiGLU clamp

class DeepseekV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.swigluLimit = swigluLimit
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let g = gateProj(x)
        let u = upProj(x)
        return downProj(DeepseekV4Math.dsv4SwiGLU(gate: g, up: u, limit: swigluLimit))
    }
}

// MARK: - mHC Hyper-Connection (per-block collapse + expand)

class DeepseekV4HyperConnection: Module {
    let hcMult: Int
    let hcIters: Int
    let hcEps: Float
    let hiddenSize: Int
    let mixHc: Int  // (2 + hcMult) * hcMult — bundle stores params at this width
    /// `hc_{attn,ffn}_fn`: shape `((2+hc)*hc, hc*hidden)`. Bundle stores
    /// `(24, 16384)` for hc=4, hidden=4096.
    @ParameterInfo(key: "fn") var fn: MLXArray
    /// `hc_{attn,ffn}_scale`: shape `(3,)` per-field scalar.
    @ParameterInfo(key: "scale") var scale: MLXArray
    /// `hc_{attn,ffn}_base`: shape `((2+hc)*hc,)` per-field bias.
    @ParameterInfo(key: "base") var base: MLXArray
    /// Constant ones-vector reused as the RMSNorm weight inside the
    /// per-block collapse (Python sets up `_hc_rms_ones = mx.ones(...)`).
    let hcRMSOnes: MLXArray

    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hcIters = config.hcSinkhornIters
        self.hcEps = config.hcEps
        self.hiddenSize = config.hiddenSize
        self.mixHc = (2 + config.hcMult) * config.hcMult
        self._fn.wrappedValue = zeros([mixHc, hcMult * hiddenSize])
        self._scale.wrappedValue = zeros([3])
        self._base.wrappedValue = zeros([mixHc])
        // Match Python `_hc_rms_ones = mx.ones(hc_dim, dtype=mx.float16)`.
        // This is a constant — not a learned parameter — so we keep it
        // as a plain stored property (not @ParameterInfo).
        self.hcRMSOnes = MLXArray.ones([config.hcMult * config.hiddenSize])
    }

    /// Collapse: `h` shape (B, L, hcMult, hiddenSize) → collapsed x
    /// (B, L, hiddenSize) plus `post` (B, L, hcMult) and `comb`
    /// (B, L, hcMult, hcMult) for the expand step.
    ///
    /// Mirrors Python `DeepseekV4DecoderLayer._hc_pre`:
    ///   x_flat   = flatten(h, axis=2)              # (B, L, hc*hidden)
    ///   x_normed = rms_norm(x_flat, ones, eps)
    ///   mixes    = x_normed @ fn.T                 # (B, L, mix_hc)
    ///   pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
    ///   y = sum(pre[..., None] * x_flat.reshape(B,L,hc,D), axis=2)
    func collapse(_ h: MLXArray) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        let dtype = h.dtype
        let B = h.dim(0)
        let L = h.dim(1)

        // Flatten the (hcMult, hidden) tail into one axis.
        let xFlat = h.reshaped(B, L, hcMult * hiddenSize)
        // Variance-only RMS norm with weight = ones.
        //
        // Force fp32 internals: the reduction `mean(square(x))` runs over
        // `hcMult * hiddenSize` (≈16384 for DSV4-Flash) elements per row
        // and bf16 rounding compounds aggressively across that axis. On
        // M3 Ultra this saturates and the rsqrt produces garbage logits
        // ("17 plus plus plus" failure mode). M4 happens to keep fp32 in
        // SIMD lanes for this reduction so MacBook tests pass — but the
        // bug is real on Mac Studio. Mirrors the Python jang_tools fix
        // at jang/research/JANGTQ-PROGRESS-LOG-2026-04-25.md §A.1 #50.
        let xNormed = MLXFast.rmsNorm(
            xFlat.asType(.float32),
            weight: hcRMSOnes.asType(.float32),
            eps: hcEps
        ).asType(xFlat.dtype)
        // mixes = x_normed @ fn.T  → (B, L, mix_hc)
        let mixes = xNormed.asType(.float32)
            .matmul(fn.asType(.float32).transposed())

        let (pre, post, comb) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: hcIters, eps: hcEps)

        // y = sum(pre[..., None] * x_flat.reshape(B, L, hc, D), axis=2)
        let preCast = pre.asType(dtype)
        let xReshape = xFlat.reshaped(B, L, hcMult, hiddenSize)
        let y = (preCast.expandedDimensions(axis: -1) * xReshape).sum(axis: -2)
        return (x: y, post: post, comb: comb)
    }

    /// Expand: given attn/ffn output `blockOut` (B, L, hiddenSize),
    /// residual (B, L, hcMult, hiddenSize), and the (post, comb) from
    /// the matching collapse, return new h (B, L, hcMult, hiddenSize).
    ///
    /// Mirrors Python `_hc_post`:
    ///   y = post[..., None] * x[..., None, :] + matmul(comb, residual)
    func expand(
        blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        let dtype = blockOut.dtype
        let postCast = post.asType(dtype)
        let combCast = comb.asType(dtype)
        // matmul(comb, residual) — comb is (B,L,hc,hc), residual is
        // (B,L,hc,D); broadcast over leading dims. MLX matmul handles
        // this directly.
        let combResid = combCast.matmul(residual)
        // post: (B,L,hc) → (B,L,hc,1); blockOut: (B,L,D) → (B,L,1,D).
        let y = postCast.expandedDimensions(axis: -1)
            * blockOut.expandedDimensions(axis: -2) + combResid
        return y
    }
}

// MARK: - HyperHead (top-of-model mHC reduce)

class DeepseekV4HyperHead: Module {
    let hcMult: Int
    let hiddenSize: Int
    let hcEps: Float
    /// Bundle stores `hc_head_fn` at `(hcMult, hcMult*hiddenSize)`.
    @ParameterInfo(key: "hc_head_fn") var fn: MLXArray
    @ParameterInfo(key: "hc_head_base") var base: MLXArray
    @ParameterInfo(key: "hc_head_scale") var scale: MLXArray
    /// Constant ones-vector for the RMS norm in `_hc_head_reduce`.
    let hcHeadRMSOnes: MLXArray

    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hiddenSize = config.hiddenSize
        self.hcEps = config.rmsNormEps
        self._fn.wrappedValue = zeros([hcMult, hcMult * hiddenSize])
        self._base.wrappedValue = zeros([hcMult])
        self._scale.wrappedValue = zeros([1])
        self.hcHeadRMSOnes = MLXArray.ones([config.hcMult * config.hiddenSize])
    }

    /// Reduce (B, L, hcMult, hiddenSize) → (B, L, hiddenSize). Mirrors
    /// Python `_hc_head_reduce`:
    ///   x_flat   = flatten(x, axis=2)            # (B, L, hc*hidden)
    ///   x_normed = rms_norm(x_flat, ones, eps)
    ///   mixes    = x_normed @ hc_head_fn.T       # (B, L, hc)
    ///   pre      = sigmoid(mixes * scale + base) + hc_eps
    ///   y        = sum(pre[..., None] * x_flat.reshape(B,L,hc,D), axis=2)
    /// NO sum-to-1 normalization — match the Python reference exactly.
    func reduce(_ h: MLXArray) -> MLXArray {
        let dtype = h.dtype
        let B = h.dim(0)
        let L = h.dim(1)
        let xFlat = h.reshaped(B, L, hcMult * hiddenSize)
        // Same dtype rule as `_hc_pre`: this RMS reduction spans
        // hcMult*hiddenSize (≈16K for DSV4-Flash). Apple GPUs differ in
        // implicit bf16 accumulation behavior, so keep the reduction and
        // the tiny gate projection in fp32, then cast the final mixed
        // residual back to the model dtype. This mirrors the jang-tools
        // HyperHead fix documented in DSV4-HC-PRE-FP32-CAST-FIX.
        let xNormed = MLXFast.rmsNorm(
            xFlat.asType(.float32),
            weight: hcHeadRMSOnes.asType(.float32),
            eps: hcEps)
        let mixes = xNormed.matmul(fn.asType(.float32).transposed())  // (B, L, hcMult)
        let pre = sigmoid(mixes * scale.asType(.float32) + base.asType(.float32))
            + MLXArray(hcEps)
        let xReshape = xFlat.reshaped(B, L, hcMult, hiddenSize)
        return (pre.asType(dtype).expandedDimensions(axis: -1) * xReshape).sum(axis: -2)
            .asType(dtype)
    }
}

// MARK: - Decoder layer (mHC wrap over attn + MoE)

class DeepseekV4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV4Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV4MoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    let layerIdx: Int

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DeepseekV4MoE(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(config: config)
    }

    /// Forward. `h` shape: (B, L, hcMult, hiddenSize).
    func callAsFunction(
        _ h: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray?
    ) -> MLXArray {
        // ---- Attention HC ----
        let residualA = h
        let (xA, postA, combA) = attnHC.collapse(h)
        let normedA = inputLayerNorm(xA)
        let attnOut = selfAttn(normedA, mask: mask, cache: cache)
        let hA = attnHC.expand(
            blockOut: attnOut, residual: residualA, post: postA, comb: combA)

        // ---- FFN HC ----
        let residualF = hA
        let (xF, postF, combF) = ffnHC.collapse(hA)
        let normedF = postAttentionLayerNorm(xF)
        mlp.currentInputIds = inputIds
        let ffnOut = mlp(normedF)
        mlp.currentInputIds = nil
        let hF = ffnHC.expand(
            blockOut: ffnOut, residual: residualF, post: postF, comb: combF)
        return hF
    }
}

// MARK: - Inner model

public class DeepseekV4ModelInner: Module {
    let config: DeepseekV4Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4DecoderLayer]
    @ModuleInfo(key: "hc_head") var hcHead: DeepseekV4HyperHead
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0..<config.numHiddenLayers).map {
            DeepseekV4DecoderLayer(config: config, layerIdx: $0)
        }
        self._hcHead.wrappedValue = DeepseekV4HyperHead(config: config)
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // embed: (B, L) → (B, L, hiddenSize)
        var h = embedTokens(inputs)
        // Tile to mHC copies: (B, L, hiddenSize) → (B, L, hcMult, hiddenSize).
        // Python tiles via broadcast; Swift uses `repeated` along axis -2.
        h = h.expandedDimensions(axis: -2)  // (B, L, 1, H)
        h = repeated(h, count: config.hcMult, axis: -2)  // (B, L, hcMult, H)

        let firstCache = cache?.first
        let hFlat2 = h.reshaped(h.dim(0), h.dim(1), -1)  // for createAttentionMask
        let mask = createAttentionMask(h: hFlat2, cache: firstCache)

        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                mask: mask,
                cache: cache?[i],
                inputIds: inputs)
        }

        // HyperHead reduce: (B, L, hcMult, H) → (B, L, H)
        var out = hcHead.reduce(h)
        out = norm(out)
        return out
    }
}

// MARK: - Outer model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]
    var config: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: DeepseekV4Configuration) {
        self.config = config
        // Single latent KV head per layer — report kvHeads as [1]*L so
        // the cache allocator sizes per-layer caches correctly.
        self.kvHeads = Array(repeating: 1, count: config.numHiddenLayers)
        self.model = DeepseekV4ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabSize, bias: false)
    }

    /// Build per-layer caches.
    ///
    /// 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass — pure long-context):
    /// DSV4-Flash IS a hybrid SWA+CSA+HSA architecture by definition.
    /// The previous "long-ctx off" fallback (a plain `RotatingKVCache`
    /// for every layer) was producing measurably degraded output on
    /// any chat that exceeded `sliding_window=128` tokens, because the
    /// `cr>0` layers lost their compressed/indexed global context after
    /// the local window rotated. The toggle is removed; every layer
    /// now allocates the canonical cache for its `compress_ratio`:
    ///   - `cr == 0` (layers 0 and n-1) → `RotatingKVCache(window=128)`
    ///   - `cr > 0`  (every other layer) → `DeepseekV4Cache(window=128, cr=cr)`
    ///
    /// `DSV4_KV_MODE` env override is preserved for diagnostics so the
    /// host can deliberately pick the local KV sizing tradeoff:
    ///   - default (unset / "sliding"): rotating window + DeepseekV4Cache pool
    ///   - "full"  : plain KVCacheSimple on every layer (no compression,
    ///               no pool — for memory-permits long-reasoning runs
    ///               that don't need the hybrid path)
    ///   - "tq"    : KVCacheSimple, BatchEngine swaps to TurboQuantKVCache
    ///               once offset > min-tokens (caller must also set
    ///               `GenerateParameters.kvMode = .turboQuant(...)`)
    ///
    /// Caller-level `GenerateParameters.kvMode = .turboQuant` is
    /// intentionally NOT enough to switch DSV4 into `"tq"` mode. Osaurus
    /// can set global TQ defaults for ordinary KV models; DSV4 must keep
    /// its SWA+CSA+HSA hybrid cache unless the operator explicitly opts
    /// into the diagnostic/simple-cache override via `DSV4_KV_MODE=tq`.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let env = ProcessInfo.processInfo.environment
        let envMode = env["DSV4_KV_MODE"]?.lowercased()
        let mode: String = envMode ?? "sliding"

        return (0..<config.numHiddenLayers).map { layerIdx in
            switch mode {
            case "full", "tq":
                return KVCacheSimple()
            default:
                let cr =
                    config.compressRatios.count > layerIdx
                    ? config.compressRatios[layerIdx]
                    : Self.defaultCompressRatio(
                        layerIdx: layerIdx,
                        numLayers: config.numHiddenLayers)
                if cr > 0 {
                    return DeepseekV4Cache(
                        slidingWindow: config.slidingWindow,
                        compressRatio: cr)
                }
                return RotatingKVCache(
                    maxSize: config.slidingWindow, keep: 0)
            }
        }
    }

    /// Mirror of the per-layer compress_ratio default in
    /// `DeepseekV4Attention.init` for bundles whose
    /// `config.compressRatios` array isn't populated. Layers 0 and n-1
    /// are pure SWA (`cr=0`); middle layers alternate `4` (HSA+CSA)
    /// and `128` (CSA only).
    static func defaultCompressRatio(layerIdx: Int, numLayers: Int) -> Int {
        if layerIdx == 0 || layerIdx == numLayers - 1 { return 0 }
        let i = layerIdx - 1
        return (i % 2 == 1) ? 4 : 128
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = model(inputs, cache: cache)
        return DeepseekV4Math.lmHeadFp32(h, lmHead: lmHead)
    }

    /// Weight sanitize — remap DSV4 bundle key names to match module
    /// attribute paths, stack per-expert weights, drop MTP + unused
    /// compressor/indexer keys.
    ///
    /// Remap rules (from §G of RUNTIME-ARCHITECTURE):
    ///   model.embed.weight            → model.embed_tokens.weight
    ///   layers.{L}.attn.*             → model.layers.{L}.self_attn.*
    ///   layers.{L}.ffn.*              → model.layers.{L}.mlp.*
    ///   layers.{L}.attn_norm.weight   → model.layers.{L}.input_layernorm.weight
    ///   layers.{L}.ffn_norm.weight    → model.layers.{L}.post_attention_layernorm.weight
    ///   layers.{L}.hc_attn_*          → model.layers.{L}.attn_hc.{fn,scale,base}
    ///   layers.{L}.hc_ffn_*           → model.layers.{L}.ffn_hc.{fn,scale,base}
    ///   hc_head_*                     → model.hc_head.{hc_head_fn,hc_head_base,hc_head_scale}
    ///   norm.weight                   → model.norm.weight
    ///   head.weight                   → lm_head.weight
    ///   ffn.experts.{E}.{w1|w2|w3}.*  → mlp.switch_mlp.{gate|down|up}_proj.* (stacked)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        // First pass: direct rename + drop MTP (training head only).
        // Compressor + Indexer weights are KEPT — they're wired into
        // DeepseekV4Attention for long-context (L > sliding_window)
        // attention. Layers with compress_ratio == 0 carry no such
        // weights; layers with >0 carry `self_attn.compressor.*` and
        // (for ratio=4) `self_attn.indexer.*`.
        // Mirrors `Model.sanitize` in
        // jang-tools/jang_tools/dsv4/mlx_model.py:1124. Per-prefix
        // structural matching — avoids over-broad string replace bugs
        // (e.g. ".w1." colliding outside MLP contexts).
        let projForW = ["w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"]
        for (rawKey, value) in weights {
            if rawKey.hasPrefix("mtp.") { continue }

            // Top-level (no `layers.N.` prefix).
            if rawKey == "embed.weight" || rawKey == "embed.scales"
                || rawKey == "embed.biases"
            {
                let suffix = String(rawKey.dropFirst("embed.".count))
                out["model.embed_tokens.\(suffix)"] = value
                continue
            }
            if rawKey.hasPrefix("head.") {
                // head.{weight,scales,biases} → lm_head.*
                let suffix = String(rawKey.dropFirst("head.".count))
                out["lm_head.\(suffix)"] = value
                continue
            }
            if rawKey == "norm.weight" {
                out["model.norm.weight"] = value
                continue
            }
            if rawKey == "hc_head_fn" || rawKey == "hc_head_base"
                || rawKey == "hc_head_scale"
            {
                // `@ParameterInfo(key: "hc_head_*")` lives at
                // `model.hc_head.hc_head_*`.
                out["model.hc_head.\(rawKey)"] = value
                continue
            }

            // layers.N.{...} branch
            guard rawKey.hasPrefix("layers.") else {
                out["model.\(rawKey)"] = value
                continue
            }
            let afterLayers = rawKey.dropFirst("layers.".count)
            guard let dotIdx = afterLayers.firstIndex(of: ".") else { continue }
            let layerStr = String(afterLayers[..<dotIdx])
            guard Int(layerStr) != nil else { continue }
            let rest = String(afterLayers[afterLayers.index(after: dotIdx)...])
            let pfx = "model.layers.\(layerStr)"

            // Norms
            if rest == "attn_norm.weight" {
                out["\(pfx).input_layernorm.weight"] = value
                continue
            }
            if rest == "ffn_norm.weight" {
                out["\(pfx).post_attention_layernorm.weight"] = value
                continue
            }

            // mHC per-layer (hc_attn_*, hc_ffn_*).
            if rest.hasPrefix("hc_attn_") {
                let field = String(rest.dropFirst("hc_attn_".count))
                out["\(pfx).attn_hc.\(field)"] = value
                continue
            }
            if rest.hasPrefix("hc_ffn_") {
                let field = String(rest.dropFirst("hc_ffn_".count))
                out["\(pfx).ffn_hc.\(field)"] = value
                continue
            }

            // Attention subtree (q_norm / kv_norm / wq_a / wq_b / wkv /
            // wo_a / wo_b / attn_sink / compressor.* / indexer.*).
            if rest.hasPrefix("attn.") {
                let inner = String(rest.dropFirst("attn.".count))
                out["\(pfx).self_attn.\(inner)"] = value
                continue
            }

            // FFN subtree.
            if rest.hasPrefix("ffn.") {
                let inner = String(rest.dropFirst("ffn.".count))
                if inner.hasPrefix("gate.") {
                    let f = String(inner.dropFirst("gate.".count))
                    out["\(pfx).mlp.gate.\(f)"] = value
                    continue
                }
                if inner.hasPrefix("shared_experts.") {
                    let f = String(inner.dropFirst("shared_experts.".count))
                    if let firstDot = f.firstIndex(of: "."),
                        let proj = projForW[String(f[..<firstDot])]
                    {
                        let suffix = String(f[f.index(after: firstDot)...])
                        out["\(pfx).mlp.shared_experts.\(proj).\(suffix)"] = value
                        continue
                    }
                    out["\(pfx).mlp.shared_experts.\(f)"] = value
                    continue
                }
                if inner.hasPrefix("experts.") {
                    let after = String(inner.dropFirst("experts.".count))
                    guard let eDot = after.firstIndex(of: ".") else { continue }
                    let eStr = String(after[..<eDot])
                    let tail = String(after[after.index(after: eDot)...])
                    if let firstDot = tail.firstIndex(of: "."),
                        let proj = projForW[String(tail[..<firstDot])]
                    {
                        let suffix = String(tail[tail.index(after: firstDot)...])
                        out["\(pfx).mlp.experts.\(eStr).\(proj).\(suffix)"] = value
                        continue
                    }
                    out["\(pfx).mlp.experts.\(eStr).\(tail)"] = value
                    continue
                }
                out["\(pfx).mlp.\(inner)"] = value
                continue
            }

            out["\(pfx).\(rest)"] = value
        }

        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
        // DSV4-Flash JANGTQ bundles ship a pre-stacked
        // `jangtq_stacked.safetensors` overlay where the routed-expert
        // weights live at
        // `layers.{L}.mlp.switch_mlp.{gate,down,up}_proj.{packed,norms}`
        // — note the missing `tq_` prefix. Older Swift JANGTQ bundles
        // and the in-tree `TurboQuantSwitchLinear` use `tq_packed` /
        // `tq_norms`. Rewrite the un-prefixed names so the
        // `@ParameterInfo` keys match. Layout-preserving rename only.
        for layerIdx in 0..<config.numHiddenLayers {
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for (src, dst) in [("packed", "tq_packed"), ("norms", "tq_norms")] {
                    let from = "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(src)"
                    let to = "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(dst)"
                    if let v = out.removeValue(forKey: from), out[to] == nil {
                        out[to] = v
                    }
                }
            }
        }

        // Second pass: stack per-expert weights into switch_mlp.{gate,
        // up,down}_proj.*. Two formats supported:
        //
        // Affine (JANG_2L / JANG4): suffixes weight / scales / biases.
        //   Source per expert: (out, in) [+ (out, in/group)] [+ (out, in/group)]
        //   Stacked shape: (n_experts, ...).
        //
        // JANGTQ (JANGTQ2 / JANGTQ4 routed experts): suffixes
        // tq_packed / tq_norms. tq_bits is a per-tensor int constant
        // — we drop it (TurboQuantSwitchLinear configures bits at
        // construction time from the model_factory).
        //   Source per expert: tq_packed (out, packed_cols), tq_norms (out,)
        //   Stacked shape: (n_experts, out, packed_cols) / (n_experts, out)
        // The first pass already rewrote `.w1.` → `.gate_proj.` (etc.)
        // globally, so per-expert keys live at
        // `model.layers.L.mlp.experts.E.gate_proj.{suffix}`. Stack into
        // `model.layers.L.mlp.switch_mlp.{gate,down,up}_proj.{suffix}`.
        let suffixes = ["weight", "scales", "biases", "tq_packed", "tq_norms"]
        let streamJANGTQExperts = JANGTQStreamingExperts.isEnabled
        for layerIdx in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx).mlp.experts"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in suffixes {
                    let first = "\(prefix).0.\(projName).\(suffix)"
                    guard out[first] != nil else { continue }
                    if streamJANGTQExperts && (suffix == "tq_packed" || suffix == "tq_norms") {
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(
                                forKey: "\(prefix).\(e).\(projName).\(suffix)")
                        }
                        continue
                    }
                    var tensors: [MLXArray] = []
                    for e in 0..<config.nRoutedExperts {
                        let key = "\(prefix).\(e).\(projName).\(suffix)"
                        guard let t = out[key] else {
                            tensors = []
                            break
                        }
                        tensors.append(t)
                    }
                    if tensors.count == config.nRoutedExperts {
                        let stackedKey =
                            "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).\(suffix)"
                        if out[stackedKey] == nil {
                            out[stackedKey] = stacked(tensors)
                        }
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(
                                forKey: "\(prefix).\(e).\(projName).\(suffix)")
                        }
                    }
                }
                // Drop per-expert + prestacked-switch_mlp tq_bits scalars
                // — TurboQuantSwitchLinear gets the bit width from the
                // JANGTQ config (`mxtq_bits.routed_expert`), not weights.
                // Legacy bundles ship `mlp.experts.{e}.{proj}.tq_bits`;
                // prestacked bundles (DSV4-Flash JANGTQ_K, etc.) ship a
                // single `mlp.switch_mlp.{proj}.tq_bits` per layer.
                // Drop both regardless of which layout.
                for e in 0..<config.nRoutedExperts {
                    out.removeValue(
                        forKey: "\(prefix).\(e).\(projName).tq_bits")
                }
                out.removeValue(
                    forKey: "model.layers.\(layerIdx).mlp.switch_mlp.\(projName).tq_bits")
            }
        }
        return out
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
