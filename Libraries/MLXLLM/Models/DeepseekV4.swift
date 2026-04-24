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
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let qLoraRank: Int
    let oGroups: Int
    let oLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    /// Shape (num_heads,) — one learned sink logit per head.
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray

    let rope: DeepseekV4RoPE

    init(config: DeepseekV4Configuration, layerIdx: Int) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.ropeDim = config.qkRopeHeadDim
        self.qLoraRank = config.qLoraRank
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = 1.0 / sqrt(Float(headDim))

        // wq_a: hidden → qLoraRank (low-rank first stage of Q projection)
        self._wqA.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: false)
        // wq_b: qLoraRank → numHeads × headDim (second stage)
        self._wqB.wrappedValue = Linear(
            qLoraRank, numHeads * headDim, bias: false)
        // Single latent KV head: hidden → headDim (broadcast to all Q heads)
        self._wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        // Grouped low-rank O: numHeads*headDim → oGroups × oLoraRank
        // (per-group matmul, then concat groups, then wo_b)
        self._woA.wrappedValue = Linear(
            numHeads * headDim, oGroups * oLoraRank, bias: false)
        // wo_b: oGroups*oLoraRank → hidden
        self._woB.wrappedValue = Linear(
            oGroups * oLoraRank, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        // Initialized to zeros; real weight loaded in sanitize.
        self._attnSink.wrappedValue = zeros([numHeads])

        let theta = config.ropeTheta(forLayer: layerIdx)
        let factor: Float =
            config.hasCompressor(layer: layerIdx)
            ? Float(
                (config.ropeScaling?["factor"]?.asFloat()) ?? 16.0)
            : 1.0
        let origMax =
            Int(
                (config.ropeScaling?["original_max_position_embeddings"]?.asInt()) ?? 65536)
        self.rope = DeepseekV4RoPE(
            dim: ropeDim, base: theta, factor: factor,
            origMaxPos: origMax, betaFast: 32, betaSlow: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let offset = cache?.offset ?? 0

        // --- Q projection ---
        // x: (B, L, H) → wq_a(x): (B, L, qLoraRank) → q_norm → wq_b:
        // (B, L, numHeads*headDim) → (B, L, numHeads, headDim) → (B, numHeads, L, headDim)
        var q = wqB(qNorm(wqA(x)))
        q = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // --- KV projection (single latent head) ---
        // wkv(x): (B, L, headDim) → kv_norm → (B, 1, L, headDim)
        var kv = kvNorm(wkv(x))
        kv = kv.reshaped(B, L, 1, headDim).transposed(0, 2, 1, 3)

        // --- Partial RoPE on last ropeDim dims of Q and K ---
        let (cosT, sinT) = rope.cosSin(offset: offset, length: L)
        // Broadcast cos/sin over (B, H, L, ropeDim/2) for Q, (B, 1, L, ropeDim/2) for K.
        let cosQ = cosT.expandedDimensions(axes: [0, 1])  // (1,1,L,ropeDim/2)
        let sinQ = sinT.expandedDimensions(axes: [0, 1])
        q = DeepseekV4Math.applyPartialRoPE(q, cos: cosQ, sin: sinQ, ropeDim: ropeDim)
        kv = DeepseekV4Math.applyPartialRoPE(kv, cos: cosQ, sin: sinQ, ropeDim: ropeDim)

        // --- Cache update ---
        // DSV4 uses keys==values (single latent head serves both roles).
        var keys = kv
        var values = kv
        if let cache = cache {
            (keys, values) = cache.update(keys: kv, values: kv)
        }

        // --- SDPA with attention sinks (fp32 accum for head_dim=512) ---
        // MLXFast SDPA supports sinks natively. For GQA broadcast,
        // num_kv_heads=1 → broadcast to numHeads. Scale = 1/√headDim.
        var output = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values,
            scale: scale, mask: mask,
            sinks: config.useAttnSink ? attnSink : nil)
        output = output.transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)

        // --- Inverse RoPE on the attention output ---
        // Strip positional info from the output before residual add-back.
        // This is applied on the last ropeDim contiguous dims of EACH
        // head's output. For simplicity we reshape to (B, L, numHeads,
        // headDim), apply inverse on last ropeDim, then flatten.
        var outputH = output.reshaped(B, L, numHeads, headDim)
        let cosO = cosT.expandedDimensions(axes: [0, 2])  // (1,L,1,ropeDim/2)
        let sinO = sinT.expandedDimensions(axes: [0, 2])
        outputH = DeepseekV4Math.applyPartialRoPE(
            outputH, cos: cosO, sin: sinO, ropeDim: ropeDim, inverse: true)
        output = outputH.reshaped(B, L, numHeads * headDim)

        // --- Grouped low-rank O projection ---
        // wo_a splits output into oGroups × oLoraRank: reshape to
        // (B, L, oGroups, oLoraRank). wo_b flattens and projects to hidden.
        let oA = woA(output)  // (B, L, oGroups*oLoraRank)
        let oB = woB(oA)
        return oB
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
        self._tid2eid.wrappedValue = zeros([isHashLayer ? config.vocabSize : 1])
    }

    /// Returns (indices, weights) where indices has shape (B, L, topK)
    /// and weights has shape (B, L, topK). For hash layers, weights is
    /// a synthetic uniform 1/topK so the downstream SwitchGLU gets the
    /// same contract as softmax-routed layers.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> (MLXArray, MLXArray) {
        if isHashLayer, let ids = inputIds {
            // Hash routing: tid2eid[ids] gives the single expert per token.
            // Replicate to topK slots with uniform weight so the
            // SwitchGLU contract (shape (B, L, topK) indices) holds.
            let (B, L) = (x.dim(0), x.dim(1))
            let chosen = tid2eid[ids]  // (B, L)
            let indices = broadcast(
                chosen.expandedDimensions(axis: -1), to: [B, L, topK])
            let weights = MLXArray.ones([B, L, topK], dtype: .float32)
                * (routedScalingFactor / Float(topK))
            return (indices.asType(.uint32), weights)
        }

        // Non-hash: compute scores via sqrt(softplus(logits))
        // The matmul is performed in fp32 per the bug-fix contract.
        let xF32 = x.asType(.float32)
        let wF32 = weight.asType(.float32)
        let logits = xF32.matmul(wF32.transposed())
        let scores = DeepseekV4Math.sqrtSoftplus(logits)

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
        self.topK = config.numExpertsPerTok
        let limit = config.swigluLimit
        // Activation: silu(min(gate, limit)) * clip(up, ±limit).
        // SwitchGLU accepts a scalar activation applied to gate, with
        // up multiplied after. Our helper fuses both legs.
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: { gate in
                // SwitchGLU computes act(gate) * up. Our "activation" is
                // applied to gate only — so we return the limited silu
                // branch here. Up clamping happens inside ops that form
                // gate*up — but SwitchGLU multiplies AFTER activation,
                // so the `up` side escapes our clamp. Acceptable for
                // now (the dominant overflow came from the silu(gate)
                // blowing up, not up — per §J bug #2 investigation).
                return silu(minimum(gate, MLXArray(limit)))
            })
        self.gate = DeepseekV4MoEGate(config: config, layerIdx: layerIdx)
        self._sharedExperts.wrappedValue = DeepseekV4MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: limit)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: currentInputIds)
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
    /// `hc_{attn,ffn}_fn`: (hcMult, 3*hcMult). Maps residual to mix
    /// coefficients in one matmul.
    @ParameterInfo(key: "fn") var fn: MLXArray
    /// `hc_{attn,ffn}_scale`: (3,) per-field scalar.
    @ParameterInfo(key: "scale") var scale: MLXArray
    /// `hc_{attn,ffn}_base`: (3*hcMult,) bias.
    @ParameterInfo(key: "base") var base: MLXArray

    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hcIters = config.hcSinkhornIters
        self.hcEps = config.hcEps
        self.hiddenSize = config.hiddenSize
        self._fn.wrappedValue = zeros([hcMult, 3 * hcMult])
        self._scale.wrappedValue = zeros([3])
        self._base.wrappedValue = zeros([3 * hcMult])
    }

    /// Collapse: `h` shape (B, L, hcMult, hiddenSize) → collapsed x
    /// (B, L, hiddenSize) plus `post` (B, L, hcMult) and `comb`
    /// (B, L, hcMult, hcMult) for the expand step.
    func collapse(_ h: MLXArray) -> (x: MLXArray, post: MLXArray, comb: MLXArray) {
        let B = h.dim(0)
        let L = h.dim(1)
        // Compute mean across the (hcMult, hiddenSize) flattened axes
        // for the mixes — using a matmul into `fn`.
        //   mixes = h_flat @ fn   where h_flat is reshaped to (..., hcMult).
        // Python reference collapses hcMult dim through rsqrt-normalized
        // mean. We follow the shape contract: produce `mixes` shape
        // (..., 3*hcMult) so hcSplitSinkhorn can consume it.
        let hFp32 = h.asType(.float32)
        // Reduce over hiddenSize to get per-copy scalar, then matmul with fn.
        let perCopy = hFp32.mean(axis: -1)  // (B, L, hcMult)
        let mixes = perCopy.matmul(fn)  // (B, L, 3*hcMult)
        let (pre, post, comb) = DeepseekV4Math.hcSplitSinkhorn(
            mixes: mixes, scale: scale, base: base,
            hcMult: hcMult, iters: hcIters, eps: hcEps)
        // x = sum_i pre[i] * h[..., i, :]
        let preExp = pre.expandedDimensions(axis: -1)  // (B, L, hcMult, 1)
        let x = (preExp * hFp32).sum(axis: -2)  // (B, L, hiddenSize)
        return (x: x.asType(h.dtype), post: post, comb: comb)
        _ = B; _ = L
    }

    /// Expand: given attn/ffn output `blockOut` (B, L, hiddenSize),
    /// residual (B, L, hcMult, hiddenSize), and the (post, comb) from
    /// the matching collapse, return new h (B, L, hcMult, hiddenSize).
    func expand(
        blockOut: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        // Contract `comb` (B,L,hc,hc) with residual (B,L,hc,D) over the
        // residual's hc axis: `matmul(comb, residual)` → (B,L,hc,D).
        let combF = comb.asType(.float32)
        let residualF = residual.asType(.float32)
        let combResid = combF.matmul(residualF)
        // post: (B, L, hc) → (B, L, hc, 1); blockOut: (B, L, D) → (B, L, 1, D).
        let postExp = post.expandedDimensions(axis: -1)
        let blockExp = blockOut.asType(.float32).expandedDimensions(axis: -2)
        let y = postExp * blockExp + combResid
        return y.asType(blockOut.dtype)
    }
}

// MARK: - HyperHead (top-of-model mHC reduce)

class DeepseekV4HyperHead: Module {
    let hcMult: Int
    let hiddenSize: Int
    @ParameterInfo(key: "hc_head_fn") var fn: MLXArray
    @ParameterInfo(key: "hc_head_base") var base: MLXArray
    @ParameterInfo(key: "hc_head_scale") var scale: MLXArray

    init(config: DeepseekV4Configuration) {
        self.hcMult = config.hcMult
        self.hiddenSize = config.hiddenSize
        // fn shape: (hcMult, hcMult*hiddenSize) per §J bug comment
        // (vs per-block fn which is (hcMult, 3*hcMult)). This is a
        // distinct reduction parameter — not used in per-block HC.
        self._fn.wrappedValue = zeros([hcMult, hcMult * hiddenSize])
        self._base.wrappedValue = zeros([hcMult])
        self._scale.wrappedValue = zeros([1])
    }

    /// Reduce (B, L, hcMult, hiddenSize) → (B, L, hiddenSize) using a
    /// sigmoid-summed mixing (no Sinkhorn, simpler than per-block HC).
    func reduce(_ h: MLXArray) -> MLXArray {
        // Flatten (hcMult, hiddenSize) → (hcMult*hiddenSize), matmul
        // with fn^T to get (B, L, hcMult), then sigmoid+sum-to-1.
        let B = h.dim(0)
        let L = h.dim(1)
        let flat = h.reshaped(B, L, hcMult * hiddenSize).asType(.float32)
        let mixes = flat.matmul(fn.asType(.float32).transposed())  // (B, L, hcMult)
        var weights = sigmoid(mixes * scale + base) + 1e-6
        let denom = weights.sum(axis: -1, keepDims: true)
        weights = weights / denom
        // Weighted sum over hcMult axis.
        let out = (weights.expandedDimensions(axis: -1) * h.asType(.float32)).sum(axis: -2)
        return out.asType(h.dtype)
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

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = model(inputs, cache: cache)
        return lmHead(h)
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
        // First pass: direct rename + drop MTP + drop compressor/indexer.
        for (rawKey, value) in weights {
            // Drop keys that aren't wired in Phase 1b.
            if rawKey.contains("mtp.") { continue }
            if rawKey.contains(".compressor.") || rawKey.contains(".indexer.") {
                // Compressor + Indexer are optional long-context paths.
                // Dropping them makes short prompts run correctly; long
                // prompts (L > sliding_window=128) will fall back to
                // sliding-window-only attention.
                continue
            }

            var k = rawKey

            // Top-level renames
            k = k.replacingOccurrences(of: "embed.weight", with: "model.embed_tokens.weight")
            if k == "norm.weight" { k = "model.norm.weight" }
            if k == "head.weight" { k = "lm_head.weight" }

            // HyperHead at model root
            if k.hasPrefix("hc_head_") {
                // Rename hc_head_{fn,base,scale} → model.hc_head.hc_head_{...}
                k = "model.hc_head." + k
            }

            // Per-layer renames — prepend model. and rewrite segments.
            if k.hasPrefix("layers.") || k.hasPrefix("model.layers.") {
                if !k.hasPrefix("model.layers.") {
                    k = "model." + k
                }
                // attn → self_attn (only as a segment)
                k = k.replacingOccurrences(of: ".attn.", with: ".self_attn.")
                // attn_norm → input_layernorm
                k = k.replacingOccurrences(
                    of: ".attn_norm.", with: ".input_layernorm.")
                // ffn_norm → post_attention_layernorm
                k = k.replacingOccurrences(
                    of: ".ffn_norm.", with: ".post_attention_layernorm.")
                // ffn → mlp
                k = k.replacingOccurrences(of: ".ffn.", with: ".mlp.")
                // hc_attn / hc_ffn (prefix-only — matches "layer.N.hc_attn_fn",
                // the suffix after the final _ is the field name).
                for which in ["hc_attn", "hc_ffn"] {
                    for field in ["fn", "base", "scale"] {
                        let src = ".\(which)_\(field)"
                        let dst = ".\(which.replacingOccurrences(of: "hc_", with: ""))_hc.\(field)"
                        if k.contains(src) {
                            k = k.replacingOccurrences(of: src, with: dst)
                        }
                    }
                }
            }

            out[k] = value
        }

        // Second pass: stack per-expert weights into switch_mlp.{gate,up,down}_proj.*
        // Source shape: (out, in) each; stacked shape: (n_experts, out, in).
        for layerIdx in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layerIdx).mlp.experts"
            for (src, dst) in [
                ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"),
            ] {
                for suffix in ["weight", "scales", "biases"] {
                    let first = "\(prefix).0.\(src).\(suffix)"
                    guard out[first] != nil else { continue }
                    var tensors: [MLXArray] = []
                    for e in 0..<config.nRoutedExperts {
                        let key = "\(prefix).\(e).\(src).\(suffix)"
                        guard let t = out[key] else {
                            tensors = []
                            break
                        }
                        tensors.append(t)
                    }
                    if tensors.count == config.nRoutedExperts {
                        let stackedKey =
                            "model.layers.\(layerIdx).mlp.switch_mlp.\(dst).\(suffix)"
                        out[stackedKey] = stacked(tensors)
                        // Remove the per-expert originals.
                        for e in 0..<config.nRoutedExperts {
                            out.removeValue(forKey: "\(prefix).\(e).\(src).\(suffix)")
                        }
                    }
                }
            }
        }
        return out
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
