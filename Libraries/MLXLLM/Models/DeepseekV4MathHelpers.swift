// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Pure-math building blocks for the DeepSeek-V4 forward pass.
// Each helper is pure (no module state, no cache) so it's trivially
// unit-testable with synthetic tensors.
//
// Reference:
//   - `jang/research/DSV4-RUNTIME-ARCHITECTURE.md` §2 (per-layer forward)
//   - `jang-tools/jang_tools/dsv4_prune/mlx_model.py` —
//       * `_hc_split_sinkhorn_ops` (lines 79-110)
//       * `_apply_partial_rope` (lines 355-362)
//       * `_dsv4_swiglu` (lines 799-814)
//       * `sqrtsoftplus_select` (lines 736-757)

import Foundation
import MLX
import MLXNN

public enum DeepseekV4Math {

    // MARK: - Per-head Q RMSNorm ones cache
    //
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // The DSV4 attention applies a unit-weight RMSNorm-like rescale per
    // head before partial RoPE: `q * rsqrt((q^2).mean(-1) + eps)`. The
    // Python reference fuses this into a single `mx.fast.rms_norm` kernel
    // with a cached ones tensor. The cache is keyed on `(headDim, dtype)`
    // and shared across all 64 attention heads × 43 layers. NSLock-guarded
    // so concurrent SwitchGLU/attention forwards never race the cache.
    private static let qNormOnesLock = NSLock()
    nonisolated(unsafe) private static var qNormOnesCache: [String: MLXArray] = [:]

    public static func qNormOnes(headDim: Int, dtype: DType) -> MLXArray {
        let key = "\(headDim)|\(dtype)"
        qNormOnesLock.lock(); defer { qNormOnesLock.unlock() }
        if let w = qNormOnesCache[key] { return w }
        let w = MLXArray.ones([headDim], dtype: dtype)
        qNormOnesCache[key] = w
        return w
    }


    // MARK: - mHC split-Sinkhorn (collapse matrices)
    //
    // Given `mixes` of shape (..., 3*hcMult) and per-block scale/base
    // parameters, produce the three matrices needed by the HC collapse
    // kernel:
    //
    //   pre   = sigmoid(mixes * scale[0] + base[:hcMult]) + eps
    //           (no normalization — used to weight residual copies)
    //
    //   post  = 2 * sigmoid(mixes * scale[1] + base[hcMult:2*hcMult])
    //           (no eps — used to scale block output before add-back)
    //
    //   comb  = softmax(mixes * scale[2] + base[2*hcMult:3*hcMult], axis=-1) + eps
    //           col-normalize
    //           repeat (iters-1)× { row-normalize; col-normalize }
    //
    // `comb` is the sinkhorn doubly-stochastic mixing matrix that
    // preserves residual norm when used for the `expand` step.
    //
    // Shape contract:
    //   mixes: (..., 3*hcMult)
    //   scale: (3,)       one learned scalar per field
    //   base:  (3*hcMult,) learned bias concatenated across fields
    //   → pre:  (..., hcMult)
    //   → post: (..., hcMult)
    //   → comb: (..., hcMult, hcMult)
    public static func hcSplitSinkhorn(
        mixes: MLXArray,
        scale: MLXArray,
        base: MLXArray,
        hcMult: Int,
        iters: Int = 20,
        eps: Float = 1e-6
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        // Match Python `_hc_split_sinkhorn_ops` exactly. Mixes width is
        // `(2 + hcMult) * hcMult`, not `3 * hcMult`. The first hcMult
        // elements form `pre`, the next hcMult form `post`, and the
        // remaining `hcMult * hcMult` are reshaped into the (hc, hc)
        // doubly-stochastic mixing matrix `comb`.
        let mh = hcMult
        let mixHc = (2 + mh) * mh
        precondition(
            mixes.shape.last == mixHc,
            "mixes last dim must be (2+hcMult)*hcMult = \(mixHc), got \(mixes.shape.last ?? -1)")

        // Bring everything to fp32 for numerical stability — the
        // sinkhorn iterations are sensitive to fp16 underflow on the
        // post-softmax row/col normalizations.
        let mixesF = mixes.asType(.float32)
        let scaleF = scale.asType(.float32)
        let baseF = base.asType(.float32)

        let preScale = scaleF[0]
        let postScale = scaleF[1]
        let combScale = scaleF[2]

        let basePre = baseF[0..<mh]
        let basePost = baseF[mh..<(2 * mh)]
        let baseComb = baseF[(2 * mh)...]  // length mh*mh

        let mixPre = mixesF[.ellipsis, 0..<mh]
        let mixPost = mixesF[.ellipsis, mh..<(2 * mh)]
        let mixCombFlat = mixesF[.ellipsis, (2 * mh)...]  // (..., mh*mh)

        let pre = sigmoid(mixPre * preScale + basePre) + eps
        let post = 2.0 * sigmoid(mixPost * postScale + basePost)

        // Reshape last axis (mh*mh) into (mh, mh) and add bias also
        // reshaped to (mh, mh).
        let leadShape = Array(mixCombFlat.shape.dropLast())
        var combRaw = mixCombFlat * combScale
        combRaw = combRaw.reshaped(leadShape + [mh, mh])
            + baseComb.reshaped([mh, mh])
        var comb = softmax(combRaw, axis: -1) + eps
        // Initial col-normalize, then (iters-1) × {row, col}.
        comb = sinkhornColNormalize(comb, eps: eps)
        for _ in 0..<max(iters - 1, 0) {
            comb = sinkhornRowNormalize(comb, eps: eps)
            comb = sinkhornColNormalize(comb, eps: eps)
        }

        return (pre: pre, post: post, comb: comb)
    }

    private static func sinkhornRowNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let rowSum = x.sum(axis: -1, keepDims: true)
        return x / (rowSum + eps)
    }

    private static func sinkhornColNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
        let colSum = x.sum(axis: -2, keepDims: true)
        return x / (colSum + eps)
    }

    // MARK: - Partial RoPE
    //
    // DSV4 applies rotary ONLY to the last `ropeDim` (default 64) of
    // the head-dim=512 Q/K vector — the first 448 dims are "no-position".
    // Forward (token -> position-rotated): standard RoPE rotate.
    // Inverse (position-rotated -> token, used on attention OUTPUT):
    //   undo the rotation via negative-angle cos/sin, so the residual
    //   stream contribution is position-agnostic.
    public static func applyPartialRoPE(
        _ x: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
        ropeDim: Int,
        inverse: Bool = false
    ) -> MLXArray {
        let headDim = x.shape.last!
        precondition(ropeDim <= headDim, "ropeDim must be ≤ headDim")
        let noPoseDim = headDim - ropeDim
        if noPoseDim == 0 {
            return rotateHalf(x, cos: cos, sin: sin, inverse: inverse)
        }
        // Split last axis: [..., :noPoseDim] keep; [..., noPoseDim:] rotate.
        let nope = x[.ellipsis, 0..<noPoseDim]
        let pe = x[.ellipsis, noPoseDim...]
        let rotated = rotateHalf(pe, cos: cos, sin: sin, inverse: inverse)
        return concatenated([nope, rotated], axis: -1)
    }

    /// Apply traditional/interleaved RoPE — DSV4 uses
    /// `traditional=True` (mx.fast.rope) which rotates ADJACENT pairs:
    /// `(x[…,0], x[…,1])`, `(x[…,2], x[…,3])`, etc. NOT split-half
    /// `(x[…,:D/2], x[…,D/2:])`. Mirror Python `_call_manual` in
    /// jang_tools/dsv4/mlx_model.py:DeepseekV4RoPE — using the wrong
    /// convention scrambles positional information across the head
    /// dim and the model decodes a repeating-token loop (verified
    /// 2026-04-24).
    ///
    /// `cos`/`sin` shape must broadcast over the leading axes and
    /// match `(L, ropeDim/2)`. `inverse=true` flips sin sign
    /// (equivalent to multiplying by conjugate of the rotation).
    private static func rotateHalf(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, inverse: Bool
    ) -> MLXArray {
        let lastDim = x.shape.last!
        let halfDim = lastDim / 2
        // Reshape last axis from D to (D/2, 2) so the trailing pair
        // is the (real, imag) tuple of each rotation.
        let xPaired = x.reshaped(x.shape.dropLast() + [halfDim, 2])
        let x0 = xPaired[.ellipsis, 0]  // (..., D/2)
        let x1 = xPaired[.ellipsis, 1]  // (..., D/2)
        let s = inverse ? -sin : sin
        let r0 = x0 * cos - x1 * s
        let r1 = x0 * s + x1 * cos
        // Stack along a new last axis (D/2, 2) then collapse → D.
        let stacked = stacked([r0, r1], axis: -1)
        return stacked.reshaped(x.shape)
    }

    // MARK: - DSV4 SwiGLU activation with `limit`
    //
    // silu(min(gate, limit)) * clip(up, -limit, +limit). The clamping
    // is essential — unclipped, silu(gate)*up overflows fp16 in the
    // MoE's down-projection matmul (same issue we hit on other MoE
    // families; see memory `mlp_bfloat16_upcast.md`).
    public static func dsv4SwiGLU(
        gate: MLXArray,
        up: MLXArray,
        limit: Float
    ) -> MLXArray {
        let gClamped = minimum(gate, MLXArray(limit))
        let uClamped = clip(up, min: -limit, max: limit)
        return silu(gClamped) * uClamped
    }

    // MARK: - sqrtsoftplus (MoE gate scoring)
    //
    // scores = sqrt(log1p(exp(logits))) — replaces softmax for DSV4's
    // routing. Monotonic, smoother gradient in the tail than softmax,
    // and doesn't require the sum-to-1 constraint that makes hash
    // routing incompatible.
    //
    // Numerical guard: log1p(exp(x)) is `softplus(x)` — mlx exposes
    // it directly and handles the overflow branch for large x.
    public static func sqrtSoftplus(_ logits: MLXArray) -> MLXArray {
        sqrt(logAddExp(logits, MLXArray(0.0)))
    }

    // MARK: - Top-k over sqrtsoftplus with bias + norm
    //
    // Production gate path (for non-hash layers):
    //   biased = scores + noauxBias
    //   topKIdx = argpartition(-biased, k)[:k]
    //   topKWeights = take_along_axis(scores, topKIdx)   — UNBIASED!
    //   normalized = topKWeights / sum(topKWeights) * routedScalingFactor
    //
    // Critical: `noauxBias` is used ONLY to pick the indices — once
    // picked, the UNBIASED score is what gets used as the expert
    // weight. This was bug #6 in the DSV-EXHAUSTIVE-VARIABLES-GUIDE;
    // using biased weights broke coherence.
    public static func sqrtSoftplusSelect(
        scores: MLXArray,
        noauxBias: MLXArray?,
        k: Int,
        normalize: Bool,
        scalingFactor: Float
    ) -> (indices: MLXArray, weights: MLXArray) {
        let biased = noauxBias != nil ? (scores + noauxBias!) : scores
        // argpartition returns unordered top-k; sort indices for
        // determinism (matters for cache-hit byte equivalence).
        let topKIdx = argPartition(-biased, kth: k - 1, axis: -1)[.ellipsis, 0..<k]
        // Gather the UNBIASED scores at those indices.
        let gathered = takeAlong(scores, topKIdx, axis: -1)
        var weights = gathered
        if normalize {
            let denom = weights.sum(axis: -1, keepDims: true) + 1e-20
            weights = weights / denom * scalingFactor
        } else {
            weights = weights * scalingFactor
        }
        return (indices: topKIdx, weights: weights)
    }

    // MARK: - YaRN RoPE freq table
    //
    // `rope_factor=16`, `original_seq_len=65536`, `beta_fast=32`,
    // `beta_slow=1` are the DSV4 defaults when compress_ratio>0.
    // Layers with compress_ratio==0 use plain (non-YaRN) RoPE with
    // `rope_theta=10000`.
    public static func yarnInvFreq(
        dim: Int,
        base: Float,
        maxPos: Int,
        origMaxPos: Int,
        factor: Float,
        betaFast: Float,
        betaSlow: Float
    ) -> MLXArray {
        // Standard inv-freq table.
        let dimF = Float(dim)
        let halfDim = dim / 2
        var invFreq = [Float]()
        invFreq.reserveCapacity(halfDim)
        for i in 0..<halfDim {
            let exponent = Float(2 * i) / dimF
            invFreq.append(1.0 / pow(base, exponent))
        }
        let invFreqArr = MLXArray(invFreq)

        if factor == 1.0 {
            return invFreqArr
        }

        // YaRN: ramp mask smooths the transition between full and
        // scaled frequencies for dims that correspond to wavelengths
        // between betaSlow and betaFast. `high = min(..., dim - 1)`
        // per the §2 bug fix (the upstream MLX had `dim/2-1`).
        let twoPi = Float.pi * 2
        func correctionDim(_ beta: Float) -> Float {
            dimF * log(Float(origMaxPos) / (beta * twoPi)) / (2.0 * log(base))
        }
        let low = max(0.0, floor(correctionDim(betaSlow)))
        let high = min(Float(dim - 1), ceil(correctionDim(betaFast)))
        let rangeWidth = max(high - low, 0.001)

        var ramp = [Float]()
        ramp.reserveCapacity(halfDim)
        for i in 0..<halfDim {
            let t = (Float(i) - low) / rangeWidth
            ramp.append(max(0.0, min(1.0, t)))
        }
        let rampArr = MLXArray(ramp)
        let smooth = MLXArray(1.0) - rampArr
        let scaled = invFreqArr / factor
        return scaled * (MLXArray(1.0) - smooth) + invFreqArr * smooth
        _ = maxPos  // reserved for future extrapolation logic
    }

    // MARK: - LM head fp32 matmul
    //
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // DeepSeek-V4 reference inference (`inference/model.py`
    // `ParallelHead.get_logits`) explicitly performs the lm_head matmul
    // in fp32 — the weights are stored as fp32 and the activations are
    // cast to fp32 before `F.linear`. Hidden contraction is 4096 wide,
    // and bf16/fp16 accumulation drops ~0.5 ULP per logit which
    // empirically flips arithmetic-style next-token answers. Mirror
    // that here: dequantize quantized lm_heads to fp32, cast `h` to
    // fp32, then matmul.
    public static func lmHeadFp32(_ h: MLXArray, lmHead: Linear) -> MLXArray {
        if let q = lmHead as? QuantizedLinear {
            let wF32 = MLX.dequantized(
                q.weight, scales: q.scales, biases: q.biases,
                groupSize: q.groupSize, bits: q.bits
            ).asType(.float32)
            let hF32 = h.asType(.float32)
            var out = hF32.matmul(wF32.transposed())
            if let b = q.bias {
                out = out + b.asType(.float32)
            }
            return out
        }
        let wF32 = lmHead.weight.asType(.float32)
        let hF32 = h.asType(.float32)
        var out = hF32.matmul(wF32.transposed())
        if let b = lmHead.bias {
            out = out + b.asType(.float32)
        }
        return out
    }

    // MARK: - Compressor + Indexer attention masks (PR #1195 port)
    //
    // The DSV4 paper §9-13 attention path is a hybrid of:
    //   1. A LOCAL sliding-window over the last `window` raw tokens
    //      (kept in a RotatingKVCache).
    //   2. A GLOBAL pooled context over compressor chunks (kept as
    //      a single (B, P, head_dim) tensor in an ArraysCache slot).
    //
    // Visibility from query at raw position `q` to a key:
    //
    //   - Window key at raw position `r`:
    //       q - window < r <= q
    //
    //   - Compressed key at chunk index `k` covering raw range
    //     [k*ratio, (k+1)*ratio):
    //       (k + 1) * ratio <= q + 1
    //
    // For compress_ratio==4 layers the Indexer adds a top-k
    // selection — only the K compressed chunks the indexer scored
    // highest are visible, ANDed with the causal staircase above.
    //
    // Both helpers return 4D bool arrays of shape (B, 1, S, L_kv)
    // that broadcast onto SDPA attention scores (B, H, S, L_kv).
    // Building 4D directly avoids the SDPA broadcast bugs the
    // previous staircase attempts hit.

    /// Per-query visibility into the local sliding-window cache.
    ///
    /// `windowLen` is the number of slots currently filled in the
    /// RotatingKVCache (== window once the buffer wraps). The
    /// trailing `windowLen` raw positions in the cache map to raw
    /// token indices `(offset + S) - windowLen + i` for slot `i`.
    /// Returns shape `(B, 1, S, windowLen)`.
    public static func buildWindowMask(
        batch B: Int, queryLen S: Int,
        offset: Int, window: Int, windowLen: Int
    ) -> MLXArray {
        // q_pos: (B, S) — broadcasted absolute raw positions of each
        // query slot. The PR #1195 Python builds (B, S) by broadcasting
        // (1, S) to (B, S); we do the same.
        let qPos =
            MLXArray(Int32(offset)..<Int32(offset + S))
            .expandedDimensions(axis: 0)        // (1, S)
            .reshaped(1, S)
        // raw_pos_at_k: (windowLen,) → (1, 1, windowLen)
        let cacheK = MLXArray(Int32(0)..<Int32(windowLen))
        let rawPosAtK = MLXArray(Int32((offset + S) - windowLen)) + cacheK
        // qPos: (1, S) → (1, S, 1) then broadcast against (1, 1, windowLen)
        let qPos3 = qPos.expandedDimensions(axis: -1)
        let raw3 = rawPosAtK.expandedDimensions(axes: [0, 1])
        let lower = raw3 .> (qPos3 - MLXArray(Int32(window)))
        let upper = raw3 .<= qPos3
        let visible = MLX.logicalAnd(lower, upper)
        // (1, S, windowLen) → broadcast to (B, 1, S, windowLen)
        let v4 = visible.expandedDimensions(axis: 1)
        let bArr = MLX.broadcast(v4, to: [B, 1, S, windowLen])
        return bArr
    }

    /// Per-query causal visibility into the compressor's pooled
    /// chunks. Chunk `k` covers raw positions `[k*ratio, (k+1)*ratio)`
    /// and is visible to query `q` once that whole chunk has been
    /// observed: `(k+1)*ratio <= q+1`.
    /// Returns shape `(B, 1, S, compressedLen)`.
    public static func compressedVisibility(
        batch B: Int, queryLen S: Int,
        offset: Int, compressedLen: Int, ratio: Int
    ) -> MLXArray {
        let qPos =
            MLXArray(Int32(offset)..<Int32(offset + S))
            .expandedDimensions(axis: 0)
            .reshaped(1, S)
        let k = MLXArray(Int32(0)..<Int32(compressedLen))
        // (k+1) * ratio <= (qPos + 1)
        let lhs =
            (k + MLXArray(Int32(1))) * MLXArray(Int32(ratio))
        let rhs = qPos + MLXArray(Int32(1))
        // lhs: (compressedLen,) → (1, 1, compressedLen)
        let lhs3 = lhs.expandedDimensions(axes: [0, 1])
        // rhs: (1, S) → (1, S, 1)
        let rhs3 = rhs.expandedDimensions(axis: -1)
        let visible = lhs3 .<= rhs3
        let v4 = visible.expandedDimensions(axis: 1)
        return MLX.broadcast(v4, to: [B, 1, S, compressedLen])
    }

    /// AND the per-query indexer top-k selection onto a compressed
    /// visibility mask. `topk` is the indexer's `(B, S, K)` int array
    /// of selected chunk indices; returns `(B, 1, S, compressedLen)`
    /// bool — true only when chunk index `c` appears in `topk[b, s, :]`.
    public static func indexerSelectionMask(
        topk: MLXArray, compressedLen: Int
    ) -> MLXArray {
        // topk: (B, S, K) → (B, S, K, 1)
        let topk4 = topk.expandedDimensions(axis: -1)
        // k_range: (compressedLen,) → (1, 1, 1, compressedLen)
        let kRange =
            MLXArray(Int32(0)..<Int32(compressedLen))
            .expandedDimensions(axes: [0, 1, 2])
        // (B, S, K, compressedLen) → (B, S, compressedLen) via any over K
        let eq = topk4 .== kRange
        let selected = eq.any(axis: -2)  // (B, S, compressedLen)
        return selected.expandedDimensions(axis: 1)  // (B, 1, S, compressedLen)
    }
}
