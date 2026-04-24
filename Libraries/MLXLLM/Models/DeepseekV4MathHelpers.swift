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
        precondition(mixes.shape.last == 3 * hcMult, "mixes last dim must be 3*hcMult")

        // Split mixes into three chunks of width `hcMult`.
        let slices = split(mixes, parts: 3, axis: -1)
        let mixPre = slices[0]
        let mixPost = slices[1]
        let mixComb = slices[2]

        let basePre = base[0..<hcMult]
        let basePost = base[hcMult..<(2 * hcMult)]
        let baseComb = base[(2 * hcMult)..<(3 * hcMult)]

        let pre = sigmoid(mixPre * scale[0] + basePre) + eps
        let post = 2.0 * sigmoid(mixPost * scale[1] + basePost)

        // `comb` starts as softmax then undergoes Sinkhorn iterations
        // alternating col-normalize and row-normalize to reach a
        // doubly-stochastic matrix. The softmax output is (..., hcMult)
        // — we broadcast it into an (hcMult, hcMult) structure by
        // treating the softmax as one *row* of the mixing matrix and
        // replicating. Python replicates via `mx.repeat`.
        let softmaxed = softmax(mixComb * scale[2] + baseComb, axis: -1) + eps
        // Expand to (..., hcMult, hcMult) — each row shares the same
        // softmax output initially; Sinkhorn iterations then mix.
        let expanded = broadcast(
            softmaxed.expandedDimensions(axis: -2),
            to: softmaxed.shape.dropLast() + [hcMult, hcMult])

        var comb = expanded
        // One initial col-normalize.
        comb = sinkhornColNormalize(comb, eps: eps)
        for _ in 0..<(iters - 1) {
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

    /// Apply the standard "rotate half" RoPE transform to a fully-rope
    /// tensor. `cos`/`sin` must broadcast over the leading shape and
    /// match the last axis (ropeDim). `inverse=true` uses conj(sin).
    private static func rotateHalf(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, inverse: Bool
    ) -> MLXArray {
        let half = x.shape.last! / 2
        let x1 = x[.ellipsis, 0..<half]
        let x2 = x[.ellipsis, half...]
        // Standard form: [x1*c - x2*s, x1*s + x2*c].
        // Inverse form: swap sign of s terms (equivalent to conj(freqs)).
        let s = inverse ? -sin : sin
        let rot1 = x1 * cos - x2 * s
        let rot2 = x1 * s + x2 * cos
        return concatenated([rot1, rot2], axis: -1)
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
}
