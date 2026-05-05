//
// JANGTQ Metal kernels — Swift port of jang-tools/jang_tools/turboquant.
// Created by Jinho Jang (eric@jangq.ai).
//
// The kernel source strings here are byte-identical to the Python kernels
// that have been validated end-to-end on MiniMax M2.7 JANGTQ_2L:
//
//   ../../../../jang/jang-tools/jang_tools/turboquant/hadamard_kernel.py
//   ../../../../jang/jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py
//   ../../../../jang/jang-tools/jang_tools/turboquant/gather_tq_kernel.py
//
// Because we use `MLXFast.metalKernel(...)` (which calls the same
// `mlx_fast_metal_kernel_*` C++ entry points as Python's `mx.fast.metal_kernel`),
// the compiled Metal pipeline is BIT-IDENTICAL to the Python runtime.
//
// What this gives us:
//   - Decode speed exactly matches Python `mlx_lm` for JANGTQ models, because
//     every other op (attention, RMSNorm, RoPE, SDPA, lm_head) already uses
//     vendored mlx-swift kernels which are the same C++ kernels as Python MLX.
//   - All optimizations from the Python side (P3 multi-block Hadamard,
//     P12/P17 thread tiling, P9 vectorized unpack, P15 compile-friendly
//     wrappers) are preserved in the kernel source.
//
// Sweet-spot tile constants (P17, M3 Ultra sweep):
//   - jangtq_fused_gate_up_swiglu : OPT = 10 outputs per thread
//   - jangtq_gather_tq_matmul     : OPT = 20 outputs per thread
//

import Foundation
import MLX

// MARK: - Hadamard multiblock

private let kHadamardMultiblockSource = """
    uint batch_idx = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint threads_per_tg = threads_per_threadgroup.x;

    uint total_d = meta[0];
    uint n_blocks = meta[1];

    // Apple Silicon caps threadgroup memory at 32 KB = 8192 floats. The
    // largest single power-of-2 block we ever decompose into is 8192
    // (e.g., Mistral-Medium-3.5 hidden=12288 → [8192, 4096]; GLM-5.1
    // hidden=6144 → [4096, 2048]; Kimi-K2.6 hidden=7168 → [4096, 2048,
    // 1024]). The shmem only needs to hold ONE block at a time —
    // butterflies are independent per block, and the output is written
    // to global memory before the next block is loaded.
    //
    // Earlier versions of this kernel loaded the entire `total_d` slab
    // into shmem up-front (the original Python prototype's design when
    // total_d ≤ 8192). On Mistral 3.5 (total_d=12288 > 8192) that
    // overran the buffer by 4096 entries, silently corrupting half the
    // rotated activations. Diagnosed via VMLX_MISTRAL3_PROJ_PROBE=1:
    // layer-0 V projection L2 was 4.3× the mxfp4 baseline; after this
    // rewrite it sits within 1.5× (residual 2-bit quant noise).
    //
    // Per-block isolation also matches the Python reference's
    // gather_tq_kernel.py (templated `threadgroup float shmem[in_features]`
    // approach) — they similarly never need >8192 floats since they
    // fuse Hadamard+gather and shmem only holds the current block's
    // post-rotation values.
    threadgroup float shmem[8192];

    uint cum_offset = 0;
    for (uint b = 0; b < n_blocks; b++) {
        uint d_b = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        // Load this block's slice of (x*signs) into shmem[0..d_b].
        for (uint i = tid; i < d_b; i += threads_per_tg) {
            shmem[i] = static_cast<float>(x[batch_idx * total_d + cum_offset + i])
                       * signs[cum_offset + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint ept = (d_b + threads_per_tg - 1u) / threads_per_tg;
        if (ept == 0u) ept = 1u;

        for (uint stage = 0; stage < log_b; stage++) {
            uint h = 1u << stage;
            uint two_h = 2u * h;

            // Stack buffer per thread for butterfly-stage values. For the
            // 8192 block + 1024 threads-per-tg this needs ≥ 8 entries;
            // smaller threads_per_tg → larger ept. Python reference uses
            // 64 for safety — match it.
            float newv[64];
            for (uint k = 0; k < 64; k++) newv[k] = 0.0f;
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    uint block_start = (i_local / two_h) * two_h;
                    uint pos = i_local - block_start;
                    float a = shmem[block_start + pos];
                    if (pos < h) {
                        newv[k] = a + shmem[block_start + pos + h];
                    } else {
                        newv[k] = shmem[block_start + pos - h] - a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < ept; k++) {
                uint i_local = tid * ept + k;
                if (i_local < d_b) {
                    shmem[i_local] = newv[k];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float norm = 1.0f / sqrt(static_cast<float>(d_b));
        for (uint i = tid; i < d_b; i += threads_per_tg) {
            out[batch_idx * total_d + cum_offset + i] = shmem[i] * norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cum_offset += d_b;
    }
"""

// MARK: - Fused gate+up+SwiGLU (P17 OPT=10)

private let kFusedSwiGLUSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 10u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];
    // 2026-05-04 (DSV4 SWA/CSA/HSA correctness pass):
    // `meta[5]` carries the SwiGLU clamp magnitude × 1000 as uint, so
    //   swiglu_limit = float(meta[5]) / 1000.0
    // A value of 0 disables the clamp (ordinary SwiGLU). DeepSeek-V4
    // sets this to 10000 → limit = 10.0, matching the codex_dsv4_fixkit
    // reference. Other models pass 0 → no clamp → byte-identical to the
    // pre-2026-05-04 kernel output.
    uint swiglu_limit_q1000 = meta[5];
    float swiglu_limit = static_cast<float>(swiglu_limit_q1000) * 0.001f;

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc_g[10];
    float acc_u[10];
    #pragma unroll
    for (uint o = 0; o < 10; o++) { acc_g[o] = 0.0f; acc_u[o] = 0.0f; }

    uint expert_base = expert * out_features * packed_cols;
    uint x_off = token_idx * in_features;

    uint n_outs = 10u;
    if (out_idx_0 + 10u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[10], pvu[10];
        #pragma unroll
        for (uint o = 0; o < 10; o++) {
            if (o < n_outs) {
                uint row_off = expert_base + (out_idx_0 + o) * packed_cols + pack_idx;
                pvg[o] = packed_gate[row_off];
                pvu[o] = packed_up[row_off];
            } else {
                pvg[o] = 0u;
                pvu[o] = 0u;
            }
        }

        // 2026-04-26: loop bound MUST be `vals_per_u32` (= 32 / bits)
        // not the hardcoded 16. Correct for bits=2 (vals_per_u32=16)
        // by coincidence but for bits=4 (vals_per_u32=8) the old
        // hardcoded 16 walked PAST the end of each packed uint32 by
        // 8 iterations: shifts 32-60 are out-of-range for uint32
        // right-shift (Metal undefined behaviour), AND `i = i_base + k`
        // for k=8..15 reads input values that belong to the NEXT
        // pack_idx — corrupting both accumulators. Reproduces as
        // garbage multilingual gibberish on Holo3-35B-A3B-JANGTQ4 and
        // Qwen3.6-35B-A3B-JANGTQ4 with suspiciously fast decode rates
        // (compute is short-circuited / corrupted, not skipped).
        // See research/QWEN36-A3B-JANGTQ4-COHERENCE-BUG-2026-04-25.md.
        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_off + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 10; o++) {
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 10; o++) {
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate[expert * out_features + oi]);
            float nu = static_cast<float>(norms_up[expert * out_features + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            // 2026-05-04: optional DSV4-style limited SwiGLU clamp.
            //   gate = min(gate, +limit)        (one-sided)
            //   up   = clamp(up,  -limit, +limit) (two-sided)
            //   y    = silu(gate) * up
            // When `swiglu_limit == 0` (every non-DSV4 caller), this
            // collapses to the original ordinary SwiGLU expression
            // exactly. See codex_dsv4_fixkit/scripts/runtime_dsv4_fixed.py
            // and jang_tools/dsv4/mlx_model.py:_dsv4_swiglu.
            if (swiglu_limit > 0.0f) {
                gv = metal::min(gv, swiglu_limit);
                uv = metal::max(metal::min(uv, swiglu_limit), -swiglu_limit);
            }
            out_act[base_off + oi] = (gv / (1.0f + metal::fast::exp(-gv))) * uv;
        }
    }
"""

// MARK: - Gather TQ matmul (P17 OPT=20)

private let kGatherTQSource = """
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * 20u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[20];
    #pragma unroll
    for (uint o = 0; o < 20; o++) acc[o] = 0.0f;

    uint expert_base = expert * out_features * packed_cols;
    uint x_offset = token_idx * in_features;

    uint n_outs = 20u;
    if (out_idx_0 + 20u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;
        uint pv[20];
        #pragma unroll
        for (uint o = 0; o < 20; o++) {
            pv[o] = (o < n_outs) ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
        }
        // Symmetric fix to the gate/up kernel: loop bound MUST be
        // vals_per_u32 (= 32 / bits), not the hardcoded 16. See the
        // comment in jangtq_fused_gate_up_swiglu_matmul above for the
        // full diagnosis.
        for (uint k = 0; k < vals_per_u32; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_offset + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < 20; o++) {
                float w = codebook[(pv[o] >> shift) & mask];
                acc[o] += xv * w;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < 20; o++) {
        acc[o] = simd_sum(acc[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[expert * out_features + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }
    }
"""

// MARK: - Public kernel access

/// Lazy-built singleton kernels. Each kernel is compiled once via
/// `MLXFast.metalKernel(...)` and cached for the lifetime of the process.
public enum JANGTQKernelLibrary {

    public static let hadamardMultiblock: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_hadamard_multiblock",
        inputNames: ["x", "signs", "meta"],
        outputNames: ["out"],
        source: kHadamardMultiblockSource
    )

    public static let fusedGateUpSwiGLU: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_fused_gate_up_swiglu",
        inputNames: [
            "x_rot", "packed_gate", "norms_gate",
            "packed_up", "norms_up",
            "codebook", "rhs_indices", "meta",
        ],
        outputNames: ["out_act"],
        source: kFusedSwiGLUSource
    )

    public static let gatherTQ: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "jangtq_gather_tq_matmul",
        inputNames: ["x_rot", "packed", "norms", "codebook", "rhs_indices", "meta"],
        outputNames: ["out"],
        source: kGatherTQSource
    )
}

// MARK: - Codebook + signs cache

/// Sign and codebook arrays are deterministic functions of (in_features, seed/bits)
/// computed at quantization time via NumPy PCG64 + Lloyd-Max iteration. They're
/// loaded once at model load from `jangtq_runtime.safetensors` and cached here
/// keyed on `(in_features, seed)` / `(in_features, bits)`.
public final class JANGTQRuntimeCache: @unchecked Sendable {
    public static let shared = JANGTQRuntimeCache()

    private var signsByKey: [String: MLXArray] = [:]
    private var codebookByKey: [String: MLXArray] = [:]
    private let lock = NSLock()

    private init() {}

    public func loadSidecar(from sidecarPath: URL) throws {
        let loaded = try MLX.loadArrays(url: sidecarPath)
        lock.lock()
        defer { lock.unlock() }
        for (name, arr) in loaded {
            if name.hasPrefix("signs.") {
                signsByKey[name] = arr
            } else if name.hasPrefix("codebook.") {
                codebookByKey[name] = arr
            }
        }
    }

    public func signs(inFeatures: Int, seed: Int) -> MLXArray? {
        lock.lock(); defer { lock.unlock() }
        return signsByKey["signs.\(inFeatures).\(seed)"]
    }

    public func codebook(inFeatures: Int, bits: Int) -> MLXArray? {
        lock.lock(); defer { lock.unlock() }
        return codebookByKey["codebook.\(inFeatures).\(bits)"]
    }

    /// Sniff the routed-MoE codebook bits directly from a sidecar
    /// safetensors file WITHOUT fully loading it into the runtime
    /// cache. Uses the `codebook.{inFeatures}.{bits}` key naming
    /// convention to read the actual bit width that was used at
    /// quantization time.
    ///
    /// This is the most reliable signal when the bundle's
    /// `jang_config.json` is missing the routed-expert bits field
    /// (e.g. some Qwen3.6-A3B-JANGTQ4 / Kimi-K2.6 bundles ship only
    /// `quantization.bits=8`, which is the affine non-routed setting,
    /// not the codebook bits). Returns the most-frequent `bits` value
    /// among the codebook keys, or `nil` if the file has no codebook
    /// entries (or doesn't exist).
    public static func sniffCodebookBits(at sidecarPath: URL) -> Int? {
        guard FileManager.default.fileExists(atPath: sidecarPath.path),
              let arrays = try? MLX.loadArrays(url: sidecarPath)
        else { return nil }
        var counts = [Int: Int]()
        for name in arrays.keys where name.hasPrefix("codebook.") {
            // Format: `codebook.{inFeatures}.{bits}`
            let parts = name.split(separator: ".")
            guard parts.count == 3, let bits = Int(parts[2]) else { continue }
            counts[bits, default: 0] += 1
        }
        return counts.max(by: { $0.value < $1.value })?.key
    }
}

/// Detect routed-MoE codebook bits from a JANG bundle's `profile`
/// string field (`JANGTQ4` → 4, `JANGTQ2`/`JANGTQ`/`MXTQ` → 2).
/// Bundle naming convention is empirically reliable: every JANG /
/// JANGTQ converter pre-2026-04 stamped the profile this way.
/// Returns `nil` for unrecognized strings so the caller falls back
/// to the next signal in the resolution chain.
public func jangtqBitsFromProfile(_ profile: String?) -> Int? {
    guard let profile, !profile.isEmpty else { return nil }
    let p = profile.lowercased()
    if p.contains("jangtq4") || p.contains("jangtq_4") || p.contains("jangtq-4") {
        return 4
    }
    if p.contains("jangtq2") || p.contains("jangtq_2") || p.contains("jangtq-2") {
        return 2
    }
    // Bare "jangtq" / "mxtq" historically meant 2-bit.
    if p == "jangtq" || p == "mxtq" {
        return 2
    }
    return nil
}

// MARK: - High-level kernel wrappers (mirror Python `make_*_decode` factories)

public enum JANGTQKernels {

    /// Decompose a non-pow2 dim into a sum of pow2 blocks (largest first).
    public static func decomposePow2(_ dim: Int) -> [Int] {
        var blocks: [Int] = []
        var rem = dim
        while rem > 0 {
            let p = 1 << (Int.bitWidth - 1 - rem.leadingZeroBitCount)
            blocks.append(p)
            rem -= p
        }
        return blocks
    }

    /// Build the `meta` array the multiblock Hadamard kernel expects:
    /// `[total_d, n_blocks, d_b0, log_b0, d_b1, log_b1, ...]`
    public static func makeHadamardMeta(totalDim: Int) -> MLXArray {
        let blocks = decomposePow2(totalDim)
        var meta: [UInt32] = [UInt32(totalDim), UInt32(blocks.count)]
        for d in blocks {
            meta.append(UInt32(d))
            meta.append(UInt32(d.trailingZeroBitCount))
        }
        return MLXArray(meta)
    }

    /// Hadamard rotate `x` (any batch shape with `dim` last). Returns fp32.
    /// `signs` must be shape `(dim,)` fp32.
    ///
    /// Apple Silicon caps threadgroup memory at 32 KB = 8192 floats, so the
    /// per-block Metal kernel can only process blocks up to 8192 elements.
    /// Mistral-Medium-3.5 hits this on `down_proj.in_features=28672` →
    /// `decomposePow2(28672) = [16384, 8192, 4096]`. The 16384-block has
    /// no in-shmem implementation; we instead split it in Swift via the
    /// well-known recursion
    ///     `H_{2n}(u,v) = [H_n((u+v)/√2), H_n((u-v)/√2)]`
    /// applying it once for each "doubling above 8192". The signs are
    /// applied to the original input ONCE before the split (signs are
    /// per-input-coordinate diagonal, so they commute with the split as
    /// long as we don't double-apply), and each leaf-call uses an all-
    /// ones sign vector.
    public static func hadamardRotate(_ x: MLXArray, signs: MLXArray, dim: Int) -> MLXArray {
        let xFlat = x.reshaped([-1, dim]).asType(.float32)
        let batch = xFlat.shape[0]

        let blocks = decomposePow2(dim)
        let maxBlock = blocks.max() ?? dim
        if maxBlock <= 8192 {
            // Fast path: every block fits in shmem — single kernel
            // dispatch processes all blocks back-to-back.
            let meta = makeHadamardMeta(totalDim: dim)
            let tgSize = min(1024, max(32, maxBlock))
            let outArrs = JANGTQKernelLibrary.hadamardMultiblock(
                [xFlat, signs, meta],
                template: nil,
                grid: (tgSize, batch, 1),
                threadGroup: (tgSize, 1, 1),
                outputShapes: [[batch, dim]],
                outputDTypes: [.float32]
            )
            var rot = outArrs[0]
            if x.ndim > 2 || (x.ndim == 2 && x.dim(0) != batch) {
                rot = rot.reshaped(x.shape)
            }
            return rot
        }

        // Slow path (shmem-overflow case): process each pow2 block
        // separately, splitting > 8192 blocks via the H_{2n} recursion.
        var blockOuts: [MLXArray] = []
        var offset = 0
        for d_b in blocks {
            let blockX = xFlat[0..., offset..<(offset + d_b)]
            let blockSigns = signs[offset..<(offset + d_b)]
            blockOuts.append(
                hadamardBlockRecursive(blockX, signs: blockSigns, d_b: d_b))
            offset += d_b
        }
        let merged = MLX.concatenated(blockOuts, axis: -1)
        if x.ndim > 2 || (x.ndim == 2 && x.dim(0) != batch) {
            return merged.reshaped(x.shape)
        }
        return merged
    }

    /// Recursive single-block Hadamard. For `d_b <= 8192` dispatches the
    /// Metal kernel directly. For `d_b > 8192` applies the H_{2n}
    /// recursion in Swift, using all-ones signs in the recursive calls
    /// (signs were consumed by the caller's sign-multiplication step).
    private static func hadamardBlockRecursive(
        _ x: MLXArray, signs: MLXArray, d_b: Int
    ) -> MLXArray {
        if d_b <= 8192 {
            // Use the multi-block kernel with n_blocks=1.
            let meta = makeHadamardMeta(totalDim: d_b)
            let tgSize = min(1024, max(32, d_b))
            let outArrs = JANGTQKernelLibrary.hadamardMultiblock(
                [x, signs, meta],
                template: nil,
                grid: (tgSize, x.shape[0], 1),
                threadGroup: (tgSize, 1, 1),
                outputShapes: [x.shape],
                outputDTypes: [.float32]
            )
            return outArrs[0]
        }
        // Apply signs ONCE to the input. The recursive halves use ones.
        let xSigned = x * signs
        let half = d_b / 2
        let u = xSigned[0..., 0..<half]
        let v = xSigned[0..., half..<d_b]
        let invSqrt2 = MLXArray(Float(1.0 / sqrt(2.0)))
        let a = (u + v) * invSqrt2
        let b = (u - v) * invSqrt2
        let onesSigns = MLXArray.ones([half], dtype: .float32)
        let halfA = hadamardBlockRecursive(a, signs: onesSigns, d_b: half)
        let halfB = hadamardBlockRecursive(b, signs: onesSigns, d_b: half)
        return MLX.concatenated([halfA, halfB], axis: -1)
    }

    /// Fused gate+up+SwiGLU.
    /// - `K` : experts per token (e.g. 8) — becomes `meta[0]` inside the kernel
    ///         so the kernel can compute `token_idx = dispatch_idx / K`.
    /// - `batchTokens` : number of input rows in `xRot` (tokens in the batch).
    ///         Total dispatches in `y` grid = `batchTokens * K`.
    /// - `xRot` shape: `(batchTokens, inFeatures)`
    /// - `rhsIndices` shape: `(batchTokens * K,)` uint32
    /// Returns fp32 of shape `(batchTokens * K, out_features)`.
    public static func fusedGateUpSwiGLU(
        xRot: MLXArray,
        packedGate: MLXArray, normsGate: MLXArray,
        packedUp: MLXArray,   normsUp: MLXArray,
        codebook: MLXArray,
        rhsIndices: MLXArray,
        batchTokens: Int, K: Int,
        inFeatures: Int, outFeatures: Int, bits: Int = 2,
        // 2026-05-04 (DSV4 SWA/CSA/HSA correctness):
        // SwiGLU clamp magnitude. 0.0 (default) preserves the historical
        // ordinary-SwiGLU output bit-for-bit. DSV4 callers must pass 10.0
        // — that activates `silu(min(gate, 10)) * clip(up, -10, 10)` per
        // jang_tools/dsv4/mlx_model.py and codex_dsv4_fixkit/scripts/
        // runtime_dsv4_fixed.py. The kernel encodes this as
        // `meta[5] = round(swigluLimit * 1000)` (uint) and divides by
        // 1000 inside Metal — small enough to fit in a uint32 for the
        // realistic range while being ~1e-3 precise.
        swigluLimit: Float = 0.0
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        let nDispatches = batchTokens * K
        let limitQ1000 = UInt32(max(0, Int((swigluLimit * 1000.0).rounded())))
        let meta = MLXArray([
            UInt32(K), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
            limitQ1000,
        ])
        let opt = 10
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.fusedGateUpSwiGLU(
            [xRot, packedGate, normsGate, packedUp, normsUp,
             codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nDispatches, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nDispatches, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }

    /// Gather TQ matmul in per-row mode (down_proj path).
    /// - `xRot` shape: `(nRows, inFeatures)` — one row per (token, expert) pair.
    /// - `rhsIndices` shape: `(nRows,)` uint32 — expert id for each row.
    /// Returns fp32 of shape `(nRows, outFeatures)`.
    public static func gatherTQ(
        xRot: MLXArray,
        packed: MLXArray, norms: MLXArray,
        codebook: MLXArray, rhsIndices: MLXArray,
        nRows: Int, inFeatures: Int, outFeatures: Int, bits: Int = 2
    ) -> MLXArray {
        let valsPerU32 = 32 / bits
        let packedCols = (inFeatures + valsPerU32 - 1) / valsPerU32
        // Per-row: K_meta = 1, so token_idx = dispatch_idx, k_idx = 0.
        let meta = MLXArray([
            UInt32(1), UInt32(inFeatures), UInt32(outFeatures),
            UInt32(packedCols), UInt32(bits),
        ])
        let opt = 20
        let outGroups = (outFeatures + opt - 1) / opt
        let gridX = outGroups * 32
        let tgX = min(gridX, 256)
        let arr = JANGTQKernelLibrary.gatherTQ(
            [xRot, packed, norms, codebook, rhsIndices, meta],
            template: nil,
            grid: (gridX, nRows, 1),
            threadGroup: (tgX, 1, 1),
            outputShapes: [[nRows, outFeatures]],
            outputDTypes: [.float32]
        )
        return arr[0]
    }
}
