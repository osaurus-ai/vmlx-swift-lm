// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/kernels.py (234 lines)
//
// Phase 0 stub — declarations only. Phase 3 lands Metal shaders for
// GatedDeltaNet per-node recurrence.

import Foundation
import MLX

/// Custom Metal kernels for tree-aware speculative decoding.
///
/// Phase 0 stub. Phase 3 will ship Metal shaders for GatedDeltaNet
/// per-node recurrence, so hybrid-SSM target models (Qwen 3.5 / 3.6,
/// Nemotron-H) see tree speedups.
///
/// The reference [kernels.py](https://github.com/humanrouter/ddtree-mlx/blob/main/ddtree_mlx/kernels.py)
/// implements:
/// - Per-position GatedDeltaNet update that branches on tree parent.
/// - Parallel sort/scatter for per-node state scatter after a tree forward.
/// - A batched-reshape RoPE apply that supports the tree's per-token
///   positions (not the scalar-offset RoPE the rest of vmlx uses).
public enum SpecDecKernels {

    /// Apply RoPE at per-token positions (the "batch-reshape trick" from
    /// verify.py). Used by the tree-verify forward to give each tree node
    /// the correct absolute-position RoPE even though they're packed in a
    /// single `(1, N+1, hidden)` tensor rather than laid out sequentially.
    ///
    /// Phase 2 implements this in pure MLX. Phase 3 may swap in a custom
    /// Metal kernel if pure MLX isn't fast enough on M-series.
    public static func ropePerToken(
        hidden: MLXArray,
        positions: MLXArray,
        ropeTheta: Float,
        headDim: Int
    ) throws -> MLXArray {
        throw SpecDecError.notImplemented("SpecDecKernels.ropePerToken — Phase 2")
    }

    /// Per-node GatedDeltaNet recurrent update that respects tree parent
    /// structure (each child's state flows from its parent's state, not
    /// the previous DFS position).
    ///
    /// Reference: `kernels.py::gated_delta_net_tree_forward` in
    /// humanrouter/ddtree-mlx.
    public static func gatedDeltaNetTreeForward(
        x: MLXArray,
        state: MLXArray,
        parents: MLXArray,
        dfsOrder: MLXArray
    ) throws -> (output: MLXArray, nextState: MLXArray) {
        throw SpecDecError.notImplemented(
            "SpecDecKernels.gatedDeltaNetTreeForward — Phase 3 hybrid SSM"
        )
    }
}
