// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/verify.py (810 lines)
//
// Phase 0 stub — public API only. Phase 2 lands the real tree-verify
// forward pass, Phase 3 adds per-node recurrent state forking.

import Foundation
import MLX

/// Result of one tree-verify forward pass.
public struct TreeVerifyResult: @unchecked Sendable {

    /// Posterior (greedy argmax) token for each tree position. Length =
    /// `compiled.treeSize`. Consumed by ``TreeBuilder/followVerifiedTree(childMaps:posteriorTokens:)``.
    public let posteriorTokens: [Int32]

    /// Target model logits for every tree position. Shape:
    /// `(1, treeSize, vocab)`. Retained so sampling strategies other than
    /// greedy argmax can swap in later.
    public let logits: MLXArray

    /// Final-accepted-node per-layer recurrent state snapshots, keyed by
    /// layer index. `nil` for pure-attention target models. Populated in
    /// Phase 3 when hybrid SSM support lands.
    public let recurrentSnapshots: [Int: RecurrentSnapshot]?

    public init(
        posteriorTokens: [Int32],
        logits: MLXArray,
        recurrentSnapshots: [Int: RecurrentSnapshot]? = nil
    ) {
        self.posteriorTokens = posteriorTokens
        self.logits = logits
        self.recurrentSnapshots = recurrentSnapshots
    }
}

/// Per-layer recurrent state captured during a tree-verify forward.
///
/// For GatedDeltaNet / Mamba / hybrid-SSM families we need to capture state
/// at every tree node (not just the final DFS position) so
/// ``SpecDecCache/treeAwarePathCommit(cacheEntries:prefixLen:acceptedIndices:treeCacheState:)``
/// can install the exact state the accepted path produced.
public struct RecurrentSnapshot: @unchecked Sendable {

    /// Convolutional state. Shape: `(treeSize, conv_kernel, hidden)`.
    public let convStates: MLXArray

    /// Recurrent / SSM state. Shape: `(treeSize, state_dim, hidden)`.
    public let states: MLXArray

    public init(convStates: MLXArray, states: MLXArray) {
        self.convStates = convStates
        self.states = states
    }
}

/// Runs one target-model forward pass over the whole compiled tree.
///
/// Phase 0 stub. Phase 2 lands the attention-only path (byte-identical vs
/// autoregressive at temp 0). Phase 3 adds hybrid-SSM per-node forking.
public enum TreeVerify {

    /// - Parameters:
    ///   - target: the target ``LanguageModel``.
    ///   - compiled: from ``TreeCompile/compile(tree:rootTokenID:prefixLen:)``.
    ///   - cache: per-layer target caches. Will be written to in-place.
    ///     Callers must have taken a ``SpecDecCache/snapshotCaches(_:)``
    ///     beforehand so they can restore on reject.
    ///   - prefixLen: number of tokens already in `cache` (context length).
    public static func verifyForward(
        target: any LanguageModel,
        compiled: CompiledTree,
        cache: [KVCache],
        prefixLen: Int
    ) throws -> TreeVerifyResult {
        throw SpecDecError.notImplemented(
            "TreeVerify.verifyForward — Phase 2 (attention-only), Phase 3 (hybrid SSM)"
        )
    }
}
