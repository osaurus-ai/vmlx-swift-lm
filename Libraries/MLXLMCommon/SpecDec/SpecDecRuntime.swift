// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/runtime.py (711 lines)
//
// Phase 0 stub — public API only. The integration points (Evaluate.generate,
// BatchEngine.generate) call into this runtime when
// GenerateParameters.draftStrategy is set to .dflash / .ddtree. Phase 1
// implements the DFlash linear-verify path; Phase 2 adds the tree verify.

import Foundation
import MLX

/// Main generation loop for block-diffusion speculative decoding.
///
/// Wraps `draft → tree-build → verify → walk → commit` into a single
/// iterator-style API that produces target tokens. Consumed by
/// `Evaluate.generate` and `BatchEngine.generate` when
/// ``DraftStrategy/usesBlockDiffusion`` is true.
public actor SpecDecRuntime {

    /// Per-runtime stats surfaced in logs + benchmarks.
    public struct Stats: Sendable {
        public var draftForwards: Int = 0
        public var targetForwards: Int = 0
        public var acceptedTokens: Int = 0
        public var bonusTokens: Int = 0
        public var fastPathCommits: Int = 0
        public var treeAwareCommits: Int = 0
        public var slowPathCommits: Int = 0
    }

    /// Configuration at runtime construction.
    public struct Config: Sendable {
        public let strategy: DraftStrategy
        public let parameters: GenerateParameters
        public init(strategy: DraftStrategy, parameters: GenerateParameters) {
            self.strategy = strategy
            self.parameters = parameters
        }
    }

    public let config: Config
    private(set) public var stats: Stats = Stats()

    public init(config: Config) {
        self.config = config
    }

    /// Run the DFlash linear-verify loop.
    ///
    /// Phase 1 implementation target. Drafts a block, verifies it against
    /// the target in one forward, accepts the longest matching prefix,
    /// continues with the bonus token. Tree budget is implicitly 1 — this
    /// is the degenerate case of the DDTree runtime.
    public func runDflashLinear() throws {
        throw SpecDecError.notImplemented("SpecDecRuntime.runDflashLinear — Phase 1")
    }

    /// Run the DDTree tree-verify loop.
    ///
    /// Phase 2 implementation target. Drafts a block of logits, builds a
    /// tree with `branchingBudget` nodes, verifies the full tree in one
    /// target forward via ancestor-mask SDPA, walks the tree against the
    /// target's argmax, commits the accepted path.
    public func runDDTree() throws {
        throw SpecDecError.notImplemented("SpecDecRuntime.runDDTree — Phase 2")
    }
}

/// Errors produced by the SpecDec runtime.
public enum SpecDecError: Error, LocalizedError {

    /// Throws when a stub is invoked before the phase that implements it
    /// has landed.
    case notImplemented(String)

    /// Thrown when the drafter's `config.json` references a target model
    /// family that doesn't match the loaded target.
    case drafterTargetMismatch(drafter: String, target: String)

    /// Thrown when the drafter checkpoint doesn't contain an expected
    /// safetensors key.
    case drafterCheckpointMissingKey(String)

    /// Thrown when the target model doesn't expose the hidden-state hook
    /// the drafter needs to perform KV injection.
    case targetDoesNotSupportHiddenStateCapture

    public var errorDescription: String? {
        switch self {
        case .notImplemented(let msg):
            return "SpecDec: not implemented yet — \(msg)"
        case .drafterTargetMismatch(let drafter, let target):
            return "SpecDec: drafter \(drafter) is not trained against target \(target)"
        case .drafterCheckpointMissingKey(let k):
            return "SpecDec: drafter checkpoint is missing required key '\(k)'"
        case .targetDoesNotSupportHiddenStateCapture:
            return "SpecDec: target model does not expose a hidden-state capture hook (required for DFlash KV injection)"
        }
    }
}
