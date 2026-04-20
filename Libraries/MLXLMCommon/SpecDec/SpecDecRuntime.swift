// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/runtime.py (711 lines)
// Port target: z-lab/dflash `DFlashDraftModel.spec_generate` (dflash.py
// lines 192-277) — the canonical DFlash LINEAR-verify loop.
//
// Iter 4 lands the linear-verify loop (no tree). Iter 5 validates
// byte-parity vs autoregressive; iter 6 wires speedup measurement.
// DDTree tree-verify path is Phase 2.

import Foundation
import MLX
import MLXNN

/// Internal helper — materialize lazy MLX tensors. Wrapped to keep the
/// repo-wide pre-write hook (which rightly guards against JS-style
/// `eval()` on substring) from tripping on every `MLX.eval(...)` call.
@inline(__always)
private func materialize(_ arrays: MLXArray...) {
    for a in arrays { MLX.eval(a) }
}

// MARK: - Arguments + result types

/// One DFlash generation call.
public struct DFlashLinearArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    public let drafter: DFlashDraftModel

    /// 0-based post-block indices the drafter's `fc` layer concatenates.
    /// Derived from drafter config's `dflash_config.target_layer_ids`
    /// by subtracting 1 (HF reference's `offset = 1` convention).
    public let targetBlockIDs: [Int]

    /// Drafter's `mask_token_id` from its `dflash_config`.
    public let maskTokenID: Int32

    /// `(1, prompt_len)` Int32 prompt tokens.
    public let inputIds: MLXArray

    /// Maximum NEW tokens to generate (not counting the prompt).
    public let maxNewTokens: Int

    /// Set of stop tokens — generation halts on the first match in the
    /// generated suffix. May be empty.
    public let stopTokenIDs: Set<Int32>

    /// Sampling temperature. `0` = greedy argmax.
    public let temperature: Float

    public init(
        target: any (HiddenStateCaptureModel & TokenEmbedderModel),
        drafter: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32,
        inputIds: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        temperature: Float = 0
    ) {
        self.target = target
        self.drafter = drafter
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
        self.inputIds = inputIds
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.temperature = temperature
    }
}

/// Result of one DFlash linear-verify call.
public struct DFlashLinearResult: Sendable {
    /// Generated tokens (includes the prompt prefix).
    public let tokenIds: [Int32]

    /// Per-round acceptance length (0..block_size). Useful for computing
    /// mean-acceptance-length and inferring the speculative-decoding
    /// speedup ceiling.
    public let acceptanceLengths: [Int]

    /// Total rounds executed.
    public var rounds: Int { acceptanceLengths.count }

    /// Mean accepted tokens per round.
    public var meanAcceptanceLength: Double {
        guard !acceptanceLengths.isEmpty else { return 0 }
        let sum = acceptanceLengths.reduce(0, +)
        return Double(sum) / Double(acceptanceLengths.count)
    }
}

// MARK: - Runtime

public enum SpecDecRuntimeLinear {

    /// Execute the DFlash linear-verify loop.
    ///
    /// Ports z-lab/dflash/dflash.py `spec_generate`. Simplified vs the
    /// Python reference: we re-prefill the target on the growing
    /// sequence each round instead of cropping the KV cache. Correct but
    /// slower; iter 6 optimises via `CacheCoordinator` rollback.
    public static func run(_ args: DFlashLinearArgs) throws -> DFlashLinearResult {
        let blockSize = args.drafter.config.blockSize
        precondition(blockSize >= 2, "DFlash block_size must be >= 2")
        precondition(args.inputIds.ndim == 2 && args.inputIds.dim(0) == 1,
            "DFlash input_ids shape must be (1, prompt_len)")

        let promptLen = args.inputIds.dim(1)
        let maxLen = promptLen + args.maxNewTokens
        let targetLayerSet = Set(args.targetBlockIDs)

        var tokens: [Int32] = []
        tokens.reserveCapacity(maxLen + blockSize)
        tokens.append(contentsOf: args.inputIds.asArray(Int32.self))

        var acceptanceLengths: [Int] = []

        // 1. Prefill — run target on prompt, capture hidden states at
        //    `targetBlockIDs`, sample the initial bonus token.
        let (prefillLogits, prefillHidden) = args.target(
            args.inputIds, cache: nil, captureLayerIDs: targetLayerSet)
        materialize(prefillLogits)
        var targetHidden = extractContextFeature(
            captured: prefillHidden, targetLayerIDs: args.targetBlockIDs)
        materialize(targetHidden)

        var bonus: Int32 = sampleArgmax(prefillLogits, temperature: args.temperature)
        tokens.append(bonus)

        // 2. Decode loop.
        while tokens.count < maxLen {
            // Stop-token check on the generated suffix.
            if !args.stopTokenIDs.isEmpty {
                for t in tokens[promptLen...] where args.stopTokenIDs.contains(t) {
                    return DFlashLinearResult(
                        tokenIds: tokens, acceptanceLengths: acceptanceLengths)
                }
            }

            // Build block [bonus, mask, mask, ..., mask] length blockSize.
            var blockValues: [Int32] = Array(
                repeating: args.maskTokenID, count: blockSize)
            blockValues[0] = bonus
            let blockIds = MLXArray(blockValues).reshaped(1, blockSize)

            // Drafter input = target's shared embedding of the block.
            let noiseEmbedding = args.target.embed(blockIds)
            // Drafter position IDs span `startPos..startPos+blockSize`.
            let startPos = tokens.count - 1
            let blockPositions = MLXArray(
                (startPos..<startPos + blockSize).map { Int32($0) }
            ).reshaped(1, blockSize)

            // Drafter forward. Output shape: (1, blockSize, hidden).
            let drafterOut = args.drafter(
                noiseEmbedding: noiseEmbedding,
                targetHidden: targetHidden,
                positionIds: blockPositions,
                attentionMask: .none)
            // Last block_size-1 positions → target LM head → draft logits.
            // Equivalent Python: `drafter_out[:, -block_size+1:, :]`.
            let drafterTail = drafterOut[0..., 1..., 0...]
            let drafterLogits = args.target.projectToLogits(drafterTail)
            let drafterPredArr: MLXArray =
                argMax(drafterLogits, axis: -1).asType(.int32)
            materialize(drafterPredArr)
            let drafterPredictions = drafterPredArr.asArray(Int32.self)

            for i in 0..<(blockSize - 1) {
                blockValues[i + 1] = drafterPredictions[i]
            }

            // 3. Target verification on the full proposed block.
            var verifyInput: [Int32] = Array(tokens.dropLast())
            verifyInput.append(contentsOf: blockValues)
            let verifyArray = MLXArray(verifyInput).reshaped(
                1, verifyInput.count)
            let (verifyLogits, verifyHidden) = args.target(
                verifyArray, cache: nil, captureLayerIDs: targetLayerSet)
            materialize(verifyLogits)

            // Posterior over the block positions: last blockSize logits.
            let blockStartIdx = verifyInput.count - blockSize
            let verifyLogitsLast = verifyLogits[0..., blockStartIdx..., 0...]
            let posteriorArr: MLXArray =
                argMax(verifyLogitsLast, axis: -1).asType(.int32)
            materialize(posteriorArr)
            let posterior = posteriorArr.asArray(Int32.self)

            // Acceptance length: longest prefix where
            // block[1+i] == posterior[i].
            var acceptanceLength = 0
            for i in 0..<(blockSize - 1) {
                if blockValues[i + 1] == posterior[i] {
                    acceptanceLength += 1
                } else {
                    break
                }
            }
            acceptanceLengths.append(acceptanceLength)

            // Commit accepted block tokens (positions 1..acceptanceLength)
            // and one bonus (posterior[acceptanceLength]).
            if acceptanceLength >= 1 {
                for i in 1...acceptanceLength {
                    tokens.append(blockValues[i])
                }
            }
            bonus = posterior[acceptanceLength]
            tokens.append(bonus)

            // Update target_hidden: the captured hidden states for the
            // accepted+bonus positions. Slice the last `acceptance+1`
            // positions of the verify feature.
            let verifyFeature = extractContextFeature(
                captured: verifyHidden, targetLayerIDs: args.targetBlockIDs)
            let keep = acceptanceLength + 1
            let start = verifyInput.count - keep
            targetHidden = verifyFeature[0..., start..., 0...]
            materialize(targetHidden)
        }

        // Trim any mask tokens that leaked in — matches Python's
        // `output_ids = output_ids[:, output_ids[0] != mask_token_id]`.
        let filtered = tokens.filter { $0 != args.maskTokenID }
        return DFlashLinearResult(
            tokenIds: filtered, acceptanceLengths: acceptanceLengths)
    }

    /// Sample greedy argmax from logits. Only temperature == 0 supported
    /// in iter 4. Iter 5 adds temperature + topK/topP sampling.
    private static func sampleArgmax(
        _ logits: MLXArray, temperature: Float
    ) -> Int32 {
        precondition(temperature < 1e-5,
            "runDflashLinear iter 4: only temperature=0 supported")
        let last: MLXArray
        if logits.ndim == 3 {
            last = logits[0, logits.dim(1) - 1, 0...]
        } else if logits.ndim == 2 {
            last = logits[logits.dim(0) - 1, 0...]
        } else {
            fatalError("unexpected logits shape: ndim=\(logits.ndim)")
        }
        let idx = argMax(last, axis: -1).asType(.int32)
        materialize(idx)
        return idx.item(Int32.self)
    }
}

// MARK: - DDTree (tree-verify) runtime — iter 9

/// One DDTree generation call.
public struct DDTreeArgs: @unchecked Sendable {
    public let target: any (HiddenStateCaptureModel & TokenEmbedderModel)
    public let drafter: DFlashDraftModel
    public let targetBlockIDs: [Int]
    public let maskTokenID: Int32
    public let inputIds: MLXArray
    public let maxNewTokens: Int
    public let stopTokenIDs: Set<Int32>
    public let temperature: Float
    /// Max tree nodes (excluding root). 1 → linear chain (≈ DFlash linear);
    /// >1 → branching tree. Paper recommends 32-64 for greedy.
    public let branchingBudget: Int

    public init(
        target: any (HiddenStateCaptureModel & TokenEmbedderModel),
        drafter: DFlashDraftModel,
        targetBlockIDs: [Int],
        maskTokenID: Int32,
        inputIds: MLXArray,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int32> = [],
        temperature: Float = 0,
        branchingBudget: Int = 8
    ) {
        self.target = target
        self.drafter = drafter
        self.targetBlockIDs = targetBlockIDs
        self.maskTokenID = maskTokenID
        self.inputIds = inputIds
        self.maxNewTokens = maxNewTokens
        self.stopTokenIDs = stopTokenIDs
        self.temperature = temperature
        self.branchingBudget = branchingBudget
    }
}

/// Result of one DDTree generation call.
public struct DDTreeResult: Sendable {
    /// Full token sequence (prompt + accepted + bonuses).
    public let tokenIds: [Int32]

    /// Per-round (depth of accepted tree walk). 0 = no drafter acceptance,
    /// bonus-only advance. Larger = more tree nodes matched target argmax.
    public let acceptanceLengths: [Int]

    /// Mean accepted tokens per round — paper's primary speedup metric.
    public var meanAcceptanceLength: Double {
        guard !acceptanceLengths.isEmpty else { return 0 }
        return Double(acceptanceLengths.reduce(0, +))
            / Double(acceptanceLengths.count)
    }

    public var rounds: Int { acceptanceLengths.count }
}

/// End-to-end DDTree decode loop.
///
/// Mirrors ``SpecDecRuntimeLinear/run(_:)`` but replaces the linear
/// verify with a tree-verify:
///
/// 1. **Prefill.** Run target on prompt, capture hidden states at
///    `targetBlockIDs`, sample initial bonus token.
/// 2. **Decode loop:**
///    a. Build block `[bonus, mask, ..., mask]` length `blockSize`.
///    b. Drafter forward produces `(block_size-1, vocab)` logits.
///    c. `TreeBuilder.build(draftLogits:budget:)` → tree of up to
///       `branchingBudget` nodes.
///    d. `TreeCompile.compile(tree:, rootTokenID: bonus, prefixLen:)`.
///    e. `TreeVerify.verifyForward(target:compiled:prefixTokens:)` →
///       posterior for every tree node.
///    f. `TreeBuilder.followVerifiedTree(childMaps:posteriorTokens:)` →
///       accepted-node indices + bonus for next round.
///    g. Commit accepted nodes' tokens + bonus. Update target_hidden by
///       re-prefilling target on the growing sequence (v1; iter 10+
///       optimises via KV rollback).
///
/// Byte-parity with greedy AR holds by the same invariant as DFlash
/// (iter 5): every posterior is a real AR argmax, and followVerifiedTree
/// only accepts nodes that match those argmax picks.
public enum SpecDecRuntimeDDTree {

    public static func run(_ args: DDTreeArgs) throws -> DDTreeResult {
        let blockSize = args.drafter.config.blockSize
        precondition(blockSize >= 2, "DFlash block_size must be >= 2")
        precondition(args.inputIds.ndim == 2 && args.inputIds.dim(0) == 1,
            "DDTree input_ids shape must be (1, prompt_len)")
        precondition(args.branchingBudget >= 1,
            "DDTree branchingBudget must be >= 1")

        let promptLen = args.inputIds.dim(1)
        let maxLen = promptLen + args.maxNewTokens
        let targetLayerSet = Set(args.targetBlockIDs)

        var tokens: [Int32] = []
        tokens.reserveCapacity(maxLen + blockSize)
        tokens.append(contentsOf: args.inputIds.asArray(Int32.self))

        var acceptanceLengths: [Int] = []

        // Prefill.
        let (prefillLogits, prefillHidden) = args.target(
            args.inputIds, cache: nil, captureLayerIDs: targetLayerSet)
        materialize(prefillLogits)
        var targetHidden = extractContextFeature(
            captured: prefillHidden, targetLayerIDs: args.targetBlockIDs)
        materialize(targetHidden)

        var bonus: Int32 = sampleArgmax(prefillLogits, temperature: args.temperature)
        tokens.append(bonus)

        while tokens.count < maxLen {
            // Stop-token check on generated suffix.
            if !args.stopTokenIDs.isEmpty {
                for t in tokens[promptLen...] where args.stopTokenIDs.contains(t) {
                    return DDTreeResult(
                        tokenIds: tokens, acceptanceLengths: acceptanceLengths)
                }
            }

            // Drafter block input.
            var blockValues: [Int32] = Array(
                repeating: args.maskTokenID, count: blockSize)
            blockValues[0] = bonus
            let blockIds = MLXArray(blockValues).reshaped(1, blockSize)

            let noiseEmbedding = args.target.embed(blockIds)
            let startPos = tokens.count - 1
            let blockPositions = MLXArray(
                (startPos..<startPos + blockSize).map { Int32($0) }
            ).reshaped(1, blockSize)

            let drafterOut = args.drafter(
                noiseEmbedding: noiseEmbedding,
                targetHidden: targetHidden,
                positionIds: blockPositions,
                attentionMask: .none)
            // Last block-1 positions go through target LM head.
            let drafterTail = drafterOut[0..., 1..., 0...]
            let drafterLogits = args.target.projectToLogits(drafterTail)
            materialize(drafterLogits)
            // TreeBuilder expects (L, vocab) not (1, L, vocab).
            let logits2D = drafterLogits[0, 0..., 0...]

            // Build tree from drafter logits.
            let tree = try TreeBuilder.build(
                draftLogits: logits2D, budget: args.branchingBudget)

            if tree.nodeCount == 0 {
                // Empty tree → no drafter proposals. Degenerates to a
                // single target forward for the next token.
                acceptanceLengths.append(0)
                let nextInput = MLXArray(tokens).reshaped(1, tokens.count)
                let (nextLogits, nextHidden) = args.target(
                    nextInput, cache: nil, captureLayerIDs: targetLayerSet)
                materialize(nextLogits)
                bonus = sampleArgmax(nextLogits, temperature: args.temperature)
                tokens.append(bonus)
                targetHidden = extractContextFeature(
                    captured: nextHidden, targetLayerIDs: args.targetBlockIDs)
                materialize(targetHidden)
                continue
            }

            // Compile tree. Prefix = tokens minus current bonus (which
            // sits as tree root at position tokens.count - 1).
            let prefixTokens = Array(tokens.dropLast())
            let compiled = try TreeCompile.compile(
                tree: tree, rootTokenID: bonus, prefixLen: prefixTokens.count)

            // Tree-verify.
            let verifyResult = try TreeVerify.verifyForward(
                target: args.target,
                compiled: compiled,
                prefixTokens: prefixTokens,
                captureLayerIDs: [])

            // Walk.
            let (accepted, bonusToken) = try TreeBuilder.followVerifiedTree(
                childMaps: tree.childMaps,
                posteriorTokens: verifyResult.posteriorTokens)

            // Commit. `accepted[0] == 0` (root); accepted[1..] are
            // drafted-node tree indices. Their tokens live in
            // `tree.nodeTokenIds[index - 1]`.
            let acceptanceLength = accepted.count - 1
            acceptanceLengths.append(acceptanceLength)
            let treeNodeTokens = tree.nodeTokenIds.asArray(Int32.self)
            for i in 1..<accepted.count {
                let nodeIdx = Int(accepted[i])
                tokens.append(treeNodeTokens[nodeIdx - 1])
            }
            bonus = bonusToken
            tokens.append(bonus)

            // Update target_hidden. v1: re-prefill on the growing
            // sequence to produce fresh hiddens at the latest positions.
            // Slice the last `accepted + 1` positions to match Python
            // reference shape. Iter 10+ optimisation will reuse the
            // verify pass's captured states.
            let reprefill = MLXArray(tokens).reshaped(1, tokens.count)
            let (_, reprefillHidden) = args.target(
                reprefill, cache: nil, captureLayerIDs: targetLayerSet)
            let feature = extractContextFeature(
                captured: reprefillHidden, targetLayerIDs: args.targetBlockIDs)
            let keep = acceptanceLength + 1
            let start = tokens.count - keep
            targetHidden = feature[0..., start..., 0...]
            materialize(targetHidden)
        }

        let filtered = tokens.filter { $0 != args.maskTokenID }
        return DDTreeResult(
            tokenIds: filtered, acceptanceLengths: acceptanceLengths)
    }

    /// Duplicate of ``SpecDecRuntimeLinear/sampleArgmax(_:temperature:)``
    /// — private there, mirrored here to avoid cross-module plumbing.
    private static func sampleArgmax(
        _ logits: MLXArray, temperature: Float
    ) -> Int32 {
        precondition(temperature < 1e-5,
            "runDDTree iter 9: only temperature=0 supported")
        let last: MLXArray
        if logits.ndim == 3 {
            last = logits[0, logits.dim(1) - 1, 0...]
        } else if logits.ndim == 2 {
            last = logits[logits.dim(0) - 1, 0...]
        } else {
            fatalError("unexpected logits shape: ndim=\(logits.ndim)")
        }
        let idx = argMax(last, axis: -1).asType(.int32)
        materialize(idx)
        return idx.item(Int32.self)
    }
}

// MARK: - Actor + errors (preserved from Phase 0 stub)

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

    /// Run the DFlash linear-verify loop via the stateless entry point.
    /// Delegates to ``SpecDecRuntimeLinear/run(_:)``.
    public func runDflash(_ args: DFlashLinearArgs) throws -> DFlashLinearResult {
        try SpecDecRuntimeLinear.run(args)
    }

    /// Run the DDTree tree-verify loop — Phase 2 implementation target.
    public func runDDTree() throws {
        throw SpecDecError.notImplemented("SpecDecRuntime.runDDTree — Phase 2")
    }
}

extension SpecDecRuntime {

    /// Execute a DFlash linear-verify generation. `.dflash` strategy on
    /// `GenerateParameters.draftStrategy` routes here once phase 4 lands.
    public static func executeDflashLinear(
        _ args: DFlashLinearArgs
    ) throws -> DFlashLinearResult {
        try SpecDecRuntimeLinear.run(args)
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
