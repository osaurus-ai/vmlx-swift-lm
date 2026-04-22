// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Describes how a transformer model is split across pipeline-parallel
/// stages: which layer range runs on this stage, whether this stage
/// owns the token embedding, and whether it owns the final norm + lm
/// head + sampler.
///
/// Phase 1 scope: uniform layer partitioning across stages, with the
/// remainder distributed to the lowest-rank stages. Stage 0 always owns
/// the embedding; the last stage always owns the final norm + lm head.
///
/// Phase 2 will extend this to heterogeneous partitions that weight
/// layers by per-device memory + compute — see
/// ``Libraries/MLXLMCommon/Distributed/DISTRIBUTED-DESIGN.md``.
public struct ModelPartition: Sendable, Equatable {

    /// Zero-based rank of this stage within the pipeline. Range:
    /// `0 ..< stageCount`.
    public let stageIndex: Int

    /// Total number of stages in the pipeline (= world size when the
    /// whole world is one pipeline group; < world size when composed
    /// with tensor parallelism in Phase 5).
    public let stageCount: Int

    /// Half-open `[start, end)` range of transformer layer indices this
    /// stage owns. `end - start` is the number of layers on this stage.
    public let layerRange: Range<Int>

    /// Total number of layers the model has, across all stages. Kept on
    /// the partition for convenience — pipelines often need this for
    /// bounds checking and KV-cache sizing.
    public let totalLayers: Int

    /// `true` when this stage owns the token embedding + any pre-layer
    /// embedding scaling. Stage 0 only. The first stage takes tokens as
    /// input and produces hidden states; later stages take hidden states
    /// from the previous stage's `send`.
    public var isFirstStage: Bool { stageIndex == 0 }

    /// `true` when this stage owns the final norm, the lm_head, and
    /// therefore the sampler. The last stage produces logits and the
    /// sampled next-token ID.
    public var isLastStage: Bool { stageIndex == stageCount - 1 }

    /// Convenience: the rank this stage will send its output to
    /// (`stageIndex + 1`). Undefined for the last stage — callers
    /// must gate via ``isLastStage``.
    public var nextRank: Int { stageIndex + 1 }

    /// Convenience: the rank this stage receives its input from
    /// (`stageIndex - 1`). Undefined for the first stage — callers
    /// must gate via ``isFirstStage``.
    public var prevRank: Int { stageIndex - 1 }

    public init(
        stageIndex: Int,
        stageCount: Int,
        layerRange: Range<Int>,
        totalLayers: Int
    ) {
        precondition(stageCount >= 1, "stageCount must be at least 1")
        precondition(
            stageIndex >= 0 && stageIndex < stageCount,
            "stageIndex \(stageIndex) out of range [0, \(stageCount))"
        )
        precondition(
            layerRange.lowerBound >= 0 && layerRange.upperBound <= totalLayers,
            "layerRange \(layerRange) out of bounds for totalLayers \(totalLayers)"
        )
        self.stageIndex = stageIndex
        self.stageCount = stageCount
        self.layerRange = layerRange
        self.totalLayers = totalLayers
    }

    /// Build a uniform partition for this stage.
    ///
    /// Layers are divided evenly across stages. If `totalLayers` is not
    /// divisible by `stageCount`, the remainder is distributed to the
    /// lowest-rank stages — stage 0 gets one extra layer first, then
    /// stage 1, etc. Example: 7 layers split 3 ways yields
    /// `[0..<3, 3..<5, 5..<7]`.
    ///
    /// This matches the exo placement strategy's uniform default. Phase 2
    /// will add a memory-weighted variant for heterogeneous hardware.
    public static func uniform(
        totalLayers: Int,
        stageCount: Int,
        stageIndex: Int
    ) -> ModelPartition {
        precondition(totalLayers >= stageCount,
            "totalLayers (\(totalLayers)) must be >= stageCount (\(stageCount))")

        let base = totalLayers / stageCount
        let remainder = totalLayers % stageCount

        // Stages with index < remainder each take one extra layer so the
        // total adds up to `totalLayers`. Compute our start by summing
        // the lengths of all earlier stages.
        let start = stageIndex * base + min(stageIndex, remainder)
        let extra = stageIndex < remainder ? 1 : 0
        let end = start + base + extra

        return ModelPartition(
            stageIndex: stageIndex,
            stageCount: stageCount,
            layerRange: start ..< end,
            totalLayers: totalLayers
        )
    }

    /// Build every partition in one pipeline. Returns `stageCount` entries,
    /// one per stage, each describing what that stage owns. Useful for
    /// inspection / logging / CI checks — the launcher itself only needs
    /// the one `ModelPartition` for its own rank.
    public static func uniformAll(
        totalLayers: Int,
        stageCount: Int
    ) -> [ModelPartition] {
        (0 ..< stageCount).map { stageIndex in
            uniform(
                totalLayers: totalLayers,
                stageCount: stageCount,
                stageIndex: stageIndex
            )
        }
    }
}
