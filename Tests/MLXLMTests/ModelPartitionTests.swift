// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("ModelPartition uniform partitioning")
struct ModelPartitionTests {

    @Test("single stage gets all layers")
    func singleStage() {
        let p = ModelPartition.uniform(
            totalLayers: 24, stageCount: 1, stageIndex: 0)
        #expect(p.layerRange == 0 ..< 24)
        #expect(p.isFirstStage)
        #expect(p.isLastStage)
    }

    @Test("evenly divisible split")
    func evenSplit() {
        // 24 layers / 2 stages = 12 each.
        let p0 = ModelPartition.uniform(
            totalLayers: 24, stageCount: 2, stageIndex: 0)
        let p1 = ModelPartition.uniform(
            totalLayers: 24, stageCount: 2, stageIndex: 1)
        #expect(p0.layerRange == 0 ..< 12)
        #expect(p1.layerRange == 12 ..< 24)
        #expect(p0.isFirstStage && !p0.isLastStage)
        #expect(!p1.isFirstStage && p1.isLastStage)
    }

    @Test("remainder distributed to low-index stages")
    func remainderSplit() {
        // 7 layers / 3 stages = base 2, remainder 1.
        // stage 0 gets 3 (base + 1), stages 1 and 2 get 2 each.
        let p0 = ModelPartition.uniform(
            totalLayers: 7, stageCount: 3, stageIndex: 0)
        let p1 = ModelPartition.uniform(
            totalLayers: 7, stageCount: 3, stageIndex: 1)
        let p2 = ModelPartition.uniform(
            totalLayers: 7, stageCount: 3, stageIndex: 2)
        #expect(p0.layerRange == 0 ..< 3)
        #expect(p1.layerRange == 3 ..< 5)
        #expect(p2.layerRange == 5 ..< 7)
    }

    @Test("larger remainder — 10 layers / 3 stages")
    func remainderTenThree() {
        // 10 / 3 = base 3, remainder 1. Stage 0 gets 4, others 3.
        let parts = ModelPartition.uniformAll(totalLayers: 10, stageCount: 3)
        #expect(parts[0].layerRange == 0 ..< 4)
        #expect(parts[1].layerRange == 4 ..< 7)
        #expect(parts[2].layerRange == 7 ..< 10)
    }

    @Test("every layer covered exactly once")
    func coverageProperty() {
        // Property test: the union of every stage's range equals
        // [0, totalLayers) with no overlap.
        for (total, count) in [(24, 2), (24, 3), (24, 4),
                               (7, 3), (10, 3), (48, 8),
                               (1, 1), (100, 7), (13, 5)] {
            let parts = ModelPartition.uniformAll(
                totalLayers: total, stageCount: count)
            let covered = parts.flatMap { Array($0.layerRange) }
            #expect(covered.count == total)
            #expect(Set(covered) == Set(0 ..< total))
        }
    }

    @Test("nextRank / prevRank helpers")
    func neighborHelpers() {
        let p0 = ModelPartition.uniform(
            totalLayers: 12, stageCount: 3, stageIndex: 0)
        let p1 = ModelPartition.uniform(
            totalLayers: 12, stageCount: 3, stageIndex: 1)
        #expect(p0.nextRank == 1)
        #expect(p1.prevRank == 0)
        #expect(p1.nextRank == 2)
    }

    @Test("uniformAll returns stageCount entries")
    func uniformAllCount() {
        let parts = ModelPartition.uniformAll(totalLayers: 48, stageCount: 4)
        #expect(parts.count == 4)
        #expect(parts.map(\.stageIndex) == [0, 1, 2, 3])
    }

    @Test("first / last stage flags are correct")
    func firstLastFlags() {
        let parts = ModelPartition.uniformAll(totalLayers: 12, stageCount: 4)
        #expect(parts[0].isFirstStage && !parts[0].isLastStage)
        #expect(!parts[1].isFirstStage && !parts[1].isLastStage)
        #expect(!parts[2].isFirstStage && !parts[2].isLastStage)
        #expect(!parts[3].isFirstStage && parts[3].isLastStage)
    }
}
