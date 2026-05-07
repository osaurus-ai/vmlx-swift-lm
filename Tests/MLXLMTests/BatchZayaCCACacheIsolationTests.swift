// BatchZayaCCACache per-slot isolation tests — gather/scatter preserves
// slot identity; B=2 update keeps independent KV histories; mask uses
// per-slot effective key lengths.

import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Suite("BatchZayaCCACache per-slot isolation", .serialized)
struct BatchZayaCCACacheIsolationTests {

    @Test("gatherCCA stacks slot-local state along batch dim")
    func gatherStacksSlots() {
        let s0 = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        let s1 = ZayaCCACache(batchSize: 1, convChannels: 4, hiddenSize: 8)
        s0.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 4)
        s1.writeCCA(
            conv: MLXArray.ones([1, 4, 2], dtype: .float32) * 5,
            prev: MLXArray.ones([1, 8], dtype: .float32) * 6)

        let batch = BatchZayaCCACache(slotCaches: [s0, s1])
        let (gConv, gPrev) = batch.gatherCCA()

        #expect(gConv.shape == [2, 4, 2])
        #expect(gPrev.shape == [2, 8])
        let row0Sum = gConv[0..<1].sum().item(Float.self)
        let row1Sum = gConv[1..<2].sum().item(Float.self)
        #expect((row0Sum - 24.0).magnitude < 1e-3)  // 4*2*3
        #expect((row1Sum - 40.0).magnitude < 1e-3)  // 4*2*5
    }

    @Test("B=2 update at independent offsets keeps slot histories independent")
    func b2UpdateIndependentOffsets() {
        let s0 = ZayaCCACache(batchSize: 1, convChannels: 2, hiddenSize: 4)
        let s1 = ZayaCCACache(batchSize: 1, convChannels: 2, hiddenSize: 4)
        // Pre-fill different offsets via separate updates
        _ = s0.update(
            keys: MLXArray.ones([1, 1, 5, 4], dtype: .bfloat16),
            values: MLXArray.ones([1, 1, 5, 4], dtype: .bfloat16))
        _ = s1.update(
            keys: MLXArray.ones([1, 1, 3, 4], dtype: .bfloat16) * 7,
            values: MLXArray.ones([1, 1, 3, 4], dtype: .bfloat16) * 9)

        let batch = BatchZayaCCACache(slotCaches: [s0, s1])
        // Sanity: offsetArray reflects per-slot offsets [5, 3]
        let initialOffsets = batch.offsetArray.asArray(Int32.self)
        #expect(initialOffsets == [5, 3])

        // One decode step
        let (kOut, _) = batch.update(
            keys:   MLXArray.ones([2, 1, 1, 4], dtype: .bfloat16) * 11,
            values: MLXArray.ones([2, 1, 1, 4], dtype: .bfloat16) * 13)

        #expect(kOut.dim(0) == 2)
        #expect(s0.offset == 6)
        #expect(s1.offset == 4)
        let updatedOffsets = batch.offsetArray.asArray(Int32.self)
        #expect(updatedOffsets == [6, 4])
    }

    @Test("scatterCCA after gather preserves per-slot identity for non-mutated slot")
    func scatterCCAPreservesNonMutatedSlot() {
        let s0 = ZayaCCACache(batchSize: 1, convChannels: 2, hiddenSize: 4)
        let s1 = ZayaCCACache(batchSize: 1, convChannels: 2, hiddenSize: 4)
        s0.writeCCA(
            conv: MLXArray.ones([1, 2, 2], dtype: .float32) * 3,
            prev: MLXArray.ones([1, 4], dtype: .float32) * 5)
        s1.writeCCA(
            conv: MLXArray.ones([1, 2, 2], dtype: .float32) * 7,
            prev: MLXArray.ones([1, 4], dtype: .float32) * 9)

        let batch = BatchZayaCCACache(slotCaches: [s0, s1])
        let (gConv, gPrev) = batch.gatherCCA()

        // Mutate ONLY slot 1's row in the gathered tensor.
        let mutConv = concatenated([gConv[0..<1], gConv[1..<2] * 2], axis: 0)
        let mutPrev = concatenated([gPrev[0..<1], gPrev[1..<2] * 2], axis: 0)
        batch.scatterCCA(conv: mutConv, prev: mutPrev)

        // Slot 0 unchanged: 2*2*3 = 12
        #expect((s0.readCCA().conv.sum().item(Float.self) - 12.0).magnitude < 1e-3)
        // Slot 1 doubled: 2*2*7*2 = 56
        #expect((s1.readCCA().conv.sum().item(Float.self) - 56.0).magnitude < 1e-3)
    }

    @Test("Cross-slot conv state never leaks: write 0 to slot 1 leaves slot 0's state alone")
    func crossSlotIsolation() {
        let s0 = ZayaCCACache(batchSize: 1, convChannels: 2, hiddenSize: 4)
        let s1 = ZayaCCACache(batchSize: 1, convChannels: 2, hiddenSize: 4)
        s0.writeCCA(
            conv: MLXArray.ones([1, 2, 2], dtype: .float32) * 99,
            prev: MLXArray.ones([1, 4], dtype: .float32) * 99)
        s1.writeCCA(
            conv: MLXArray.zeros([1, 2, 2], dtype: .float32),
            prev: MLXArray.zeros([1, 4], dtype: .float32))

        let batch = BatchZayaCCACache(slotCaches: [s0, s1])
        let (gConv, gPrev) = batch.gatherCCA()
        // Push the same gather back unchanged.
        batch.scatterCCA(conv: gConv, prev: gPrev)

        // Slot 0 must still be 99-everywhere.
        let s0Sum = s0.readCCA().conv.sum().item(Float.self)
        #expect((s0Sum - Float(99 * 2 * 2)).magnitude < 1e-2)
        // Slot 1 must still be 0.
        let s1Sum = s1.readCCA().conv.sum().item(Float.self)
        #expect(s1Sum.magnitude < 1e-3)
    }
}
