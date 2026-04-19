// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 1B.4 scaffolding tests — BucketRowAllocator + BucketDeadRow.
// Pure-logic, no compile invocation.

import Foundation
import MLX
import MLXLMCommon
import Testing

// MARK: - BucketRowAllocator

@Suite("BucketRowAllocator")
struct BucketRowAllocatorTests {

    @Test("initial state: all rows free, no live")
    func testInitialState() {
        let a = BucketRowAllocator(bucketSize: 4)
        #expect(a.bucketSize == 4)
        #expect(a.liveCount == 0)
        #expect(a.freeCount == 4)
        #expect(a.isEmpty == true)
        #expect(a.isFull == false)
        #expect(a.liveRows.isEmpty)
    }

    @Test("claim returns smallest free row and records ownership")
    func testClaimOrdering() {
        var a = BucketRowAllocator(bucketSize: 4)
        let s1 = BatchRequestID()
        let s2 = BatchRequestID()

        let row1 = a.claim(slotID: s1)
        #expect(row1 == 0)
        #expect(a.row(for: s1) == 0)
        #expect(a.liveCount == 1)
        #expect(a.freeCount == 3)

        let row2 = a.claim(slotID: s2)
        #expect(row2 == 1)
        #expect(a.row(for: s2) == 1)
    }

    @Test("release frees row and subsequent claim reuses it")
    func testReleaseAndReclaim() {
        var a = BucketRowAllocator(bucketSize: 4)
        let s1 = BatchRequestID()
        let s2 = BatchRequestID()
        let s3 = BatchRequestID()

        _ = a.claim(slotID: s1)
        _ = a.claim(slotID: s2)

        let released = a.release(slotID: s1)
        #expect(released == 0)
        #expect(a.row(for: s1) == nil)

        let row3 = a.claim(slotID: s3)
        #expect(row3 == 0, "Reclaim should reuse the released row")
    }

    @Test("release of unknown slot is a no-op")
    func testReleaseUnknown() {
        var a = BucketRowAllocator(bucketSize: 4)
        let unknown = BatchRequestID()
        #expect(a.release(slotID: unknown) == nil)
    }

    @Test("full bucket: isFull=true, cannot claim more")
    func testFullBucket() {
        var a = BucketRowAllocator(bucketSize: 2)
        _ = a.claim(slotID: BatchRequestID())
        _ = a.claim(slotID: BatchRequestID())
        #expect(a.isFull == true)
        #expect(a.isEmpty == false)
        #expect(a.freeCount == 0)
    }

    @Test("liveRows stays sorted ascending")
    func testLiveRowsSorted() {
        var a = BucketRowAllocator(bucketSize: 4)
        let s0 = BatchRequestID()
        let s1 = BatchRequestID()
        let s2 = BatchRequestID()

        _ = a.claim(slotID: s0)
        _ = a.claim(slotID: s1)
        _ = a.claim(slotID: s2)
        _ = a.release(slotID: s1)

        let live = a.liveRows
        #expect(live == [0, 2])
    }

    @Test("release re-inserts sorted so smallest free row wins")
    func testReinsertionOrdering() {
        var a = BucketRowAllocator(bucketSize: 4)
        let s0 = BatchRequestID()
        let s1 = BatchRequestID()
        let s2 = BatchRequestID()
        let s3 = BatchRequestID()

        _ = a.claim(slotID: s0)
        _ = a.claim(slotID: s1)
        _ = a.claim(slotID: s2)
        _ = a.release(slotID: s0)

        let row = a.claim(slotID: s3)
        #expect(row == 0)
    }
}

// MARK: - BucketDeadRow

@Suite("BucketDeadRow", .serialized)
struct BucketDeadRowTests {

    @Test("decodeInput places live tokens at live rows, placeholders at dead")
    func testDecodeInputStructure() {
        let input = BucketDeadRow.decodeInput(
            bucketSize: 4, liveRows: [0, 2], liveTokens: [42, 99])
        MLX.eval(input)

        #expect(input.shape == [4, 1])
        #expect(input[0, 0].item(Int32.self) == 42)
        #expect(input[1, 0].item(Int32.self) == 0)
        #expect(input[2, 0].item(Int32.self) == 99)
        #expect(input[3, 0].item(Int32.self) == 0)
    }

    @Test("decodeInput with no live rows is all placeholders")
    func testDecodeInputAllDead() {
        let input = BucketDeadRow.decodeInput(
            bucketSize: 3, liveRows: [], liveTokens: [])
        MLX.eval(input)
        #expect(input.shape == [3, 1])
        for i in 0 ..< 3 {
            #expect(input[i, 0].item(Int32.self) == 0)
        }
    }

    @Test("liveFlags marks live rows as 1, dead as 0")
    func testLiveFlags() {
        let flags = BucketDeadRow.liveFlags(bucketSize: 4, liveRows: [0, 2])
        MLX.eval(flags)
        #expect(flags.shape == [4])
        #expect(flags[0].item(Int32.self) == 1)
        #expect(flags[1].item(Int32.self) == 0)
        #expect(flags[2].item(Int32.self) == 1)
        #expect(flags[3].item(Int32.self) == 0)
    }

    @Test("liveFlags with all rows live")
    func testLiveFlagsAllLive() {
        let flags = BucketDeadRow.liveFlags(bucketSize: 3, liveRows: [0, 1, 2])
        MLX.eval(flags)
        for i in 0 ..< 3 {
            #expect(flags[i].item(Int32.self) == 1)
        }
    }

    @Test("liveFlags with no rows live")
    func testLiveFlagsNoneLive() {
        let flags = BucketDeadRow.liveFlags(bucketSize: 3, liveRows: [])
        MLX.eval(flags)
        for i in 0 ..< 3 {
            #expect(flags[i].item(Int32.self) == 0)
        }
    }
}
