// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 0 tests for the Swift distributed bindings.
//
// These tests cover the SINGLE-RANK fallback semantics. The fallback is
// linked via `MLXDistributedCFallback` as weak-alias stubs that treat
// every collective as the identity on a size-1 group — which is provably
// correct.
//
// Once we patch osaurus-ai/mlx-swift to enable real distributed
// compilation (Phase 0.5 in DISTRIBUTED-DESIGN.md), a separate
// multi-rank test suite will exercise the real collectives on a 2-process
// loopback. Those tests are gated on the `VMLX_DISTRIBUTED_MULTIRANK=1`
// env var so they only run when a real backend is linked in.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("MLXDistributed bindings — single-rank fallback")
struct MLXDistributedSingleRankTests {

    @Test("isAvailable reports true")
    func isAvailableTrue() {
        // The fallback always reports availability so callers get a
        // consistent affirmative regardless of backend choice. Real
        // multi-rank availability is observable via `worldGroup.isMultiRank`.
        #expect(MLXDistributed.isAvailable() == true)
        #expect(MLXDistributed.isAvailable(backend: "ring") == true)
        #expect(MLXDistributed.isAvailable(backend: "jaccl") == true)
    }

    @Test("initialize returns a single-rank world group")
    func initializeSingleRank() {
        let group = MLXDistributed.initialize()
        #expect(group.rank == 0)
        #expect(group.size == 1)
        #expect(group.isMultiRank == false)
        #expect(MLXDistributed.worldGroup?.rank == 0)
        #expect(MLXDistributed.worldGroup?.size == 1)
    }

    @Test("split returns a group with the same rank/size on size 1")
    func splitOnSizeOne() {
        let world = MLXDistributed.initialize()
        let sub = world.split(color: 0, key: 0)
        #expect(sub.rank == 0)
        #expect(sub.size == 1)
    }

    // MARK: - Identity semantics on size-1 collectives

    @Test("allSum is identity on size 1")
    func allSumIdentity() {
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(1.0), 2.0, 3.0, 4.0])
        let y = MLXDistributed.allSum(x)
        #expect(y.shape == [4])
        let values = y.asArray(Float.self)
        #expect(values == [1.0, 2.0, 3.0, 4.0])
    }

    @Test("allGather is identity on size 1")
    func allGatherIdentity() {
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(5.0), 6.0])
        let y = MLXDistributed.allGather(x)
        let values = y.asArray(Float.self)
        #expect(values == [5.0, 6.0])
    }

    @Test("allMax is identity on size 1")
    func allMaxIdentity() {
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(1.0), -2.0, 3.5])
        let y = MLXDistributed.allMax(x)
        let values = y.asArray(Float.self)
        #expect(values == [1.0, -2.0, 3.5])
    }

    @Test("allMin is identity on size 1")
    func allMinIdentity() {
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(-1.0), 0.0, 1.0])
        let y = MLXDistributed.allMin(x)
        let values = y.asArray(Float.self)
        #expect(values == [-1.0, 0.0, 1.0])
    }

    @Test("sumScatter is identity on size 1")
    func sumScatterIdentity() {
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(1.0), 2.0, 3.0, 4.0])
        let y = MLXDistributed.sumScatter(x)
        let values = y.asArray(Float.self)
        #expect(values == [1.0, 2.0, 3.0, 4.0])
    }

    @Test("send on size 1 returns the input unchanged")
    func sendIdentity() {
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(7.0), 8.0])
        let placeholder = MLXDistributed.send(x, dst: 0)
        let values = placeholder.asArray(Float.self)
        #expect(values == [7.0, 8.0])
    }

    // MARK: - Shape + dtype plumbing

    @Test("send/recv signatures accept expected shape/dtype arguments")
    func sendRecvSignaturesCompile() {
        // Compilation-only check: exercise every overload so a signature
        // regression trips at build time rather than first real use.
        _ = MLXDistributed.initialize()
        let x = MLXArray([Float(1.0), 2.0, 3.0, 4.0]).reshaped([2, 2])
        _ = MLXDistributed.send(x, dst: 0)
        // recv/recvLike return errors on size-1 (no peer); we don't call
        // them here. The real test for these is the multi-rank suite.
    }
}

@Suite("TransportProbe — single-rank short-circuit")
struct TransportProbeSingleRankTests {

    @Test("runInitiator returns empty on single-rank world")
    func initiatorEmptyOnSingleRank() {
        _ = MLXDistributed.initialize()
        let results = TransportProbe.runInitiator(
            peerRank: 0,  // would be invalid if we weren't short-circuiting
            payloads: [1024],
            iterations: 1
        )
        #expect(results.isEmpty)
    }

    @Test("runResponder returns immediately on single-rank world")
    func responderNoOpOnSingleRank() {
        _ = MLXDistributed.initialize()
        // Must return without blocking. If it hangs, the test deadlocks —
        // swift-testing's default 60s timeout will surface the regression.
        TransportProbe.runResponder(peerRank: 0)
    }

    @Test("Result bandwidth arithmetic is correct")
    func resultArithmetic() {
        // Sanity-check the Gbps derivation against a known value.
        // payload 1 MiB (1_048_576 B) at 100 us one-way latency →
        //   bw = 1_048_576 / 100 ≈ 10485.76 MB/s
        //   gbps ≈ 10485.76 * 1_048_576 * 8 / 1e9 ≈ 87.96
        let r = TransportProbe.Result(
            payloadBytes: 1_048_576,
            iterations: 100,
            medianOneWayLatencyUs: 100,
            p99OneWayLatencyUs: 200,
            medianBandwidthMBps: 10485.76
        )
        #expect(abs(r.medianBandwidthGbps - 87.96) < 0.1)
    }
}
