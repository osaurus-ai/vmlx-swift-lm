// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 0 tests for the Swift distributed bindings.
//
// On a dev box running a single process (no MLX_HOSTS env var, no peer
// process), mlx-core returns an "EmptyGroup" whose collective methods
// deliberately throw "Communication not implemented in an empty
// distributed group" — see mlx/distributed/distributed.cpp. That's the
// documented behaviour: real collectives require a real multi-rank world.
// These tests cover exactly that surface:
//
//   1. isAvailable / initialize work on a single process.
//   2. split on a size-1 group is rejected cleanly.
//   3. Transport probe short-circuits on size 1.
//
// The real collectives are exercised only in the multi-rank suite
// `MLXDistributedMultiRankTests` below, which is gated on the env var
// `VMLX_DISTRIBUTED_MULTIRANK=1` and typically run via a two-process
// launcher against two Macs over bridge0 / TB.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("MLXDistributed bindings — single-rank")
struct MLXDistributedSingleRankTests {

    @Test("isAvailable reports availability for `any`")
    func isAvailableAny() {
        // `any` always picks a backend — with the ring backend compiled
        // in (osaurus-ai/mlx-swift @ osaurus-0.31.3-distributed-phase0)
        // this returns true. Specific backends may return false on
        // hosts without the required runtime (e.g. `mpi` without an MPI
        // install), which is correct — that's what `isAvailable(backend:)`
        // is for. We don't assert specific backend availability because
        // it's host-dependent.
        #expect(MLXDistributed.isAvailable() == true)
        _ = MLXDistributed.isAvailable(backend: "ring")
        _ = MLXDistributed.isAvailable(backend: "jaccl")
    }

    @Test("initialize returns a single-rank world group on dev host")
    func initializeSingleRank() {
        // On a dev box with no MLX_HOSTS env var and no peer process,
        // init(strict: false) returns the EmptyGroup singleton —
        // rank 0, size 1. A real multi-rank launcher gives size > 1,
        // covered by the multi-rank suite.
        let group = MLXDistributed.initialize()
        #expect(group.isValid)
        #expect(group.rank == 0)
        #expect(group.size == 1)
        #expect(group.isMultiRank == false)
        #expect(MLXDistributed.worldGroup?.rank == 0)
        #expect(MLXDistributed.worldGroup?.size == 1)
    }

    @Test("split on a size-1 group returns nil with a logged error")
    func splitOnSizeOne() {
        // mlx-core's `Group::split` throws "Cannot split the distributed
        // group further" on a size-1 parent. The mlx-c bridge catches
        // and returns a zero-initialized group; our Swift wrapper maps
        // that to `nil`. Callers decide whether to treat this as fatal.
        let world = MLXDistributed.initialize()
        let sub = world.split(color: 0, key: 0)
        #expect(sub == nil)
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

// MARK: - Multi-rank suite (gated)
//
// When running under a real multi-rank launcher (two processes with
// `MLX_HOSTS=ip1,ip2` or equivalent), set `VMLX_DISTRIBUTED_MULTIRANK=1`
// and this suite activates. Otherwise the tests return without asserting
// — cheap to keep in the suite, dormant on the dev box.
//
// The real collective tests belong here because the EmptyGroup returned
// on a single process throws "Communication not implemented" when any
// collective is invoked; those throws are the documented behaviour and
// testing against them is covered at the `split` boundary above.

@Suite("MLXDistributed — multi-rank (gated on VMLX_DISTRIBUTED_MULTIRANK=1)")
struct MLXDistributedMultiRankTests {

    private var isEnabled: Bool {
        ProcessInfo.processInfo.environment["VMLX_DISTRIBUTED_MULTIRANK"] == "1"
    }

    @Test("allSum sums contributions across ranks")
    func allSumAcrossRanks() {
        guard isEnabled else { return }
        _ = MLXDistributed.initialize()
        guard let world = MLXDistributed.worldGroup, world.isMultiRank else {
            return
        }
        // Each rank contributes `rank + 1`. The sum across N ranks is
        // N*(N+1)/2 — a stable scalar result.
        let x = MLXArray([Float(world.rank + 1)])
        let y = MLXDistributed.allSum(x)
        let expected = Float((world.size * (world.size + 1)) / 2)
        #expect(abs(y.item(Float.self) - expected) < 1e-3)
    }

    @Test("allGather concatenates rank order")
    func allGatherConcatenation() {
        guard isEnabled else { return }
        _ = MLXDistributed.initialize()
        guard let world = MLXDistributed.worldGroup, world.isMultiRank else {
            return
        }
        let x = MLXArray([Float(world.rank)])
        let y = MLXDistributed.allGather(x)
        let values = y.asArray(Float.self)
        let expected = (0 ..< world.size).map { Float($0) }
        #expect(values == expected)
    }

    @Test("send/recv round-trip between adjacent ranks")
    func sendRecvRoundTrip() {
        guard isEnabled else { return }
        _ = MLXDistributed.initialize()
        guard let world = MLXDistributed.worldGroup, world.isMultiRank else {
            return
        }
        // Rank 0 → rank 1 handshake, then rank 1 → rank 0. Values chosen
        // so a byte-flip shows up immediately.
        let payload = MLXArray([Float(1.0), 2.0, 3.0, 4.0])
        if world.rank == 0 {
            _ = MLXDistributed.send(payload, dst: 1)
            let echo = MLXDistributed.recv(
                shape: [4], dtype: .float32, src: 1)
            #expect(echo.asArray(Float.self) == [1.0, 2.0, 3.0, 4.0])
        } else if world.rank == 1 {
            let received = MLXDistributed.recv(
                shape: [4], dtype: .float32, src: 0)
            _ = MLXDistributed.send(received, dst: 0)
        }
    }
}
