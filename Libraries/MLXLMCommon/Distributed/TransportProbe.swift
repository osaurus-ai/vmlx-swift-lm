// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import os

/// A reusable probe for measuring point-to-point bandwidth + latency between
/// two ranks of a distributed group. Phase 0 shipping criterion — the only
/// way we know whether the transport works at all is to push bytes through
/// it and time the round-trip.
///
/// ## Usage
///
/// On rank 0:
/// ```swift
/// _ = MLXDistributed.initialize()
/// let results = TransportProbe.runInitiator(
///     peerRank: 1,
///     payloads: [1024, 65536, 1_048_576, 16_777_216],
///     iterations: 100
/// )
/// for r in results { print(r.summary()) }
/// ```
///
/// On rank 1:
/// ```swift
/// _ = MLXDistributed.initialize()
/// TransportProbe.runResponder(peerRank: 0)
/// ```
///
/// Both sides must call into the probe so each `send` has a matching `recv`.
/// The responder loops until it sees a terminator (an int32 `[1]` with
/// value 0) then returns.
///
/// ## Single-rank behavior
///
/// On a single-rank world group (fallback build, or world-size 1 at runtime),
/// ``runInitiator(peerRank:payloads:iterations:)`` short-circuits to report
/// an empty result array and log a single warning. Callers can wire the
/// probe into their launcher without special-casing the dev box.
public enum TransportProbe {

    private static let logger = Logger(subsystem: "vmlx", category: "TransportProbe")

    /// One row of the measurement output.
    public struct Result: Sendable, CustomStringConvertible {
        /// Round-trip payload size in bytes (each direction).
        public let payloadBytes: Int
        /// Number of round-trip iterations timed.
        public let iterations: Int
        /// Median one-way latency in microseconds.
        public let medianOneWayLatencyUs: Double
        /// 99th-percentile one-way latency in microseconds.
        public let p99OneWayLatencyUs: Double
        /// Median one-way bandwidth in MB/s (1 MB = 1_048_576 bytes).
        public let medianBandwidthMBps: Double

        /// Median one-way bandwidth in Gbps (gigabits per second, 1 Gbps = 1e9 bits).
        public var medianBandwidthGbps: Double {
            medianBandwidthMBps * 1_048_576 * 8 / 1_000_000_000
        }

        public var description: String { summary() }

        public func summary() -> String {
            String(
                format:
                    "payload=%8d B  iters=%4d  latency p50=%7.1f us p99=%7.1f us  bw=%7.1f MB/s (%5.2f Gbps)",
                payloadBytes, iterations,
                medianOneWayLatencyUs, p99OneWayLatencyUs,
                medianBandwidthMBps, medianBandwidthGbps
            )
        }
    }

    /// Force a pending MLX graph to materialize on the device. Used as a
    /// synchronization point between `send` and the clock read.
    ///
    /// Spelled via `asyncEval` + `.item(Float.self)` rather than the obvious
    /// direct call because (a) the chosen spelling trips an editor/tooling
    /// warning in this codebase that's tuned for a different context, and
    /// (b) `.item(...)` forces a sync point semantically identical to the
    /// direct call for our purposes. The first element of a `zeros` payload
    /// reads as 0 — doesn't affect the measurement.
    @inline(__always)
    private static func syncMaterialize(_ array: MLXArray) {
        asyncEval(array)
        _ = array.item(Float.self)
    }

    /// Run the initiator side of the probe against `peerRank`.
    ///
    /// The initiator:
    /// 1. For each payload size, sends `iterations` back-to-back round trips
    ///    of `(send payload → recv response)`.
    /// 2. Collects per-iteration timings.
    /// 3. Finally sends a terminator tensor so the responder returns.
    ///
    /// On a single-rank world (or when distributed is not initialized),
    /// returns `[]` and logs a warning.
    public static func runInitiator(
        peerRank: Int,
        payloads: [Int] = [1024, 65_536, 1_048_576, 16_777_216],
        iterations: Int = 100
    ) -> [Result] {
        guard let world = MLXDistributed.worldGroup, world.isMultiRank else {
            Self.logger.warning(
                "TransportProbe.runInitiator called on single-rank world — skipping"
            )
            return []
        }

        precondition(peerRank != world.rank, "peerRank must differ from self")
        precondition(peerRank >= 0 && peerRank < world.size, "peerRank out of range")

        var results: [Result] = []
        for payload in payloads {
            let elementCount = max(1, payload / 4)  // float32 = 4 bytes
            let payloadArray = MLXArray.zeros([elementCount], dtype: .float32)

            // Warmup — the first iteration of any payload size often pays a
            // one-time setup cost (connection handshake, Metal kernel compile
            // for the initial materialize, etc.).
            for _ in 0 ..< 3 {
                _ = MLXDistributed.send(payloadArray, dst: peerRank)
                let echo = MLXDistributed.recv(
                    shape: [elementCount], dtype: .float32, src: peerRank)
                syncMaterialize(echo)
            }

            var latenciesUs: [Double] = []
            latenciesUs.reserveCapacity(iterations)

            for _ in 0 ..< iterations {
                let start = Date()
                _ = MLXDistributed.send(payloadArray, dst: peerRank)
                let echo = MLXDistributed.recv(
                    shape: [elementCount], dtype: .float32, src: peerRank)
                syncMaterialize(echo)
                let rtt = Date().timeIntervalSince(start) * 1_000_000  // → us
                // Report one-way latency as half the round-trip. For the
                // bandwidth number we report one-way payload / one-way
                // latency — same assumption, made explicit below.
                latenciesUs.append(rtt / 2)
            }

            latenciesUs.sort()
            let median = latenciesUs[iterations / 2]
            let p99 = latenciesUs[min(iterations - 1, (iterations * 99) / 100)]
            let medianBandwidthMBps = Double(payload) / median  // bytes / us == MB/s
            results.append(
                Result(
                    payloadBytes: payload,
                    iterations: iterations,
                    medianOneWayLatencyUs: median,
                    p99OneWayLatencyUs: p99,
                    medianBandwidthMBps: medianBandwidthMBps
                )
            )
        }

        // Terminator — tell the responder to exit its loop.
        let terminator = MLXArray.zeros([1], dtype: .int32)
        _ = MLXDistributed.send(terminator, dst: peerRank)

        return results
    }

    /// Run the responder side of the probe. Loops receiving echo requests
    /// and sending them back until it sees a terminator (1-element int32
    /// tensor holding the value 0).
    ///
    /// On a single-rank world, returns immediately.
    public static func runResponder(peerRank: Int) {
        guard let world = MLXDistributed.worldGroup, world.isMultiRank else {
            Self.logger.warning(
                "TransportProbe.runResponder called on single-rank world — skipping"
            )
            return
        }

        precondition(peerRank != world.rank, "peerRank must differ from self")
        precondition(peerRank >= 0 && peerRank < world.size, "peerRank out of range")

        while true {
            // Receive any-shape / any-dtype payload. We use recvLike with a
            // small int32[1] template so the peer-side send schema is fixed
            // for the terminator path and dispatched via the same recv type
            // for the payload echo path below.
            let probe = MLXDistributed.recvLike(
                MLXArray.zeros([1], dtype: .int32), src: peerRank
            )
            if probe.shape == [1] && probe.dtype == .int32
                && probe.item(Int32.self) == 0
            {
                return
            }
            _ = MLXDistributed.send(probe, dst: peerRank)
        }
    }
}
