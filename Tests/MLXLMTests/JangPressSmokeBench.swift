// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressSmokeBench — small live test that runs in CI / on a
// developer laptop without OOM. Allocates ~2 GB of synthetic tiles,
// drives a routing simulation, measures acquire latency, and
// reports kernel pressure response.
//
// Unlike the heavier `JangPressPressureBench`, this one is enabled
// by default — it's small enough to run in any test pass.

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JangPressSmokeBench")
struct JangPressSmokeBench {

    static let tileSize = 4 * 1024 * 1024              // 4 MB tiles
    static let numTiles = 512                          // → ~2 GB total
    static let layers = 16
    static let expertsPerLayer = 32                    // 16 × 32 = 512 tiles
    static let topK = 4
    static let decodeSteps = 200

    @Test("smoke: register + Zipfian routing + balloon + latency sample")
    func smokeRunFullPipeline() async throws {
        let cache = JangPressMachCache()

        // 1. Register tiles
        let registerStart = Date()
        for layer in 0..<Self.layers {
            for expert in 0..<Self.expertsPerLayer {
                let bytes = Self.makePattern(seed: UInt8((layer * 32 + expert) & 0xFF))
                try bytes.withUnsafeBytes { buf in
                    _ = try cache.register(layer: layer, expert: expert, bytes: buf)
                }
            }
        }
        print("[smoke] registered \(Self.numTiles) tiles (\(Self.numTiles * Self.tileSize / 1024 / 1024) MB) in \(String(format: "%.1f", Date().timeIntervalSince(registerStart))) s")

        // 2. Drive Zipfian routing
        let hot = (0..<Int(Double(Self.expertsPerLayer) * 0.30)).map { _ in
            Int.random(in: 0..<Self.expertsPerLayer)
        }
        let routeStart = Date()
        for _ in 0..<Self.decodeSteps {
            for layer in 0..<Self.layers {
                let pickedHot = (0..<2).map { _ in hot.randomElement()! }
                let pickedCold = (0..<2).map { _ in Int.random(in: 0..<Self.expertsPerLayer) }
                let picked = Array(Set(pickedHot + pickedCold)).prefix(Self.topK).map { $0 }
                _ = try cache.acquire(layer: layer, experts: picked)
                cache.release(layer: layer, experts: picked)
            }
        }
        print("[smoke] simulated \(Self.decodeSteps) decode steps in \(String(format: "%.1f", Date().timeIntervalSince(routeStart))) s")

        // 3. Measure baseline latency on resident tiles
        let baseline = sampleLatencyUs(cache: cache, count: 50)
        print("[smoke] baseline latency: p50=\(Int(baseline.p50)) µs  p95=\(Int(baseline.p95)) µs (no pressure)")

        // 4. Synthesize light pressure (allocate same-size balloon)
        let balloonSize = Self.numTiles * Self.tileSize
        let balloon = malloc(balloonSize)
        defer { free(balloon) }
        memset(balloon, 0xCC, balloonSize)

        try await Task.sleep(nanoseconds: 2_000_000_000)

        // 5. Re-measure after pressure
        let pressured = sampleLatencyUs(cache: cache, count: 50)
        print("[smoke] under-pressure latency: p50=\(Int(pressured.p50)) µs  p95=\(Int(pressured.p95)) µs")

        let stats = cache.snapshot()
        print("[smoke] cache stats:")
        print("  tiles=\(stats.totalTiles) bytes=\(stats.totalBytesAllocated / 1024 / 1024) MB")
        print("  acquire=\(stats.acquireCount) release=\(stats.releaseCount)")
        print("  refault=\(stats.refaultCount) discard=\(stats.discardCount)")
        print("  pressure events: low=\(stats.pressureLowCount) warn=\(stats.pressureWarnCount) crit=\(stats.pressureCriticalCount)")

        // ACCEPTANCE: even under light pressure, p95 acquire latency
        // should stay well under 100 ms. If it spikes above that the
        // path is unviable.
        #expect(pressured.p95 < 100_000)  // 100 ms in µs
    }

    // MARK: - helpers

    static func makePattern(seed: UInt8) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: Self.tileSize)
        for i in stride(from: 0, to: Self.tileSize, by: 4) {
            out[i] = UInt8((Int(seed) + i) & 0xFF)
            if i + 1 < out.count { out[i + 1] = seed }                 // repetitive (compresses)
            if i + 2 < out.count { out[i + 2] = UInt8(i & 0xF0) }
            if i + 3 < out.count { out[i + 3] = seed ^ UInt8(i & 0xFF) }
        }
        return out
    }

    func sampleLatencyUs(cache: JangPressMachCache, count: Int) -> (p50: Double, p95: Double) {
        var lat: [Double] = []
        for _ in 0..<count {
            let l = Int.random(in: 0..<Self.layers)
            let e = Int.random(in: 0..<Self.expertsPerLayer)
            let t0 = Date()
            _ = try? cache.acquire(layer: l, experts: [e])
            lat.append(Date().timeIntervalSince(t0) * 1_000_000)
            cache.release(layer: l, experts: [e])
        }
        lat.sort()
        let p50 = lat[count / 2]
        let p95 = lat[Int(Double(count) * 0.95)]
        return (p50, p95)
    }
}
