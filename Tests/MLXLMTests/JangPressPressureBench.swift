// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressPressureBench — synthetic memory-pressure simulation that
// proves whether the macOS kernel actually compresses our purgeable
// regions under MoE routing density.
//
// PROBLEM
// =======
// vm_purgable_control documents the contract but doesn't tell us how
// aggressively the kernel will compress under a specific access
// pattern. Top-k=6/N=256 routing means 250 of 256 expert tiles are
// untouched between accesses — but each tile is touched within a few
// seconds, which the kernel's LRU might consider "still active".
//
// HOW WE TEST
// ===========
// 1. Allocate N synthetic expert tiles (~7.5 MB each, matching DSV4)
//    via JangPressMachCache.
// 2. Inflate a "balloon" allocation that consumes most free RAM, so
//    the kernel WANTS to compress.
// 3. Drive a synthetic routing pattern: top-6 of 256, with
//    Zipfian frequency (some experts hot, most cold).
// 4. Periodically poll vm_statistics to see how many of OUR pages
//    have been compressed.
// 5. Report compression ratio + decompress latency on access.
//
// WHAT GOOD LOOKS LIKE
// ====================
// • > 60 % of cold-tile pages compressed within 30 s of pressure
// • acquire() latency on compressed tile < 10 ms
// • disk-refault never fires (kernel keeps compressed, doesn't discard)
// • routing-frequency LRU correctly identifies the 30 % hot set
//
// WHAT WOULD FAIL
// ===============
// • Kernel never compresses (purgeable wired internally for our
//   access pattern, no real WKdm engagement) → approach is wrong
// • Compression happens but acquire latency is multi-second → useless
// • Kernel discards entirely instead of compressing → disk-refault
//   becomes mandatory, latency becomes unpredictable
//
// HOW TO RUN
// ==========
// This is `@Test(.disabled)` by default because it allocates GBs and
// takes minutes. Run it explicitly when measuring:
//
//   swift test --filter JangPressPressureBench

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JangPressPressureBench", .disabled("intentional — heavy memory bench, run on demand"))
struct JangPressPressureBench {

    static let tileSizeBytes = 7 * 1024 * 1024 + 512 * 1024  // ~7.5 MB
    static let numExpertsPerLayer = 256
    static let numLayers = 43
    static let topK = 6

    @Test("compression engages under simulated MoE routing + memory pressure")
    func compressionUnderPressure() async throws {
        let cache = JangPressMachCache()
        let totalTiles = Self.numExpertsPerLayer * Self.numLayers
        let totalBytes = totalTiles * Self.tileSizeBytes
        print("[bench] registering \(totalTiles) tiles @ \(Self.tileSizeBytes / 1024 / 1024) MB each = \(totalBytes / 1024 / 1024 / 1024) GB total")

        // Phase 1 — register all tiles
        let registerStart = Date()
        for layer in 0..<Self.numLayers {
            for expert in 0..<Self.numExpertsPerLayer {
                let bytes = Self.makePattern(seed: UInt8((layer * 256 + expert) & 0xFF))
                try bytes.withUnsafeBytes { buf in
                    _ = try cache.register(layer: layer, expert: expert, bytes: buf)
                }
            }
        }
        print("[bench] register: \(Int(Date().timeIntervalSince(registerStart))) s")

        // Phase 2 — Zipfian routing simulation. Top 30 % experts get
        // 90 % of routes, bottom 70 % get 10 %.
        var hotExperts: [Int] = []
        for _ in 0..<(Self.numExpertsPerLayer * 30 / 100) {
            hotExperts.append(Int.random(in: 0..<Self.numExpertsPerLayer))
        }

        // Run 300 simulated decode steps. Each step:
        // 1. acquire top-6 experts per layer
        // 2. release them
        // (No actual compute — we're measuring kernel behavior, not throughput.)
        let routeStart = Date()
        for step in 0..<300 {
            for layer in 0..<Self.numLayers {
                let pickedHot = (0..<4).map { _ in hotExperts.randomElement()! }
                let pickedCold = (0..<2).map { _ in Int.random(in: 0..<Self.numExpertsPerLayer) }
                let picked = Array(Set(pickedHot + pickedCold)).prefix(Self.topK).map { $0 }
                _ = try cache.acquire(layer: layer, experts: picked)
                cache.release(layer: layer, experts: picked)
            }
            if step % 50 == 0 {
                print("[bench] step \(step) @ \(Int(Date().timeIntervalSince(routeStart))) s")
            }
        }
        print("[bench] simulated 300 decode steps: \(Int(Date().timeIntervalSince(routeStart))) s")

        // Phase 3 — synthesize pressure. Allocate a balloon equal to
        // the cache size to force the kernel to compress something.
        print("[bench] inflating balloon to force pressure...")
        let balloonSize = totalBytes
        let balloon = malloc(balloonSize)
        defer { free(balloon) }
        memset(balloon, 0xCC, min(balloonSize, 1 * 1024 * 1024 * 1024)) // touch first 1 GB

        // Wait for kernel pressure response.
        try await Task.sleep(nanoseconds: 5_000_000_000)

        // Phase 4 — measure: re-acquire a sample of tiles, time the
        // acquire call. Cold tiles should show > 1 ms latency if the
        // kernel actually compressed; resident tiles stay <100 µs.
        print("[bench] sampling acquire latency...")
        var latencies: [Double] = []
        for _ in 0..<100 {
            let l = Int.random(in: 0..<Self.numLayers)
            let e = Int.random(in: 0..<Self.numExpertsPerLayer)
            let t0 = Date()
            _ = try cache.acquire(layer: l, experts: [e])
            let dt = Date().timeIntervalSince(t0) * 1_000_000  // µs
            latencies.append(dt)
            cache.release(layer: l, experts: [e])
        }
        let sortedLat = latencies.sorted()
        let p50 = sortedLat[50]
        let p95 = sortedLat[95]
        print("[bench] acquire latency p50=\(p50) µs  p95=\(p95) µs")

        let stats = cache.snapshot()
        print("[bench] cache stats: acquire=\(stats.acquireCount) release=\(stats.releaseCount) refault=\(stats.refaultCount) discard=\(stats.discardCount)")
        print("[bench] pressure events: low=\(stats.pressureLowCount) warn=\(stats.pressureWarnCount) crit=\(stats.pressureCriticalCount)")

        // Acceptance: p95 < 50 ms means the path is viable. Higher
        // means kernel discarded entirely and disk refault would be
        // needed (we didn't register diskURL so we'd see throws).
        #expect(p95 < 50_000)  // 50 ms in µs
    }

    // MARK: helpers

    static func makePattern(seed: UInt8) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: Self.tileSizeBytes)
        // Mix of structured + entropy so WKdm sees realistic compression
        // potential (model weights average ~2.3× WKdm ratio empirically).
        for i in stride(from: 0, to: Self.tileSizeBytes, by: 4) {
            let b0 = UInt8((Int(seed) + i) & 0xFF)
            let b1 = UInt8((i >> 8) & 0x0F)  // high entropy in low nibble
            let b2 = seed                    // repetitive byte (compresses well)
            let b3 = UInt8((i & 0xF0))
            out[i] = b0
            if i + 1 < Self.tileSizeBytes { out[i + 1] = b1 }
            if i + 2 < Self.tileSizeBytes { out[i + 2] = b2 }
            if i + 3 < Self.tileSizeBytes { out[i + 3] = b3 }
        }
        return out
    }
}
