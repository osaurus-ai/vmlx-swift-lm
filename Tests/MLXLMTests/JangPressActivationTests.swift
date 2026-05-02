// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Tests for `JangPressActivation` — the SDK-level entry point that
// instantiates the JangPress tiers from a `JangPressLoadOptions`
// configuration.
//
// These tests exercise the activation logic without requiring a real
// model bundle on disk: they construct a synthetic safetensors-shaped
// directory tree and verify the tier wiring honors every combination
// of `enabled`, `compressPct`, `backend`, `forceMode`, `enablePrefetch`.
//
// For real-bundle smoke tests with actual models from /Volumes, see
// the bench harnesses in `RunBench/` (added in a follow-up).

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JangPressActivation")
struct JangPressActivationTests {

    // MARK: - Disabled paths

    @Test("disabled options return JangPressRuntime.none")
    func disabledReturnsNone() async throws {
        let runtime = JangPressActivation.activate(
            bundleURL: URL(fileURLWithPath: "/nonexistent"),
            options: .disabled)
        #expect(runtime.controller == nil)
        #expect(runtime.mmap == nil)
        #expect(runtime.mach == nil)
        #expect(runtime.embed == nil)
        #expect(runtime.isActive == false)
    }

    @Test("explicit enabled=false returns JangPressRuntime.none")
    func explicitFalseReturnsNone() async throws {
        let opts = JangPressLoadOptions(enabled: false, compressPct: 70)
        let runtime = JangPressActivation.activate(
            bundleURL: URL(fileURLWithPath: "/nonexistent"),
            options: opts)
        #expect(runtime.isActive == false)
    }

    @Test("backend=.mach falls back to disabled with stderr warning")
    func machBackendFallsBack() async throws {
        let opts = JangPressLoadOptions(enabled: true, backend: .mach)
        let runtime = JangPressActivation.activate(
            bundleURL: URL(fileURLWithPath: "/nonexistent"),
            options: opts)
        // .mach is gated on the MLX-swift fork; falls back to .none
        // until the MachCache.register path is wired.
        #expect(runtime.isActive == false)
    }

    // MARK: - Enabled paths (synthetic bundle)

    /// Build a temp directory with one routed-expert safetensors shard
    /// so the mmap-tier ctor has something to scan. Returns the dir
    /// URL; caller is responsible for `removeItem` cleanup.
    private func makeSyntheticBundle() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangpress-test-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)

        // Minimal safetensors file with one routed-expert tensor + one
        // attention tensor (so the sniff path can correctly identify
        // expert vs non-expert shards).
        let shardURL = dir.appendingPathComponent("model-00001-of-00001.safetensors")

        // Tensor name patterns recognized by JangPressMmapTier:
        //   model.layers.0.mlp.experts.0.gate_proj.weight  (Mistral 4 / DSV3 style — pattern B)
        //   model.layers.0.self_attn.q_proj.weight         (NOT a routed expert)
        let header: [String: Any] = [
            "model.layers.0.mlp.experts.0.gate_proj.weight": [
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32]
            ],
            "model.layers.0.mlp.experts.0.up_proj.weight": [
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [32, 64]
            ],
            "model.layers.0.mlp.experts.0.down_proj.weight": [
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [64, 96]
            ],
            "model.layers.0.self_attn.q_proj.weight": [
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [96, 128]
            ],
        ]
        let headerJSON = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        var data = Data()
        var headerSize = UInt64(headerJSON.count)
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerJSON)
        // 128 bytes of zero payload (4 × 32-byte tensors).
        data.append(Data(count: 128))
        try data.write(to: shardURL)
        return dir
    }

    @Test("enabled mmap backend returns active runtime with controller + tier")
    func enabledMmapReturnsActive() async throws {
        let bundle = try makeSyntheticBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let opts = JangPressLoadOptions(
            enabled: true, compressPct: 70,
            backend: .mmap, forceMode: .soft, enablePrefetch: true)
        let runtime = JangPressActivation.activate(
            bundleURL: bundle, options: opts)

        #expect(runtime.isActive == true)
        #expect(runtime.controller != nil)
        #expect(runtime.mmap != nil)
        #expect(runtime.mach == nil)
        // embed-tier may be nil if the synthetic shard has no
        // embed_tokens key — that's fine, it's a non-fatal failure.

        // Controller's stats should be reachable.
        let stats = runtime.controller?.snapshot()
        #expect(stats != nil)

        // Disarm cleanly — no crash, no leak.
        JangPressActivation.deactivate(runtime)
    }

    @Test("forceMode=.force is accepted (controller uses MS_INVALIDATE)")
    func forceModeAccepted() async throws {
        let bundle = try makeSyntheticBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let opts = JangPressLoadOptions(
            enabled: true, compressPct: 100,
            backend: .mmap, forceMode: .force, enablePrefetch: false)
        let runtime = JangPressActivation.activate(
            bundleURL: bundle, options: opts)
        #expect(runtime.isActive == true)
        JangPressActivation.deactivate(runtime)
    }

    @Test("compressPct out-of-range gets clamped to [0, 100]")
    func compressPctClamped() async throws {
        let opts1 = JangPressLoadOptions(enabled: true, compressPct: -5)
        #expect(opts1.compressPct == 0)
        let opts2 = JangPressLoadOptions(enabled: true, compressPct: 150)
        #expect(opts2.compressPct == 100)
    }

    @Test("inference brackets fire without crash on an active runtime")
    func bracketHooks() async throws {
        let bundle = try makeSyntheticBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let opts = JangPressLoadOptions(enabled: true, compressPct: 70)
        let runtime = JangPressActivation.activate(
            bundleURL: bundle, options: opts)
        #expect(runtime.controller != nil)

        // Simulate a single inference cycle.
        runtime.controller?.willStartInference(layerExpertHints: [])
        runtime.controller?.recordRoute(layer: 0, experts: [0])
        runtime.controller?.didFinishInference()

        // Repeat — no crash, stats accumulate.
        for _ in 0 ..< 3 {
            runtime.controller?.willStartInference()
            runtime.controller?.recordRoute(layer: 0, experts: [0, 1])
            runtime.controller?.didFinishInference()
        }

        let stats = runtime.controller?.snapshot()
        #expect(stats?.totalRoutesObserved ?? 0 > 0)

        JangPressActivation.deactivate(runtime)
    }

    @Test("idempotent deactivate — safe to call multiple times")
    func deactivateIdempotent() async throws {
        let bundle = try makeSyntheticBundle()
        defer { try? FileManager.default.removeItem(at: bundle) }

        let opts = JangPressLoadOptions(enabled: true, compressPct: 70)
        let runtime = JangPressActivation.activate(
            bundleURL: bundle, options: opts)

        JangPressActivation.deactivate(runtime)
        // Second call on an already-disarmed runtime must not crash.
        JangPressActivation.deactivate(runtime)
        // .none is also safe.
        JangPressActivation.deactivate(.none)
    }
}
