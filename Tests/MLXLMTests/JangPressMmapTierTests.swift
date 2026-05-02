// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressMmapTierTests — verify the bundle-aware mmap+madvise
// tier finds routed-expert tiles by name pattern and exposes
// acquire/release that issues the right advise calls.
//
// We synthesize a tiny fake "bundle" with two safetensors shards:
//   shard-001: model.layers.0.mlp.experts.0.gate_proj.weight
//              model.layers.0.mlp.experts.0.up_proj.weight
//              model.layers.0.mlp.experts.0.down_proj.weight
//              model.layers.0.mlp.experts.1.gate_proj.weight
//              model.layers.0.mlp.experts.1.up_proj.weight
//              model.layers.0.mlp.experts.1.down_proj.weight
//   shard-002: model.layers.1.mlp.switch_mlp.gate_proj.weight
//              model.layers.1.mlp.switch_mlp.up_proj.weight
//              model.layers.1.mlp.switch_mlp.down_proj.weight
//              model.norm.weight   (non-routed, should be ignored)

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JangPressMmapTier")
struct JangPressMmapTierTests {

    // MARK: - Helpers

    static func makeBundleDir() -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mmap-tier-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Build a one-shard safetensors file with the given tensors at
    /// 32 bytes each. Returns the file URL.
    @discardableResult
    static func writeShard(at dir: URL, name: String, tensorNames: [String]) throws -> URL {
        var header: [String: Any] = [:]
        var offset: UInt64 = 0
        for tn in tensorNames {
            header[tn] = [
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [offset, offset + 32],
            ]
            offset += 32
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let headerSize = UInt64(json.count)

        var fileBytes = Data()
        fileBytes.append(contentsOf: withUnsafeBytes(of: headerSize.littleEndian) { Array($0) })
        fileBytes.append(json)
        for (i, _) in tensorNames.enumerated() {
            fileBytes.append(contentsOf: (0..<32).map { UInt8(($0 + i * 17) & 0xFF) })
        }

        let url = dir.appendingPathComponent(name)
        try fileBytes.write(to: url)
        return url
    }

    static func standardBundle() throws -> URL {
        let dir = makeBundleDir()
        try writeShard(at: dir, name: "model-001.safetensors", tensorNames: [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.1.up_proj.weight",
            "model.layers.0.mlp.experts.1.down_proj.weight",
        ])
        try writeShard(at: dir, name: "model-002.safetensors", tensorNames: [
            "model.layers.1.mlp.switch_mlp.gate_proj.weight",
            "model.layers.1.mlp.switch_mlp.up_proj.weight",
            "model.layers.1.mlp.switch_mlp.down_proj.weight",
            "model.norm.weight",
        ])
        return dir
    }

    // MARK: - Tests

    @Test("regex parses all 13 expert tile layouts (A-M)")
    func regexParsesAllLayouts() {
        // Pattern A — Qwen/GLM/MiniMax fp16 stacked
        let switchMlp = "model.layers.13.mlp.switch_mlp.up_proj.weight"
        let r1 = JangPressMmapTier.parseRoutedExpertName(switchMlp)
        #expect(r1?.layer == 13)
        #expect(r1?.expert == 0)

        // Pattern B — Mistral 4 / Kimi / DSV3 per-expert
        let perExpertMlp = "model.layers.7.mlp.experts.42.gate_proj.weight"
        let r2 = JangPressMmapTier.parseRoutedExpertName(perExpertMlp)
        #expect(r2?.layer == 7)
        #expect(r2?.expert == 42)

        // Pattern C — Laguna / Qwen3.6 JANGTQ stacked
        let jangtqStacked = "model.layers.5.mlp.experts.gate_up_proj.tq_packed"
        let r3 = JangPressMmapTier.parseRoutedExpertName(jangtqStacked)
        #expect(r3?.layer == 5)
        #expect(r3?.expert == 0)

        // Pattern D — JANG_2L affine stacked
        let affineStacked = "model.layers.9.mlp.experts.down_proj.weight"
        let r4 = JangPressMmapTier.parseRoutedExpertName(affineStacked)
        #expect(r4?.layer == 9)
        #expect(r4?.expert == 0)

        // Pattern E — DSV4 per-expert JANGTQ (NEW iter 12)
        let dsv4Tq = "layers.3.ffn.experts.17.w2.tq_packed"
        let r5 = JangPressMmapTier.parseRoutedExpertName(dsv4Tq)
        #expect(r5?.layer == 3)
        #expect(r5?.expert == 17)

        // Pattern F — DSV4 per-expert affine
        let dsv4Affine = "layers.0.ffn.experts.5.w1.weight"
        let r6 = JangPressMmapTier.parseRoutedExpertName(dsv4Affine)
        #expect(r6?.layer == 0)
        #expect(r6?.expert == 5)

        // Pattern G — Holo3 / Qwen3.5MoE switch_mlp JANGTQ (NEW iter 16)
        let holo3 = "language_model.model.layers.20.mlp.switch_mlp.up_proj.tq_packed"
        let r7 = JangPressMmapTier.parseRoutedExpertName(holo3)
        #expect(r7?.layer == 20)
        #expect(r7?.expert == 0)

        // Pattern G also without VL prefix
        let qwen35moe = "model.layers.4.mlp.switch_mlp.gate_proj.tq_packed"
        let r7b = JangPressMmapTier.parseRoutedExpertName(qwen35moe)
        #expect(r7b?.layer == 4)
        #expect(r7b?.expert == 0)

        // Pattern D / Qwen3.6 JANG_2L deep-VL prefix:
        //   model.language_model.layers.<L>... (NEW iter 18)
        let qwen36deep = "model.language_model.layers.21.mlp.switch_mlp.down_proj.weight"
        let r7c = JangPressMmapTier.parseRoutedExpertName(qwen36deep)
        #expect(r7c?.layer == 21)
        #expect(r7c?.expert == 0)

        // Pattern H — MiniMax M2/M2.7 JANGTQ per-expert (NEW iter 17)
        let minimax = "model.layers.30.block_sparse_moe.experts.150.w1.tq_packed"
        let r8 = JangPressMmapTier.parseRoutedExpertName(minimax)
        #expect(r8?.layer == 30)
        #expect(r8?.expert == 150)

        // Pattern I — MiniMax affine
        let minimaxAff = "model.layers.4.block_sparse_moe.experts.0.w3.weight"
        let r9 = JangPressMmapTier.parseRoutedExpertName(minimaxAff)
        #expect(r9?.layer == 4)
        #expect(r9?.expert == 0)

        // Pattern J — Nemotron Omni JANGTQ per-expert (NEW iter 17)
        let nemotron = "backbone.layers.34.mixer.experts.17.up_proj.tq_packed"
        let r10 = JangPressMmapTier.parseRoutedExpertName(nemotron)
        #expect(r10?.layer == 34)
        #expect(r10?.expert == 17)

        // Pattern K — Nemotron affine per-expert
        let nemotronAff = "backbone.layers.5.mixer.experts.42.gate_proj.weight"
        let r11 = JangPressMmapTier.parseRoutedExpertName(nemotronAff)
        #expect(r11?.layer == 5)
        #expect(r11?.expert == 42)

        // Pattern L — Nemotron Omni MXFP4 stacked switch_mlp
        let nemotronMx = "backbone.layers.31.mixer.switch_mlp.fc1.weight"
        let r12 = JangPressMmapTier.parseRoutedExpertName(nemotronMx)
        #expect(r12?.layer == 31)
        #expect(r12?.expert == 0)

        // Pattern M — Nemotron Cascade-2 affine stacked switch_mlp
        let cascade2 = "backbone.layers.29.mixer.switch_mlp.down_proj.weight"
        let r13 = JangPressMmapTier.parseRoutedExpertName(cascade2)
        #expect(r13?.layer == 29)
        #expect(r13?.expert == 0)

        // DSV4 hash-routed layers L0-L2 use the same physical naming
        // as routed layers — distinguished only at routing time, not
        // tile structure. So they match pattern E/F too. This is by
        // design: same tier, both routing modes.

        // Negative cases — non-routed tensors
        #expect(JangPressMmapTier.parseRoutedExpertName("model.norm.weight") == nil)
        #expect(JangPressMmapTier.parseRoutedExpertName(
            "model.layers.0.self_attn.q_proj.weight") == nil)
        #expect(JangPressMmapTier.parseRoutedExpertName(
            "model.layers.0.mlp.shared_expert.gate_proj.weight") == nil)
        // DSV4 attention tensors (NOT routed-expert tiles)
        #expect(JangPressMmapTier.parseRoutedExpertName(
            "layers.0.attn.wq_a.weight") == nil)
        #expect(JangPressMmapTier.parseRoutedExpertName(
            "layers.0.ffn.shared_experts.w1.weight") == nil)
    }

    @Test("opens shards and indexes routed-expert tiles")
    func opensAndIndexesShards() throws {
        let dir = try Self.standardBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressMmapTier(
            config: .init(bundleURL: dir, hotPercent: 30, startCold: false))

        let stats = tier.snapshot()
        // 2 shards opened
        #expect(stats.shardCount == 2)
        // 3 routed experts: (layer=0, exp=0), (layer=0, exp=1),
        // (layer=1, exp=0) — the stacked switch_mlp is indexed once.
        #expect(stats.expertCount == 3)
        // Layer 0 has 2 experts; layer 1 has 1 (the stacked one)
        #expect(stats.byLayer[0] == 2)
        #expect(stats.byLayer[1] == 1)
        // 9 expert tile bytes (3 layers/experts × 3 projections × 32 bytes)
        // = 3 × 3 × 32 minus 1 expert (layer=1 is stacked, also 3×32).
        // Our model: 6 (per-expert L0) + 3 (stacked L1) = 9 tiles × 32 B = 288 B
        #expect(stats.totalRoutedBytes == 288)
    }

    @Test("acquire/release issues madvise without crashing")
    func acquireRelease() throws {
        let dir = try Self.standardBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressMmapTier(
            config: .init(bundleURL: dir, hotPercent: 30, startCold: true))

        // Should not throw / crash.
        tier.acquire(layer: 0, experts: [0, 1])
        tier.release(layer: 0, experts: [0, 1])
        tier.acquire(layer: 1, experts: [0])
        tier.release(layer: 1, experts: [0])

        // Acquiring a non-existent expert is a no-op (no throw).
        tier.acquire(layer: 99, experts: [99])
    }

    @Test("startCold flag triggers initial dontNeed pass")
    func startCold() throws {
        let dir = try Self.standardBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Just verify the constructor doesn't blow up with startCold=true.
        // The actual page-state observation requires kernel/vmstat hooks.
        let tier = try JangPressMmapTier(
            config: .init(bundleURL: dir, hotPercent: 0, startCold: true))
        #expect(tier.snapshot().expertCount == 3)
    }

    @Test("forceRelease + reacquire produces byte-identical data")
    func forceReleaseRoundTrip() throws {
        let dir = try Self.standardBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressMmapTier(
            config: .init(bundleURL: dir, hotPercent: 0, startCold: false))

        // Pick the first stored expert tile and snapshot its bytes BEFORE
        // any release. This is the "ground truth".
        let expertKey = JangPressMmapTier.TileKey(layer: 0, expert: 0)
        guard let ranges = tier.experts[expertKey] else {
            Issue.record("expected expert (layer=0, e=0) to exist")
            return
        }
        guard let firstPart = ranges.parts.first else {
            Issue.record("expected at least one part for expert")
            return
        }
        guard let shard = tier.shards[firstPart.shard] else {
            Issue.record("expected shard to be opened")
            return
        }

        let original = Array(shard.bytes(in: firstPart.range))
        #expect(original.count == 32)

        // Force-release: msync(MS_INVALIDATE). Pages dropped from cache.
        tier.forceRelease(layer: 0, experts: [0])

        // Re-acquire: madvise(WILLNEED) + read again.
        tier.acquire(layer: 0, experts: [0])
        let reread = Array(shard.bytes(in: firstPart.range))
        #expect(reread == original, "post-forceRelease re-read must be byte-identical to pre-release")
    }

    @Test("sniff path skips non-expert shards from mmap")
    func sniffSkipsNonExpertShards() throws {
        let dir = Self.makeBundleDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Shard 1: contains routed experts.
        try Self.writeShard(at: dir, name: "model-001.safetensors", tensorNames: [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
        ])
        // Shard 2: ONLY non-expert tensors.
        try Self.writeShard(at: dir, name: "model-002.safetensors", tensorNames: [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ])

        let tier = try JangPressMmapTier(
            config: .init(bundleURL: dir, hotPercent: 0, startCold: false))

        // Only shard 1 should be opened — shard 2 has no routed experts.
        #expect(tier.shards.count == 1)
        #expect(tier.snapshot().shardCount == 1)
        // 1 expert tile (synthetic id 0 since per-expert with single tensor).
        #expect(tier.snapshot().expertCount == 1)
    }
}
