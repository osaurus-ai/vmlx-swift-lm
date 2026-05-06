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

    @Test("regex parses current expert tile layouts")
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

        // Pattern B also covers Ling-style per-expert TQ tensors.
        let lingTq = "model.layers.7.mlp.experts.42.gate_proj.tq_packed"
        let r2b = JangPressMmapTier.parseRoutedExpertName(lingTq)
        #expect(r2b?.layer == 7)
        #expect(r2b?.expert == 42)

        let lingNorms = "model.layers.7.mlp.experts.42.up_proj.tq_norms"
        let r2c = JangPressMmapTier.parseRoutedExpertName(lingNorms)
        #expect(r2c?.layer == 7)
        #expect(r2c?.expert == 42)

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

        let qwen35Norms = "model.layers.4.mlp.switch_mlp.gate_proj.tq_norms"
        let r7n = JangPressMmapTier.parseRoutedExpertName(qwen35Norms)
        #expect(r7n?.layer == 4)
        #expect(r7n?.expert == 0)

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

        // Pattern O — MiniMax prestacker overlay, stacked on axis 0.
        let minimaxPrestacked =
            "model.layers.12.block_sparse_moe.switch_mlp.gate_proj.tq_packed"
        let r9b = JangPressMmapTier.parseRoutedExpertName(minimaxPrestacked)
        #expect(r9b?.layer == 12)
        #expect(r9b?.expert == 0)

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

        // Pattern N — Gemma 4 text/VLM omits `.mlp.` before switch_mlp.
        let gemma4 = "model.language_model.layers.8.switch_mlp.up_proj.tq_norms"
        let r14 = JangPressMmapTier.parseRoutedExpertName(gemma4)
        #expect(r14?.layer == 8)
        #expect(r14?.expert == 0)

        // Pattern P — DeepSeek V3/V4 canonical prestacked overlay.
        let dsv4Stacked = "layers.19.ffn.switch_mlp.down_proj.tq_packed"
        let r15 = JangPressMmapTier.parseRoutedExpertName(dsv4Stacked)
        #expect(r15?.layer == 19)
        #expect(r15?.expert == 0)

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
        // iter 25 (Issue 4): stacked tiles in layer 1 are split into
        // per-expert byte sub-ranges using shape[0] (=2 in the test
        // fixture). So layer 1 now reports 2 split-experts instead of 1
        // synthetic-whole-tile entry.
        //   Layer 0: 2 per-expert entries (unchanged)
        //   Layer 1: 2 split-from-stacked entries (was 1)
        //   Total = 4 (was 3)
        #expect(stats.expertCount == 4)
        #expect(stats.byLayer[0] == 2)
        #expect(stats.byLayer[1] == 2)
        // Total routed bytes is unchanged by the split (whole-tile
        // bytes are now distributed across N per-expert sub-ranges).
        // 6 (per-expert L0) × 32 + 3 projections × 2 experts × 16 (L1
        // split-half) = 192 + 96 = 288 B.
        #expect(stats.totalRoutedBytes == 288)
    }

    // (former acquire/release test deleted in iter 26 — those methods
    //  were removed from JangPressMmapTier alongside the orphaned
    //  controller. The tier is now a tile-classification probe; see
    //  docs/WIRED-LIMIT-INVESTIGATION-2026-05-03.md.)

    @Test("startCold flag triggers initial dontNeed pass")
    func startCold() throws {
        let dir = try Self.standardBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Just verify the constructor doesn't blow up with startCold=true.
        // The actual page-state observation requires kernel/vmstat hooks.
        let tier = try JangPressMmapTier(
            config: .init(bundleURL: dir, hotPercent: 0, startCold: true))
        // iter 25: stacked tile in layer 1 splits into 2 per-expert
        // entries (shape[0]=2 in the test fixture), so total = 2 + 2 = 4.
        #expect(tier.snapshot().expertCount == 4)
    }

    // (former forceRelease test deleted in iter 26 — see acquire/release
    //  note above. Byte-level shard reads are still covered by the
    //  JangPressShard test suite.)

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
