// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressEmbedTierTests — verify the embed/lm_head Zipfian tier
// (component F) opens shards, identifies the embedding/lm_head
// tensors, records token activity, and applies WILLNEED/DONTNEED
// advise per the configured hotPercent.
//
// Synthesizes a fake bundle with one shard:
//   model.embed_tokens.weight  bf16 shape=[100, 16]  (100-vocab × 16-hidden)
//   lm_head.weight             bf16 shape=[100, 16]
//   model.norm.weight          bf16 shape=[16]      (irrelevant)
//
// Then drives `recordTokenActivity` and `applyZipfianAdvise` and
// verifies the snapshot matches.

import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JangPressEmbedTier")
struct JangPressEmbedTierTests {

    static func makeBundle() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("zipfian-tier-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Two 2D tensors at 100×16 bf16 = 100 × 16 × 2 = 3200 bytes each.
        // Plus a 1-D norm tensor.
        let header: [String: Any] = [
            "model.embed_tokens.weight": [
                "dtype": "BF16",
                "shape": [100, 16],
                "data_offsets": [0, 3200],
            ],
            "lm_head.weight": [
                "dtype": "BF16",
                "shape": [100, 16],
                "data_offsets": [3200, 6400],
            ],
            "model.norm.weight": [
                "dtype": "BF16",
                "shape": [16],
                "data_offsets": [6400, 6432],
            ],
        ]
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let headerSize = UInt64(json.count)

        var bytes = Data()
        bytes.append(contentsOf: withUnsafeBytes(of: headerSize.littleEndian) { Array($0) })
        bytes.append(json)
        bytes.append(Data(repeating: 0xAB, count: 6432))   // tensor data

        let shardURL = dir.appendingPathComponent("model.safetensors")
        try bytes.write(to: shardURL)
        return dir
    }

    @Test("opens bundle + finds embed_tokens and lm_head")
    func opensBundle() throws {
        let dir = try Self.makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressEmbedTier(
            config: .init(bundleURL: dir, hotPercent: 1, skipLMHead: false))
        let s = tier.snapshot()
        #expect(s.hasEmbedTokens == true)
        #expect(s.hasLMHead == true)
        #expect(s.vocabSize == 100)
        #expect(s.hiddenSize == 16)
        #expect(s.hotPercent == 1)
    }

    @Test("skipLMHead config drops lm_head")
    func skipLMHead() throws {
        let dir = try Self.makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressEmbedTier(
            config: .init(bundleURL: dir, hotPercent: 5, skipLMHead: true))
        let s = tier.snapshot()
        #expect(s.hasEmbedTokens == true)
        #expect(s.hasLMHead == false)   // intentionally skipped
    }

    @Test("recordTokenActivity tracks frequencies")
    func recordTokenActivity() throws {
        let dir = try Self.makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressEmbedTier(
            config: .init(bundleURL: dir, hotPercent: 5))
        // Hot tokens (will be in top 5 %): 0, 1, 2
        for _ in 0..<100 {
            tier.recordTokenActivity([0, 1, 2])
        }
        // Cold tokens (used once)
        for t in 50..<60 {
            tier.recordTokenActivity([t])
        }
        let s = tier.snapshot()
        #expect(s.observedTokenSamples == 100 * 3 + 10)
        #expect(s.distinctTokensSeen == 3 + 10)
    }

    @Test("applyZipfianAdvise runs without crashing")
    func applyZipfianAdvise() throws {
        let dir = try Self.makeBundle()
        defer { try? FileManager.default.removeItem(at: dir) }

        let tier = try JangPressEmbedTier(
            config: .init(bundleURL: dir, hotPercent: 5))
        for _ in 0..<50 { tier.recordTokenActivity([0, 1, 2]) }
        for t in 80..<100 { tier.recordTokenActivity([t]) }

        // No throw, no crash. Per-row madvise issues 100 calls per
        // tensor so this exercises the full loop.
        tier.applyZipfianAdvise()
        // Idempotent — second call is fine.
        tier.applyZipfianAdvise()
    }
}
