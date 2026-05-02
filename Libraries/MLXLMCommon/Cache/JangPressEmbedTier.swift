// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressEmbedTier — page-level Zipfian compression for the
// embedding table and lm_head. Component F from
// `Cache/CACHE-ARCHITECTURE.md`.
//
// MOTIVATION
// ==========
// `model.embed_tokens.weight` and `model.lm_head.weight` are the two
// vocab-sized matrices in the model. Per decode step we touch:
//
//   • embed_tokens: ONE row (the input token id)
//   • lm_head:      ONE row when greedy / sampling; the entire matrix
//                   for argmax over vocab. Most production sampling
//                   uses temperature/top-p which DOES touch every row,
//                   but the post-softmax distribution is Zipfian — a
//                   handful of rows dominate the probability mass.
//
// On a 128 K vocab × 4096 hidden bf16 model that's ~1 GB per matrix.
// If we identify the top-1 % most-frequent token rows (~1.3 K rows),
// pin them MADV_WILLNEED, and let the rest be MADV_DONTNEED, the
// kernel can keep ~99 % of the vocab evictable. Practical save:
// ~1-2 GB across embed + lm_head depending on activation pattern.
//
// COMPATIBILITY WITH JANGPRESS ROUTED-EXPERT TIER
// ===============================================
// This tier is independent of `JangPressMmapTier`. They both use
// `JangPressShard` for the underlying mmap+madvise primitive but
// don't share state. A bundle can have both active simultaneously:
//
//   JangPressMmapTier   — covers routed-expert tiles
//   JangPressEmbedTier   — covers embed_tokens + lm_head rows
//   (JangPressMachCache for .mach backend, mutually exclusive
//    with JangPressMmapTier per Engine.LoadOptions selection)
//
// This module is **scaffold-only** in iter 8 — the public API + tests
// land but it's not yet integrated into the engine. Once the routed-
// expert path proves out on a real bundle the same wiring pattern
// extends here.

import Foundation

// MARK: - Errors

public enum JangPressEmbedError: Error, CustomStringConvertible {
    case missingEmbeddingTensor(URL)
    case rowSizeUnknown(tensor: String)

    public var description: String {
        switch self {
        case .missingEmbeddingTensor(let url):
            return "no embed_tokens.weight or lm_head.weight in \(url.lastPathComponent)"
        case .rowSizeUnknown(let t):
            return "cannot infer row size for \(t)"
        }
    }
}

// MARK: - Config

public struct JangPressEmbedConfig: Sendable {
    public let bundleURL: URL

    /// 0..100 — fraction of vocab kept MADV_WILLNEED. The remainder
    /// is MADV_DONTNEED-eligible. Default 1 % — Zipfian distributions
    /// concentrate most activations in the top ~1 % of vocab.
    public var hotPercent: Int

    /// If true, scan only `model.embed_tokens.weight`. Skips
    /// `model.lm_head.weight` (e.g. for tied embeddings where it
    /// doesn't exist as a separate tensor).
    public var skipLMHead: Bool

    public init(bundleURL: URL, hotPercent: Int = 1, skipLMHead: Bool = false) {
        self.bundleURL = bundleURL
        self.hotPercent = max(0, min(100, hotPercent))
        self.skipLMHead = skipLMHead
    }
}

// MARK: - Tier

public final class JangPressEmbedTier: @unchecked Sendable {

    public let config: JangPressEmbedConfig

    /// Shards that hold embed_tokens and/or lm_head.
    public private(set) var shards: [URL: JangPressShard] = [:]

    /// Per-tensor metadata we need at acquire/release time.
    public struct TensorView: Sendable {
        public let name: String
        public let shard: URL
        public let dtypeBytes: Int        // bytes per scalar (bf16=2, fp32=4)
        public let vocabSize: Int
        public let hiddenSize: Int
        public let dataOffset: UInt64     // absolute byte offset of row 0 in shard file
    }
    public private(set) var embedTokens: TensorView?
    public private(set) var lmHead: TensorView?

    /// Per-token-id activation count. Updated by `recordTokenActivity`
    /// during the first ~1000 decode steps; the warm-up window builds
    /// the Zipfian profile.
    private var tokenFrequency: [Int: UInt64] = [:]
    private var observedSamples: UInt64 = 0

    public init(config: JangPressEmbedConfig) throws {
        self.config = config

        // Embed + LM head tensor name candidates. Different model
        // families use different canonical names; we accept all common
        // variants.
        let embedCandidates: Set<String> = [
            "model.embed_tokens.weight",     // Llama, Mistral, Qwen, Gemma, GLM, etc.
            "embed_tokens.weight",           // some bundles drop the model. prefix
            "embed.weight",                  // DeepSeek-V4
            "language_model.embed_tokens.weight",  // VL wrappers
            "model.embed.weight",            // edge case
        ]
        let headCandidates: Set<String> = [
            "lm_head.weight",                // most architectures
            "head.weight",                   // DeepSeek-V4
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ]
        let allCandidates = embedCandidates.union(headCandidates)

        // 1. iter 20: header-sniff each shard FIRST. Only mmap the shards
        //    that actually contain embed_tokens or lm_head — typically
        //    just one shard out of 86 on DSV4. Saves ~37 seconds at load
        //    by eliminating 85 redundant JangPressShard.init mmap+parse
        //    cycles. Same fix as iter 19 sniff in JangPressMmapTier.
        let fm = FileManager.default
        let shardURLs = (try? fm.contentsOfDirectory(
            at: config.bundleURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
        )?.filter { $0.pathExtension == "safetensors" } ?? []

        var skippedCount = 0
        for url in shardURLs {
            // Header-only sniff: which tensor names are in this shard?
            guard let names = JangPressShard.sniffTensorNames(at: url) else {
                // Header parse failed — fall back to full open (rare).
                do {
                    self.shards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressEmbedTier] sniff+open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
                continue
            }
            let hasEmbedOrHead = names.contains(where: { allCandidates.contains($0) })
            if hasEmbedOrHead {
                do {
                    self.shards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressEmbedTier] open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
            } else {
                skippedCount += 1
            }
        }
        if skippedCount > 0 {
            FileHandle.standardError.write(Data(
                "[JangPressEmbedTier] sniffed \(shardURLs.count) shards, mmap'd \(self.shards.count), skipped \(skippedCount) (no embed/lm_head)\n".utf8))
        }

        // 2. Locate the embedding + LM head tensors among the (small)
        // set of shards we actually opened.
        for (url, shard) in shards {
            if self.embedTokens == nil {
                for name in embedCandidates {
                    if let d = shard.descriptor(for: name) {
                        self.embedTokens = Self.makeView(name: name, shard: url, descriptor: d)
                        break
                    }
                }
            }
            if self.lmHead == nil, !config.skipLMHead {
                for name in headCandidates {
                    if let d = shard.descriptor(for: name) {
                        self.lmHead = Self.makeView(name: name, shard: url, descriptor: d)
                        break
                    }
                }
            }
        }
    }

    /// Compute row-size from descriptor. Defaults to bf16 if dtype
    /// can't be parsed.
    private static func makeView(
        name: String, shard: URL, descriptor: TensorDescriptor
    ) -> TensorView? {
        guard descriptor.shape.count == 2 else { return nil }
        let vocab = descriptor.shape[0]
        let hidden = descriptor.shape[1]
        let dtypeBytes: Int
        switch descriptor.dtype {
        case "F32", "I32", "U32": dtypeBytes = 4
        case "F16", "BF16", "I16", "U16": dtypeBytes = 2
        case "I8", "U8", "F8_E4M3", "F8_E5M2": dtypeBytes = 1
        default: dtypeBytes = 2  // assume bf16 — most JANGTQ embeds
        }
        return TensorView(
            name: name, shard: shard, dtypeBytes: dtypeBytes,
            vocabSize: vocab, hiddenSize: hidden,
            dataOffset: descriptor.dataOffset)
    }

    // MARK: - Routing-time API

    /// Per-decode-step hook. Records token activity for the warm-up
    /// profile. Cheap (single dict update).
    public func recordTokenActivity(_ tokenIds: [Int]) {
        for t in tokenIds {
            tokenFrequency[t, default: 0] &+= 1
            observedSamples &+= 1
        }
    }

    /// After warm-up, set advise on the bottom (1 - hotPercent)% of
    /// vocab rows to MADV_DONTNEED. The hottest rows are kept
    /// MADV_WILLNEED. Idempotent — safe to call multiple times.
    public func applyZipfianAdvise() {
        guard !tokenFrequency.isEmpty else { return }
        let sorted = tokenFrequency.sorted { $0.value > $1.value }
        let total = sorted.count
        guard let embed = embedTokens else { return }

        let hotCount = max(1, Int(Double(embed.vocabSize) * Double(config.hotPercent) / 100.0))
        let hotIds = Set(sorted.prefix(hotCount).map { $0.key })

        // Mark hot rows WILLNEED, the rest DONTNEED, on each tensor view.
        for view in [embed, lmHead].compactMap({ $0 }) {
            guard let shard = shards[view.shard] else { continue }
            let rowBytes = UInt64(view.hiddenSize * view.dtypeBytes)

            // For each row in vocab, advise based on hot/cold status.
            // We could batch into runs of consecutive cold rows for
            // fewer madvise calls — TODO performance.
            for rowId in 0..<view.vocabSize {
                let start = view.dataOffset + UInt64(rowId) * rowBytes
                let end = start + rowBytes
                let advice: JangPressAdvice = hotIds.contains(rowId) ? .willNeed : .dontNeed
                shard.advise(advice, range: start..<end)
            }
        }
        _ = total
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var hasEmbedTokens: Bool
        public var hasLMHead: Bool
        public var vocabSize: Int
        public var hiddenSize: Int
        public var observedTokenSamples: UInt64
        public var distinctTokensSeen: Int
        public var hotPercent: Int
    }

    public func snapshot() -> Stats {
        Stats(
            hasEmbedTokens: embedTokens != nil,
            hasLMHead: lmHead != nil,
            vocabSize: embedTokens?.vocabSize ?? 0,
            hiddenSize: embedTokens?.hiddenSize ?? 0,
            observedTokenSamples: observedSamples,
            distinctTokensSeen: tokenFrequency.count,
            hotPercent: config.hotPercent)
    }
}
