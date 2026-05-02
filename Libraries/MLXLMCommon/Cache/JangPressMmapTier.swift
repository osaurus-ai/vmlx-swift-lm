// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressMmapTier — bundle-aware mmap+madvise tier for routed
// MoE expert weights.
//
// PURPOSE
// =======
// Open every safetensors shard in a bundle once at load time as a
// read-only `JangPressShard`. Walk the shards' tensor indexes to
// identify which entries are routed-expert tiles (per-architecture
// regex patterns). Build an in-memory map of (layer, expert) →
// (shard, byteRange). At inference time:
//
//   • acquire(layer, experts) → call madvise(.willNeed) on those
//     ranges so the kernel pre-faults.
//   • release(layer, experts) → call madvise(.dontNeed) so the
//     kernel can reclaim those pages under pressure.
//
// COMPARED TO JangPressMachCache
// =======================================
// Different tradeoff space:
//
//   JangPressMachCache (vm_purgable_control)
//     • Owns its own copy of weights in fresh purgeable VM regions.
//     • Independent of how MLX stores tensors.
//     • Doubles RAM at load (until MLX integration replaces the
//       canonical storage with our region — gated on MLX-swift fork).
//     • Kernel uses WKdm to compress dormant pages.
//
//   JangPressMmapTier (file-backed mmap + madvise)
//     • Uses the bundle file as the source of truth.
//     • No extra RAM — pages are file-backed, shared with the kernel
//       page cache. Multiple opens of the same file share pages.
//     • Discard reclaims pages back to the file (no compression);
//       refault re-reads from disk.
//     • Doesn't conflict with MLX — they hold ANOTHER copy in their
//       allocator. Our mmap is a parallel read-only view.
//     • Win comes when memory pressure removes our mmap pages —
//       MLX's copies are still resident, so the model still works.
//       BUT: in the future, if MLX is taught to read from our mmap
//       directly (replacing its allocator), we save the duplicate
//       copy entirely.
//
// REGEX PATTERNS PER FAMILY
// =========================
// Routed-expert tile names follow architecture-family conventions.
// We support the most common shapes today:
//
//   model.layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.weight
//     (Qwen 3.5/3.6, GLM 4/5, MiniMax, Laguna stacked-expert format)
//
//   model.layers.<L>.mlp.experts.<E>.<gate|up|down>_proj.weight
//     (DSV3 / DSV4 / Kimi K2.x per-expert format)
//
// The patterns are matched per-bundle at construction time — we
// detect which scheme is in use by counting tensor names that match
// each.

import Foundation

public struct JangPressMmapConfig: Sendable {
    /// Path to a bundle directory holding safetensors shards. We open
    /// every `*.safetensors` file in this directory.
    public let bundleURL: URL

    /// 0..100 — fraction of the routed-expert pool to keep MADV_WILLNEED
    /// ("hot") at all times. Bottom (1 - hotPct/100) gets MADV_DONTNEED
    /// when idle. Default 30 % hot.
    public var hotPercent: Int

    /// When true, the entire data area of every shard starts with
    /// MADV_DONTNEED. The first inference's `acquire()` calls switch
    /// the active set to MADV_WILLNEED. Use this for memory-tight
    /// startups; under low pressure on a roomy machine prefer false
    /// (default) so all weights start resident.
    public var startCold: Bool

    public init(bundleURL: URL, hotPercent: Int = 30, startCold: Bool = false) {
        self.bundleURL = bundleURL
        self.hotPercent = max(0, min(100, hotPercent))
        self.startCold = startCold
    }
}

public final class JangPressMmapTier: @unchecked Sendable {

    public let config: JangPressMmapConfig

    /// Shards opened from the bundle. Held strongly so the mmap
    /// regions stay alive for the lifetime of the tier.
    public private(set) var shards: [URL: JangPressShard] = [:]

    /// (layer, expert) → list of (shard, byteRange) for the three
    /// expert projections (gate / up / down). One expert may contribute
    /// multiple ranges (one per projection).
    public struct ExpertRanges: Sendable {
        public let layer: Int
        public let expert: Int
        public let parts: [(shard: URL, range: Range<UInt64>)]
        public var totalBytes: UInt64 {
            parts.reduce(0) { $0 + ($1.range.upperBound - $1.range.lowerBound) }
        }
    }
    public private(set) var experts: [TileKey: ExpertRanges] = [:]

    /// Tile identifier — matches the shape used by
    /// `JangPressMachCache`.
    public struct TileKey: Hashable, Sendable {
        public let layer: Int
        public let expert: Int
        public init(layer: Int, expert: Int) {
            self.layer = layer
            self.expert = expert
        }
    }

    public init(config: JangPressMmapConfig) throws {
        self.config = config

        // 1. Find every *.safetensors shard.
        let fm = FileManager.default
        let shardURLs = try fm.contentsOfDirectory(
            at: config.bundleURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ).filter { $0.pathExtension == "safetensors" }

        // 2. iter 19: header-sniff each shard FIRST to identify which
        //    contain routed-expert tensors. Skip mmap'ing shards that
        //    have only attention/embed/lm_head/etc — that reduces
        //    page-cache competition with MLX-swift's own mmap views
        //    of those same files, and addresses the OFF→ON output drift
        //    documented in JANGPRESS-DEEP-TRACE.md Issue 5b.
        var openedShards: [URL: JangPressShard] = [:]
        var skippedCount = 0
        for url in shardURLs {
            // Header-only sniff: which tensor names are in this shard?
            guard let names = JangPressShard.sniffTensorNames(at: url) else {
                // Header parse failed — fall back to full open, log.
                do {
                    openedShards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressMmapTier] sniff+open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
                continue
            }
            // Does this shard contain ANY routed-expert tensor?
            let hasRoutedExpert = names.contains(where: { name in
                Self.parseRoutedExpertName(name) != nil
            })
            if hasRoutedExpert {
                do {
                    openedShards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressMmapTier] open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
            } else {
                skippedCount += 1
            }
        }
        self.shards = openedShards
        if skippedCount > 0 {
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] sniffed \(shardURLs.count) shards, mmap'd \(openedShards.count), skipped \(skippedCount) (no routed experts)\n".utf8))
        }

        // 3. Walk every tensor name across all shards, attribute to
        //    (layer, expert) via the two known regex patterns.
        var byKey: [TileKey: [(URL, Range<UInt64>)]] = [:]
        for (url, shard) in openedShards {
            for name in shard.tensors.keys {
                guard let (layer, expert) = Self.parseRoutedExpertName(name),
                      let range = shard.byteRange(for: name)
                else { continue }
                byKey[TileKey(layer: layer, expert: expert), default: []].append((url, range))
            }
        }

        var built: [TileKey: ExpertRanges] = [:]
        for (key, parts) in byKey {
            built[key] = ExpertRanges(
                layer: key.layer,
                expert: key.expert,
                parts: parts.map { (shard: $0.0, range: $0.1) }
            )
        }
        self.experts = built

        // 4. Initial advise. If startCold, mark every routed range
        //    as DONTNEED — the first inference will WILLNEED its top-k.
        if config.startCold {
            for (_, ranges) in built {
                for part in ranges.parts {
                    if let shard = openedShards[part.shard] {
                        shard.advise(.dontNeed, range: part.range)
                    }
                }
            }
        }
    }

    // MARK: - Routing API

    /// Pre-fault the given experts. Does the corresponding number of
    /// `madvise(MADV_WILLNEED)` syscalls — one per (gate/up/down) part.
    public func acquire(layer: Int, experts: [Int]) {
        for e in experts {
            guard let r = self.experts[TileKey(layer: layer, expert: e)] else { continue }
            for part in r.parts {
                shards[part.shard]?.advise(.willNeed, range: part.range)
            }
        }
    }

    /// Mark the given experts as MADV_DONTNEED — kernel can reclaim
    /// their pages under pressure.
    public func release(layer: Int, experts: [Int]) {
        for e in experts {
            guard let r = self.experts[TileKey(layer: layer, expert: e)] else { continue }
            for part in r.parts {
                shards[part.shard]?.advise(.dontNeed, range: part.range)
            }
        }
    }

    /// Stronger version of `release` — uses `msync(MS_INVALIDATE)` to
    /// force the kernel to drop pages immediately rather than treating
    /// it as a hint. Use during quiesce-time compaction when you're
    /// confident these experts will stay dormant for ≥30 s. Cost: the
    /// next acquire pays a disk re-fault (~ms per tile).
    public func forceRelease(layer: Int, experts: [Int]) {
        for e in experts {
            guard let r = self.experts[TileKey(layer: layer, expert: e)] else { continue }
            for part in r.parts {
                shards[part.shard]?.forceInvalidate(range: part.range)
            }
        }
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var shardCount: Int
        public var expertCount: Int
        public var totalRoutedBytes: UInt64
        public var byLayer: [Int: Int]
    }

    public func snapshot() -> Stats {
        var byLayer: [Int: Int] = [:]
        var total: UInt64 = 0
        for (key, r) in experts {
            byLayer[key.layer, default: 0] += 1
            total += r.totalBytes
        }
        return Stats(
            shardCount: shards.count,
            expertCount: experts.count,
            totalRoutedBytes: total,
            byLayer: byLayer)
    }

    // MARK: - Tensor name parsing

    // VL bundles wrap the language tower under various namespace prefixes:
    //   • Plain text: `model.layers.<L>...`
    //   • Holo3 VL outside: `language_model.model.layers.<L>...`
    //   • Qwen3.6 VL inside: `model.language_model.layers.<L>...`
    //   • Plain (some affines): `language_model.layers.<L>...`
    // The shared prefix matches zero or more `model.` / `language_model.`
    // chunks before the trailing `layers.` anchor. This costs one
    // backtrack step per call (cheap) and covers all observed layouts.
    private static let vlPrefix = #"(?:(?:model|language_model)\.)*"#

    // Pattern A — Qwen/GLM/MiniMax fp16 stacked layout:
    //   [<vlPrefix>]layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.weight
    private static let switchMlpRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern B — Mistral 4 / DSV3.x / Kimi K2 per-expert layout:
    //   [<vlPrefix>]layers.<L>.mlp.experts.<E>.<gate|up|down>_proj.weight
    private static let perExpertMlpRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern C — Laguna / Qwen3.6 / MiniMax JANGTQ stacked:
    //   [<vlPrefix>]layers.<L>.mlp.experts.<gate_up_proj|down_proj>.tq_packed
    private static let jangtqStackedRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.tq_packed$"#)

    // Pattern D — JANG_2L / MXFP4 affine stacked:
    //   [<vlPrefix>]layers.<L>.mlp.experts.<gate_up_proj|down_proj>.weight
    private static let affineStackedRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.experts\.(?:gate_up_proj|down_proj|gate_proj|up_proj)\.weight$"#)

    // Pattern G — Holo3 / Qwen3.5MoE JANGTQ switch_mlp (per-projection
    // TQ-packed, one stacked tile per layer per projection):
    //   [<vlPrefix>]layers.<L>.mlp.switch_mlp.<gate|up|down>_proj.tq_packed
    private static let switchMlpJangtqRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.tq_packed$"#)

    // Pattern H — MiniMax M2 / M2.7 per-expert JANGTQ:
    //   [<vlPrefix>]layers.<L>.block_sparse_moe.experts.<E>.w[123].tq_packed
    private static let minimaxBlockSparseRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.tq_packed$"#)

    // Pattern I — MiniMax affine JANG (no .tq_packed suffix):
    //   [<vlPrefix>]layers.<L>.block_sparse_moe.experts.<E>.w[123].weight
    private static let minimaxBlockSparseAffineRegex = try! NSRegularExpression(
        pattern: #"^"# + vlPrefix + #"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w[123]\.weight$"#)

    // Pattern J — Nemotron Omni / Cascade nemotron_h JANGTQ:
    //   backbone.layers.<L>.mixer.experts.<E>.<gate|up|down>_proj.tq_packed
    // Nvidia uses `backbone.layers` + `mixer` (since hybrid SSM/attn
    // mixer pattern), with the same projection trio as Qwen.
    private static let nemotronMixerRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.tq_packed$"#)

    // Pattern K — Nemotron affine variant:
    //   backbone.layers.<L>.mixer.experts.<E>.<gate|up|down>_proj.weight
    private static let nemotronMixerAffineRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern L — Nemotron MXFP4 stacked switch_mlp (one tile per layer):
    //   backbone.layers.<L>.mixer.switch_mlp.<fc1|fc2>.weight
    // The .biases and .scales sidecars are tiny — only .weight matters
    // for compression-target purposes.
    private static let nemotronSwitchMlpRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.fc[12]\.weight$"#)

    // Pattern M — Nemotron Cascade-2 affine stacked switch_mlp:
    //   backbone.layers.<L>.mixer.switch_mlp.<gate|up|down>_proj.weight
    private static let nemotronSwitchMlpAffineRegex = try! NSRegularExpression(
        pattern: #"^backbone\.layers\.(\d+)\.mixer\.switch_mlp\.(?:gate|up|down)_proj\.weight$"#)

    // Pattern E — DeepSeek V4 per-expert JANGTQ (NEW iter 12).
    // Note the differences from pattern B:
    //   • NO `model.` prefix (DSV4's own naming convention)
    //   • `ffn` instead of `mlp`
    //   • `w1` / `w2` / `w3` instead of gate/up/down_proj
    //   • `.tq_packed` (or `.tq_norms` / `.tq_bits`) suffix
    //
    // Catches both routed AND hash-routed (DSV4 L0-L2) layers since
    // they share the same physical naming — only the router upstream
    // differs. Component H from CACHE-ARCHITECTURE.md.
    //
    //   layers.<L>.ffn.experts.<E>.<w1|w2|w3>.tq_packed
    private static let dsv4PerExpertRegex = try! NSRegularExpression(
        pattern: #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.tq_packed$"#)

    // Pattern F — DeepSeek V4 per-expert affine (e.g. JANG_2L of DSV4):
    //   layers.<L>.ffn.experts.<E>.<w1|w2|w3>.weight
    private static let dsv4PerExpertAffineRegex = try! NSRegularExpression(
        pattern: #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.weight$"#)

    /// Parse a tensor name and return (layer, expert) if it's a routed
    /// expert tile, else nil. For stacked-expert layouts (patterns A,
    /// C, D) where one tensor holds ALL N experts, we synthesize
    /// expert id 0 (the caller acquires/releases the whole stack via
    /// experts=[0] for that layer).
    public static func parseRoutedExpertName(_ name: String) -> (layer: Int, expert: Int)? {
        let range = NSRange(name.startIndex..<name.endIndex, in: name)

        // Per-expert patterns (B + E + F + H + I) FIRST — they have
        // numeric expert ids in path so we want fine-grained per-expert
        // tracking, not the synthetic id 0 of the stacked layouts.
        for perExpertRegex in [perExpertMlpRegex, dsv4PerExpertRegex, dsv4PerExpertAffineRegex,
                               minimaxBlockSparseRegex, minimaxBlockSparseAffineRegex,
                               nemotronMixerRegex, nemotronMixerAffineRegex] {
            if let m = perExpertRegex.firstMatch(in: name, range: range), m.numberOfRanges >= 3 {
                guard
                    let lr = Range(m.range(at: 1), in: name),
                    let er = Range(m.range(at: 2), in: name),
                    let layer = Int(name[lr]),
                    let expert = Int(name[er])
                else { return nil }
                return (layer, expert)
            }
        }

        // Stacked patterns (A + C + D + G + L). Each whole layer = one tile;
        // synthetic expert id 0.
        for regex in [switchMlpRegex, jangtqStackedRegex, affineStackedRegex,
                      switchMlpJangtqRegex, nemotronSwitchMlpRegex, nemotronSwitchMlpAffineRegex] {
            if let m = regex.firstMatch(in: name, range: range), m.numberOfRanges >= 2 {
                guard
                    let lr = Range(m.range(at: 1), in: name),
                    let layer = Int(name[lr])
                else { continue }
                return (layer, 0)
            }
        }

        return nil
    }
}
