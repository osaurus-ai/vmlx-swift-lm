// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressMmapTier — bundle-aware mmap probe for routed MoE expert
// weights.
//
// PURPOSE
// =======
// Open safetensors shards in a bundle as read-only `JangPressShard`s
// and walk their tensor indexes to identify routed-expert tiles
// (per-architecture regex patterns). Build an in-memory map of
// (layer, expert) → (shard, byteRange) so status surfaces can report
// routed tile counts and routed bytes.
//
// This tier is a probe/status utility. It is not the canonical MLX
// weight store, and it does not drive per-token acquire/release in
// production. Real steady-state RAM savings come only when the pinned
// osaurus mlx-swift runtime honors `MLX_SAFETENSORS_MMAP=1` and loads
// safetensors through the C++ whole-shard mmap loader.
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
//   JangPressMmapTier (file-backed mmap probe)
//     • Uses the bundle file as the source of truth.
//     • No durable extra RAM — pages are file-backed and can be
//       reclaimed by the kernel. Multiple opens of the same file share
//       pages.
//     • Doesn't conflict with MLX — they hold ANOTHER copy in their
//       allocator. Our mmap is a parallel read-only view.
//     • The probe alone does not save canonical model RAM. The win
//       comes from the MLX C++ mmap safetensors loader replacing the
//       stock `pread()` copy with mmap-backed tensor storage.
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

    /// 0..100 legacy hot fraction retained for API/source
    /// compatibility. The production probe no longer has per-token
    /// acquire/release call sites, so this value is informational.
    public var hotPercent: Int

    /// When true, routed ranges in this probe mapping are marked
    /// `MADV_DONTNEED` after indexing. This can release redundant file
    /// cache pages created by the probe, but it does not release the
    /// canonical MLX tensor storage unless the patched MLX mmap loader
    /// owns that storage.
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
    /// Lazily populated by `ensureBuilt()` on first use.
    public var shards: [URL: JangPressShard] {
        ensureBuilt()
        return _shards
    }
    private var _shards: [URL: JangPressShard] = [:]

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
    /// Lazily populated by `ensureBuilt()` on first use.
    public var experts: [TileKey: ExpertRanges] {
        ensureBuilt()
        return _experts
    }
    private var _experts: [TileKey: ExpertRanges] = [:]

    /// iter 25: layers whose routed-expert tensor was a STACKED tile
    /// (one safetensor of shape `[N_experts, ...]`) → number of experts
    /// the tile was split into. Used by `acquireLayer(_:)` to WILLNEED
    /// the whole stack when the caller doesn't know per-route experts.
    /// Lazily populated by `ensureBuilt()` on first use.
    public var stackedLayers: [Int: Int] {
        ensureBuilt()
        return _stackedLayerExpertCount
    }
    private var _stackedLayerExpertCount: [Int: Int] = [:]

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

    /// iter 24: build state machine. Init is now O(1) and does no I/O.
    /// All shard opens, header parses, and tile indexing are deferred
    /// to `ensureBuilt()` which fires on first acquire/release/snapshot.
    /// This addresses the load-time SIGKILL on memory-tight hosts where
    /// JangPress was racing MLX's pread for limited RAM.
    private let buildLock = NSLock()
    private var didBuild = false
    private var buildError: Error?

    public init(config: JangPressMmapConfig) throws {
        self.config = config
        // No work in init. The bundle URL is validated lazily inside
        // ensureBuilt() so that Engine.load completes regardless of
        // whether the bundle actually exists / is readable. A missing
        // bundle becomes a noop tier (acquire/release no-op cleanly)
        // rather than a hard load failure.
    }

    /// Lazily perform shard sniff + mmap + tile indexing. Idempotent;
    /// concurrent callers will block on `buildLock` and observe the
    /// already-built state. Errors are captured + re-thrown on the
    /// first call only (subsequent calls observe no-op state).
    private func ensureBuilt() {
        // Fast path — already built.
        if didBuild { return }
        buildLock.lock(); defer { buildLock.unlock() }
        if didBuild { return }
        defer { didBuild = true }

        let fm = FileManager.default
        let shardURLs: [URL]
        do {
            shardURLs = try fm.contentsOfDirectory(
                at: config.bundleURL,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            ).filter { $0.pathExtension == "safetensors" }
        } catch {
            buildError = error
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] enumerate \(config.bundleURL.path) failed: \(error) — tier inert\n".utf8))
            return
        }

        // iter 19: header-sniff each shard FIRST to identify which
        // contain routed-expert tensors. Skip mmap'ing shards that have
        // only attention/embed/lm_head/etc.
        var openedShards: [URL: JangPressShard] = [:]
        var skippedCount = 0
        for url in shardURLs {
            guard let names = JangPressShard.sniffTensorNames(at: url) else {
                do {
                    openedShards[url] = try JangPressShard(path: url)
                } catch {
                    FileHandle.standardError.write(Data(
                        "[JangPressMmapTier] sniff+open failed \(url.lastPathComponent): \(error)\n".utf8))
                }
                continue
            }
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
        self._shards = openedShards
        if skippedCount > 0 {
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] lazy-built: sniffed \(shardURLs.count) shards, mmap'd \(openedShards.count), skipped \(skippedCount) (no routed experts)\n".utf8))
        }

        // Walk tensor names + build (layer, expert) → byte-range map.
        //
        // iter 25 (Issue 4): stacked-tile patterns (A, C, D, G, L, M)
        // hold ALL experts of a layer in one safetensors tensor of
        // shape `[N_experts, ...]`. Previously we registered the WHOLE
        // tile under synthetic expert id 0, so `acquire(layer, [e])`
        // WILLNEED-faulted the entire 67-304 MB tile (60-200 ms cold-
        // fault per layer). We now split stacked tensors into per-
        // expert byte sub-ranges using shape[0] as the stacked-axis
        // dim, so `acquire(layer, [e])` only faults
        // ~total_bytes / N_experts (1-2 MB) for each routed expert.
        //
        // Per-expert patterns (B, E, F, H, I, J, K) already have the
        // expert id encoded in the tensor name; they take the no-split
        // path unchanged.
        //
        // Layers indexed via the stacked path are tracked in
        // `_stackedLayerExpertCount` so `acquireLayer(_:)` can WILLNEED
        // the whole stack when the caller doesn't know the routing
        // decision (legacy hint path, controller wake-all).
        var byKey: [TileKey: [(URL, Range<UInt64>)]] = [:]
        var stackedLayerN: [Int: Int] = [:]
        var descriptorCount = 0
        var parsedRoutedNames = 0
        var missingDescriptors = 0
        var stackedTensorNames = 0
        var wholeTensorNames = 0
        for (url, shard) in openedShards {
            descriptorCount += shard.tensors.count
            for name in shard.tensors.keys {
                guard let (layer, expert) = Self.parseRoutedExpertName(name) else {
                    continue
                }
                parsedRoutedNames += 1
                guard let desc = shard.descriptor(for: name) else {
                    missingDescriptors += 1
                    continue
                }
                let isStacked = expert == 0 && Self.isStackedTensorName(name)
                if isStacked, !desc.shape.isEmpty, desc.shape[0] > 1 {
                    stackedTensorNames += 1
                    // Split [N, …] tensor into N per-expert sub-ranges.
                    let nExperts = desc.shape[0]
                    let perExpertBytes = desc.dataLength / UInt64(nExperts)
                    // Sanity check: total must be evenly divisible by N.
                    // If not (corrupt header / unexpected layout), fall
                    // back to whole-tile registration under id 0.
                    if perExpertBytes * UInt64(nExperts) != desc.dataLength {
                        let fullRange = desc.dataOffset
                            ..< desc.dataOffset + desc.dataLength
                        byKey[TileKey(layer: layer, expert: 0),
                              default: []].append((url, fullRange))
                        continue
                    }
                    stackedLayerN[layer] = max(stackedLayerN[layer] ?? 0,
                                               nExperts)
                    for e in 0..<nExperts {
                        let off = desc.dataOffset
                            + UInt64(e) * perExpertBytes
                        let subrange = off ..< off + perExpertBytes
                        byKey[TileKey(layer: layer, expert: e),
                              default: []].append((url, subrange))
                    }
                } else {
                    wholeTensorNames += 1
                    let fullRange = desc.dataOffset
                        ..< desc.dataOffset + desc.dataLength
                    byKey[TileKey(layer: layer, expert: expert),
                          default: []].append((url, fullRange))
                }
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
        self._experts = built
        self._stackedLayerExpertCount = stackedLayerN

        let debug = ProcessInfo.processInfo.environment["JANGPRESS_DEBUG"] == "1"
            || ProcessInfo.processInfo.environment["JANGPRESS_MMAP_DEBUG"] == "1"
        if debug {
            FileHandle.standardError.write(Data(
                "[JangPressMmapTier] index: descriptors=\(descriptorCount) parsed=\(parsedRoutedNames) missingDesc=\(missingDescriptors) stacked=\(stackedTensorNames) whole=\(wholeTensorNames) experts=\(built.count) layers=\(stackedLayerN.count) routedBytes=\(built.values.reduce(UInt64(0)) { $0 + $1.totalBytes })\n".utf8))
        }

        // startCold: mark every routed probe range DONTNEED right after
        // building. This releases only the probe's file-backed pages; it
        // does not affect stock MLX pread copies.
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

    // MARK: - Probe API
    //
    // The acquire/release/madvise surface that lived here was deleted
    // in iter 26 — none of it had production call sites and the
    // parallel-mmap design couldn't actually compress anything (see
    // docs/WIRED-LIMIT-INVESTIGATION-2026-05-03.md). What remains is
    // the tile-classification probe (`snapshot()` / `snapshotIfBuilt()`)
    // which `LoadBundleFacts.inspect` and tests use to count routed
    // bytes per layer/expert without faulting in the data segment.

    // MARK: - Stats

    public struct Stats: Sendable {
        public var shardCount: Int
        public var expertCount: Int
        public var totalRoutedBytes: UInt64
        public var byLayer: [Int: Int]
        public var built: Bool
    }

    /// Returns a stats snapshot. Will trigger `ensureBuilt()` to populate
    /// real data; for the "is this initialized?" probe use
    /// `snapshotIfBuilt()` instead.
    public func snapshot() -> Stats {
        ensureBuilt()
        var byLayer: [Int: Int] = [:]
        var total: UInt64 = 0
        for (key, r) in _experts {
            byLayer[key.layer, default: 0] += 1
            total += r.totalBytes
        }
        return Stats(
            shardCount: _shards.count,
            expertCount: _experts.count,
            totalRoutedBytes: total,
            byLayer: byLayer,
            built: didBuild)
    }

    /// Probe-only snapshot: does NOT trigger ensureBuilt(). Returns
    /// zeros + built=false until something actually uses the tier.
    public func snapshotIfBuilt() -> Stats {
        buildLock.lock(); defer { buildLock.unlock() }
        if !didBuild {
            return Stats(shardCount: 0, expertCount: 0, totalRoutedBytes: 0, byLayer: [:], built: false)
        }
        var byLayer: [Int: Int] = [:]
        var total: UInt64 = 0
        for (key, r) in _experts {
            byLayer[key.layer, default: 0] += 1
            total += r.totalBytes
        }
        return Stats(
            shardCount: _shards.count,
            expertCount: _experts.count,
            totalRoutedBytes: total,
            byLayer: byLayer,
            built: true)
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
    /// iter 25: returns true iff the tensor name matches one of the
    /// STACKED-expert layout patterns (A, C, D, G, L, M). Stacked
    /// tensors hold all experts of a layer in a single tensor of shape
    /// `[N_experts, ...]`; the indexer splits these into per-expert
    /// byte sub-ranges so `acquire(layer, [e])` only WILLNEEDs the
    /// routed expert's slice.
    public static func isStackedTensorName(_ name: String) -> Bool {
        let range = NSRange(name.startIndex..<name.endIndex, in: name)
        for regex in [switchMlpRegex, jangtqStackedRegex, affineStackedRegex,
                      switchMlpJangtqRegex, nemotronSwitchMlpRegex,
                      nemotronSwitchMlpAffineRegex] {
            if regex.firstMatch(in: name, range: range) != nil { return true }
        }
        return false
    }

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
