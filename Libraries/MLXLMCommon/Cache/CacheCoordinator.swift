// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation
@preconcurrency import MLX
import os

// MARK: - CacheDetail

/// Identifies which cache tier satisfied a lookup.
public enum CacheDetail: String, Sendable {
    /// The in-memory paged KV cache.
    case paged
    /// The on-disk L2 cache.
    case disk
    /// No cache tier had a match.
    case miss
}

// MARK: - CacheFetchResult

/// The result of a unified cache lookup across all tiers.
public enum CacheFetchResult: Sendable {
    /// A cache hit with the matched prefix data.
    ///
    /// - Parameters:
    ///   - matchedTokens: Number of tokens matched from the cache.
    ///   - remainingTokens: Tokens that still need to be computed.
    ///   - detail: Which cache tier provided the hit.
    ///   - blocks: Paged cache blocks covering the matched prefix (empty for disk hits).
    ///   - ssmStates: Companion SSM states for hybrid models, if available.
    case hit(
        matchedTokens: Int,
        remainingTokens: [Int],
        detail: CacheDetail,
        blocks: [CacheBlock],
        ssmStates: [MLXArray]?,
        diskArrays: [String: MLXArray]? = nil
    )

    /// No cache tier had a match for the given tokens.
    case miss
}

// MARK: - CacheCoordinator

/// Unified cache coordinator that cascades lookups across paged (L1),
/// disk (L2), and SSM companion caches.
///
/// The coordinator implements a tiered fetch strategy:
/// 1. Try the in-memory paged cache first (fastest).
/// 2. Fall back to the on-disk cache if the paged cache misses.
/// 3. For hybrid models (with SSM layers), also fetch companion SSM state.
///
/// Thread safety for the `_isHybrid` flag is provided by `OSAllocatedUnfairLock`.
/// Individual sub-caches handle their own internal locking.
public final class CacheCoordinator: @unchecked Sendable {

    // MARK: - Properties

    /// The configuration used to create this coordinator.
    public let config: CacheCoordinatorConfig

    /// The in-memory paged KV cache, or `nil` if disabled.
    public let pagedCache: PagedCacheManager?

    /// The on-disk L2 cache, or `nil` if disabled.
    public let diskCache: DiskCache?

    /// The SSM state companion cache for hybrid models.
    public let ssmStateCache: SSMStateCache

    /// Whether the model has hybrid (attention + SSM) layers.
    private var _isHybrid: Bool = false

    /// Lock protecting `_isHybrid`.
    private let lock = OSAllocatedUnfairLock()

    // MARK: - Initialization

    /// Creates a new cache coordinator.
    ///
    /// Sub-caches are instantiated based on the configuration flags.
    ///
    /// - Parameter config: The cache configuration to use.
    public init(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        self.config = config

        if config.usePagedCache {
            self.pagedCache = PagedCacheManager(
                blockSize: config.pagedBlockSize,
                maxBlocks: config.maxCacheBlocks,
                modelKey: config.modelKey
            )
        } else {
            self.pagedCache = nil
        }

        if config.enableDiskCache {
            let dir = config.diskCacheDir
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx_disk_cache")
            self.diskCache = DiskCache(cacheDir: dir, maxSizeGB: config.diskCacheMaxGB, modelKey: config.modelKey)
        } else {
            self.diskCache = nil
        }

        self.ssmStateCache = SSMStateCache(maxEntries: config.ssmMaxEntries)
    }

    // MARK: - Hybrid Flag

    /// Set whether the model is hybrid (has both attention and SSM layers).
    ///
    /// When hybrid mode is active, the coordinator will also fetch/store
    /// SSM companion states alongside the KV cache data.
    ///
    /// - Parameter isHybrid: `true` for hybrid models.
    public func setHybrid(_ isHybrid: Bool) {
        lock.withLock { _isHybrid = isHybrid }
    }

    /// Whether the model is hybrid (has both attention and SSM layers).
    public var isHybrid: Bool {
        lock.withLock { _isHybrid }
    }

    // MARK: - Fetch

    /// Perform a tiered cache lookup for the given token sequence.
    ///
    /// The lookup cascades through cache tiers in order:
    /// 1. **Paged cache** (in-memory, block-aligned prefix matching).
    /// 2. **Disk cache** (exact match on full token sequence, then with one fewer token).
    /// 3. If all tiers miss, returns `.miss`.
    ///
    /// For hybrid models, SSM companion states are fetched alongside paged cache hits.
    ///
    /// - Parameter tokens: The full token sequence to look up.
    /// - Returns: A ``CacheFetchResult`` describing the outcome.
    public func fetch(tokens: [Int]) -> CacheFetchResult {
        // Tier 1: Paged cache (in-memory)
        if let pagedCache, let result = pagedCache.fetchPrefix(tokens: tokens) {
            var ssmStates: [MLXArray]? = nil

            if isHybrid {
                ssmStates = ssmStateCache.fetch(
                    tokens: tokens,
                    boundary: result.matchedTokens
                )
            }

            return .hit(
                matchedTokens: result.matchedTokens,
                remainingTokens: result.remainingTokens,
                detail: .paged,
                blocks: result.blocks,
                ssmStates: ssmStates,
                diskArrays: nil
            )
        }

        // Tier 2: Disk cache (exact match, then one-shorter fallback)
        if let diskCache {
            if let arrays = diskCache.fetch(tokens: tokens) {
                var ssmStates: [MLXArray]? = nil
                if isHybrid {
                    ssmStates = ssmStateCache.fetch(tokens: tokens, boundary: tokens.count)
                }
                return .hit(
                    matchedTokens: tokens.count,
                    remainingTokens: [],
                    detail: .disk,
                    blocks: [],
                    ssmStates: ssmStates,
                    diskArrays: arrays
                )
            }

            if tokens.count > 1 {
                let shorter = Array(tokens.dropLast())
                if let arrays = diskCache.fetch(tokens: shorter) {
                    var ssmStates: [MLXArray]? = nil
                    if isHybrid {
                        ssmStates = ssmStateCache.fetch(tokens: tokens, boundary: shorter.count)
                    }
                    return .hit(
                        matchedTokens: shorter.count,
                        remainingTokens: [tokens.last!],
                        detail: .disk,
                        blocks: [],
                        ssmStates: ssmStates,
                        diskArrays: arrays
                    )
                }
            }
        }

        // All tiers missed
        return .miss
    }

    // MARK: - Store

    /// Store cache data after generation completes.
    ///
    /// Distributes the data to each enabled cache tier:
    /// 1. Paged cache receives the token sequence and per-block layer data.
    /// 2. Disk cache receives flattened KV arrays keyed by token hash.
    ///    When TurboQuant-compressed layers are detected in `cache`, the disk
    ///    tier stores the 26x-smaller compressed representation via
    ///    ``TQDiskSerializer`` instead of raw float16 arrays.
    /// 3. SSM companion cache receives states for hybrid models.
    ///
    /// The `perLayerData` is the full-sequence per-layer output from
    /// ``extractLayerData(from:)``. This method splits it into block-sized
    /// chunks internally before passing to the paged cache.
    ///
    /// - Parameters:
    ///   - promptTokens: The full prompt token sequence.
    ///   - perLayerData: Per-layer KV tensors covering the entire prompt sequence.
    ///     Layers without KV data (SSM layers) are `nil`.
    ///   - ssmStates: SSM layer states for hybrid models, or `nil`.
    ///   - cache: The raw per-layer KV cache array from the model. When provided
    ///     and any layer is a TurboQuant cache in compressed phase, the disk tier
    ///     stores the compressed representation. Pass `nil` (default) to use the
    ///     standard float16 disk path.
    public func storeAfterGeneration(
        promptTokens: [Int],
        perLayerData: [(keys: MLXArray, values: MLXArray)?],
        ssmStates: [MLXArray]?,
        cache: [any KVCache]? = nil
    ) {
        let totalTokens = promptTokens.count
        let blockSize = config.pagedBlockSize

        // Split per-layer full-sequence data into per-block chunks.
        let blockLayerData = splitLayerDataIntoBlocks(
            perLayerData, blockSize: blockSize, totalTokens: totalTokens)

        // Store in paged cache
        if let pagedCache {
            pagedCache.storeTokenSequence(tokens: promptTokens, layerData: blockLayerData)
        }

        // Store in disk cache
        if let diskCache {
            // Check if any layer in the raw cache is TQ-compressed.
            let hasTQ = cache?.contains(where: { TQDiskSerializer.isTQCompressed($0) }) ?? false

            if hasTQ, let cache {
                // Use TQDiskSerializer for 26x smaller disk footprint.
                let arrays = TQDiskSerializer.serialize(cache: cache)
                if !arrays.isEmpty {
                    diskCache.store(tokens: promptTokens, arrays: arrays)
                }
            } else {
                // Standard float16 disk path: flatten block layer data into a dictionary.
                var arrays: [String: MLXArray] = [:]
                for (blockIdx, block) in blockLayerData.enumerated() {
                    for (layerIdx, kv) in block.enumerated() {
                        arrays["b\(blockIdx)_l\(layerIdx)_keys"] = kv.keys
                        arrays["b\(blockIdx)_l\(layerIdx)_values"] = kv.values
                    }
                }
                if !arrays.isEmpty {
                    diskCache.store(tokens: promptTokens, arrays: arrays)
                }
            }
        }

        // Store SSM companion states for hybrid models
        if isHybrid, let ssmStates, !ssmStates.isEmpty {
            let boundary = min(totalTokens, blockLayerData.count * blockSize)
            ssmStateCache.store(
                ssmStates: ssmStates,
                tokens: promptTokens,
                boundary: boundary
            )
        }
    }

    /// Split full-sequence per-layer KV data into block-sized chunks.
    ///
    /// Each block spans `blockSize` tokens along the sequence dimension (axis 2
    /// for the standard `[B, H, T, D]` layout). The last block may be shorter
    /// if `totalTokens` is not a multiple of `blockSize`.
    ///
    /// Layers that are `nil` (SSM layers without KV data) are skipped in
    /// the output — only layers with actual KV data are included.
    ///
    /// - Parameters:
    ///   - layerData: Per-layer `(keys, values)` for the full sequence, from ``extractLayerData(from:)``.
    ///   - blockSize: Number of tokens per block.
    ///   - totalTokens: Total number of tokens in the sequence.
    /// - Returns: Per-block array of per-layer `(keys, values)` tuples (non-optional, nil layers filtered out).
    private func splitLayerDataIntoBlocks(
        _ layerData: [(keys: MLXArray, values: MLXArray)?],
        blockSize: Int,
        totalTokens: Int
    ) -> [[(keys: MLXArray, values: MLXArray)]] {
        guard totalTokens > 0, !layerData.isEmpty else { return [] }

        var blocks: [[(keys: MLXArray, values: MLXArray)]] = []
        var offset = 0

        while offset < totalTokens {
            let end = min(offset + blockSize, totalTokens)
            var blockData: [(keys: MLXArray, values: MLXArray)] = []

            for kv in layerData {
                guard let kv else { continue }
                // KV tensors are [B, H, T, D] — slice along axis 2 (sequence dim)
                let slicedKeys = kv.keys[.ellipsis, offset ..< end, 0...]
                let slicedValues = kv.values[.ellipsis, offset ..< end, 0...]
                blockData.append((keys: slicedKeys, values: slicedValues))
            }

            blocks.append(blockData)
            offset = end
        }

        return blocks
    }

    // MARK: - Clear

    /// Clear all cache tiers, releasing all cached data.
    public func clear() {
        pagedCache?.clear()
        diskCache?.clear()
        ssmStateCache.clear()
    }
}
