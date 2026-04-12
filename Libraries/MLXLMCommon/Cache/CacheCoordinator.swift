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
        ssmStates: [MLXArray]?
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
                maxBlocks: config.maxCacheBlocks
            )
        } else {
            self.pagedCache = nil
        }

        if config.enableDiskCache {
            let dir = config.diskCacheDir
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("vmlx_disk_cache")
            self.diskCache = DiskCache(cacheDir: dir, maxSizeGB: config.diskCacheMaxGB)
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
                ssmStates: ssmStates
            )
        }

        // Tier 2: Disk cache (exact match, then one-shorter fallback)
        if let diskCache {
            if let _ = diskCache.fetch(tokens: tokens) {
                return .hit(
                    matchedTokens: tokens.count,
                    remainingTokens: [],
                    detail: .disk,
                    blocks: [],
                    ssmStates: nil
                )
            }

            if tokens.count > 1 {
                let shorter = Array(tokens.dropLast())
                if let _ = diskCache.fetch(tokens: shorter) {
                    return .hit(
                        matchedTokens: shorter.count,
                        remainingTokens: [tokens.last!],
                        detail: .disk,
                        blocks: [],
                        ssmStates: nil
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
    /// 3. SSM companion cache receives states for hybrid models.
    ///
    /// - Parameters:
    ///   - promptTokens: The full prompt token sequence.
    ///   - layerData: Per-block, per-layer KV tensors for the paged cache.
    ///   - ssmStates: SSM layer states for hybrid models, or `nil`.
    public func storeAfterGeneration(
        promptTokens: [Int],
        layerData: [[(keys: MLXArray, values: MLXArray)]],
        ssmStates: [MLXArray]?
    ) {
        // Store in paged cache
        if let pagedCache {
            pagedCache.storeTokenSequence(tokens: promptTokens, layerData: layerData)
        }

        // Store in disk cache
        if let diskCache {
            // Flatten layer data into a dictionary keyed by layer/type
            var arrays: [String: MLXArray] = [:]
            for (blockIdx, block) in layerData.enumerated() {
                for (layerIdx, kv) in block.enumerated() {
                    arrays["b\(blockIdx)_l\(layerIdx)_keys"] = kv.keys
                    arrays["b\(blockIdx)_l\(layerIdx)_values"] = kv.values
                }
            }
            if !arrays.isEmpty {
                diskCache.store(tokens: promptTokens, arrays: arrays)
            }
        }

        // Store SSM companion states for hybrid models
        if isHybrid, let ssmStates, !ssmStates.isEmpty {
            let boundary = min(
                promptTokens.count,
                layerData.count * config.pagedBlockSize
            )
            ssmStateCache.store(
                ssmStates: ssmStates,
                tokens: promptTokens,
                boundary: boundary
            )
        }
    }

    // MARK: - Clear

    /// Clear all cache tiers, releasing all cached data.
    public func clear() {
        pagedCache?.clear()
        diskCache?.clear()
        ssmStateCache.clear()
    }
}
