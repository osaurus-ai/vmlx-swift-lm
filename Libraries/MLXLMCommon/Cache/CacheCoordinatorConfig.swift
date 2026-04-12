// Copyright © 2025 Apple Inc. All rights reserved.

import Foundation

/// Configuration for ``CacheCoordinator``, controlling which cache tiers
/// are enabled and their sizing parameters.
public struct CacheCoordinatorConfig: Sendable {

    /// Whether the in-memory paged KV cache is enabled.
    public var usePagedCache: Bool

    /// Whether the on-disk L2 cache (SQLite + safetensors) is enabled.
    public var enableDiskCache: Bool

    /// Number of tokens per paged cache block.
    public var pagedBlockSize: Int

    /// Maximum number of blocks in the paged cache pool (including sentinel).
    public var maxCacheBlocks: Int

    /// Maximum disk cache size in gigabytes.
    public var diskCacheMaxGB: Float

    /// Directory for disk cache files. If `nil`, a default temp directory is used.
    public var diskCacheDir: URL?

    /// Maximum number of SSM state entries in the companion LRU cache.
    public var ssmMaxEntries: Int

    /// Model-specific key to prevent cross-model cache poisoning.
    /// Include model path, type, or a unique identifier. When set, cache hashes
    /// incorporate this key so different models with the same tokenizer cannot
    /// return each other's cached KV state.
    public var modelKey: String?

    public init(
        usePagedCache: Bool = true,
        enableDiskCache: Bool = false,
        pagedBlockSize: Int = 64,
        maxCacheBlocks: Int = 1000,
        diskCacheMaxGB: Float = 10.0,
        diskCacheDir: URL? = nil,
        ssmMaxEntries: Int = 50,
        modelKey: String? = nil
    ) {
        self.usePagedCache = usePagedCache
        self.enableDiskCache = enableDiskCache
        self.pagedBlockSize = pagedBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.ssmMaxEntries = ssmMaxEntries
        self.modelKey = modelKey
    }
}
