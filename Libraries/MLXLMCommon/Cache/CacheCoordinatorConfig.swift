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

    /// Creates a new cache coordinator configuration.
    ///
    /// - Parameters:
    ///   - usePagedCache: Enable the in-memory paged KV cache. Defaults to `true`.
    ///   - enableDiskCache: Enable the on-disk L2 cache. Defaults to `false`.
    ///   - pagedBlockSize: Tokens per paged cache block. Defaults to 64.
    ///   - maxCacheBlocks: Maximum blocks in the paged cache pool. Defaults to 1000.
    ///   - diskCacheMaxGB: Maximum disk cache size in GB. Defaults to 10.0.
    ///   - diskCacheDir: Directory for disk cache files. Defaults to `nil`.
    ///   - ssmMaxEntries: Maximum SSM state cache entries. Defaults to 50.
    public init(
        usePagedCache: Bool = true,
        enableDiskCache: Bool = false,
        pagedBlockSize: Int = 64,
        maxCacheBlocks: Int = 1000,
        diskCacheMaxGB: Float = 10.0,
        diskCacheDir: URL? = nil,
        ssmMaxEntries: Int = 50
    ) {
        self.usePagedCache = usePagedCache
        self.enableDiskCache = enableDiskCache
        self.pagedBlockSize = pagedBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.ssmMaxEntries = ssmMaxEntries
    }
}
