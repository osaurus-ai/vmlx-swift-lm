// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Port target: humanrouter/ddtree-mlx/ddtree_mlx/cache.py (188 lines)
//
// Phase 0 stub — public API only. Phase 2 ports the fast-path commit
// (DFS prefix), Phase 3 lands the tree-aware commit for SSM layers.

import Foundation
import MLX

/// Commit strategies for propagating the accepted path into the target's
/// long-running KV cache.
///
/// Chosen by `SpecDecRuntime.runDDTreeLoop` based on the shape of
/// `acceptedIndices` relative to `compiled.dfsOrder`.
public enum SpecDecCommitStrategy: Sendable {

    /// Accepted path equals `dfsOrder[:n]`. Trim KV offsets + tape-rollback
    /// recurrent layers. No re-forward needed. Typical case ~80% of rounds.
    case fastPath

    /// Accepted path is arbitrary depth-first — pack accepted KV entries
    /// with `mx.take` and install the final-accepted recurrent state
    /// snapshot. Used when tree-verify captured per-node state.
    case treeAware

    /// Rare fallback — restore cache snapshot and re-forward the accepted
    /// tokens sequentially. Guaranteed lossless but costs one extra target
    /// forward.
    case slowPath
}

/// Lazy snapshot of cache state taken before a tree-verify forward.
/// Callers restore via ``SpecDecCache/restoreCaches(_:snapshots:)`` if the
/// tree produced zero accepted tokens and we need to fall back to AR.
public struct SpecDecCacheSnapshot: @unchecked Sendable {

    /// Saved state per cache entry. For KV caches this is typically an
    /// offset integer; for recurrent caches it is an array-refs list.
    /// Phase 2 details the concrete layout — kept opaque here.
    public let entries: [CacheSnapshotEntry]

    public init(entries: [CacheSnapshotEntry]) {
        self.entries = entries
    }
}

/// Discriminated payload for a single cache entry's snapshot.
public enum CacheSnapshotEntry: @unchecked Sendable {
    /// Simple integer offset (KVCache). Restoring sets
    /// `cacheEntry.offset = value`.
    case offset(Int)

    /// List of MLXArray refs (recurrent / SSM cache). Restoring sets
    /// `cacheEntry.state = value`.
    case stateRefs([MLXArray])

    /// No snapshot captured — the cache entry doesn't need rollback.
    case none
}

/// Snapshot + restore + commit primitives for the SpecDec runtime.
///
/// Phase 0 stub — Phase 2 ports the real implementation.
public enum SpecDecCache {

    /// Take a lazy snapshot of every cache entry prior to tree-verify.
    public static func snapshotCaches(
        _ cacheEntries: [KVCache]
    ) throws -> SpecDecCacheSnapshot {
        throw SpecDecError.notImplemented(
            "SpecDecCache.snapshotCaches — Phase 2 will port humanrouter/ddtree-mlx cache.py"
        )
    }

    /// Restore every cache entry from a snapshot taken by
    /// ``snapshotCaches(_:)``. Called when the tree produced zero accepted
    /// tokens and we need to bail out to AR decode.
    public static func restoreCaches(
        _ cacheEntries: [KVCache],
        snapshots: SpecDecCacheSnapshot
    ) throws {
        throw SpecDecError.notImplemented("SpecDecCache.restoreCaches — Phase 2")
    }

    /// Fast-path commit — accepted path is a DFS prefix.
    ///
    /// - Trims KV cache offsets to `prefixLen + nAccepted`.
    /// - Replays the recurrent tape forward `nAccepted` steps.
    ///
    /// Reference: `cache.py::fast_path_commit`.
    public static func fastPathCommit(
        cacheEntries: [KVCache],
        prefixLen: Int,
        nAccepted: Int
    ) throws {
        throw SpecDecError.notImplemented("SpecDecCache.fastPathCommit — Phase 2")
    }

    /// Tree-aware commit — accepted path is arbitrary depth-first.
    ///
    /// Packs accepted KV entries after the prefix, installs the final
    /// accepted node's recurrent state.
    ///
    /// Reference: `cache.py::tree_aware_path_commit`.
    public static func treeAwarePathCommit(
        cacheEntries: [KVCache],
        prefixLen: Int,
        acceptedIndices: [Int32],
        recurrentSnapshots: [Int: RecurrentSnapshot]?
    ) throws {
        throw SpecDecError.notImplemented(
            "SpecDecCache.treeAwarePathCommit — Phase 3 hybrid SSM"
        )
    }

    /// Slow-path commit — re-forward accepted tokens sequentially.
    ///
    /// Restores `cacheEntries` from `snapshots`, then runs the target
    /// model's normal forward on the accepted token sequence. Guaranteed
    /// lossless.
    ///
    /// Reference: `cache.py::slow_path_commit`.
    public static func slowPathCommit(
        target: any LanguageModel,
        cacheEntries: [KVCache],
        snapshots: SpecDecCacheSnapshot,
        acceptedTokenIds: MLXArray
    ) throws -> MLXArray {
        throw SpecDecError.notImplemented("SpecDecCache.slowPathCommit — Phase 2")
    }
}
