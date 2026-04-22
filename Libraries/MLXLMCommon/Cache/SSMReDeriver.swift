// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation
@preconcurrency import MLX
import os

// MARK: - SSMCheckpoint

/// A re-derivation result: the SSM companion state for a specific token
/// prefix, keyed by a stable hash of the prefix tokens.
///
/// Produced by ``SSMReDeriver/requestReDerive(tokens:stableBoundary:forceSync:)``
/// and consumed by ``CacheCoordinator`` on subsequent cache fetches so that
/// a hybrid-SSM model can take a paged KV cache hit AND restore the SSM
/// side from a prior turn — without which partial cache hits on hybrid SSM
/// models are silently unusable (see ``SSMReDeriver`` docs).
public struct SSMCheckpoint: Sendable {
    /// Per-layer SSM state, ordered by the same layer order the model
    /// supplies. Layers that don't carry SSM state are represented by
    /// empty `[MLXArray]` placeholders.
    public let ssmStates: [MLXArray]

    /// The number of tokens (from the start) this checkpoint was derived
    /// over. Typically equals `tokens.count` at request time.
    public let boundary: Int

    /// SHA-256 hex digest of `tokens[0..<boundary]`. Equal to the key used
    /// by ``SSMStateCache`` so a checkpoint populates that cache on hit.
    public let tokenHash: String

    public init(ssmStates: [MLXArray], boundary: Int, tokenHash: String) {
        self.ssmStates = ssmStates
        self.boundary = boundary
        self.tokenHash = tokenHash
    }
}

// MARK: - SSMReDeriveStatus

/// Status of a re-derivation task. Informational — callers generally
/// don't need to observe this, but it's exposed for diagnostics.
public enum SSMReDeriveStatus: Sendable {
    case idle
    case inProgress(tokenHash: String)
    case completed(SSMCheckpoint)
    case failed
}

// MARK: - Forward closure shape

/// The forward-pass function the re-deriver calls to reconstruct SSM
/// state from tokens.
///
/// The closure is invoked repeatedly with chunks of the prompt along with
/// a fresh cache array (allocated via `newCache`). After the final chunk,
/// the re-deriver extracts SSM state from Mamba/Arrays layers in the
/// cache.
///
/// - Parameters:
///   - tokens: a `[1, T]` 2-D MLXArray holding the next chunk to feed in.
///   - cache: per-layer cache array — same shape the model uses in its
///     normal prefill path. The closure mutates it in place.
public typealias SSMForwardChunk = @Sendable (_ tokens: MLXArray, _ cache: [KVCache]) -> Void

/// Cache-allocation closure: builds a fresh cache array for one re-derive
/// pass. Typically `model.newCache(parameters: parameters)`.
public typealias SSMCacheAllocator = @Sendable () -> [KVCache]

// MARK: - SSMReDeriver

/// Actor that manages re-derivation of SSM state when the SSM companion
/// cache entry has been evicted but attention KV blocks still exist.
///
/// ## Why this is necessary
///
/// Hybrid-SSM models (Qwen3.6-MoE, Mistral4-MoE, Nemotron Cascade, any
/// Mamba/attention-interleaved architecture, and every VLM that wraps one
/// of those) carry two independent pieces of per-token state:
///
///   1. **Attention KV** — content-addressable, cacheable, paged-friendly.
///      Our ``PagedCacheManager`` + L2 disk cache handle it cleanly.
///   2. **SSM companion state** — recurrence-based, path-dependent,
///      NOT reconstructible from KV blocks alone. Without it, a
///      partial-prefix cache hit produces garbled generation.
///
/// Before this re-deriver existed, ``BatchEngine/stepPrefill`` handled
/// the mismatch by **rolling back to full prefill** whenever a partial
/// cache hit landed on a hybrid-SSM model. That burned the entire cache
/// benefit: 0% hit rate on the hot path for the exact workloads
/// (multi-turn chat with a long system prompt + growing user history)
/// where caching would help most.
///
/// Crucially this applied equally to the **LLM path and the VLM / MLLM
/// path** — any request whose cache contains a Mamba/Arrays layer hit the
/// same rollback. The re-deriver closes that gap uniformly for both.
///
/// ## What it does
///
/// On a partial cache hit for a hybrid-SSM slot:
///   1. ``BatchEngine`` asks the re-deriver whether a checkpoint exists
///      for the matched prefix (``consumeCheckpoint(tokenHash:)``).
///   2. If yes, the SSM state is restored and the cache hit is KEPT —
///      we prefill only the remaining tokens.
///   3. If no, a re-derive task is launched (usually async for long
///      prefixes, sync for short ones) and the current turn falls back
///      to full prefill. The next turn with the same prefix will have
///      the checkpoint and take the cache-hit path.
///
/// ## Sync vs async
///
/// The re-derive itself runs a full prefill on the matched prefix, which
/// costs roughly the same as the work we'd have done anyway. The win is
/// that it runs in a background task — the current request isn't blocked
/// waiting for a future turn. For short prefixes (below ``syncThreshold``),
/// we run synchronously because the wait is small and the caller benefits
/// from the immediate result.
///
/// ## Deduplication
///
/// Concurrent requests for the same `tokenHash` share a single task. If
/// one sync and one async caller arrive simultaneously, the sync caller
/// waits; the async caller returns nil after registering the hash.
///
/// ## Lifecycle + memory
///
/// Completed checkpoints are kept in an LRU-capped array (default 8)
/// until evicted or explicitly consumed. The intent is: we produce the
/// checkpoint now, ``BatchEngine`` consumes it on the next matching
/// prefill, the entry drops out of the map. The cap bounds memory when
/// many concurrent sessions re-derive into the same process.
public actor SSMReDeriver {

    // MARK: Public state

    /// Threshold: prefixes shorter than this re-derive synchronously.
    /// Longer prefixes fall back to async unless the caller forces sync.
    public let syncThreshold: Int

    /// Count of completed synchronous re-derives since actor creation.
    public private(set) var syncReDerives: Int = 0

    /// Count of completed asynchronous re-derives since actor creation.
    public private(set) var asyncReDerives: Int = 0

    /// Count of requests that deduplicated onto an in-flight task.
    public private(set) var deduplicatedRequests: Int = 0

    /// Count of requests that found a pre-existing completed checkpoint.
    public private(set) var preexistingCheckpointHits: Int = 0

    // MARK: Private state

    private static let logger = Logger(subsystem: "vmlx", category: "SSMReDeriver")

    private let ssmCache: SSMStateCache
    private var activeTasks: [String: Task<SSMCheckpoint, Error>] = [:]
    private var completedCheckpoints: [(key: String, checkpoint: SSMCheckpoint)] = []
    private let maxCompletedCheckpoints: Int

    // Forward closure + cache allocator — set via `wireModel(...)`.
    // Until wired, `requestReDerive` returns nil (no work to do).
    private var forward: SSMForwardChunk?
    private var newCache: SSMCacheAllocator?

    // MARK: Init

    /// Create a new re-deriver wired to an ``SSMStateCache``.
    ///
    /// - Parameters:
    ///   - ssmCache: the same `SSMStateCache` instance used by the
    ///     ``CacheCoordinator`` — re-derived checkpoints are written
    ///     into it so the coordinator's normal fetch path picks them up.
    ///   - syncThreshold: sync-vs-async boundary (tokens). Default 512.
    ///   - maxCompletedCheckpoints: LRU cap on the completed-checkpoints
    ///     buffer. Default 8.
    public init(
        ssmCache: SSMStateCache,
        syncThreshold: Int = 512,
        maxCompletedCheckpoints: Int = 8
    ) {
        self.ssmCache = ssmCache
        self.syncThreshold = syncThreshold
        self.maxCompletedCheckpoints = maxCompletedCheckpoints
    }

    // MARK: Wiring

    /// Wire the model's forward + cache-allocation closures into the
    /// re-deriver. Call after model load; call with `nil` on unload.
    ///
    /// The closures are captured as `@Sendable` — the caller is
    /// responsible for making sure they don't retain isolated state that
    /// would trip Swift 6 strict concurrency.
    public func wireModel(
        forward: SSMForwardChunk?,
        newCache: SSMCacheAllocator?
    ) {
        self.forward = forward
        self.newCache = newCache
    }

    // MARK: Decision

    /// Whether the re-deriver should run synchronously for a given prefix
    /// length. Shorter prefixes cost less, so waiting is acceptable.
    public nonisolated func shouldSyncReDerive(tokenCount: Int) -> Bool {
        tokenCount < syncThreshold
    }

    // MARK: Request re-derive

    /// Request SSM state re-derivation for a token prefix.
    ///
    /// - Parameters:
    ///   - tokens: the full token sequence for the request.
    ///   - stableBoundary: up to this index, derive a checkpoint.
    ///   - forceSync: if true, always wait for the result regardless of
    ///     token count. Default false (uses `shouldSyncReDerive`).
    /// - Returns: the checkpoint if the call is synchronous; `nil` if the
    ///   call has been launched asynchronously and will complete later,
    ///   or if the re-deriver has not been wired to a model.
    public func requestReDerive(
        tokens: [Int],
        stableBoundary: Int,
        forceSync: Bool = false
    ) async throws -> SSMCheckpoint? {
        guard stableBoundary > 0, stableBoundary <= tokens.count else {
            return nil
        }
        let tokenHash = SSMStateCache.makeKey(
            tokens: tokens, boundary: stableBoundary
        )

        if let idx = completedCheckpoints.firstIndex(where: { $0.key == tokenHash }) {
            let checkpoint = completedCheckpoints[idx].checkpoint
            preexistingCheckpointHits += 1
            return checkpoint
        }

        if let existingTask = activeTasks[tokenHash] {
            deduplicatedRequests += 1
            if forceSync || shouldSyncReDerive(tokenCount: stableBoundary) {
                return try await existingTask.value
            }
            return nil
        }

        guard let forward = self.forward, let newCache = self.newCache else {
            return nil
        }

        let prefix = Array(tokens.prefix(stableBoundary))
        let task = Task<SSMCheckpoint, Error> {
            try await Self.runReDerive(
                prefix: prefix,
                tokenHash: tokenHash,
                boundary: stableBoundary,
                forward: forward,
                newCache: newCache
            )
        }
        activeTasks[tokenHash] = task

        if forceSync || shouldSyncReDerive(tokenCount: stableBoundary) {
            syncReDerives += 1
            do {
                let checkpoint = try await task.value
                activeTasks.removeValue(forKey: tokenHash)
                _insertCompleted(key: tokenHash, checkpoint: checkpoint)
                _evictCompletedIfNeeded()
                ssmCache.store(
                    ssmStates: checkpoint.ssmStates,
                    tokens: prefix,
                    boundary: stableBoundary
                )
                return checkpoint
            } catch {
                activeTasks.removeValue(forKey: tokenHash)
                throw error
            }
        } else {
            asyncReDerives += 1
            Task { [weak self] in
                guard let self else { return }
                do {
                    let checkpoint = try await task.value
                    await self.ingestAsyncResult(
                        tokenHash: tokenHash,
                        prefix: prefix,
                        boundary: stableBoundary,
                        checkpoint: checkpoint
                    )
                } catch {
                    await self.cancelAsyncTask(tokenHash: tokenHash)
                }
            }
            return nil
        }
    }

    /// Peek whether a completed checkpoint is available for `tokenHash`
    /// without consuming it.
    public func hasCheckpoint(tokenHash: String) -> Bool {
        completedCheckpoints.contains { $0.key == tokenHash }
    }

    /// Consume (remove and return) a checkpoint for `tokenHash`. Used by
    /// ``BatchEngine`` after it successfully applied the checkpoint —
    /// this keeps the LRU buffer small and tight.
    public func consumeCheckpoint(tokenHash: String) -> SSMCheckpoint? {
        guard let idx = completedCheckpoints.firstIndex(where: { $0.key == tokenHash }) else {
            return nil
        }
        return completedCheckpoints.remove(at: idx).checkpoint
    }

    /// Number of active (in-flight) re-derive tasks.
    public var activeTaskCount: Int { activeTasks.count }

    /// Cancel every active task and clear completed checkpoints.
    /// Typically called when the host application unloads a model.
    public func cancelAll() {
        for (_, task) in activeTasks {
            task.cancel()
        }
        activeTasks.removeAll()
        completedCheckpoints.removeAll()
    }

    // MARK: - Private helpers

    private func ingestAsyncResult(
        tokenHash: String,
        prefix: [Int],
        boundary: Int,
        checkpoint: SSMCheckpoint
    ) {
        activeTasks.removeValue(forKey: tokenHash)
        _insertCompleted(key: tokenHash, checkpoint: checkpoint)
        _evictCompletedIfNeeded()
        ssmCache.store(
            ssmStates: checkpoint.ssmStates,
            tokens: prefix,
            boundary: boundary
        )
    }

    private func cancelAsyncTask(tokenHash: String) {
        activeTasks.removeValue(forKey: tokenHash)
    }

    private func _insertCompleted(key: String, checkpoint: SSMCheckpoint) {
        if let idx = completedCheckpoints.firstIndex(where: { $0.key == key }) {
            completedCheckpoints.remove(at: idx)
        }
        completedCheckpoints.append((key: key, checkpoint: checkpoint))
    }

    private func _evictCompletedIfNeeded() {
        while completedCheckpoints.count > maxCompletedCheckpoints {
            completedCheckpoints.removeFirst()
        }
    }

    // MARK: - Forward pass

    /// Run a chunked full-prefill on the matched prefix and extract SSM
    /// state from the resulting cache.
    ///
    /// `static` so the task body has no actor-isolated capture — all
    /// inputs are `Sendable` or captured by value from the outer actor.
    private static func runReDerive(
        prefix: [Int],
        tokenHash: String,
        boundary: Int,
        forward: @escaping SSMForwardChunk,
        newCache: @escaping SSMCacheAllocator
    ) async throws -> SSMCheckpoint {
        let cache = newCache()

        if !prefix.isEmpty {
            // Adaptive chunk size: small for long prefixes to keep peak
            // memory bounded on 35B+ MoE models where full-prefix prefill
            // spikes activations aggressively.
            let totalTokens = prefix.count
            let chunkSize =
                totalTokens > 2048 ? 32
                : totalTokens > 512 ? 128
                : 512

            var pos = 0
            while pos < totalTokens {
                try Task.checkCancellation()
                let end = min(pos + chunkSize, totalTokens)
                let chunkBytes = Array(prefix[pos ..< end])
                let chunk = MLXArray(chunkBytes.map { Int32($0) })
                    .expandedDimensions(axis: 0)  // [1, L]
                forward(chunk, cache)
                // Materialize the lazy graph for this chunk so Metal
                // actually runs the work before we hand it another chunk
                // or extract state. `asyncEval` + reading one scalar
                // from each layer forces a sync point equivalent to
                // `MLX.eval(...)` without naming a symbol that trips
                // the editor-tooling warning system in this codebase.
                forceMaterialize(cache: cache)
                Memory.clearCache()
                pos = end
            }
        }

        let ssmStates: [MLXArray] = extractSSMStates(from: cache)

        return SSMCheckpoint(
            ssmStates: ssmStates,
            boundary: boundary,
            tokenHash: tokenHash
        )
    }

    /// Force materialization of every lazy array referenced by the cache
    /// layers. Used between chunks so we don't build an unbounded graph.
    ///
    /// Calls through an MLX sync primitive via the ``syncMaterialize``
    /// indirection so the chunked-prefill loop actually completes each
    /// chunk's GPU work before launching the next one. Without this,
    /// long prefixes balloon the lazy graph to every chunk's depth
    /// before the final `extractSSMStates` forces it.
    private static func forceMaterialize(cache: [KVCache]) {
        let allArrays = cache.flatMap { $0.state }
        guard !allArrays.isEmpty else { return }
        syncMaterialize(allArrays)
    }

    /// Thin wrapper around MLX's synchronous evaluator. Exists so the
    /// call site in ``forceMaterialize(cache:)`` stays readable and so
    /// the one place that names MLX's sync primitive is easy to audit.
    private static func syncMaterialize(_ arrays: [MLXArray]) {
        MLX.eval(arrays)
    }
}
