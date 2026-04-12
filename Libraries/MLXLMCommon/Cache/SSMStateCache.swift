// Copyright © 2024 Apple Inc.

import CryptoKit
import Foundation
import MLX
import os

/// An LRU companion cache for SSM layer state in hybrid models
/// (Nemotron-H, Qwen3.5-A3B, Jamba).
///
/// SSM state is cumulative (path-dependent) and cannot be reconstructed
/// from KV cache alone, so it must be cached separately. Entries are keyed
/// by a SHA-256 hash of the token prefix up to a given boundary, and the
/// cache uses LRU eviction when the entry limit is reached.
///
/// All public methods are thread-safe via `OSAllocatedUnfairLock`.
///
/// **Deep-copy semantics**: ``fetch(tokens:boundary:)`` returns independent
/// copies of the stored state arrays because model forward passes modify
/// SSM state in-place; sharing would corrupt the cached snapshot.
public final class SSMStateCache: @unchecked Sendable {

    // MARK: - Properties

    private let lock = OSAllocatedUnfairLock()
    private let maxEntries: Int
    private var entries: [(key: String, states: [MLXArray])]

    /// Number of successful cache hits since creation (or last ``clear()``).
    public private(set) var hits: Int = 0

    /// Number of cache misses since creation (or last ``clear()``).
    public private(set) var misses: Int = 0

    // MARK: - Initialization

    /// Creates a new SSM state cache.
    /// - Parameter maxEntries: Maximum number of entries before LRU eviction
    ///   kicks in. Defaults to 50.
    public init(maxEntries: Int = 50) {
        self.maxEntries = maxEntries
        self.entries = []
    }

    // MARK: - Public API

    /// Store SSM layer states for a given token prefix.
    ///
    /// Each state array is materialized (evaluated) immediately so that the
    /// stored snapshot is independent of the lazy computation graph.
    ///
    /// - Parameters:
    ///   - ssmStates: The per-layer SSM state arrays to cache.
    ///   - tokens: The full token sequence for the current generation.
    ///   - boundary: The number of tokens (from the start) to include in the
    ///     cache key.
    public func store(ssmStates: [MLXArray], tokens: [Int], boundary: Int) {
        let key = Self.makeKey(tokens: tokens, boundary: boundary)

        lock.lock()
        defer { lock.unlock() }

        // Remove existing entry with same key (if any)
        entries.removeAll { $0.key == key }

        // Materialize each state array (lazy-graph safety)
        // Use [.ellipsis] identity slice to create an independent copy,
        // then eval() to detach from the lazy computation graph.
        let copies = ssmStates.map { arr -> MLXArray in
            let copy = arr[.ellipsis]
            MLX.eval(copy)
            return copy
        }

        // Append to end (most recently used position)
        entries.append((key: key, states: copies))

        // Evict oldest if over capacity
        if entries.count > maxEntries {
            entries.removeFirst()
        }
    }

    /// Fetch cached SSM states for a given token prefix.
    ///
    /// Returns deep copies of the stored arrays so that in-place mutations
    /// during model forward passes do not corrupt the cache.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence for the current generation.
    ///   - boundary: The number of tokens (from the start) to include in the
    ///     cache key.
    /// - Returns: Deep copies of the cached state arrays, or `nil` on a miss.
    public func fetch(tokens: [Int], boundary: Int) -> [MLXArray]? {
        let key = Self.makeKey(tokens: tokens, boundary: boundary)

        lock.lock()
        defer { lock.unlock() }

        guard let index = entries.firstIndex(where: { $0.key == key }) else {
            misses += 1
            return nil
        }

        let entry = entries[index]

        // Empty states array is treated as a miss (bug fix from osa-jang ba07392)
        guard !entry.states.isEmpty else {
            misses += 1
            return nil
        }

        // LRU touch: move to end
        entries.remove(at: index)
        entries.append(entry)

        hits += 1

        // Return deep copies — model forward passes modify SSM state in-place
        return entry.states.map { $0[.ellipsis] }
    }

    /// Remove all entries and reset hit/miss statistics.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }

        entries.removeAll()
        hits = 0
        misses = 0
    }

    // MARK: - Key Generation

    /// Compute a deterministic cache key from the first `boundary` tokens.
    ///
    /// The key is the SHA-256 hash of the raw bytes of `tokens[0..<boundary]`,
    /// returned as a 64-character lowercase hex string.
    ///
    /// - Parameters:
    ///   - tokens: The full token sequence.
    ///   - boundary: How many tokens from the start to include in the hash.
    /// - Returns: A 64-character lowercase hex string.
    public static func makeKey(tokens: [Int], boundary: Int) -> String {
        let prefix = Array(tokens.prefix(boundary))
        var hasher = SHA256()

        prefix.withUnsafeBufferPointer { buffer in
            let rawBuffer = UnsafeRawBufferPointer(buffer)
            hasher.update(bufferPointer: rawBuffer)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
