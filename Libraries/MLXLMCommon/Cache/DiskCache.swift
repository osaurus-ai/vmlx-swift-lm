// Copyright © 2025 Apple Inc. All rights reserved.

import CryptoKit
import Foundation
import MLX
import SQLite3
import os

/// L2 SSD cache with SQLite index and safetensors file storage.
///
/// `DiskCache` provides persistent KV cache storage on disk using safetensors
/// files for tensor data and a SQLite database for indexing. Writes are
/// dispatched to a background task to avoid blocking the caller. Reads are
/// synchronous since they typically feed directly into model inference.
public final class DiskCache: @unchecked Sendable {

    // MARK: - Properties

    /// Root directory for cache files and the SQLite index.
    public let cacheDir: URL

    /// Maximum total cache size in bytes.
    public let maxSizeBytes: Int

    /// SQLite database handle.
    private var db: OpaquePointer?

    /// Lock for thread-safe access to mutable state.
    private let lock = OSAllocatedUnfairLock()

    /// Number of successful cache hits.
    public private(set) var hits: Int = 0

    /// Number of cache misses.
    public private(set) var misses: Int = 0

    /// Number of store operations initiated.
    public private(set) var stores: Int = 0

    // MARK: - Initialization

    /// Creates a new disk cache.
    ///
    /// - Parameters:
    ///   - cacheDir: Directory where safetensors files and the SQLite index are stored.
    ///   - maxSizeGB: Maximum cache size in gigabytes. Defaults to 10 GB.
    public init(cacheDir: URL, maxSizeGB: Float = 10.0) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = Int(maxSizeGB * 1_073_741_824)

        // Create cache directory if needed
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        // Open SQLite database
        let dbPath = cacheDir.appendingPathComponent("cache_index.db").path
        if sqlite3_open(dbPath, &db) != SQLITE_OK {
            db = nil
            return
        }

        // Enable WAL mode for better concurrent read performance
        executeSQL("PRAGMA journal_mode=WAL")

        // Create the index table
        executeSQL("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                hash TEXT PRIMARY KEY,
                token_count INTEGER,
                file_size INTEGER,
                created_at REAL DEFAULT (julianday('now'))
            )
            """)
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    // MARK: - Public API

    /// Store token arrays to disk as a safetensors file.
    ///
    /// Arrays are evaluated on the calling thread, then the file write and
    /// SQLite insert are dispatched to a background task.
    ///
    /// - Parameters:
    ///   - tokens: Token IDs used to compute the cache key hash.
    ///   - arrays: Dictionary of named MLX arrays to persist.
    public func store(tokens: [Int], arrays: [String: MLXArray]) {
        let hash = DiskCache.hashTokens(tokens)
        let url = safetensorsURL(for: hash)
        let tokenCount = tokens.count

        // Pre-evaluate arrays on calling thread so GPU work completes
        // before handing off to the background writer.
        MLX.eval(Array(arrays.values))

        lock.withLock {
            stores += 1
        }

        // Background write to avoid blocking the caller.
        // Use DispatchQueue to avoid Swift 6 Sendable constraints on Task.detached,
        // since MLXArray is not Sendable but is safe to use cross-thread after eval.
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }

            do {
                try save(arrays: arrays, metadata: ["format": "mlx"], url: url)

                // Record file size
                let fileSize: Int
                if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
                   let size = attrs[.size] as? Int
                {
                    fileSize = size
                } else {
                    fileSize = 0
                }

                // Insert/replace into SQLite
                self.insertEntry(hash: hash, tokenCount: tokenCount, fileSize: fileSize)
            } catch {
                // Silently ignore write failures -- cache is best-effort
            }
        }
    }

    /// Fetch cached arrays for the given token sequence.
    ///
    /// - Parameter tokens: Token IDs to look up.
    /// - Returns: The cached arrays if found, or `nil` on a miss.
    public func fetch(tokens: [Int]) -> [String: MLXArray]? {
        let hash = DiskCache.hashTokens(tokens)
        let url = safetensorsURL(for: hash)

        guard FileManager.default.fileExists(atPath: url.path) else {
            lock.withLock { misses += 1 }
            return nil
        }

        do {
            let (arrays, _) = try loadArraysAndMetadata(url: url)
            lock.withLock { hits += 1 }
            return arrays
        } catch {
            lock.withLock { misses += 1 }
            return nil
        }
    }

    /// Remove all cached entries and safetensors files.
    public func clear() {
        // Delete all SQLite entries
        executeSQL("DELETE FROM cache_entries")

        // Remove all .safetensors files in the cache directory
        if let enumerator = FileManager.default.enumerator(
            at: cacheDir,
            includingPropertiesForKeys: nil,
            options: [.skipsSubdirectoryDescendants]
        ) {
            for case let fileURL as URL in enumerator {
                if fileURL.pathExtension == "safetensors" {
                    try? FileManager.default.removeItem(at: fileURL)
                }
            }
        }

        // Reset stats
        lock.withLock {
            hits = 0
            misses = 0
            stores = 0
        }
    }

    // MARK: - Hashing

    /// Compute a deterministic hash from a token sequence.
    ///
    /// Uses SHA-256 over the raw byte representation of the token array
    /// and returns the first 32 hex characters.
    ///
    /// - Parameter tokens: The token IDs to hash.
    /// - Returns: A 32-character lowercase hex string.
    public static func hashTokens(_ tokens: [Int]) -> String {
        var hasher = SHA256()
        tokens.withUnsafeBufferPointer { buffer in
            let rawBuffer = UnsafeRawBufferPointer(buffer)
            hasher.update(bufferPointer: rawBuffer)
        }
        let digest = hasher.finalize()
        let fullHex = digest.map { String(format: "%02x", $0) }.joined()
        return String(fullHex.prefix(32))
    }

    // MARK: - Private Helpers

    /// Build the file URL for a given hash.
    private func safetensorsURL(for hash: String) -> URL {
        cacheDir.appendingPathComponent("\(hash).safetensors")
    }

    /// Execute a simple SQL statement with no bindings.
    private func executeSQL(_ sql: String) {
        guard let db else { return }
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK {
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    /// Insert or replace a cache entry in the SQLite index.
    private func insertEntry(hash: String, tokenCount: Int, fileSize: Int) {
        guard let db else { return }

        let sql = """
            INSERT OR REPLACE INTO cache_entries (hash, token_count, file_size)
            VALUES (?, ?, ?)
            """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }

        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, hash, -1, transient)
        sqlite3_bind_int64(stmt, 2, Int64(tokenCount))
        sqlite3_bind_int64(stmt, 3, Int64(fileSize))
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }
}
