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

    /// Model key for cache isolation (prevents cross-model hash collisions).
    public let modelKey: String?

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
    public init(cacheDir: URL, maxSizeGB: Float = 10.0, modelKey: String? = nil) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = Int(maxSizeGB * 1_073_741_824)
        self.modelKey = modelKey

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
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
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

                // Evict oldest entries if total cache exceeds size limit
                self.evictIfNeeded()
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
        let hash = DiskCache.hashTokens(tokens, modelKey: modelKey)
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
    /// and returns the first 32 hex characters. When `modelKey` is provided,
    /// it is hashed first to prevent cross-model cache collisions.
    ///
    /// - Parameters:
    ///   - tokens: The token IDs to hash.
    ///   - modelKey: Optional model identifier for cache isolation.
    /// - Returns: A 32-character lowercase hex string.
    public static func hashTokens(_ tokens: [Int], modelKey: String? = nil) -> String {
        var hasher = SHA256()
        if let modelKey {
            hasher.update(data: Data(modelKey.utf8))
        }
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

        hash.withCString { cStr in
            sqlite3_bind_text(stmt, 1, cStr, -1, nil)
            sqlite3_bind_int64(stmt, 2, Int64(tokenCount))
            sqlite3_bind_int64(stmt, 3, Int64(fileSize))
            sqlite3_step(stmt)
        }
        sqlite3_finalize(stmt)
    }

    /// Evict oldest entries until total cache size is under `maxSizeBytes`.
    private func evictIfNeeded() {
        guard let db else { return }

        // Query total size
        var totalSize: Int64 = 0
        var stmt: OpaquePointer?
        if sqlite3_prepare_v2(db, "SELECT COALESCE(SUM(file_size), 0) FROM cache_entries", -1, &stmt, nil) == SQLITE_OK {
            if sqlite3_step(stmt) == SQLITE_ROW {
                totalSize = sqlite3_column_int64(stmt, 0)
            }
        }
        sqlite3_finalize(stmt)

        guard totalSize > Int64(maxSizeBytes) else { return }

        // Fetch oldest entries (by creation time) to evict
        var toEvict: [(hash: String, fileSize: Int64)] = []
        var accumulated: Int64 = 0
        let excess = totalSize - Int64(maxSizeBytes)

        if sqlite3_prepare_v2(db, "SELECT hash, file_size FROM cache_entries ORDER BY created_at ASC", -1, &stmt, nil) == SQLITE_OK {
            while sqlite3_step(stmt) == SQLITE_ROW, accumulated < excess {
                if let cStr = sqlite3_column_text(stmt, 0) {
                    let hash = String(cString: cStr)
                    let size = sqlite3_column_int64(stmt, 1)
                    toEvict.append((hash: hash, fileSize: size))
                    accumulated += size
                }
            }
        }
        sqlite3_finalize(stmt)

        // Delete evicted entries and their files
        for entry in toEvict {
            let url = safetensorsURL(for: entry.hash)
            try? FileManager.default.removeItem(at: url)

            entry.hash.withCString { cStr in
                if sqlite3_prepare_v2(db, "DELETE FROM cache_entries WHERE hash = ?", -1, &stmt, nil) == SQLITE_OK {
                    sqlite3_bind_text(stmt, 1, cStr, -1, nil)
                    sqlite3_step(stmt)
                }
                sqlite3_finalize(stmt)
            }
        }
    }
}
