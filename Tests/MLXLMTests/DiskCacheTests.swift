import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Test func diskCacheStoreAndFetch() async throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx_test_\(UUID())")
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)

    let tokens = [1, 2, 3, 4, 5]
    let arrays: [String: MLXArray] = [
        "keys": MLXArray.ones([2, 4, 8]),
        "values": MLXArray.zeros([2, 4, 8]),
    ]

    cache.store(tokens: tokens, arrays: arrays)

    // Wait for background write to complete
    try await Task.sleep(nanoseconds: 500_000_000)

    let result = cache.fetch(tokens: tokens)
    #expect(result != nil)
    #expect(result?.keys.sorted() == ["keys", "values"])
    #expect(cache.hits == 1)
    #expect(cache.stores == 1)
}

@Test func diskCacheMiss() {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx_test_\(UUID())")
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)

    let result = cache.fetch(tokens: [99, 100, 101])
    #expect(result == nil)
    #expect(cache.misses == 1)
    #expect(cache.hits == 0)
}

@Test func diskCacheClear() async throws {
    let tempDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx_test_\(UUID())")
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let cache = DiskCache(cacheDir: tempDir, maxSizeGB: 0.1)

    let tokens = [10, 20, 30]
    let arrays: [String: MLXArray] = [
        "data": MLXArray.ones([4, 4]),
    ]

    cache.store(tokens: tokens, arrays: arrays)

    // Wait for background write to complete
    try await Task.sleep(nanoseconds: 500_000_000)

    // Verify the entry exists
    let beforeClear = cache.fetch(tokens: tokens)
    #expect(beforeClear != nil)

    // Clear the cache
    cache.clear()

    // Verify the entry is gone
    let afterClear = cache.fetch(tokens: tokens)
    #expect(afterClear == nil)
}

@Test func diskCacheHashDeterminism() {
    let tokens = [42, 43, 44, 45]
    let hash1 = DiskCache.hashTokens(tokens)
    let hash2 = DiskCache.hashTokens(tokens)
    #expect(hash1 == hash2)
    #expect(hash1.count == 32)

    // Different tokens produce different hashes
    let hash3 = DiskCache.hashTokens([42, 43, 44, 46])
    #expect(hash1 != hash3)
}
