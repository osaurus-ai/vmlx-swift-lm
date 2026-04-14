import Foundation
import MLX
@testable import MLXLMCommon
import Testing

@Test func blockHashChaining() {
    let tokens = [1, 2, 3, 4, 5]

    // Same tokens with same parent produce the same hash
    let hashA = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: tokens)
    let hashB = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: tokens)
    #expect(hashA == hashB)

    // Same tokens with different parents produce different hashes
    let hashC = CacheBlock.computeBlockHash(parentHash: "abc", tokenIds: tokens)
    let hashD = CacheBlock.computeBlockHash(parentHash: "xyz", tokenIds: tokens)
    #expect(hashC != hashD)

    // Chaining: child hashes depend on parent
    let parent1 = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [10, 20])
    let parent2 = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [10, 21])
    let child1 = CacheBlock.computeBlockHash(parentHash: parent1, tokenIds: [30, 40])
    let child2 = CacheBlock.computeBlockHash(parentHash: parent2, tokenIds: [30, 40])
    #expect(child1 != child2)

    // Hash is 64 hex characters (SHA-256)
    #expect(hashA.count == 64)
}

@Test func blockHashMediaSaltIsolation() {
    // VLM cache keying: mediaSalt must distinguish otherwise-identical
    // token sequences so "same text + different image" doesn't collide.
    let tokens = [1, 2, 3, 4, 5]

    // Backward-compat: omitting mediaSalt matches explicit-nil exactly.
    let hashNoSalt = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: tokens)
    let hashExplicitNil = CacheBlock.computeBlockHash(
        parentHash: nil, tokenIds: tokens, modelKey: nil, mediaSalt: nil)
    #expect(hashNoSalt == hashExplicitNil)

    // Text-only (nil salt) is distinct from any salted variant — prevents
    // a text-only entry from ever satisfying a VLM lookup or vice versa.
    let hashSaltA = CacheBlock.computeBlockHash(
        parentHash: nil, tokenIds: tokens, mediaSalt: "imageA-sha256")
    #expect(hashNoSalt != hashSaltA)

    // Different salts diverge.
    let hashSaltB = CacheBlock.computeBlockHash(
        parentHash: nil, tokenIds: tokens, mediaSalt: "imageB-sha256")
    #expect(hashSaltA != hashSaltB)

    // Same salt is deterministic.
    let hashSaltAAgain = CacheBlock.computeBlockHash(
        parentHash: nil, tokenIds: tokens, mediaSalt: "imageA-sha256")
    #expect(hashSaltA == hashSaltAAgain)

    // mediaSalt and modelKey are independent salt sources.
    let hashSaltAModelX = CacheBlock.computeBlockHash(
        parentHash: nil, tokenIds: tokens,
        modelKey: "model-x", mediaSalt: "imageA-sha256")
    let hashSaltAModelY = CacheBlock.computeBlockHash(
        parentHash: nil, tokenIds: tokens,
        modelKey: "model-y", mediaSalt: "imageA-sha256")
    #expect(hashSaltAModelX != hashSaltAModelY)
    #expect(hashSaltAModelX != hashSaltA)
}

@Test func computeMediaSaltDeterminism() {
    // Build a fake "image" from a constant seed so the test is reproducible.
    // 3x8x8 uint8 mimics a tiny patch tensor.
    let pixelsA = MLXArray(
        Array(0..<(3 * 8 * 8)).map { UInt8($0 % 256) },
        [3, 8, 8])
    let pixelsA2 = MLXArray(
        Array(0..<(3 * 8 * 8)).map { UInt8($0 % 256) },
        [3, 8, 8])
    let pixelsB = MLXArray(
        Array(0..<(3 * 8 * 8)).map { UInt8((255 - $0) % 256) },
        [3, 8, 8])

    let imageA = LMInput.ProcessedImage(pixels: pixelsA)
    let imageA2 = LMInput.ProcessedImage(pixels: pixelsA2)
    let imageB = LMInput.ProcessedImage(pixels: pixelsB)

    let tokens = MLXArray([Int32(1), 2, 3])
    let text = LMInput.Text(tokens: tokens)

    let inputTextOnly = LMInput(text: text)
    let inputWithA = LMInput(text: text, image: imageA)
    let inputWithA2 = LMInput(text: text, image: imageA2)
    let inputWithB = LMInput(text: text, image: imageB)

    let saltNone = computeMediaSalt(for: inputTextOnly)
    let saltA = computeMediaSalt(for: inputWithA)
    let saltA2 = computeMediaSalt(for: inputWithA2)
    let saltB = computeMediaSalt(for: inputWithB)

    // Text-only inputs get nil so we preserve the exact text-only hashing.
    #expect(saltNone == nil)

    // Identical pixel contents produce identical salts regardless of which
    // MLXArray instance holds them.
    #expect(saltA != nil)
    #expect(saltA == saltA2)

    // Different pixel contents produce different salts.
    #expect(saltA != saltB)

    // Salts are 64-hex SHA256 digests.
    #expect(saltA?.count == 64)
}

@Test func diskCacheHashMediaSaltIsolation() {
    let tokens = [10, 20, 30, 40]

    // Backward-compat: nil salt matches omitted salt.
    let noSalt = DiskCache.hashTokens(tokens)
    let explicitNil = DiskCache.hashTokens(tokens, modelKey: nil, mediaSalt: nil)
    #expect(noSalt == explicitNil)

    // Salted variant differs from unsalted.
    let salted = DiskCache.hashTokens(tokens, mediaSalt: "abc123")
    #expect(noSalt != salted)

    // Determinism and pairwise distinctness across salts.
    let saltedAgain = DiskCache.hashTokens(tokens, mediaSalt: "abc123")
    #expect(salted == saltedAgain)
    let otherSalt = DiskCache.hashTokens(tokens, mediaSalt: "def456")
    #expect(salted != otherSalt)
}

@Test func cacheBlockRefCounting() {
    let block = CacheBlock(blockId: 0, blockSize: 32)
    #expect(block.refCount == 0)

    block.incrementRef()
    #expect(block.refCount == 1)

    block.incrementRef()
    #expect(block.refCount == 2)

    block.decrementRef()
    #expect(block.refCount == 1)

    block.decrementRef()
    #expect(block.refCount == 0)

    // Floor at zero — should not go negative
    block.decrementRef()
    #expect(block.refCount == 0)
}

@Test func cacheBlockReset() {
    let block = CacheBlock(blockId: 7, blockSize: 16)

    // Populate the block
    block.blockHash = "somehash"
    block.tokenIds = [1, 2, 3]
    block.cacheData = [
        (keys: MLXArray.zeros([1, 4, 8, 64]), values: MLXArray.zeros([1, 4, 8, 64]))
    ]
    block.incrementRef()
    block.incrementRef()

    #expect(!block.isEmpty)
    #expect(block.refCount == 2)
    #expect(block.blockHash != nil)
    #expect(block.cacheData != nil)

    // Reset and verify everything is cleared
    block.reset()

    #expect(block.isEmpty)
    #expect(block.tokenCount == 0)
    #expect(!block.isFull)
    #expect(block.blockHash == nil)
    #expect(block.tokenIds.isEmpty)
    #expect(block.cacheData == nil)
    #expect(block.refCount == 0)

    // Immutable properties remain
    #expect(block.blockId == 7)
    #expect(block.blockSize == 16)
}

@Test func blockHashDeterministic() {
    let parentHash = "a1b2c3d4e5f6"
    let tokens = [100, 200, 300, 400, 500]

    // Run multiple times to confirm determinism
    let hashes = (0..<10).map { _ in
        CacheBlock.computeBlockHash(parentHash: parentHash, tokenIds: tokens)
    }

    let first = hashes[0]
    for hash in hashes {
        #expect(hash == first)
    }

    // Nil parent is also deterministic
    let nilParentHashes = (0..<10).map { _ in
        CacheBlock.computeBlockHash(parentHash: nil, tokenIds: tokens)
    }
    let firstNil = nilParentHashes[0]
    for hash in nilParentHashes {
        #expect(hash == firstNil)
    }

    // Nil parent and non-nil parent produce different results
    #expect(first != firstNil)
}

// MARK: - BlockHashMap Tests

@Test func blockHashMapLookup() {
    let map = BlockHashMap()

    let block = CacheBlock(blockId: 0, blockSize: 16)
    block.blockHash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3])

    map.insert(block)
    #expect(map.count == 1)

    let found = map.find(hash: block.blockHash!)
    #expect(found != nil)
    #expect(found?.blockId == 0)

    map.remove(block)
    #expect(map.count == 0)
    #expect(map.find(hash: block.blockHash!) == nil)
}

@Test func blockHashMapNilHash() {
    let map = BlockHashMap()

    let block = CacheBlock(blockId: 1, blockSize: 16)
    // blockHash is nil by default
    #expect(block.blockHash == nil)

    // Insert with nil hash should be a no-op
    map.insert(block)
    #expect(map.count == 0)

    // Remove with nil hash should also be a no-op (no crash)
    map.remove(block)
    #expect(map.count == 0)
}

@Test func blockHashMapMultipleBlocks() {
    let map = BlockHashMap()

    let block0 = CacheBlock(blockId: 0, blockSize: 16)
    block0.blockHash = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3])

    let block1 = CacheBlock(blockId: 1, blockSize: 16)
    block1.blockHash = CacheBlock.computeBlockHash(parentHash: block0.blockHash, tokenIds: [4, 5, 6])

    map.insert(block0)
    map.insert(block1)
    #expect(map.count == 2)

    // Both should be findable
    #expect(map.find(hash: block0.blockHash!)?.blockId == 0)
    #expect(map.find(hash: block1.blockHash!)?.blockId == 1)

    // removeAll clears everything
    map.removeAll()
    #expect(map.count == 0)
    #expect(map.find(hash: block0.blockHash!) == nil)
    #expect(map.find(hash: block1.blockHash!) == nil)
}

// MARK: - FreeBlockQueue Tests

@Test func freeBlockQueueLRU() {
    let queue = FreeBlockQueue()
    let b0 = CacheBlock(blockId: 0, blockSize: 32)
    let b1 = CacheBlock(blockId: 1, blockSize: 32)
    let b2 = CacheBlock(blockId: 2, blockSize: 32)

    queue.append(b0)
    queue.append(b1)
    queue.append(b2)
    #expect(queue.count == 3)

    // popFirst returns the oldest (least recently used)
    let first = queue.popFirst()
    #expect(first?.blockId == 0)
    #expect(queue.count == 2)

    // Remove from middle
    queue.remove(b1)
    #expect(queue.count == 1)

    // Only b2 remains
    let remaining = queue.popFirst()
    #expect(remaining?.blockId == 2)
    #expect(queue.count == 0)

    // Empty queue returns nil
    #expect(queue.popFirst() == nil)
}

@Test func freeBlockQueueTouch() {
    let queue = FreeBlockQueue()
    let b0 = CacheBlock(blockId: 0, blockSize: 32)
    let b1 = CacheBlock(blockId: 1, blockSize: 32)
    let b2 = CacheBlock(blockId: 2, blockSize: 32)

    queue.append(b0)
    queue.append(b1)
    queue.append(b2)

    // Touch b0 -- moves it to the back
    queue.touch(b0)

    // popFirst should now return b1 (oldest non-touched)
    let first = queue.popFirst()
    #expect(first?.blockId == 1)

    // Then b2
    let second = queue.popFirst()
    #expect(second?.blockId == 2)

    // Then b0 (was touched, moved to back)
    let third = queue.popFirst()
    #expect(third?.blockId == 0)

    #expect(queue.count == 0)
}
