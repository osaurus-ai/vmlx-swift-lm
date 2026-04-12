import Foundation
import MLX
import MLXLMCommon
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
