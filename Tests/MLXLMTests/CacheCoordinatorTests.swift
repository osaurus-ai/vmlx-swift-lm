import Foundation
import MLX
@testable import MLXLMCommon
import Testing

// MARK: - CacheCoordinator Tests

@Test func coordinatorMiss() {
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: 4,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    let result = coordinator.fetch(tokens: [1, 2, 3, 4, 5, 6, 7, 8])

    switch result {
    case .miss:
        break  // expected
    case .hit:
        Issue.record("Empty coordinator should return .miss")
    }
}

@Test func coordinatorPagedHit() {
    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    // Store 8 tokens (2 full blocks of size 4)
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    // Per-layer data covering the full 8-token sequence (coordinator splits into blocks)
    let perLayerData: [(keys: MLXArray, values: MLXArray)?] = [
        (keys: MLXArray.zeros([1, 1, tokens.count, 8]),
         values: MLXArray.zeros([1, 1, tokens.count, 8]))
    ]

    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: perLayerData,
        ssmStates: nil
    )

    // Fetch with the same prefix plus extra tokens
    let query = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let result = coordinator.fetch(tokens: query)

    switch result {
    case .hit(let matchedTokens, let remainingTokens, let detail, let blocks, let ssmStates):
        #expect(matchedTokens == 8)
        #expect(remainingTokens == [9, 10])
        #expect(detail == .paged)
        #expect(blocks.count == 2)
        #expect(ssmStates == nil)
    case .miss:
        Issue.record("Should have hit the paged cache")
    }
}

@Test func coordinatorSSMCompanion() {
    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20,
        ssmMaxEntries: 10
    )
    let coordinator = CacheCoordinator(config: config)
    coordinator.setHybrid(true)

    #expect(coordinator.isHybrid == true)

    // Store 8 tokens with SSM states
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    let perLayerData: [(keys: MLXArray, values: MLXArray)?] = [
        (keys: MLXArray.zeros([1, 1, tokens.count, 8]),
         values: MLXArray.zeros([1, 1, tokens.count, 8]))
    ]
    let ssmStates = [MLXArray.ones([2, 4]), MLXArray.zeros([2, 4])]

    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: perLayerData,
        ssmStates: ssmStates
    )

    // Fetch should return SSM states alongside the paged hit
    let result = coordinator.fetch(tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    switch result {
    case .hit(let matchedTokens, _, let detail, _, let fetchedSSM):
        #expect(matchedTokens == 8)
        #expect(detail == .paged)
        #expect(fetchedSSM != nil)
        #expect(fetchedSSM?.count == 2)
    case .miss:
        Issue.record("Should have hit the paged cache with SSM companion")
    }
}

@Test func coordinatorClear() {
    let blockSize = 4
    let config = CacheCoordinatorConfig(
        usePagedCache: true,
        enableDiskCache: false,
        pagedBlockSize: blockSize,
        maxCacheBlocks: 20
    )
    let coordinator = CacheCoordinator(config: config)

    // Store tokens
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    let perLayerData: [(keys: MLXArray, values: MLXArray)?] = [
        (keys: MLXArray.zeros([1, 1, tokens.count, 8]),
         values: MLXArray.zeros([1, 1, tokens.count, 8]))
    ]
    coordinator.storeAfterGeneration(
        promptTokens: tokens,
        perLayerData: perLayerData,
        ssmStates: nil
    )

    // Verify the data is cached
    let beforeClear = coordinator.fetch(tokens: tokens)
    switch beforeClear {
    case .hit:
        break  // expected
    case .miss:
        Issue.record("Data should be cached before clear")
    }

    // Clear all caches
    coordinator.clear()

    // Verify the data is gone
    let afterClear = coordinator.fetch(tokens: tokens)
    switch afterClear {
    case .miss:
        break  // expected
    case .hit:
        Issue.record("Data should be gone after clear")
    }
}
