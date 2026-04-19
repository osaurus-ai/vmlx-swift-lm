// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 1A unit tests for `BatchCompile` scaffolding — pure-logic helpers.
// Covers:
//  - `CacheFamily.classify(_:)` on homogeneous arrays and hybrids
//  - `BatchCompile.nextBucket(activeCount:buckets:)` bucket selection
//  - `BatchCompile.makeLiveMask(bucketSize:liveIndices:)` shape + values
//  - `BucketKey` Hashable contract
//
// Stage 1B tests will exercise the actual compile trace; this file is pure
// logic on top of the types. No model load, no MLX GPU traffic beyond the
// single `MLXArray` created by `makeLiveMask`.

import Foundation
import MLX
import MLXLMCommon
import Testing

// MARK: - CacheFamily classification

@Suite("CacheFamily classification", .serialized)
struct CacheFamilyClassificationTests {

    @Test("single KVCacheSimple classifies as .simple")
    func testSingleSimple() {
        let cache: [KVCache] = [KVCacheSimple()]
        #expect(CacheFamily.classify(cache) == .simple)
    }

    @Test("pure KVCacheSimple array classifies as .simple")
    func testPureSimple() {
        let cache: [KVCache] = (0 ..< 4).map { _ in KVCacheSimple() }
        #expect(CacheFamily.classify(cache) == .simple)
    }

    @Test("pure CompilableKVCache array classifies as .simple")
    func testPureCompilable() {
        let cache: [KVCache] = (0 ..< 4).map { _ in CompilableKVCache(maxLength: 256) }
        #expect(CacheFamily.classify(cache) == .simple)
    }

    @Test("mixed KVCacheSimple + CompilableKVCache both classify as .simple")
    func testMixedSimpleCompilable() {
        let cache: [KVCache] = [
            KVCacheSimple(),
            CompilableKVCache(maxLength: 256),
            KVCacheSimple(),
            CompilableKVCache(maxLength: 256),
        ]
        #expect(CacheFamily.classify(cache) == .simple,
            "CompilableKVCache and KVCacheSimple share the .simple family")
    }

    @Test("pure RotatingKVCache array classifies as .rotating")
    func testPureRotating() {
        let cache: [KVCache] = (0 ..< 4).map { _ in
            RotatingKVCache(maxSize: 256, keep: 4)
        }
        #expect(CacheFamily.classify(cache) == .rotating)
    }

    @Test("pure MambaCache array classifies as .mamba")
    func testPureMamba() {
        let cache: [KVCache] = (0 ..< 4).map { _ in MambaCache() }
        #expect(CacheFamily.classify(cache) == .mamba)
    }

    @Test("hybrid Mamba + KVCacheSimple classifies as .heterogeneous")
    func testHybridMambaSimple() {
        // Matches Qwen3.5 / Qwen3Next (GDN) / LFM2 / Jamba which alternate
        // Mamba (SSM state) and attention (KV) layers.
        let cache: [KVCache] = [
            MambaCache(),
            KVCacheSimple(),
            MambaCache(),
            KVCacheSimple(),
        ]
        #expect(CacheFamily.classify(cache) == .heterogeneous)
    }

    @Test("pure TurboQuantKVCache classifies as .turboQuant")
    func testPureTurboQuant() {
        let cache: [KVCache] = (0 ..< 4).map { _ in
            TurboQuantKVCache(keyBits: 3, valueBits: 3)
        }
        #expect(CacheFamily.classify(cache) == .turboQuant)
    }

    @Test("TurboQuantKVCache + KVCacheSimple mixed classifies as .heterogeneous")
    func testMixedTQSimple() {
        // Can happen during the "some slots compressed, some still below
        // threshold" window that BatchKVCache handles via shared shape
        // contract (Stage 0). Under the compile path (Stage 2+), this mixed
        // state requires the uncompiled fallback.
        let cache: [KVCache] = [
            KVCacheSimple(),
            TurboQuantKVCache(keyBits: 3, valueBits: 3),
        ]
        #expect(CacheFamily.classify(cache) == .heterogeneous)
    }

    @Test("family compile-eligibility stamps reflect current stage")
    func testEligibilityStamps() {
        #expect(CacheFamily.simple.isCompileEligibleAtCurrentStage == true,
            "Stage 1 ships .simple compile support")
        #expect(CacheFamily.turboQuant.isCompileEligibleAtCurrentStage == true,
            "Stage 2 SHIPPED (iter 21). Root cause was applyRotaryPosition routing TQ through Int offset instead of MLXArray offsetArray — fixed in RoPEApplication.swift. Drift dropped from 6-13% to FP precision (~5e-7).")
        #expect(CacheFamily.rotating.isCompileEligibleAtCurrentStage == true,
            "Stage 3 shipped (iter 13) — rotating compile wired in BatchEngine")
        #expect(CacheFamily.mamba.isCompileEligibleAtCurrentStage == false,
            "Stage 4 pending")
        #expect(CacheFamily.cacheList.isCompileEligibleAtCurrentStage == true,
            "Stage 5 shipped (iter 22 wiring) — CompilableCacheList composite wired via BatchEngine")
        #expect(CacheFamily.heterogeneous.isCompileEligibleAtCurrentStage == false,
            "Heterogeneous state is never traceable as one trace")
    }
}

// MARK: - BatchCompile.nextBucket

@Suite("BatchCompile.nextBucket")
struct BatchCompileNextBucketTests {

    @Test("picks smallest bucket >= activeCount")
    func testBasicSelection() {
        let buckets = [1, 2, 4]
        #expect(BatchCompile.nextBucket(activeCount: 1, buckets: buckets) == 1)
        #expect(BatchCompile.nextBucket(activeCount: 2, buckets: buckets) == 2)
        #expect(BatchCompile.nextBucket(activeCount: 3, buckets: buckets) == 4)
        #expect(BatchCompile.nextBucket(activeCount: 4, buckets: buckets) == 4)
    }

    @Test("returns nil when activeCount exceeds largest bucket")
    func testExceedsLargest() {
        #expect(BatchCompile.nextBucket(activeCount: 5, buckets: [1, 2, 4]) == nil)
        #expect(BatchCompile.nextBucket(activeCount: 9, buckets: [1, 2, 4, 8]) == nil)
    }

    @Test("returns nil when activeCount <= 0")
    func testZeroOrNegativeActive() {
        #expect(BatchCompile.nextBucket(activeCount: 0, buckets: [1, 2, 4]) == nil)
        #expect(BatchCompile.nextBucket(activeCount: -1, buckets: [1, 2, 4]) == nil)
    }

    @Test("returns nil when buckets is empty")
    func testEmptyBuckets() {
        #expect(BatchCompile.nextBucket(activeCount: 3, buckets: []) == nil)
    }

    @Test("handles unsorted input defensively")
    func testUnsortedBuckets() {
        #expect(BatchCompile.nextBucket(activeCount: 3, buckets: [8, 1, 4, 2]) == 4,
            "Caller passed unsorted buckets — helper must still pick smallest >= activeCount")
    }

    @Test("handles duplicate entries defensively")
    func testDuplicateBuckets() {
        #expect(BatchCompile.nextBucket(activeCount: 2, buckets: [1, 1, 2, 2, 4, 4]) == 2)
        #expect(BatchCompile.nextBucket(activeCount: 3, buckets: [4, 4, 4]) == 4)
    }

    @Test("skips non-positive entries defensively")
    func testNonPositiveBuckets() {
        #expect(BatchCompile.nextBucket(activeCount: 1, buckets: [0, -1, 1, 2]) == 1,
            "Zero/negative buckets should be filtered out")
    }

    @Test("larger bucket sets stretch further")
    func testLargerSet() {
        let buckets = [1, 2, 4, 8, 16]
        #expect(BatchCompile.nextBucket(activeCount: 5, buckets: buckets) == 8)
        #expect(BatchCompile.nextBucket(activeCount: 9, buckets: buckets) == 16)
        #expect(BatchCompile.nextBucket(activeCount: 17, buckets: buckets) == nil)
    }
}

// MARK: - BatchCompile.makeLiveMask

@Suite("BatchCompile.makeLiveMask")
struct BatchCompileLiveMaskTests {

    @Test("all rows live produces all-true mask")
    func testAllLive() {
        let mask = BatchCompile.makeLiveMask(bucketSize: 4, liveIndices: [0, 1, 2, 3])
        MLX.eval(mask)
        #expect(mask.shape == [4])
        for i in 0 ..< 4 {
            #expect(mask[i].item(Int32.self) == 1)
        }
    }

    @Test("all rows dead produces all-false mask")
    func testAllDead() {
        let mask = BatchCompile.makeLiveMask(bucketSize: 4, liveIndices: [])
        MLX.eval(mask)
        #expect(mask.shape == [4])
        for i in 0 ..< 4 {
            #expect(mask[i].item(Int32.self) == 0)
        }
    }

    @Test("partial liveness marks only specified rows")
    func testPartialLive() {
        // Bucket of 4 slots, 2 live at rows 0 and 2
        let mask = BatchCompile.makeLiveMask(bucketSize: 4, liveIndices: [0, 2])
        MLX.eval(mask)
        #expect(mask.shape == [4])
        #expect(mask[0].item(Int32.self) == 1)
        #expect(mask[1].item(Int32.self) == 0)
        #expect(mask[2].item(Int32.self) == 1)
        #expect(mask[3].item(Int32.self) == 0)
    }
}

// MARK: - BucketKey

@Suite("BucketKey")
struct BucketKeyTests {

    @Test("equality requires all three fields to match")
    func testEquality() {
        let a = BucketKey(batchSize: 4, maxCacheLength: 4096, family: .simple)
        let b = BucketKey(batchSize: 4, maxCacheLength: 4096, family: .simple)
        let c = BucketKey(batchSize: 4, maxCacheLength: 4096, family: .turboQuant)
        let d = BucketKey(batchSize: 4, maxCacheLength: 2048, family: .simple)
        let e = BucketKey(batchSize: 2, maxCacheLength: 4096, family: .simple)

        #expect(a == b)
        #expect(a != c, "different family")
        #expect(a != d, "different maxCacheLength")
        #expect(a != e, "different batchSize")
    }

    @Test("hashable — distinct keys dedup in a Set")
    func testHashable() {
        let a = BucketKey(batchSize: 1, maxCacheLength: 4096, family: .simple)
        let b = BucketKey(batchSize: 2, maxCacheLength: 4096, family: .simple)
        let c = BucketKey(batchSize: 4, maxCacheLength: 4096, family: .simple)

        let set: Set<BucketKey> = [a, b, c, a, b, c]
        #expect(set.count == 3, "Distinct keys must deduplicate correctly")
    }

    @Test("description reflects all three fields")
    func testDescription() {
        let k = BucketKey(batchSize: 4, maxCacheLength: 4096, family: .simple)
        let desc = k.description
        #expect(desc.contains("B=4"))
        #expect(desc.contains("maxLen=4096"))
        #expect(desc.contains("simple"))
    }
}
