// Regression test for hybrid SSM/GLA models that must mark the cache
// coordinator paged-incompatible. See `docs/HYBRID-PAGED-CACHE-BUG-2026-05-09.md`
// for the bench evidence: Bailing/Ling (ArraysCache GLA) and Nemotron-H
// (MambaCache) produced garbled Turn 2 output on warm-cache because the
// paged-incompatible auto-flip in Evaluate.swift / BatchEngine.swift
// missed those cache types. The right detector is
// `cacheContainsPathDependentState` (CacheHelpers.swift:105-119) — these
// tests pin its contract so the auto-flip can call it directly.

import Foundation
@testable import MLXLMCommon
import Testing

@Test func mambaCacheTriggersPathDependent() {
    let cache: [any KVCache] = [MambaCache(), KVCacheSimple()]
    #expect(cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
}

@Test func arraysCacheTriggersPathDependent() {
    let cache: [any KVCache] = [ArraysCache(size: 2), KVCacheSimple()]
    #expect(cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
}

@Test func zayaCCACacheTriggersPathDependent() {
    let cache: [any KVCache] = [ZayaCCACache(), KVCacheSimple()]
    #expect(cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
}

@Test func cacheListWrappingMambaTriggersPathDependent() {
    let composite = CacheList(MambaCache(), KVCacheSimple())
    let cache: [any KVCache] = [composite]
    #expect(cacheContainsPathDependentState(cache))
}

@Test func cacheListWrappingArraysTriggersPathDependent() {
    let composite = CacheList(ArraysCache(size: 2), KVCacheSimple())
    let cache: [any KVCache] = [composite]
    #expect(cacheContainsPathDependentState(cache))
}

@Test func plainKVDoesNotTriggerPathDependent() {
    let cache: [any KVCache] = [KVCacheSimple(), KVCacheSimple()]
    #expect(!cacheContainsPathDependentState(cache))
    #expect(!cacheRequiresDiskBackedCoordinatorRestore(cache))
}

@Test func emptyCacheDoesNotTriggerPathDependent() {
    let cache: [any KVCache] = []
    #expect(!cacheContainsPathDependentState(cache))
    #expect(!cacheRequiresDiskBackedCoordinatorRestore(cache))
}

@Test func rotatingCacheRequiresDiskBackedCoordinatorRestore() {
    let cache: [any KVCache] = [RotatingKVCache(maxSize: 32), KVCacheSimple()]
    #expect(!cacheContainsPathDependentState(cache))
    #expect(cacheRequiresDiskBackedCoordinatorRestore(cache))
}

@Test func mixedHybridLayerOrderingTriggersPathDependent() {
    // Real hybrid models (Nemotron-H, Bailing) interleave KV layers with
    // recurrent state layers. Path-dependent detection must fire regardless
    // of which position the recurrent layer occupies.
    let cacheStart: [any KVCache] = [MambaCache(), KVCacheSimple(), KVCacheSimple()]
    let cacheMiddle: [any KVCache] = [KVCacheSimple(), MambaCache(), KVCacheSimple()]
    let cacheEnd: [any KVCache] = [KVCacheSimple(), KVCacheSimple(), MambaCache()]
    #expect(cacheContainsPathDependentState(cacheStart))
    #expect(cacheContainsPathDependentState(cacheMiddle))
    #expect(cacheContainsPathDependentState(cacheEnd))
}
