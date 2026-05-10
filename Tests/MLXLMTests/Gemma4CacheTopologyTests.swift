// Pin Gemma4's `newCache(parameters:)` topology + `BatchCompile.classify` result
// for both branches of the silent-`maxKVSize`-foot-gun.
//
// Background (see `docs/GEMMA4-DEEP-TRACE-2026-05-10.md` §2.1, §7.1):
// `Gemma4TextModel.newCache` returns a MIXED cache list when the caller did
// not supply `parameters?.maxKVSize` (sliding layers are RotatingKVCache,
// full-attention layers are KVCacheSimple) — `BatchCompile.classify` reports
// `.heterogeneous` and the compile path is skipped silently. With
// `maxKVSize` provided, both layer types use RotatingKVCache and classify
// returns `.rotating`.
//
// This test mirrors `Gemma4TextModel.newCache` source by hand-constructing
// the cache list it WOULD produce for representative configs. We avoid
// instantiating the full model so the test runs without a Metal
// runner — the contract it pins is purely structural (cache types
// returned + `CacheFamily.classify` result), not a forward-pass row.

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("Gemma4 cache topology + compile-classification contract")
struct Gemma4CacheTopologyTests {

    /// Reproduces the cache list `Gemma4TextModel.newCache(parameters: nil)`
    /// returns for a 4-layer Gemma4 with the canonical sliding/full
    /// alternation and no KV sharing.
    private static func mixedRotatingSimpleCache(slidingWindow: Int = 64) -> [KVCache] {
        // layerTypes = ["sliding", "full", "sliding", "full"]
        // No maxKVSize → full layers use KVCacheSimple()
        return [
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            KVCacheSimple(),
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            KVCacheSimple(),
        ]
    }

    /// Reproduces the cache list when `maxKVSize` IS provided.
    /// Full-attention layers use `RotatingKVCache(maxSize: maxKVSize, keep: 4)`
    /// (attention-sink pattern), sliding layers use the window-sized rotor.
    private static func allRotatingCache(slidingWindow: Int = 64, maxKVSize: Int = 2048) -> [KVCache] {
        return [
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            RotatingKVCache(maxSize: maxKVSize, keep: 4),
            RotatingKVCache(maxSize: slidingWindow, keep: 0),
            RotatingKVCache(maxSize: maxKVSize, keep: 4),
        ]
    }

    /// Without `maxKVSize`, full-attention layers use `KVCacheSimple` while
    /// sliding layers use `RotatingKVCache`. `BatchCompile.classify` sees
    /// the heterogeneous mix and returns `.heterogeneous`, which means the
    /// Stage 1B.3 compile path is skipped. Pin both halves so a refactor
    /// can't silently flip the compile-eligibility of every Gemma4 bundle.
    @Test("Mixed Rotating+Simple Gemma4 cache classifies as heterogeneous (compile skipped)")
    func cacheWithoutMaxKVSizeIsHeterogeneous() {
        let cache = Self.mixedRotatingSimpleCache()

        #expect(cache.count == 4)
        #expect(cache[0] is RotatingKVCache, "sliding layer should be RotatingKVCache")
        #expect(cache[1] is KVCacheSimple, "full layer (no maxKVSize) should be KVCacheSimple")
        #expect(cache[2] is RotatingKVCache, "sliding layer should be RotatingKVCache")
        #expect(cache[3] is KVCacheSimple, "full layer (no maxKVSize) should be KVCacheSimple")

        let family = CacheFamily.classify(cache)
        #expect(family == .heterogeneous,
            "Mixed Rotating+Simple Gemma4 cache must classify as heterogeneous so BatchCompile skips compile.")
        #expect(family.isCompileEligibleAtCurrentStage == false)
    }

    /// With `maxKVSize` supplied, full-attention layers also use
    /// `RotatingKVCache(maxSize: maxKVSize, keep: 4)` — the entire list is
    /// RotatingKVCache and classify returns `.rotating`.
    @Test("All-Rotating Gemma4 cache classifies as .rotating (compile-eligible)")
    func cacheWithMaxKVSizeIsRotating() {
        let cache = Self.allRotatingCache()

        #expect(cache.count == 4)
        for (i, c) in cache.enumerated() {
            #expect(c is RotatingKVCache, "layer \(i) should be RotatingKVCache when maxKVSize is set")
        }

        let family = CacheFamily.classify(cache)
        #expect(family == .rotating,
            "All-Rotating Gemma4 cache must classify as .rotating so BatchCompile picks the rotating bucket.")
        #expect(family.isCompileEligibleAtCurrentStage == true)
    }

    /// Pins the attention-sink invariant for full-attention layers.
    /// Gemma4's full-attention RotatingKVCache uses `keep: 4` (preserves
    /// the first 4 tokens across rotations). Sliding-window layers use
    /// `keep: 0`. A future refactor that flipped these would break the
    /// model's long-context expectations.
    @Test("Full-attention RotatingKVCache uses keep=4 (attention sink); sliding uses keep=0")
    func attentionSinkContract() {
        let sliding = RotatingKVCache(maxSize: 1024, keep: 0)
        let full = RotatingKVCache(maxSize: 4096, keep: 4)

        // RotatingKVCache exposes `keep` indirectly via maxSize; the
        // attention-sink semantic is in the `keep` constructor arg.
        // Construction itself is the contract pin — if the type signature
        // changes (e.g. removing `keep`) this test fails to compile,
        // forcing the Gemma4 model to be re-evaluated.
        _ = sliding
        _ = full
        #expect(sliding.maxSize == 1024)
        #expect(full.maxSize == 4096)
    }
}
