// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for RotatingKVCache L2 disk persistence in
// `CacheCoordinator.storeAfterGeneration`.
//
// HISTORY: this file originally pinned a SKIP guard — the central
// `hasRotatingLayer` check that suppressed disk + memory writes when
// any layer was a `RotatingKVCache`. That guard existed because
// `TQDiskSerializer` v2 had no `.rotating` LayerKind tag and would
// emit `.skip` placeholders that lost the wrap state silently.
//
// SLIDING-1 (2026-04-15) added the missing `.rotating` LayerKind to
// the v2 schema. RotatingKVCache now round-trips cleanly via
// `serializeRotatingLayer` / `restoreRotatingLayer`, with the full
// 5-tuple metaState `(keep, maxSize, step, offset, idx)` captured in
// `__rot_{i}_meta__`. This file now pins the OPPOSITE contract:
// disk store MUST happen when a RotatingKVCache is present, AND the
// stored entry MUST be retrievable from disk afterwards.

import XCTest
import MLX
@testable import MLXLMCommon

final class CacheCoordinatorRotatingGuardTests: XCTestCase {

    private func makeCoordWithDisk() -> (CacheCoordinator, URL) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_rotating_guard_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: tmp, withIntermediateDirectories: true)
        var cfg = CacheCoordinatorConfig()
        cfg.enableDiskCache = true
        cfg.diskCacheDir = tmp
        cfg.diskCacheMaxGB = 1.0
        cfg.modelKey = "rotating-guard-test"
        return (CacheCoordinator(config: cfg), tmp)
    }

    /// SLIDING-1: cache lists containing a `RotatingKVCache` MUST persist
    /// to disk. The previous skip guard is gone — the v2 schema now has
    /// a `.rotating` LayerKind that captures both the ring buffer and
    /// the wrap-state metaState.
    func testStorePersistsRotatingCacheToDisk() {
        let (coord, dir) = makeCoordWithDisk()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Build a small cache list: one KVCacheSimple + one prefilled
        // RotatingKVCache. Both must serialize.
        let kv = KVCacheSimple()
        let rot = RotatingKVCache(maxSize: 1024, keep: 0)
        // Prefill the rotating layer so it has actual state to capture.
        let k = MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16)
        let v = MLXArray.ones([1, 4, 8, 16], dtype: .bfloat16) * Float(0.5)
        _ = rot.update(keys: k, values: v)

        coord.storeAfterGeneration(
            promptTokens: [1, 2, 3, 4, 5, 6, 7, 8],
            perLayerData: [],
            ssmStates: nil,
            cache: [kv, rot],
            mediaSalt: nil
        )

        let fetched = coord.diskCache?.fetch(
            tokens: [1, 2, 3, 4, 5, 6, 7, 8], mediaSalt: nil)
        XCTAssertNotNil(
            fetched,
            "SLIDING-1: disk store must include cache lists containing a " +
            "RotatingKVCache — the v2 .rotating LayerKind round-trips the " +
            "ring buffer + wrap state. If this fails, the central skip " +
            "guard returned. See CacheCoordinator.swift line 390 (gone).")

        // Verify the rotating layer's kind tag is present in the dict.
        if let arrays = fetched {
            XCTAssertTrue(
                arrays.keys.contains("__layer_kind_1__"),
                "RotatingKVCache layer must be tagged in the disk dict.")
            if let kindArr = arrays["__layer_kind_1__"] {
                let kind = kindArr.shape.isEmpty
                    ? kindArr.item(Int32.self)
                    : kindArr[0].item(Int32.self)
                XCTAssertEqual(
                    kind,
                    TQDiskSerializer.LayerKind.rotating.rawValue,
                    "Rotating layer must be tagged .rotating, got \(kind).")
            }
            XCTAssertTrue(
                arrays.keys.contains("rot_1_keys"),
                "Ring buffer keys must be present at rot_1_keys.")
            XCTAssertTrue(
                arrays.keys.contains("__rot_1_meta__"),
                "5-tuple metaState must be present at __rot_1_meta__.")
        }
    }

    /// SLIDING-1 edge case: `CacheList` wrapping a `RotatingKVCache`
    /// (BaichuanM1 / FalconH1 sliding+mamba mix). The serializer's
    /// outer dispatch sees the CacheList not the inner rotating, so
    /// this currently lands on `.skip` — restore re-prefills the
    /// wrapped layer naturally on the next turn. Pinned to confirm
    /// the disk write does NOT throw and the outer store returns
    /// successfully (no central skip guard remains).
    func testStoreCacheListWrappedRotatingDoesNotThrow() {
        let (coord, dir) = makeCoordWithDisk()
        defer { try? FileManager.default.removeItem(at: dir) }

        let rot = RotatingKVCache(maxSize: 1024, keep: 0)
        let kv = KVCacheSimple()
        let wrapped = CacheList(rot, kv)

        // Call must not throw; the outer CacheList layer lands on
        // `.skip` in v2 (no LayerKind for CacheList), but the OTHER
        // layers in the cache (none here) would still serialize.
        coord.storeAfterGeneration(
            promptTokens: [7, 8, 9],
            perLayerData: [],
            ssmStates: nil,
            cache: [wrapped],
            mediaSalt: nil
        )
        // No assertion on the disk fetch shape — empty stores can fall
        // through to no-op. The contract here is "doesn't crash".
    }
}
