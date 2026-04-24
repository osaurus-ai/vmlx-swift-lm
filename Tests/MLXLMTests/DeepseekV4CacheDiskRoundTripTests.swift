// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// L2 disk round-trip tests for DSV4 per-layer cache types.
//
// Verifies:
//   - plain `RotatingKVCache` (default DSV4 path) encodes via
//     `TQDiskSerializer` and decodes via `restoreRotatingLayer`
//   - `DeepseekV4Cache` (DSV4_LONG_CTX=1 opt-in path) conforms to
//     `RotatingKVCacheWrapper` so its inner rotating state round-trips
//     transparently; compressor/indexer buffer state is ephemeral
//     and correctly NOT persisted (recomputable from prompt tokens)
//   - A mixed per-layer array of both types encodes and restores
//     without kind-tag drift.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@Suite("DSV4 L2 disk round-trip", .serialized)
struct DeepseekV4CacheDiskRoundTripTests {

    /// Fill a RotatingKVCache with two (keys, values) steps so its
    /// state has shape-preserving content for the roundtrip check.
    static func fillRotating(
        _ rot: RotatingKVCache,
        B: Int = 1, H: Int = 1, headDim: Int = 8
    ) {
        let step1Keys = MLXArray.ones([B, H, 3, headDim])
        let step1Vals = MLXArray.ones([B, H, 3, headDim]) * 2.0
        _ = rot.update(keys: step1Keys, values: step1Vals)
        let step2Keys = MLXArray.ones([B, H, 2, headDim]) * 3.0
        let step2Vals = MLXArray.ones([B, H, 2, headDim]) * 4.0
        _ = rot.update(keys: step2Keys, values: step2Vals)
    }

    @Test("RotatingKVCache (default DSV4 path) disk round-trips")
    func rotatingRoundTrip() {
        let rot = RotatingKVCache(maxSize: 16, keep: 0)
        Self.fillRotating(rot)
        let originalState = rot.state
        let originalMeta = rot.metaState
        let originalOffset = rot.offset

        // Encode via TQDiskSerializer.
        let encoded = TQDiskSerializer.serialize(cache: [rot])
        #expect(encoded["__layer_kind_0__"] != nil,
            "encode must tag layer 0 kind")
        #expect(encoded["rot_0_keys"] != nil)
        #expect(encoded["rot_0_values"] != nil)
        #expect(encoded["__rot_0_meta__"] != nil)

        // Decode into a fresh cache via restoreFromDiskArrays.
        let target = RotatingKVCache(maxSize: 16, keep: 0)
        _ = restoreFromDiskArrays(encoded, into: [target])
        #expect(target.state.count == originalState.count)
        #expect(target.metaState == originalMeta)
        #expect(target.offset == originalOffset,
            "offset must survive disk round-trip")
    }

    @Test("DeepseekV4Cache disk round-trip: rotating state persists, compressor cleared")
    func deepseekV4CacheRoundTrip() {
        let v4 = DeepseekV4Cache(slidingWindow: 16)
        Self.fillRotating(v4.local)
        // Stash ephemeral buffer state via the internal API — prove it
        // does NOT survive the round-trip (documented contract).
        let dummyBuf = MLXArray.ones([1, 5, 8])
        v4.setBuffers(.compressor, kv: dummyBuf, gate: dummyBuf)
        v4.setPooled(.compressor, value: dummyBuf)

        let originalLocalState = v4.local.state
        let originalOffset = v4.offset
        let originalMeta = v4.local.metaState

        // Encode — should go through the RotatingKVCacheWrapper path.
        let encoded = TQDiskSerializer.serialize(cache: [v4])
        #expect(encoded["rot_0_keys"] != nil,
            "DeepseekV4Cache must serialize as rotating via wrapper protocol")
        #expect(encoded["rot_0_values"] != nil)

        // Decode into a fresh v4 cache.
        let target = DeepseekV4Cache(slidingWindow: 16)
        _ = restoreFromDiskArrays(encoded, into: [target])
        #expect(target.offset == originalOffset,
            "inner offset must survive round-trip")
        #expect(target.local.metaState == originalMeta,
            "inner rotating metaState must survive")
        #expect(target.local.state.count == originalLocalState.count)
        // Compressor/Indexer buffer state must be CLEAR on the restored
        // cache (ephemeral — not serialized).
        let (bufKV, bufGate) = target.getBuffers(.compressor)
        #expect(bufKV == nil && bufGate == nil,
            "compressor buffers must be nil on restore — they recompute on next prefill")
        #expect(target.getPooled(.compressor) == nil)
    }

    @Test("Mixed per-layer array: RotatingKVCache + DeepseekV4Cache round-trip together")
    func mixedPerLayerRoundTrip() {
        let layer0 = RotatingKVCache(maxSize: 16, keep: 0)
        Self.fillRotating(layer0)
        let layer1 = DeepseekV4Cache(slidingWindow: 16)
        Self.fillRotating(layer1.local)

        let caches: [any KVCache] = [layer0, layer1]
        let encoded = TQDiskSerializer.serialize(cache: caches)
        #expect(encoded["rot_0_keys"] != nil)
        #expect(encoded["rot_1_keys"] != nil)

        let target: [any KVCache] = [
            RotatingKVCache(maxSize: 16, keep: 0),
            DeepseekV4Cache(slidingWindow: 16),
        ]
        _ = restoreFromDiskArrays(encoded, into: target)
        #expect(target[0].offset == layer0.offset)
        #expect(target[1].offset == layer1.offset)
    }

    @Test("DeepseekV4Cache conforms to RotatingKVCacheWrapper protocol")
    func wrapperProtocolConformance() {
        let v4 = DeepseekV4Cache(slidingWindow: 16)
        let wrapper: RotatingKVCacheWrapper? = v4
        #expect(wrapper != nil,
            "DeepseekV4Cache must conform to RotatingKVCacheWrapper")
        #expect(wrapper?.rotating === v4.local,
            "wrapper.rotating must return the exact inner RotatingKVCache")
    }
}
