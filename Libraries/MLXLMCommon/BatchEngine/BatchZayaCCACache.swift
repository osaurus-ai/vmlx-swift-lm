// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXNN

// MARK: - BatchZayaCCACache

/// A per-layer batch wrapper for ZAYA CCA-attention slots that mirrors
/// `BatchKVCache`'s split/pad/stack contract for ordinary KV AND adds
/// gather/scatter for the path-dependent CCA state.
///
/// ## How It Works
///
/// Each active sequence in the batch engine owns its own `ZayaCCACache`
/// (B=1). During a batched decode step, the engine constructs one
/// `BatchZayaCCACache` per CCA-attention layer by collecting each
/// sequence's cache for that layer.
///
/// 1. The model calls `cache.update(keys:values:)` with `[B, H, L, D]`
///    KV tensors. Per-slot KV state is updated independently and the
///    padded/stacked result returned for the attention call.
/// 2. The model calls `gatherCCA()` to read each slot's `(conv_state,
///    prev_hs)` stacked along batch dim.
/// 3. After the conv_qk math + new state derivation, the model calls
///    `scatterCCA(conv:prev:)` to push per-slot updates back to each
///    `ZayaCCACache`.
///
/// State isolation guarantee: gather concatenates per-slot tensors along
/// dim 0, so slot A's data is never mixed with slot B's. Mutating row i
/// in the gathered tensor and scattering back affects only `slotCaches[i]`.
public final class BatchZayaCCACache: BaseKVCache {

    /// Per-sequence ZAYA caches. Index matches batch dim ordering.
    private let slotCaches: [ZayaCCACache]

    /// Number of sequences in this batch.
    public let batchSize: Int

    /// Per-sequence position offsets as `[B]`-shaped `MLXArray`. Mirrors
    /// `BatchKVCache.offsetArray` so RoPE per-slot routing keeps working.
    public private(set) var offsetArray: MLXArray

    /// Convenience identity for the model: all slots share the same CCA
    /// geometry (conv_channels + hidden_size) — the BatchEngine constructs
    /// the wrapper from same-layer slots so this is a structural invariant.
    public var convChannels: Int { slotCaches[0].convChannels }
    public var hiddenSize: Int { slotCaches[0].hiddenSize }

    public init(slotCaches: [ZayaCCACache]) {
        precondition(!slotCaches.isEmpty,
            "BatchZayaCCACache requires at least one slot cache")
        let cc = slotCaches[0].convChannels
        let hs = slotCaches[0].hiddenSize
        precondition(slotCaches.allSatisfy { $0.convChannels == cc && $0.hiddenSize == hs },
            "BatchZayaCCACache: all slot caches must share convChannels/hiddenSize")
        self.slotCaches = slotCaches
        self.batchSize = slotCaches.count
        self.offsetArray = MLXArray(slotCaches.map { Int32($0.offset) })
        super.init()
        self.offset = slotCaches.map(\.offset).max() ?? 0
    }

    // MARK: - KVCache protocol

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        precondition(B == batchSize, "Key batch size \(B) != expected \(batchSize)")

        var allKeys = [MLXArray]()
        var allValues = [MLXArray]()
        allKeys.reserveCapacity(B)
        allValues.reserveCapacity(B)

        for i in 0..<B {
            let ki = keys[i ..< i + 1]
            let vi = values[i ..< i + 1]
            let (ck, cv) = slotCaches[i].update(keys: ki, values: vi)
            allKeys.append(ck)
            allValues.append(cv)
        }

        let paddedKeys = padAndConcatenate(allKeys, along: 2)
        let paddedValues = padAndConcatenate(allValues, along: 2)

        self.offsetArray = MLXArray(slotCaches.map { Int32($0.offset) })
        self.offset = slotCaches.map(\.offset).max() ?? 0

        return (paddedKeys, paddedValues)
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let offsets = slotCaches.map(\.offset)
        let effectiveKeyLens: [Int] = slotCaches.map { slot in
            let logical = slot.offset + n
            if let maxSize = slot.maxSize, logical > maxSize {
                return maxSize
            }
            return logical
        }
        return .array(createBatchCausalMask(
            queryLen: n,
            offsets: offsets,
            effectiveKeyLens: effectiveKeyLens,
            windowSize: windowSize))
    }

    public override var maxSize: Int? { nil }

    // MARK: - CCA gather/scatter

    /// Stack each slot's `(conv_state, prev_hs)` along batch dim so the
    /// model's CCA-attention forward can run as if it had a B-wide cache.
    public func gatherCCA() -> (conv: MLXArray, prev: MLXArray) {
        let convs = slotCaches.map { $0.readCCA().conv }
        let prevs = slotCaches.map { $0.readCCA().prev }
        return (concatenated(convs, axis: 0), concatenated(prevs, axis: 0))
    }

    /// Split the batched CCA state along dim 0 and write each row back to
    /// the corresponding slot. Per-slot identity is preserved by construction.
    public func scatterCCA(conv: MLXArray, prev: MLXArray) {
        precondition(conv.dim(0) == batchSize,
            "scatterCCA: conv has B=\(conv.dim(0)), expected \(batchSize)")
        precondition(prev.dim(0) == batchSize,
            "scatterCCA: prev has B=\(prev.dim(0)), expected \(batchSize)")
        for i in 0..<batchSize {
            slotCaches[i].writeCCA(
                conv: conv[i ..< i + 1],
                prev: prev[i ..< i + 1])
        }
    }

    // MARK: - Unsupported Operations

    // BatchZayaCCACache is a transient view — it is not serializable,
    // trimmable, or copyable. Per-slot ZayaCCACaches are the durable record.

    public override var state: [MLXArray] {
        get { [] }
        set { }
    }

    public override var metaState: [String] {
        get { [""] }
        set { }
    }

    public override var isTrimmable: Bool { false }

    public override func copy() -> any KVCache {
        fatalError("BatchZayaCCACache is a transient view and cannot be copied")
    }

    // MARK: - Internal Helpers

    private func padAndConcatenate(_ arrays: [MLXArray], along axis: Int) -> MLXArray {
        let maxLen = arrays.map { $0.dim(axis) }.max() ?? 0
        let padded: [MLXArray] = arrays.map { arr in
            let currentLen = arr.dim(axis)
            guard currentLen < maxLen else { return arr }
            var paddingShape = arr.shape
            paddingShape[axis] = maxLen - currentLen
            let pad = MLXArray.zeros(paddingShape, dtype: arr.dtype)
            return concatenated([arr, pad], axis: axis)
        }
        return concatenated(padded, axis: 0)
    }
}
