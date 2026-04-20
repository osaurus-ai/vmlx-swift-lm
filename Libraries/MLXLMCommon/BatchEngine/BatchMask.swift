// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX

// MARK: - Batch Causal Mask Generation

/// Create a batch-aware causal attention mask for sequences at different positions.
///
/// Each sequence in the batch may be at a different generation step (different
/// cache offset). This function builds a per-sequence causal mask that:
/// - Allows each query to attend only to keys at positions <= its own position
/// - Masks out padding positions from shorter sequences in the batch
/// - Optionally applies sliding window constraints
///
/// The returned mask has shape `[B, 1, queryLen, totalKeyLen]` where `totalKeyLen`
/// is the maximum effective key length across all sequences.
///
/// ## Rotating / sliding-window caches
///
/// When a slot's underlying cache is a `RotatingKVCache`, the number of keys
/// actually returned by `cache.update(...)` is capped at `maxCacheSize` once
/// the ring wraps — it is NOT `offset + n`. Callers MUST pass per-slot
/// `effectiveKeyLens` in that case, otherwise the mask's last axis will not
/// match the attention scores' last axis and MLX will crash in
/// `broadcast_shapes`. See `GEMMA4-SLIDING-WINDOW-CRASH.md` next to this file.
///
/// - Parameters:
///   - queryLen: Number of new tokens being processed (1 for decode, chunk size for prefill)
///   - offsets: Per-sequence cache offsets — each element is the number of tokens
///     already in that sequence's cache **before** this step's tokens are added.
///     Shape requirement: must have exactly `B` elements.
///   - effectiveKeyLens: Optional per-slot effective key length (in keys actually
///     present in the cache after this step's update). If nil, defaults to
///     `offset_i + n` for each slot (correct for unbounded caches such as
///     `KVCacheSimple`). For `RotatingKVCache` slots that have wrapped, pass
///     `min(offset_i + n, maxCacheSize_i)`.
///   - windowSize: Optional sliding window size for sliding-window attention layers.
///     When set, keys outside the window are masked out. Ignored for slots that
///     have wrapped their ring buffer — post-wrap, every stored key is within
///     the window by construction, and keys are in ring order (not logical
///     position order) so a logical-position window test does not apply.
/// - Returns: Boolean `MLXArray` of shape `[B, 1, queryLen, totalKeyLen]`.
public func createBatchCausalMask(
    queryLen n: Int,
    offsets: [Int],
    effectiveKeyLens: [Int]? = nil,
    windowSize: Int? = nil
) -> MLXArray {
    let B = offsets.count
    precondition(B > 0, "createBatchCausalMask requires at least one sequence")

    // Effective key length per slot = number of keys present in cache.update's
    // return for that slot. For unbounded caches this is `offset + n`; for
    // rotating caches it is capped at `maxCacheSize` once wrapped.
    let keyLens: [Int] = effectiveKeyLens ?? offsets.map { $0 + n }
    precondition(
        keyLens.count == B,
        "effectiveKeyLens.count \(keyLens.count) must equal offsets.count \(B)")

    // Total key length = max effective key length across all sequences.
    // Pad shorter ones to this total. Matches `BatchKVCache.padAndConcatenate`.
    let maxTotal = keyLens.max()!

    // Key column indices: [0, 1, ..., maxTotal - 1], shape [1, maxTotal]
    let rinds = MLXArray(Int32(0) ..< Int32(maxTotal)).reshaped(1, maxTotal)

    // Build per-sequence masks and stack
    var masks = [MLXArray]()
    masks.reserveCapacity(B)

    for (offset, keyLen) in zip(offsets, keyLens) {
        let isWrapped = keyLen < (offset + n)

        let mask: MLXArray
        if isWrapped {
            // Rotating cache has wrapped: every stored key is a valid attention
            // target for every query in this chunk. Keys are in ring order
            // (not logical position order), so the logical causal test does
            // not apply to [0..keyLen). Mask = all-true on valid keys, false
            // on padding (positions [keyLen..maxTotal)).
            //
            // The common case is n == 1 (decode) with windowSize == maxCacheSize
            // (Gemma-3/4 SWA layers where sliding_window equals cache maxSize).
            // In that regime, every ring slot is within the window by
            // construction and all-true is the correct mask.
            mask = MLX.broadcast(
                (rinds .< Int32(keyLen)).reshaped(1, maxTotal),
                to: [n, maxTotal])
        } else {
            // Query row indices for this sequence: [offset, offset+1, ..., offset+n-1]
            // Shape: [n, 1]
            let linds = (MLXArray(Int32(0) ..< Int32(n)) + Int32(offset)).reshaped(n, 1)

            // Standard causal: query at position q can attend to key at position k if k <= q
            var m = linds .>= rinds

            // Sliding window: additionally require k >= q - windowSize + 1
            if let windowSize {
                m = m & (rinds .>= (linds - Int32(windowSize - 1)))
            }

            // Also mask out positions beyond this sequence's actual cached range.
            // Keys at positions >= keyLen are padding from other (longer) sequences.
            m = m & (rinds .< Int32(keyLen))
            mask = m
        }

        // Shape: [1, 1, n, maxTotal] — one mask per sequence
        masks.append(mask.reshaped(1, 1, n, maxTotal))
    }

    // Stack to [B, 1, n, maxTotal]
    return concatenated(masks, axis: 0)
}
