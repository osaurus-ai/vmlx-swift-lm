// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - KV Cache Extraction

/// Extract per-layer KV tensors from a model's cache array.
///
/// Returns per-layer `(keys, values)` tuples. SSM/MambaCache layers return `nil`.
/// Used to populate ``CacheBlock/cacheData`` for paged cache storage.
///
/// - Parameter cache: The model's per-layer cache array.
/// - Returns: An array of optional `(keys, values)` tuples, one per layer.
public func extractLayerData(from cache: [any KVCache]) -> [(keys: MLXArray, values: MLXArray)?] {
    cache.map { layer in
        if let simple = layer as? KVCacheSimple {
            let state = simple.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let cacheList = layer as? CacheList {
            // CacheList: check sub-caches for KV data.
            // Sub-cache[0] is typically MambaCache, Sub-cache[1] is KVCacheSimple.
            // We only extract the KV part; SSM state is handled separately.
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    let state = simple.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
            }
            return nil
        }
        // MambaCache, ArraysCache, RotatingKVCache — no KV extraction
        return nil
    }
}

// MARK: - KV Cache Restoration

/// Restore per-layer KV tensors from cached blocks into a model's cache array.
///
/// Concatenates KV data from multiple blocks per layer, then sets cache state
/// via the public ``KVCache/state`` setter.
///
/// - Parameters:
///   - blocks: The cache blocks to restore from, ordered by sequence position.
///   - cache: The model's per-layer cache array to restore into.
/// - Returns: The total number of tokens restored across all blocks.
@discardableResult
public func restoreLayerData(from blocks: [CacheBlock], into cache: [any KVCache]) -> Int {
    guard let firstBlock = blocks.first, let firstData = firstBlock.cacheData else { return 0 }
    let numLayers = firstData.count
    guard numLayers == cache.count else { return 0 }

    for layerIdx in 0..<numLayers {
        // Collect KV slices from all blocks for this layer
        var keySlices: [MLXArray] = []
        var valueSlices: [MLXArray] = []

        for block in blocks {
            guard let data = block.cacheData, layerIdx < data.count,
                  let kv = data[layerIdx] else { continue }
            keySlices.append(kv.keys)
            valueSlices.append(kv.values)
        }

        guard !keySlices.isEmpty else { continue }

        // Concatenate along sequence dimension (axis 2: [B, H, T, D])
        let restoredKeys = keySlices.count == 1 ? keySlices[0] : concatenated(keySlices, axis: 2)
        let restoredValues = valueSlices.count == 1 ? valueSlices[0] : concatenated(valueSlices, axis: 2)

        // Restore into the cache layer
        if let simple = cache[layerIdx] as? KVCacheSimple {
            simple.state = [restoredKeys, restoredValues]
        } else if let cacheList = cache[layerIdx] as? CacheList {
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    simple.state = [restoredKeys, restoredValues]
                    break
                }
            }
        }
    }

    let totalTokens = blocks.reduce(0) { $0 + $1.tokenCount }
    return totalTokens
}

// MARK: - SSM State Extraction

/// Extract SSM (MambaCache/ArraysCache) states from a model's cache array.
///
/// Returns the state arrays from each SSM layer. Non-SSM layers are skipped.
/// Used to populate ``SSMStateCache`` for hybrid model companion storage.
///
/// - Parameter cache: The model's per-layer cache array.
/// - Returns: All SSM state arrays, flattened across layers.
public func extractSSMStates(from cache: [any KVCache]) -> [MLXArray] {
    var states: [MLXArray] = []
    for layer in cache {
        if let mamba = layer as? MambaCache {
            // MambaCache.state returns [conv_state, hidden_state]
            states.append(contentsOf: mamba.state)
        } else if let arrays = layer as? ArraysCache {
            states.append(contentsOf: arrays.state)
        } else if let cacheList = layer as? CacheList {
            // Extract SSM sub-cache from composite layers
            for i in 0..<cacheList.count {
                if let mamba = cacheList[i] as? MambaCache {
                    states.append(contentsOf: mamba.state)
                } else if let arrays = cacheList[i] as? ArraysCache {
                    states.append(contentsOf: arrays.state)
                }
            }
        }
    }
    return states
}

// MARK: - SSM State Restoration

/// Restore SSM states into a model's cache array.
///
/// The `states` array should match the output order of ``extractSSMStates(from:)``.
/// Each MambaCache consumes 2 state arrays (conv state + hidden state).
///
/// - Parameters:
///   - states: The SSM state arrays to restore.
///   - cache: The model's per-layer cache array to restore into.
public func restoreSSMStates(_ states: [MLXArray], into cache: [any KVCache]) {
    var stateIdx = 0
    for layer in cache {
        if let mamba = layer as? MambaCache {
            let existingCount = mamba.state.count
            if existingCount == 0 {
                // Fresh cache — MambaCache always has 2 slots (conv + hidden)
                let slotCount = 2
                if stateIdx + slotCount <= states.count {
                    mamba.state = Array(states[stateIdx..<(stateIdx + slotCount)])
                        .map { $0[.ellipsis] }
                    stateIdx += slotCount
                }
            } else if stateIdx + existingCount <= states.count {
                mamba.state = Array(states[stateIdx..<(stateIdx + existingCount)])
                    .map { $0[.ellipsis] }
                stateIdx += existingCount
            }
        } else if let arrays = layer as? ArraysCache {
            // ArraysCache (non-Mamba variant) — restore however many slots it has
            let existingCount = arrays.state.count
            if existingCount > 0, stateIdx + existingCount <= states.count {
                arrays.state = Array(states[stateIdx..<(stateIdx + existingCount)])
                    .map { $0[.ellipsis] }
                stateIdx += existingCount
            }
        } else if let cacheList = layer as? CacheList {
            for i in 0..<cacheList.count {
                if let mamba = cacheList[i] as? MambaCache {
                    let slotCount = 2
                    if stateIdx + slotCount <= states.count {
                        mamba.state = Array(states[stateIdx..<(stateIdx + slotCount)])
                            .map { $0[.ellipsis] }
                        stateIdx += slotCount
                    }
                } else if let arrays = cacheList[i] as? ArraysCache {
                    let existingCount = arrays.state.count
                    if existingCount > 0, stateIdx + existingCount <= states.count {
                        arrays.state = Array(states[stateIdx..<(stateIdx + existingCount)])
                            .map { $0[.ellipsis] }
                        stateIdx += existingCount
                    }
                }
            }
        }
    }
}
