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
        if let quantized = layer as? QuantizedKVCache {
            // QuantizedKVCache stores quantized tuples, not raw KV.
            // Dequantize back to float keys/values for cache block storage.
            let unquantized = quantized.toUnquantized()
            let state = unquantized.state
            guard state.count == 2 else { return nil }
            return (keys: state[0], values: state[1])
        }
        if let tq = layer as? TurboQuantKVCache {
            // TurboQuantKVCache.state returns float KV in both fill and compressed phases.
            // In fill phase: returns raw float keys/values.
            // In compressed phase: returns unified (decompressed prefix + float window).
            let state = tq.state
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
                if let quantized = cacheList[i] as? QuantizedKVCache {
                    let unquantized = quantized.toUnquantized()
                    let state = unquantized.state
                    if state.count == 2 { return (keys: state[0], values: state[1]) }
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    let state = tq.state
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
/// Blocks only contain KV-bearing layers (SSM/RotatingKVCache layers are filtered
/// during storage). This function maps block layer indices to the KV-bearing
/// cache layers, skipping non-KV layers.
///
/// - Parameters:
///   - blocks: The cache blocks to restore from, ordered by sequence position.
///   - cache: The model's per-layer cache array to restore into.
/// - Returns: The total number of tokens restored across all blocks.
@discardableResult
public func restoreLayerData(from blocks: [CacheBlock], into cache: [any KVCache]) -> Int {
    guard let firstBlock = blocks.first, let firstData = firstBlock.cacheData else { return 0 }
    let numBlockLayers = firstData.count

    // Build mapping: block layer index → cache layer index
    // Only KVCacheSimple, QuantizedKVCache, TurboQuantKVCache, and CacheList-with-KV layers are KV-bearing
    var kvCacheIndices: [Int] = []
    for (i, layer) in cache.enumerated() {
        if layer is KVCacheSimple {
            kvCacheIndices.append(i)
        } else if layer is QuantizedKVCache {
            kvCacheIndices.append(i)
        } else if layer is TurboQuantKVCache {
            kvCacheIndices.append(i)
        } else if let cacheList = layer as? CacheList {
            // Check if any sub-cache is KV-bearing
            for j in 0..<cacheList.count {
                if cacheList[j] is KVCacheSimple || cacheList[j] is QuantizedKVCache
                    || cacheList[j] is TurboQuantKVCache
                {
                    kvCacheIndices.append(i)
                    break
                }
            }
        }
    }

    // Block layers should match KV-bearing cache layers
    guard numBlockLayers == kvCacheIndices.count else { return 0 }

    for (blockLayerIdx, cacheLayerIdx) in kvCacheIndices.enumerated() {
        var keySlices: [MLXArray] = []
        var valueSlices: [MLXArray] = []

        for block in blocks {
            guard let data = block.cacheData, blockLayerIdx < data.count,
                  let kv = data[blockLayerIdx] else { continue }
            keySlices.append(kv.keys)
            valueSlices.append(kv.values)
        }

        guard !keySlices.isEmpty else { continue }

        var restoredKeys = keySlices.count == 1 ? keySlices[0] : concatenated(keySlices, axis: 2)
        var restoredValues = valueSlices.count == 1 ? valueSlices[0] : concatenated(valueSlices, axis: 2)

        // Ensure restored KV matches bfloat16 (prevents dtype mismatch from stale
        // disk cache entries created before the universal bfloat16 conversion)
        if restoredKeys.dtype == .float16 {
            restoredKeys = restoredKeys.asType(.bfloat16)
            restoredValues = restoredValues.asType(.bfloat16)
        }

        if let simple = cache[cacheLayerIdx] as? KVCacheSimple {
            simple.state = [restoredKeys, restoredValues]
        } else if let quantizedCache = cache[cacheLayerIdx] as? QuantizedKVCache {
            let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
            if let biases = qKeys.biases { stateArrays.append(biases) }
            stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
            if let biases = qValues.biases { stateArrays.append(biases) }
            quantizedCache.state = stateArrays
            quantizedCache.offset = restoredKeys.dim(2)
        } else if let tq = cache[cacheLayerIdx] as? TurboQuantKVCache {
            // Setting state transitions TQ to fill phase with the restored float KV.
            // The model will re-compress during the next generation cycle if needed.
            tq.state = [restoredKeys, restoredValues]
        } else if let cacheList = cache[cacheLayerIdx] as? CacheList {
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    simple.state = [restoredKeys, restoredValues]
                    break
                }
                if let quantizedCache = cacheList[i] as? QuantizedKVCache {
                    let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
                    if let biases = qKeys.biases { stateArrays.append(biases) }
                    stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
                    if let biases = qValues.biases { stateArrays.append(biases) }
                    quantizedCache.state = stateArrays
                    quantizedCache.offset = restoredKeys.dim(2)
                    break
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    tq.state = [restoredKeys, restoredValues]
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

// MARK: - Disk Cache KV Restoration

/// Restore KV state from disk cache arrays into model cache.
///
/// Disk arrays use keys `b{block}_l{layer}_keys` and `b{block}_l{layer}_values`.
/// Groups by layer across all blocks, concatenates along the sequence dimension, and
/// restores into the corresponding KVCacheSimple (or QuantizedKVCache / CacheList) layers.
///
/// Also handles TQ-native format: if the arrays contain a TQ marker, they are
/// deserialized via ``TQDiskSerializer`` first and the resulting standard KV pairs
/// are restored.
///
/// - Parameters:
///   - arrays: The disk cache dictionary (keyed by `b{block}_l{layer}_keys/values`
///     or TQ-native keys).
///   - cache: The model's per-layer KV cache array to restore into.
/// - Returns: The total number of tokens restored (sequence length of layer 0), or 0 on failure.
@discardableResult
public func restoreFromDiskArrays(_ arrays: [String: MLXArray], into cache: [any KVCache]) -> Int {
    // Handle TQ-native format by converting to standard kv_{i}_keys/values dict
    var kvByLayer: [Int: (keys: MLXArray, values: MLXArray)] = [:]

    if TQDiskSerializer.isTQNative(arrays) {
        let layers = TQDiskSerializer.deserialize(arrays)
        for (i, layerData) in layers.enumerated() {
            switch layerData {
            case .standard(let kv):
                kvByLayer[i] = (keys: kv.keys, values: kv.values)
            case .tq:
                // TQ-compressed layers need full TurboQuantKVCache reconstruction,
                // which is not yet supported. Skip for now.
                continue
            }
        }
    } else {
        // Standard block-layer format: b{block}_l{layer}_keys / b{block}_l{layer}_values
        // Parse all keys to discover block and layer indices
        var layerBlocks: [Int: [(blockIdx: Int, keys: MLXArray, values: MLXArray)]] = [:]

        for (key, array) in arrays {
            // Match pattern: b{N}_l{M}_keys
            guard key.hasSuffix("_keys") else { continue }
            let base = String(key.dropLast(5))  // remove "_keys"
            let parts = base.split(separator: "_")
            guard parts.count == 2,
                  parts[0].hasPrefix("b"), parts[1].hasPrefix("l"),
                  let blockIdx = Int(parts[0].dropFirst()),
                  let layerIdx = Int(parts[1].dropFirst())
            else { continue }

            let valuesKey = "b\(blockIdx)_l\(layerIdx)_values"
            guard let valuesArray = arrays[valuesKey] else { continue }

            layerBlocks[layerIdx, default: []].append(
                (blockIdx: blockIdx, keys: array, values: valuesArray))
        }

        // For each layer, sort blocks by index and concatenate along axis 2 (sequence dim)
        for (layerIdx, blocks) in layerBlocks {
            let sorted = blocks.sorted { $0.blockIdx < $1.blockIdx }
            let keySlices = sorted.map(\.keys)
            let valueSlices = sorted.map(\.values)

            let concatKeys = keySlices.count == 1
                ? keySlices[0] : concatenated(keySlices, axis: 2)
            let concatValues = valueSlices.count == 1
                ? valueSlices[0] : concatenated(valueSlices, axis: 2)

            kvByLayer[layerIdx] = (keys: concatKeys, values: concatValues)
        }
    }

    guard !kvByLayer.isEmpty else { return 0 }

    // Build mapping of KV-bearing cache layer indices (same logic as restoreLayerData)
    var kvCacheIndices: [Int] = []
    for (i, layer) in cache.enumerated() {
        if layer is KVCacheSimple {
            kvCacheIndices.append(i)
        } else if layer is QuantizedKVCache {
            kvCacheIndices.append(i)
        } else if layer is TurboQuantKVCache {
            kvCacheIndices.append(i)
        } else if let cacheList = layer as? CacheList {
            for j in 0..<cacheList.count {
                if cacheList[j] is KVCacheSimple || cacheList[j] is QuantizedKVCache
                    || cacheList[j] is TurboQuantKVCache
                {
                    kvCacheIndices.append(i)
                    break
                }
            }
        }
    }

    // Disk layer indices should map 1:1 to KV-bearing cache layers
    let sortedLayers = kvByLayer.keys.sorted()
    guard sortedLayers.count == kvCacheIndices.count else { return 0 }

    var totalTokens = 0

    for (diskLayerIdx, cacheLayerIdx) in zip(sortedLayers, kvCacheIndices) {
        guard var (restoredKeys, restoredValues) = kvByLayer[diskLayerIdx] else { continue }

        // Cast stale float16 disk entries to bfloat16
        if restoredKeys.dtype == .float16 {
            restoredKeys = restoredKeys.asType(.bfloat16)
            restoredValues = restoredValues.asType(.bfloat16)
        }

        if totalTokens == 0 {
            totalTokens = restoredKeys.dim(2)
        }

        if let simple = cache[cacheLayerIdx] as? KVCacheSimple {
            simple.state = [restoredKeys, restoredValues]
        } else if let quantizedCache = cache[cacheLayerIdx] as? QuantizedKVCache {
            let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
            var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
            if let biases = qKeys.biases { stateArrays.append(biases) }
            stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
            if let biases = qValues.biases { stateArrays.append(biases) }
            quantizedCache.state = stateArrays
            quantizedCache.offset = restoredKeys.dim(2)
        } else if let tq = cache[cacheLayerIdx] as? TurboQuantKVCache {
            tq.state = [restoredKeys, restoredValues]
        } else if let cacheList = cache[cacheLayerIdx] as? CacheList {
            for i in 0..<cacheList.count {
                if let simple = cacheList[i] as? KVCacheSimple {
                    simple.state = [restoredKeys, restoredValues]
                    break
                }
                if let quantizedCache = cacheList[i] as? QuantizedKVCache {
                    let qKeys = quantized(restoredKeys, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    let qValues = quantized(restoredValues, groupSize: quantizedCache.groupSize, bits: quantizedCache.bits)
                    var stateArrays: [MLXArray] = [qKeys.wq, qKeys.scales]
                    if let biases = qKeys.biases { stateArrays.append(biases) }
                    stateArrays.append(contentsOf: [qValues.wq, qValues.scales])
                    if let biases = qValues.biases { stateArrays.append(biases) }
                    quantizedCache.state = stateArrays
                    quantizedCache.offset = restoredKeys.dim(2)
                    break
                }
                if let tq = cacheList[i] as? TurboQuantKVCache {
                    tq.state = [restoredKeys, restoredValues]
                    break
                }
            }
        }
    }

    return totalTokens
}
