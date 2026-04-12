// Copyright © 2025 Osaurus & JANG. All rights reserved.
// TQ-native disk serialization for 26x compressed cache storage.

import Foundation
import MLX

/// Serializes TurboQuantKVCache compressed data into a flat `[String: MLXArray]` dictionary
/// for disk persistence via `DiskCache`. This stores the COMPRESSED data directly —
/// 26x smaller than storing decoded float16 arrays.
///
/// ## Key Format
///
/// Per-layer keys use the pattern `tq_{layerIdx}_{component}`:
///
///     ck_indices      — EncodedKeys.indicesPacked (uint32)
///     ck_qjl          — EncodedKeys.qjlPacked (uint32)
///     ck_res_norms    — EncodedKeys.residualNorms (float16)
///     ck_vec_norms    — EncodedKeys.vectorNorms (float16)
///     ck_sink         — EncodedKeys.sinkData (float16, optional)
///     cv_indices      — EncodedValues.indicesPacked (uint32)
///     cv_norms        — EncodedValues.vectorNorms (float16)
///     cv_sink         — EncodedValues.sinkData (float16, optional)
///
/// Standard (non-TQ) layers use `kv_{layerIdx}_keys` / `kv_{layerIdx}_values`.
///
/// ## Metadata Keys
///
///     __tq_native_marker__    — presence indicates TQ-native format
///     __tq_{i}_ck_shape__     — original compressed key shape as int32 array
///     __tq_{i}_ck_index_bits__— key index bits as int32 scalar
///     __tq_{i}_ck_seed__      — key encoding seed as int32 scalar
///     __tq_{i}_cv_shape__     — original compressed value shape as int32 array
///     __tq_{i}_cv_index_bits__— value index bits as int32 scalar
///     __tq_{i}_cv_seed__      — value encoding seed as int32 scalar
///
/// ## Usage
///
/// ```swift
/// // Store
/// let arrays = TQDiskSerializer.serialize(cache: model.cache)
/// diskCache.store(tokens: tokenIds, arrays: arrays)
///
/// // Load
/// if let loaded = diskCache.fetch(tokens: tokenIds),
///    TQDiskSerializer.isTQNative(loaded) {
///     let components = TQDiskSerializer.deserialize(loaded)
///     // Reconstruct TurboQuantKVCache from components...
/// }
/// ```
public enum TQDiskSerializer {

    // MARK: - Detection

    /// Check if a cache layer is a `TurboQuantKVCache` in compressed phase.
    ///
    /// Returns `false` for TQ caches still in fill phase (they behave like
    /// `KVCacheSimple` and should be stored as standard float16).
    public static func isTQCompressed(_ cache: any KVCache) -> Bool {
        guard let tq = cache as? TurboQuantKVCache else { return false }
        return tq.phase == .compressed
    }

    /// Check if a loaded dictionary contains TQ-native compressed data.
    public static func isTQNative(_ arrays: [String: MLXArray]) -> Bool {
        arrays.keys.contains("__tq_native_marker__")
    }

    // MARK: - Serialize

    /// Serialize cache layers into a flat `[String: MLXArray]` dictionary
    /// suitable for safetensors persistence.
    ///
    /// - `TurboQuantKVCache` layers in compressed phase are stored as their
    ///   compact encoded arrays (indices, QJL signs, norms).
    /// - `TurboQuantKVCache` layers still in fill phase and `KVCacheSimple`
    ///   layers are stored as standard float16 key/value pairs.
    /// - Other cache types (e.g., `QuantizedKVCache`) are skipped.
    ///
    /// - Parameter cache: Array of per-layer KV caches from the model.
    /// - Returns: Flat dictionary ready for `DiskCache.store()`.
    public static func serialize(cache: [any KVCache]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]

        for (i, layer) in cache.enumerated() {
            if let tq = layer as? TurboQuantKVCache, tq.phase == .compressed {
                serializeTQLayer(tq, index: i, into: &result)
            } else {
                // Fall back to standard float16 state for KVCacheSimple,
                // TQ in fill phase, or any BaseKVCache subclass.
                let state = layer.state
                if state.count >= 2 {
                    result["kv_\(i)_keys"] = state[0]
                    result["kv_\(i)_values"] = state[1]
                }
            }
        }

        // Marker so loaders can detect TQ-native format
        result["__tq_native_marker__"] = MLXArray([Int32(1)])

        return result
    }

    /// Serialize a single TQ-compressed layer's encoded data.
    private static func serializeTQLayer(
        _ tq: TurboQuantKVCache,
        index i: Int,
        into result: inout [String: MLXArray]
    ) {
        // --- Compressed keys ---
        if let ck = tq.compressedKeys {
            result["tq_\(i)_ck_indices"] = ck.indicesPacked
            result["tq_\(i)_ck_qjl"] = ck.qjlPacked
            result["tq_\(i)_ck_res_norms"] = ck.residualNorms
            result["tq_\(i)_ck_vec_norms"] = ck.vectorNorms

            if let sink = ck.sinkData {
                result["tq_\(i)_ck_sink"] = sink
            }

            // Store shape, indexBits, seed as metadata arrays
            result["__tq_\(i)_ck_shape__"] = MLXArray(ck.shape.map { Int32($0) })
            result["__tq_\(i)_ck_index_bits__"] = MLXArray(Int32(ck.indexBits))
            result["__tq_\(i)_ck_seed__"] = MLXArray(Int32(ck.seed))
        }

        // --- Compressed values ---
        if let cv = tq.compressedValues {
            result["tq_\(i)_cv_indices"] = cv.indicesPacked
            result["tq_\(i)_cv_norms"] = cv.vectorNorms

            if let sink = cv.sinkData {
                result["tq_\(i)_cv_sink"] = sink
            }

            // Store shape, indexBits, seed as metadata arrays
            result["__tq_\(i)_cv_shape__"] = MLXArray(cv.shape.map { Int32($0) })
            result["__tq_\(i)_cv_index_bits__"] = MLXArray(Int32(cv.indexBits))
            result["__tq_\(i)_cv_seed__"] = MLXArray(Int32(cv.seed))
        }
    }

    // MARK: - Deserialize

    /// Parsed TQ components for a single attention layer.
    public struct TQLayerComponents {
        /// Reconstructed `EncodedKeys` from the serialized arrays.
        public let encodedKeys: EncodedKeys
        /// Reconstructed `EncodedValues` from the serialized arrays.
        public let encodedValues: EncodedValues
    }

    /// Standard (non-TQ) KV data for a single attention layer.
    public struct KVLayerComponents {
        public let keys: MLXArray
        public let values: MLXArray
    }

    /// Result of deserializing a TQ-native cache dictionary.
    ///
    /// Each layer is either TQ-compressed or standard float16 KV.
    public enum LayerData {
        case tq(TQLayerComponents)
        case standard(KVLayerComponents)
    }

    /// Deserialize a TQ-native dictionary into per-layer components.
    ///
    /// Returns an ordered array of `LayerData`, one per layer found in the
    /// dictionary. Layer indices are inferred from the key naming convention.
    ///
    /// - Parameter arrays: Dictionary loaded from safetensors via `DiskCache.fetch()`.
    /// - Returns: Per-layer components, ordered by layer index.
    public static func deserialize(_ arrays: [String: MLXArray]) -> [LayerData] {
        // Discover all layer indices
        var tqIndices = Set<Int>()
        var kvIndices = Set<Int>()

        for key in arrays.keys {
            if key.hasPrefix("tq_"), let idx = parseLayerIndex(from: key, prefix: "tq_") {
                tqIndices.insert(idx)
            } else if key.hasPrefix("kv_"), let idx = parseLayerIndex(from: key, prefix: "kv_") {
                kvIndices.insert(idx)
            }
        }

        let allIndices = tqIndices.union(kvIndices).sorted()
        var layers: [LayerData] = []

        for i in allIndices {
            if tqIndices.contains(i) {
                if let components = deserializeTQLayer(index: i, from: arrays) {
                    layers.append(.tq(components))
                }
            } else if kvIndices.contains(i) {
                if let keys = arrays["kv_\(i)_keys"],
                   let values = arrays["kv_\(i)_values"]
                {
                    layers.append(.standard(KVLayerComponents(keys: keys, values: values)))
                }
            }
        }

        return layers
    }

    /// Deserialize a single TQ layer's encoded data from the flat dictionary.
    private static func deserializeTQLayer(
        index i: Int,
        from arrays: [String: MLXArray]
    ) -> TQLayerComponents? {
        // --- Keys ---
        guard let ckIndices = arrays["tq_\(i)_ck_indices"],
              let ckQjl = arrays["tq_\(i)_ck_qjl"],
              let ckResNorms = arrays["tq_\(i)_ck_res_norms"],
              let ckVecNorms = arrays["tq_\(i)_ck_vec_norms"],
              let ckShapeArr = arrays["__tq_\(i)_ck_shape__"],
              let ckIndexBitsArr = arrays["__tq_\(i)_ck_index_bits__"],
              let ckSeedArr = arrays["__tq_\(i)_ck_seed__"]
        else {
            return nil
        }

        // --- Values ---
        guard let cvIndices = arrays["tq_\(i)_cv_indices"],
              let cvNorms = arrays["tq_\(i)_cv_norms"],
              let cvShapeArr = arrays["__tq_\(i)_cv_shape__"],
              let cvIndexBitsArr = arrays["__tq_\(i)_cv_index_bits__"],
              let cvSeedArr = arrays["__tq_\(i)_cv_seed__"]
        else {
            return nil
        }

        // Extract scalar/array metadata
        let ckShape = ckShapeArr.asArray(Int32.self).map { Int($0) }
        let ckIndexBits = Int(ckIndexBitsArr.item(Int32.self))
        let ckSeed = Int(ckSeedArr.item(Int32.self))

        let cvShape = cvShapeArr.asArray(Int32.self).map { Int($0) }
        let cvIndexBits = Int(cvIndexBitsArr.item(Int32.self))
        let cvSeed = Int(cvSeedArr.item(Int32.self))

        let ckSink = arrays["tq_\(i)_ck_sink"]
        let cvSink = arrays["tq_\(i)_cv_sink"]

        let encodedKeys = EncodedKeys(
            indicesPacked: ckIndices,
            qjlPacked: ckQjl,
            residualNorms: ckResNorms,
            vectorNorms: ckVecNorms,
            shape: ckShape,
            indexBits: ckIndexBits,
            seed: ckSeed,
            sinkData: ckSink
        )

        let encodedValues = EncodedValues(
            indicesPacked: cvIndices,
            vectorNorms: cvNorms,
            shape: cvShape,
            indexBits: cvIndexBits,
            seed: cvSeed,
            sinkData: cvSink
        )

        return TQLayerComponents(encodedKeys: encodedKeys, encodedValues: encodedValues)
    }

    // MARK: - Helpers

    /// Extract the layer index from a key like "tq_42_ck_indices" or "kv_7_keys".
    private static func parseLayerIndex(from key: String, prefix: String) -> Int? {
        let remainder = key.dropFirst(prefix.count)
        guard let underscoreIdx = remainder.firstIndex(of: "_") else { return nil }
        let indexStr = String(remainder[remainder.startIndex..<underscoreIdx])
        return Int(indexStr)
    }
}
