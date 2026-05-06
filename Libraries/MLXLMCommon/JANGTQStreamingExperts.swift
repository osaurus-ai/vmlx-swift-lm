import Foundation
import MLX
import MLXNN

public enum JANGTQStreamingExperts {
    public static var isEnabled: Bool {
        let env = ProcessInfo.processInfo.environment
        guard let raw = env["MLXPRESS_STREAMING_EXPERTS"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }

    public static func hasStreamableExperts(in modelDirectory: URL) -> Bool {
        let resolved = modelDirectory.resolvingSymlinksInPath()
        guard let index = try? JANGTQStreamingExpertIndex.build(modelDirectory: resolved) else {
            return false
        }
        return !index.layers.isEmpty
    }
}

private func mlXPressStreamingTokenChunkSize() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_TOKEN_CHUNK_SIZE"]
        ?? env["MLXPRESS_STREAMING_TOKEN_CHUNK"]
        ?? "16"
    return max(1, Int(raw) ?? 16)
}

private func mlXPressStreamingReduceTokenChunkSize() -> Int {
    let env = ProcessInfo.processInfo.environment
    let raw =
        env["MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK_SIZE"]
        ?? env["MLXPRESS_STREAMING_REDUCE_TOKEN_CHUNK"]
        ?? "1"
    return max(1, Int(raw) ?? 1)
}

private enum StreamingProjection: String, CaseIterable {
    case gate = "gate_proj"
    case up = "up_proj"
    case down = "down_proj"
}

private enum StreamingSuffix: String, CaseIterable {
    case packed = "tq_packed"
    case norms = "tq_norms"
}

private struct StreamingTensorRef {
    var fileURL: URL
    var offset: UInt64
    var byteCount: Int
    var dtype: String
    var shape: [Int]
}

private struct StreamingExpertRef {
    var tensors: [StreamingProjection: [StreamingSuffix: StreamingTensorRef]]
}

private struct StreamingLayerRef {
    var experts: [Int: StreamingExpertRef]
}

private final class JANGTQStreamingExpertIndex: @unchecked Sendable {
    let modelDirectory: URL
    let layers: [Int: StreamingLayerRef]

    init(modelDirectory: URL, layers: [Int: StreamingLayerRef]) {
        self.modelDirectory = modelDirectory
        self.layers = layers
    }

    static func build(modelDirectory: URL) throws -> JANGTQStreamingExpertIndex {
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "safetensors" }
            .filter { $0.lastPathComponent != "jangtq_runtime.safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var layers: [Int: StreamingLayerRef] = [:]
        for file in files {
            let header = try readSafetensorsHeader(file)
            for (key, value) in header.tensors {
                guard let match = matchPerExpertTQKey(key),
                      let dtype = value["dtype"] as? String,
                      let shape = value["shape"] as? [Int],
                      let offsets = value["data_offsets"] as? [UInt64],
                      offsets.count == 2,
                      offsets[1] >= offsets[0]
                else { continue }

                var layer = layers[match.layer] ?? StreamingLayerRef(experts: [:])
                var expert = layer.experts[match.expert] ?? StreamingExpertRef(tensors: [:])
                var projection = expert.tensors[match.projection] ?? [:]
                projection[match.suffix] = StreamingTensorRef(
                    fileURL: file,
                    offset: header.dataBase + offsets[0],
                    byteCount: Int(offsets[1] - offsets[0]),
                    dtype: dtype,
                    shape: shape)
                expert.tensors[match.projection] = projection
                layer.experts[match.expert] = expert
                layers[match.layer] = layer
            }
        }
        return JANGTQStreamingExpertIndex(modelDirectory: modelDirectory, layers: layers)
    }

    private struct HeaderRead {
        var dataBase: UInt64
        var tensors: [String: [String: Any]]
    }

    private static func readSafetensorsHeader(_ url: URL) throws -> HeaderRead {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let prefix = try handle.read(upToCount: 8) ?? Data()
        guard prefix.count == 8 else {
            throw NSError(domain: "JANGTQStreamingExperts", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "short safetensors header: \(url.path)"])
        }
        let headerLength = prefix.withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerData = try handle.read(upToCount: Int(headerLength)) ?? Data()
        guard headerData.count == Int(headerLength) else {
            throw NSError(domain: "JANGTQStreamingExperts", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "truncated safetensors header: \(url.path)"])
        }
        let json = try JSONSerialization.jsonObject(with: headerData)
        guard let dict = json as? [String: Any] else {
            throw NSError(domain: "JANGTQStreamingExperts", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "invalid safetensors header: \(url.path)"])
        }
        var tensors: [String: [String: Any]] = [:]
        for (key, value) in dict where key != "__metadata__" {
            guard var entry = value as? [String: Any] else { continue }
            if let rawShape = entry["shape"] as? [NSNumber] {
                entry["shape"] = rawShape.map(\.intValue)
            }
            if let rawOffsets = entry["data_offsets"] as? [NSNumber] {
                entry["data_offsets"] = rawOffsets.map(\.uint64Value)
            }
            tensors[key] = entry
        }
        return HeaderRead(dataBase: 8 + headerLength, tensors: tensors)
    }

    private struct KeyMatch {
        var layer: Int
        var expert: Int
        var projection: StreamingProjection
        var suffix: StreamingSuffix
    }

    private static func matchPerExpertTQKey(_ key: String) -> KeyMatch? {
        let patterns: [(String, [String: StreamingProjection])] = [
            (
                #"^(?:language_model\.)?model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
                [
                    "gate_proj": .gate,
                    "up_proj": .up,
                    "down_proj": .down,
                ]
            ),
            (
                #"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
                [
                    "w1": .gate,
                    "w2": .down,
                    "w3": .up,
                ]
            ),
            (
                #"^(?:language_model\.)?model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(w1|w2|w3)\.(tq_packed|tq_norms)$"#,
                [
                    "w1": .gate,
                    "w2": .down,
                    "w3": .up,
                ]
            ),
            (
                #"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(tq_packed|tq_norms)$"#,
                [
                    "up_proj": .up,
                    "down_proj": .down,
                ]
            ),
        ]
        for (pattern, projectionMap) in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let nsRange = NSRange(key.startIndex..<key.endIndex, in: key)
            guard let match = regex.firstMatch(in: key, range: nsRange),
                  match.numberOfRanges == 5,
                  let layerRange = Range(match.range(at: 1), in: key),
                  let expertRange = Range(match.range(at: 2), in: key),
                  let projectionRange = Range(match.range(at: 3), in: key),
                  let suffixRange = Range(match.range(at: 4), in: key),
                  let layer = Int(key[layerRange]),
                  let expert = Int(key[expertRange]),
                  let projection = projectionMap[String(key[projectionRange])],
                  let suffix = StreamingSuffix(rawValue: String(key[suffixRange]))
            else { continue }
            return KeyMatch(layer: layer, expert: expert, projection: projection, suffix: suffix)
        }
        return nil
    }
}

public final class StreamingTurboQuantSwitchReLUSquaredMLP: Module {
    @ModuleInfo(key: "fc1") public var fc1: TurboQuantSwitchLinear
    @ModuleInfo(key: "fc2") public var fc2: TurboQuantSwitchLinear

    private let layerIdx: Int
    private let inputDims: Int
    private let hiddenDims: Int
    private let evaluateEachLayer: Bool
    private let tokenChunkSize: Int

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bits: Int = 2,
        seed: Int = 42,
        layerIdx: Int
    ) {
        self.layerIdx = layerIdx
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.evaluateEachLayer =
            ProcessInfo.processInfo.environment["MLXPRESS_STREAMING_EVAL_EACH_LAYER"] != "0"
        self.tokenChunkSize = mlXPressStreamingTokenChunkSize()
        self._fc1.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: inputDims,
            outFeatures: hiddenDims,
            numExperts: numExperts,
            bits: bits,
            seed: seed)
        self._fc2.wrappedValue = TurboQuantSwitchLinear(
            inFeatures: hiddenDims,
            outFeatures: inputDims,
            numExperts: numExperts,
            bits: bits,
            seed: seed)
        super.init()
        _ = JANGTQStreamingExpertStore.shared.index()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let totalTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([totalTokens, inputDims])
        let indicesFlat = indices.reshaped([totalTokens, kSlots])
        let chunkSize = max(1, tokenChunkSize)
        if totalTokens <= chunkSize {
            let chunk = callChunk(
                xFlat: xFlat,
                indicesFlat: indicesFlat,
                tokenCount: totalTokens,
                kSlots: kSlots)
            var outShape = Array(indices.shape)
            outShape.append(inputDims)
            return chunk.reshaped(outShape)
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((totalTokens + chunkSize - 1) / chunkSize)
        var start = 0
        while start < totalTokens {
            let end = min(start + chunkSize, totalTokens)
            chunks.append(callChunk(
                xFlat: xFlat[start ..< end, 0...],
                indicesFlat: indicesFlat[start ..< end, 0...],
                tokenCount: end - start,
                kSlots: kSlots))
            start = end
        }
        let joined = concatenated(chunks, axis: 0)
        var outShape = Array(indices.shape)
        outShape.append(inputDims)
        return joined.reshaped(outShape)
    }

    private func callChunk(
        xFlat: MLXArray,
        indicesFlat: MLXArray,
        tokenCount: Int,
        kSlots: Int
    ) -> MLXArray {
        let indexValues = indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let uniqueExperts = Array(Set(indexValues)).sorted()
        guard !uniqueExperts.isEmpty else {
            fatalError("[MLXPressStreaming] empty routed expert set in layer \(layerIdx)")
        }

        func stack(_ projection: StreamingProjection, _ suffix: StreamingSuffix) -> MLXArray? {
            var arrays: [MLXArray] = []
            arrays.reserveCapacity(uniqueExperts.count)
            for expert in uniqueExperts {
                guard let array = JANGTQStreamingExpertStore.shared.load(
                    layerIdx: layerIdx,
                    expertIdx: expert,
                    projection: projection,
                    suffix: suffix)
                else { return nil }
                arrays.append(array)
            }
            return MLX.stacked(arrays, axis: 0)
        }

        guard let signsIn = JANGTQRuntimeCache.shared.signs(inFeatures: inputDims, seed: fc1.mxtqSeed),
              let signsInter = JANGTQRuntimeCache.shared.signs(inFeatures: hiddenDims, seed: fc2.mxtqSeed),
              let cbIn = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: fc1.bits),
              let cbInter = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: fc2.bits)
        else {
            fatalError("[MLXPressStreaming] missing active Nemotron JANGTQ tensors for layer \(layerIdx)")
        }

        var remap: [Int: Int32] = [:]
        for (local, expert) in uniqueExperts.enumerated() {
            remap[expert] = Int32(local)
        }
        let localIndices = indexValues.map { remap[$0] ?? 0 }
        let rhsIndices = MLXArray(localIndices, indicesFlat.shape).asType(.uint32).reshaped([-1])

        guard let fc1Packed = stack(.up, .packed),
              let fc1Norms = stack(.up, .norms)
        else {
            fatalError("[MLXPressStreaming] missing active Nemotron JANGTQ fc1 tensors for layer \(layerIdx)")
        }
        let xRot1 = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)
        var hidden = JANGTQKernels.gatherTQTopK(
            xRot: xRot1,
            packed: fc1Packed,
            norms: fc1Norms,
            codebook: cbIn,
            rhsIndices: rhsIndices,
            batchTokens: tokenCount,
            K: kSlots,
            inFeatures: inputDims,
            outFeatures: hiddenDims,
            bits: fc1.bits)
        let relu = MLX.maximum(hidden, MLXArray(0, dtype: hidden.dtype))
        hidden = relu * relu
        if evaluateEachLayer {
            MLX.eval(hidden)
            MLX.Memory.clearCache()
        }

        guard let fc2Packed = stack(.down, .packed),
              let fc2Norms = stack(.down, .norms)
        else {
            fatalError("[MLXPressStreaming] missing active Nemotron JANGTQ fc2 tensors for layer \(layerIdx)")
        }

        let xRot2 = JANGTQKernels.hadamardRotate(hidden, signs: signsInter, dim: hiddenDims)
        let out = JANGTQKernels.gatherTQ(
            xRot: xRot2,
            packed: fc2Packed,
            norms: fc2Norms,
            codebook: cbInter,
            rhsIndices: rhsIndices,
            nRows: tokenCount * kSlots,
            inFeatures: hiddenDims,
            outFeatures: inputDims,
            bits: fc2.bits)

        let shaped = out.reshaped([tokenCount, kSlots, inputDims]).asType(xFlat.dtype)
        if evaluateEachLayer {
            MLX.eval(shaped)
            MLX.Memory.clearCache()
        }
        return shaped
    }
}

private final class JANGTQStreamingExpertStore: @unchecked Sendable {
    static let shared = JANGTQStreamingExpertStore()
    private let lock = NSLock()
    private var cachedIndex: JANGTQStreamingExpertIndex?

    func index() -> JANGTQStreamingExpertIndex? {
        lock.lock()
        if let cachedIndex {
            lock.unlock()
            return cachedIndex
        }
        lock.unlock()

        guard let path = ProcessInfo.processInfo.environment["MLXPRESS_MODEL_DIR"],
              !path.isEmpty
        else { return nil }

        let modelDirectory = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        guard let built = try? JANGTQStreamingExpertIndex.build(modelDirectory: modelDirectory) else {
            return nil
        }
        lock.lock()
        cachedIndex = built
        lock.unlock()
        let layers = built.layers.count
        let experts = built.layers.values.map { $0.experts.count }.max() ?? 0
        FileHandle.standardError.write(Data(
            "[MLXPressStreaming] indexed active-expert JANGTQ tensors layers=\(layers) experts=\(experts)\n".utf8))
        return built
    }

    func load(
        layerIdx: Int,
        expertIdx: Int,
        projection: StreamingProjection,
        suffix: StreamingSuffix
    ) -> MLXArray? {
        guard let ref = index()?.layers[layerIdx]?.experts[expertIdx]?.tensors[projection]?[suffix],
              let data = readBytes(from: ref.fileURL, offset: ref.offset, count: ref.byteCount)
        else { return nil }
        return makeArray(data: data, shape: ref.shape, dtype: ref.dtype)
    }

    private func readBytes(from url: URL, offset: UInt64, count: Int) -> Data? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        do {
            try handle.seek(toOffset: offset)
            let data = try handle.read(upToCount: count) ?? Data()
            return data.count == count ? data : nil
        } catch {
            return nil
        }
    }

    private func makeArray(data: Data, shape: [Int], dtype: String) -> MLXArray? {
        switch dtype {
        case "U32":
            return data.withUnsafeBytes {
                MLXArray(Array($0.bindMemory(to: UInt32.self)), shape)
            }
        case "I32":
            return data.withUnsafeBytes {
                MLXArray(Array($0.bindMemory(to: Int32.self)), shape)
            }
        case "F16":
            return data.withUnsafeBytes {
                MLXArray(Array($0.bindMemory(to: Float16.self)), shape)
            }
        case "BF16":
            return data.withUnsafeBytes {
                MLXArray(Array($0.bindMemory(to: Float16.self)), shape).asType(.bfloat16)
            }
        case "F32":
            return data.withUnsafeBytes {
                MLXArray(Array($0.bindMemory(to: Float.self)), shape)
            }
        default:
            return nil
        }
    }
}

public final class StreamingTurboQuantSwitchGLU: TurboQuantSwitchGLU {
    private let layerIdx: Int
    private let evaluateEachLayer: Bool
    private let tokenChunkSize: Int

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        gateUpBits: Int,
        downBits: Int,
        seed: Int = 42,
        swigluLimit: Float = 0.0,
        layerIdx: Int
    ) {
        self.layerIdx = layerIdx
        self.evaluateEachLayer =
            ProcessInfo.processInfo.environment["MLXPRESS_STREAMING_EVAL_EACH_LAYER"] != "0"
        self.tokenChunkSize = mlXPressStreamingTokenChunkSize()
        super.init(
            inputDims: inputDims,
            hiddenDims: hiddenDims,
            numExperts: numExperts,
            gateUpBits: gateUpBits,
            downBits: downBits,
            seed: seed,
            swigluLimit: swigluLimit)
        _ = JANGTQStreamingExpertStore.shared.index()
    }

    public override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let batchTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([batchTokens, inputDims])
        let indicesFlat = indices.reshaped([batchTokens, kSlots])
        let chunkSize = max(1, tokenChunkSize)
        if batchTokens <= chunkSize {
            let chunk = callChunk(
                xFlat: xFlat,
                indicesFlat: indicesFlat,
                tokenCount: batchTokens,
                kSlots: kSlots)
            var outShape = indices.shape
            outShape.append(inputDims)
            return chunk.reshaped(outShape)
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((batchTokens + chunkSize - 1) / chunkSize)
        var start = 0
        while start < batchTokens {
            let end = min(start + chunkSize, batchTokens)
            chunks.append(callChunk(
                xFlat: xFlat[start ..< end, 0...],
                indicesFlat: indicesFlat[start ..< end, 0...],
                tokenCount: end - start,
                kSlots: kSlots))
            start = end
        }
        let joined = concatenated(chunks, axis: 0)
        var outShape = indices.shape
        outShape.append(inputDims)
        return joined.reshaped(outShape)
    }

    public func reduced(_ x: MLXArray, indices: MLXArray, scores: MLXArray) -> MLXArray {
        let batchTokens = x.size / inputDims
        let kSlots = indices.dim(-1)
        let xFlat = x.reshaped([batchTokens, inputDims])
        let indicesFlat = indices.reshaped([batchTokens, kSlots])
        let scoresFlat = scores.reshaped([batchTokens, kSlots])
        let chunkSize = mlXPressStreamingReduceTokenChunkSize()

        func reduceChunk(start: Int, end: Int) -> MLXArray {
            let tokenCount = end - start
            let xChunk = xFlat[start ..< end, 0...]
            var accumulated: MLXArray?
            for slot in 0 ..< kSlots {
                let expertOutput = callChunk(
                    xFlat: xChunk,
                    indicesFlat: indicesFlat[start ..< end, slot ..< (slot + 1)],
                    tokenCount: tokenCount,
                    kSlots: 1)
                let scoreChunk = scoresFlat[start ..< end, slot ..< (slot + 1)]
                    .asType(expertOutput.dtype)
                let reducedSlot = (expertOutput * scoreChunk[.ellipsis, .newAxis])
                    .sum(axis: -2)
                let next = accumulated.map { $0 + reducedSlot } ?? reducedSlot
                if evaluateEachLayer {
                    MLX.eval(next)
                    MLX.Memory.clearCache()
                }
                accumulated = next
            }
            return accumulated ?? MLXArray.zeros([tokenCount, inputDims])
        }

        if batchTokens <= chunkSize {
            return reduceChunk(start: 0, end: batchTokens).reshaped(x.shape)
        }

        var chunks: [MLXArray] = []
        chunks.reserveCapacity((batchTokens + chunkSize - 1) / chunkSize)
        var start = 0
        while start < batchTokens {
            let end = min(start + chunkSize, batchTokens)
            chunks.append(reduceChunk(start: start, end: end))
            start = end
        }
        return concatenated(chunks, axis: 0).reshaped(x.shape)
    }

    private func callChunk(
        xFlat: MLXArray,
        indicesFlat: MLXArray,
        tokenCount: Int,
        kSlots: Int
    ) -> MLXArray {
        let indexValues = indicesFlat.reshaped([-1]).asArray(Int32.self).map(Int.init)
        let uniqueExperts = Array(Set(indexValues)).sorted()
        guard !uniqueExperts.isEmpty else {
            fatalError("[MLXPressStreaming] empty routed expert set in layer \(layerIdx)")
        }

        func stack(_ projection: StreamingProjection, _ suffix: StreamingSuffix) -> MLXArray? {
            var arrays: [MLXArray] = []
            arrays.reserveCapacity(uniqueExperts.count)
            for expert in uniqueExperts {
                guard let array = JANGTQStreamingExpertStore.shared.load(
                    layerIdx: layerIdx,
                    expertIdx: expert,
                    projection: projection,
                    suffix: suffix)
                else { return nil }
                arrays.append(array)
            }
            return MLX.stacked(arrays, axis: 0)
        }

        guard let signsIn = JANGTQRuntimeCache.shared.signs(inFeatures: inputDims, seed: mxtqSeed),
              let signsDn = JANGTQRuntimeCache.shared.signs(inFeatures: hiddenDims, seed: mxtqSeed),
              let cbGate = JANGTQRuntimeCache.shared.codebook(inFeatures: inputDims, bits: gateUpBits),
              let cbDown = JANGTQRuntimeCache.shared.codebook(inFeatures: hiddenDims, bits: downBits)
        else {
            fatalError("[MLXPressStreaming] missing active JANGTQ tensors for layer \(layerIdx)")
        }

        var remap: [Int: Int32] = [:]
        for (local, expert) in uniqueExperts.enumerated() {
            remap[expert] = Int32(local)
        }
        let localIndices = indexValues.map { remap[$0] ?? 0 }
        let rhsIndices = MLXArray(localIndices, indicesFlat.shape).asType(.uint32).reshaped([-1])

        let xAct = {
            guard let gatePacked = stack(.gate, .packed),
                  let gateNorms = stack(.gate, .norms),
                  let upPacked = stack(.up, .packed),
                  let upNorms = stack(.up, .norms)
            else {
                fatalError("[MLXPressStreaming] missing active JANGTQ gate/up tensors for layer \(layerIdx)")
            }
            let xRot = JANGTQKernels.hadamardRotate(xFlat, signs: signsIn, dim: inputDims)
            let xAct = JANGTQKernels.fusedGateUpSwiGLU(
                xRot: xRot,
                packedGate: gatePacked, normsGate: gateNorms,
                packedUp: upPacked, normsUp: upNorms,
                codebook: cbGate, rhsIndices: rhsIndices,
                batchTokens: tokenCount, K: kSlots,
                inFeatures: inputDims, outFeatures: hiddenDims,
                bits: gateUpBits,
                swigluLimit: swigluLimit)
            if evaluateEachLayer {
                MLX.eval(xAct)
                MLX.Memory.clearCache()
            }
            return xAct
        }()

        guard let downPacked = stack(.down, .packed),
              let downNorms = stack(.down, .norms)
        else {
            fatalError("[MLXPressStreaming] missing active JANGTQ down tensors for layer \(layerIdx)")
        }

        let xActRot = JANGTQKernels.hadamardRotate(xAct, signs: signsDn, dim: hiddenDims)
        let y = JANGTQKernels.gatherTQ(
            xRot: xActRot,
            packed: downPacked, norms: downNorms,
            codebook: cbDown, rhsIndices: rhsIndices,
            nRows: tokenCount * kSlots,
            inFeatures: hiddenDims, outFeatures: inputDims,
            bits: downBits)
        let out = y.reshaped([tokenCount, kSlots, inputDims]).asType(xFlat.dtype)
        if evaluateEachLayer {
            MLX.eval(out)
            MLX.Memory.clearCache()
        }
        return out
    }
}
