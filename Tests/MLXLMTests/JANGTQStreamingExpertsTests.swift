import Foundation
import Testing
@testable import MLXLMCommon

@Suite("JANGTQ streaming expert index")
struct JANGTQStreamingExpertsTests {
    @Test("prestacked switch_mlp tensors are streamable")
    func prestackedSwitchMLPTensorsAreStreamable() throws {
        let bundle = try Self.makeBundle(name: "prestacked")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.up_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.up_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.down_proj.tq_packed",
                    dtype: "U32", shape: [2, 2, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.switch_mlp.down_proj.tq_norms",
                    dtype: "F16", shape: [2, 2]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    @Test("Zaya zaya_block switch_mlp tensors are streamable")
    func zayaSwitchMLPTensorsAreStreamable() throws {
        let bundle = try Self.makeBundle(name: "zaya")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.gate_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.gate_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.up_proj.tq_packed",
                    dtype: "U32", shape: [2, 3, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.up_proj.tq_norms",
                    dtype: "F16", shape: [2, 3]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.down_proj.tq_packed",
                    dtype: "U32", shape: [2, 2, 1]),
                TensorSpec(
                    name: "model.layers.1.zaya_block.experts.switch_mlp.down_proj.tq_norms",
                    dtype: "F16", shape: [2, 2]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    @Test("legacy per-expert tensors remain streamable")
    func perExpertTensorsRemainStreamable() throws {
        let bundle = try Self.makeBundle(name: "per-expert")
        defer { try? FileManager.default.removeItem(at: bundle) }

        try Self.writeSafetensors(
            at: bundle.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: [
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.tq_packed",
                    dtype: "U32", shape: [3, 1]),
                TensorSpec(
                    name: "model.layers.0.block_sparse_moe.experts.0.w1.tq_norms",
                    dtype: "F16", shape: [3]),
            ])

        #expect(JANGTQStreamingExperts.hasStreamableExperts(in: bundle))
    }

    private struct TensorSpec {
        var name: String
        var dtype: String
        var shape: [Int]
    }

    private static func makeBundle(name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangtq-streaming-\(name)-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func writeSafetensors(at url: URL, tensors: [TensorSpec]) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for tensor in tensors {
            let byteCount = tensor.shape.reduce(1, *) * byteWidth(dtype: tensor.dtype)
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, offset + byteCount],
            ]
            offset += byteCount
        }
        let headerJSON = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var data = Data()
        var headerSize = UInt64(headerJSON.count)
        withUnsafeBytes(of: &headerSize) { data.append(contentsOf: $0) }
        data.append(headerJSON)
        data.append(Data(count: offset))
        try data.write(to: url)
    }

    private static func byteWidth(dtype: String) -> Int {
        switch dtype {
        case "F16", "BF16", "I16", "U16": return 2
        case "F32", "I32", "U32": return 4
        case "F64", "I64", "U64": return 8
        default: return 1
        }
    }
}
