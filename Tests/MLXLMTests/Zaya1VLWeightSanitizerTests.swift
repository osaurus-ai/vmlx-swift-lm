import Foundation
import MLX
@testable import MLXVLM
import Testing

@Suite("ZAYA1-VL weight sanitizer")
struct Zaya1VLWeightSanitizerTests {
    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    private static let repoRoot: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()

    private static func bundlePath(_ name: String) -> String {
        let roots = [
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/JANGQ").path,
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/Osaurus").path,
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("models/Zyphra").path,
        ]
        for root in roots {
            let path = URL(fileURLWithPath: root).appendingPathComponent(name).path
            if FileManager.default.fileExists(atPath: path) {
                return path
            }
        }
        return URL(fileURLWithPath: roots[0]).appendingPathComponent(name).path
    }

    @Test("Sanitizer source pins every required native-loading rewrite")
    func sanitizerSourcePinsRequiredRewrites() throws {
        let source = try String(
            contentsOf: Self.repoRoot
                .appendingPathComponent("Libraries")
                .appendingPathComponent("MLXVLM")
                .appendingPathComponent("Models")
                .appendingPathComponent("Zaya1VL.swift"),
            encoding: .utf8)

        #expect(source.contains("public enum Zaya1VLWeightSanitizer"))
        #expect(source.contains("language_model.model."))
        #expect(source.contains("language_model.lm_head."))
        #expect(source.contains("model.layers.\\(layer).zaya_block.experts.switch_mlp."))
        #expect(source.contains("language_model.model.layers.\\(layer).mlp.zaya_block.experts.switch_mlp."))
        #expect(source.contains("key.hasSuffix(\".tq_bits\")"))
        #expect(source.contains("value.movedAxis(source: 2, destination: 1)"))
        #expect(source.contains("compressRouterMLPSequentialIndices"))
        #expect(source.contains("(\"\\(prefix)2.\", \"\\(prefix)1.\")"))
        #expect(source.contains("(\"\\(prefix)4.\", \"\\(prefix)2.\")"))
        #expect(source.contains("router_states_scale"))
        #expect(source.contains("router_mlp.2.bias"))
        #expect(source.contains("fillResidualScaleDefaults"))
        #expect(source.contains("lm_head.weight"))
    }

    @Test("Real JANGTQ bundles expose the non-native keys the sanitizer rewrites",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-JANGTQ2") + "/model.safetensors.index.json")))
    func realJANGTQBundleNeedsSanitizerRewrites() throws {
        for bundle in ["ZAYA1-VL-8B-JANGTQ2", "ZAYA1-VL-8B-JANGTQ4"] {
            let indexURL = URL(fileURLWithPath: Self.bundlePath(bundle))
                .appendingPathComponent("model.safetensors.index.json")
            guard FileManager.default.fileExists(atPath: indexURL.path) else { continue }

            let index = try JSONDecoder.json5().decode(
                SafetensorsIndex.self,
                from: Data(contentsOf: indexURL))
            let keys = Set(index.weightMap.keys)

            #expect(keys.contains(
                "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.tq_packed"))
            let config = try JSONDecoder.json5().decode(
                Zaya1VLConfiguration.self,
                from: Data(contentsOf: URL(fileURLWithPath: Self.bundlePath(bundle))
                    .appendingPathComponent("config.json")))
            let sanitized = Zaya1VLWeightSanitizer.sanitize(
                weights: [
                    "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.tq_packed":
                        MLXArray.zeros([1]),
                    "model.layers.0.mlp.zaya_block.router.router_mlp.4.weight":
                        MLXArray.zeros([17, 256]),
                    "lm_head.weight": MLXArray.zeros([1]),
                ],
                configuration: config)
            #expect(sanitized[
                "language_model.model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.tq_packed"
            ] != nil)
            #expect(sanitized["language_model.model.layers.0.mlp.zaya_block.router.router_mlp.2.weight"] != nil)
            #expect(sanitized["language_model.lm_head.weight"] == nil)
            #expect(!keys.contains(
                "model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.tq_packed"))
            #expect(keys.contains("model.layers.0.attn.self_attn.qkv.conv_qk.0.weight"))
            #expect(keys.contains("model.layers.0.mlp.zaya_block.router.router_mlp.2.weight"))
            #expect(keys.contains("model.layers.0.mlp.zaya_block.router.router_mlp.4.weight"))
            // Some converted bundles already include residual/router defaults and
            // some older bundles do not. The sanitizer must be idempotent: it fills
            // missing defaults without treating their presence as an error.
        }
    }

    @Test("Real MXFP4 bundle uses the same native rewrite boundary without JANGTQ sidecar",
          .enabled(if: FileManager.default.fileExists(
              atPath: bundlePath("ZAYA1-VL-8B-MXFP4") + "/model.safetensors.index.json")))
    func realMXFP4BundleNeedsSharedSanitizerRewrites() throws {
        let indexURL = URL(fileURLWithPath: Self.bundlePath("ZAYA1-VL-8B-MXFP4"))
            .appendingPathComponent("model.safetensors.index.json")
        let index = try JSONDecoder.json5().decode(
            SafetensorsIndex.self,
            from: Data(contentsOf: indexURL))
        let keys = Set(index.weightMap.keys)

        #expect(keys.contains(
            "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.weight"))
        let config = try JSONDecoder.json5().decode(
            Zaya1VLConfiguration.self,
            from: Data(contentsOf: URL(fileURLWithPath: Self.bundlePath("ZAYA1-VL-8B-MXFP4"))
                .appendingPathComponent("config.json")))
        let sanitized = Zaya1VLWeightSanitizer.sanitize(
            weights: [
                "model.layers.0.zaya_block.experts.switch_mlp.gate_proj.weight":
                    MLXArray.zeros([1]),
                "model.layers.0.mlp.zaya_block.router.router_mlp.4.weight":
                    MLXArray.zeros([17, 256]),
            ],
            configuration: config)
        #expect(sanitized[
            "language_model.model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.weight"
        ] != nil)
        #expect(sanitized["language_model.model.layers.0.mlp.zaya_block.router.router_mlp.2.weight"] != nil)
        #expect(!keys.contains(
            "model.layers.0.mlp.zaya_block.experts.switch_mlp.gate_proj.weight"))
        #expect(keys.contains("model.layers.0.mlp.zaya_block.router.router_mlp.4.weight"))
        // Default tensors are intentionally not part of this boundary assertion:
        // current bundles may already carry them, while stale bundles rely on the
        // sanitizer to add them.
    }
}
