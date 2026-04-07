// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(from:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and updates the model with the weights.
///
/// When a JANG model is detected (via `jangConfig`), per-layer bit widths are
/// inferred from tensor shapes automatically. Standard MLX models are unaffected.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    jangConfig: JangConfig? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // Resolve symlinks (mlxstudio uses symlinked model directories)
    let modelDirectory = modelDirectory.resolvingSymlinksInPath()

    // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
    if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let (w, m) = try loadArraysAndMetadata(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
                if metadata.isEmpty {
                    metadata = m
                }
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // JANG: dequantize MoE gate weights from quantized uint32 → float.
    // Gates are stored at 8-bit (CRITICAL tier) but may have different group_size
    // than the body. Dequantizing resolves ambiguous bit/group_size inference.
    if let jangConfig {
        JangLoader.dequantizeMoEGates(
            weights: &weights, groupSize: jangConfig.quantization.blockSize,
            bitWidthsUsed: jangConfig.quantization.bitWidthsUsed)
    }

    // Determine quantization: JANG models infer per-layer bit widths from tensor shapes.
    // Standard MLX models use the quantization from config.json as before.
    let effectivePerLayerQuantization: BaseConfiguration.PerLayerQuantization?
    if let jangConfig {
        effectivePerLayerQuantization = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig)
    } else if let perLayerQuantization {
        // Remap perLayerQuantization keys to match sanitized weight paths.
        // Config.json uses VLM-prefixed keys like "language_model.model.layers.0..."
        // LLM sanitize strips to "model.layers.0..." but VLM keeps "language_model.model.layers.0..."
        // Keep BOTH original and stripped keys so it works for both paths.
        var remappedPerLayer = perLayerQuantization.perLayerQuantization
        for (key, value) in perLayerQuantization.perLayerQuantization {
            if key.hasPrefix("language_model.model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                remappedPerLayer[stripped] = value
            } else if key.hasPrefix("language_model.") {
                let stripped = String(key.dropFirst("language_model.".count))
                remappedPerLayer[stripped] = value
            }
        }
        effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: perLayerQuantization.quantization,
            perLayerQuantization: remappedPerLayer
        )
    } else {
        effectivePerLayerQuantization = nil
    }

    // quantize if needed
    if quantization != nil || effectivePerLayerQuantization != nil {
        // Inline quantize with error logging instead of try! crash
        let updates = model.leafModules().flattened().compactMap { (path, m) -> (String, Module)? in
            guard weights["\(path).scales"] != nil else { return nil }
            let tup: (groupSize: Int, bits: Int, mode: QuantizationMode)?
            if let effectivePerLayerQuantization {
                tup = effectivePerLayerQuantization.quantization(layer: path)?.asTuple
            } else {
                tup = quantization?.asTuple
            }
            guard let (gs, b, mode) = tup else { return nil }
            if let q = quantizeSingle(layer: m, groupSize: gs, bits: b, mode: mode) {
                return (path, q)
            }
            return nil
        }
        do {
            try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
        } catch {
            print("[loadWeights] quantize model.update failed: \(error)")
            for (path, mod) in updates.prefix(5) {
                print("  update path: \(path) → \(type(of: mod))")
            }
            throw error
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    // MoE optimization: convert float16 weights to bfloat16 to prevent Metal's
    // automatic float16→float32 promotion on mixed-dtype operations.
    // MoE gate routing runs at float32 for numerical stability, but if weights are
    // float16, Metal promotes ALL ops to float32 (2x bandwidth, ~50% speed drop).
    // bfloat16 shares float32's exponent range, avoiding the promotion.
    let isMoE = weights.keys.contains { $0.contains("switch_mlp") || $0.contains("switch_glu")
        || ($0.contains(".gate.weight") && !$0.contains("gate_proj")) }
    if isMoE {
        convertToBFloat16(model: model)
    }

    eval(model)
}

/// Convert float16/float32 model parameters to bfloat16 for MoE performance.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing ~50% speed regression for MoE models where gate routing runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
/// Quantization scales/biases are left as-is (they're used directly by Metal kernels).
private func convertToBFloat16(model: Module) {
    var converted = [String: MLXArray]()
    for (key, array) in model.parameters().flattened() {
        if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
            continue  // Leave quantization metadata unchanged
        }
        if array.dtype == .float16 || array.dtype == .float32 {
            converted[key] = array.asType(.bfloat16)
        }
    }
    if !converted.isEmpty {
        let params = ModuleParameters.unflattened(converted)
        try? model.update(parameters: params, verify: [])
    }
}
