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

    // JANGTQ-native detection: presence of the runtime sidecar means the
    // bundle ships tq_packed/tq_norms tensors that should be consumed RAW
    // by TurboQuantSwitchGLU — we must NOT run the MXTQ→affine expander,
    // the per-layer quant inference, or the MoE-gate dequant.
    let jangtqSidecarURL = modelDirectory.appendingPathComponent("jangtq_runtime.safetensors")
    let isJANGTQNative = FileManager.default.fileExists(atPath: jangtqSidecarURL.path)

    if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                // Skip the JANGTQ sidecar — it contains runtime signs/codebook
                // arrays that go into JANGTQRuntimeCache, not module params.
                if url.lastPathComponent == "jangtq_runtime.safetensors" {
                    continue
                }
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

    // JANGTQ native: load the signs/codebook sidecar into the runtime cache
    // before model.update() so TurboQuantSwitchGLU has everything it needs
    // on first forward.
    if isJANGTQNative {
        do {
            try JANGTQRuntimeCache.shared.loadSidecar(from: jangtqSidecarURL)
        } catch {
            print("[loadWeights] JANGTQ sidecar load failed: \(error)")
            throw error
        }
    }

    // Fail-fast guard: if jang_config.json declares `weight_format: "mxtq"`
    // (which routes the factory to the JANGTQ model class with
    // `TurboQuantSwitchGLU`), the bundle MUST also ship
    // `jangtq_runtime.safetensors`. Without it the runtime cache stays
    // empty and `TurboQuantSwitchLinear` `fatalError`s on the first
    // forward pass — a SIGABRT that's hard to trace from a server log.
    // Surface the missing file at load time with a clear message instead.
    if let jangConfigURL = JangLoader.findConfigPath(at: modelDirectory),
        let configData = try? Data(contentsOf: jangConfigURL),
        let configJSON = try? JSONSerialization.jsonObject(with: configData)
            as? [String: Any],
        (configJSON["weight_format"] as? String) == "mxtq",
        !isJANGTQNative
    {
        throw NSError(
            domain: "MLXLMCommon.JANGTQ", code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "JANGTQ bundle is incomplete: jang_config.json at "
                    + "\(jangConfigURL.path) declares weight_format=\"mxtq\" "
                    + "but jangtq_runtime.safetensors is missing from "
                    + "\(modelDirectory.path). Re-download the bundle "
                    + "including the sidecar (signs.{N}.{seed} + "
                    + "codebook.{N}.{bits} arrays)."
            ])
    }

    // JANG: dequantize MoE gate weights from quantized uint32 → float.
    // Gates are stored at 8-bit (CRITICAL tier) but may have different group_size
    // than the body. Dequantizing resolves ambiguous bit/group_size inference.
    // Safe for JANGTQ-native too: the dequant only touches `.*.gate.*` keys,
    // not the `tq_packed`/`tq_norms` expert projections.
    if let jangConfig {
        JangLoader.dequantizeMoEGates(
            weights: &weights, groupSize: jangConfig.quantization.blockSize,
            bitWidthsUsed: jangConfig.quantization.bitWidthsUsed)
    }

    // Determine quantization: JANG models infer per-layer bit widths from tensor shapes.
    // Standard MLX models use the quantization from config.json as before.
    // Safe for JANGTQ-native: infer only walks `.scales` keys, so it picks
    // up the affine 8-bit attention / embed / lm_head and ignores the
    // tq_packed expert projections.
    let effectivePerLayerQuantization: BaseConfiguration.PerLayerQuantization?
    if let jangConfig {
        // Prefer config.json's explicit `quantization.group_size` over
        // jangConfig.blockSize when the jang_config doesn't carry quant
        // metadata of its own (e.g., DSV4-Flash bundles ship
        // `weight_format: "bf16"` even on quantized variants — the
        // global group_size is in config.json instead).
        //
        // 2026-04-28: when jangConfig has explicit `bit_widths_used`
        // (signals real JANG conversion), `inferPerLayerQuantization`
        // ignores `overrideGroupSize` and uses jangConfig.blockSize as
        // the authoritative prior — see `JangLoader.swift` for the
        // root-cause writeup of the (8, 32) ≡ (4, 64) shape ambiguity
        // that crashed Cascade-2 JANG_4M / Nemotron-Omni MXFP4 with
        // mid-prefill rmsNorm. We do NOT merge config.json's per-layer
        // dict here because that dict can be HF-tooling-stale (claims
        // `(gs=32, bits=8)` for layers actually packed at `(gs=64,
        // bits=4)`) and would re-introduce the wrong values.
        let configGS: Int? = quantization?.groupSize
        let inferred = JangLoader.inferPerLayerQuantization(
            weights: weights, jangConfig: jangConfig,
            overrideGroupSize: configGS)

        if !inferred.perLayerQuantization.isEmpty {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] JANG shape walk produced \(inferred.perLayerQuantization.count) per-layer quant override(s) over default (bits=\(b), gs=\(g))\n".utf8))
        }
        effectivePerLayerQuantization = inferred
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

        // Defense-in-depth: cross-check the config-supplied per-layer
        // overrides against actual safetensors shapes. If a bundle's
        // config.json was re-stamped (or a converter bug emitted wrong
        // bits / group_size), the runtime would otherwise silently
        // corrupt dequant. Shape walk is authoritative — it overrides
        // disagreeing entries and logs a one-line summary so users can
        // see when a patch was applied.
        if let shapeInferred = JangLoader.inferPerLayerQuantizationFromShapes(
            weights: weights,
            defaultBits: perLayerQuantization.quantization?.bits,
            defaultGroupSize: perLayerQuantization.quantization?.groupSize)
        {
            var corrections = 0
            for (path, expected) in shapeInferred.perLayerQuantization {
                if case .quantize(let q) = expected {
                    if let configured = remappedPerLayer[path],
                        case .quantize(let cq) = configured,
                        cq.bits == q.bits && cq.groupSize == q.groupSize
                    { continue }
                    remappedPerLayer[path] = expected
                    corrections += 1
                }
            }
            if corrections > 0 {
                FileHandle.standardError.write(
                    Data("[Load] config per-layer quant disagreed with safetensors shapes — patched \(corrections) layer(s) from shape walk\n".utf8))
            }
        }
        effectivePerLayerQuantization = BaseConfiguration.PerLayerQuantization(
            quantization: perLayerQuantization.quantization,
            perLayerQuantization: remappedPerLayer
        )
    } else if let quantization {
        // Bundle has top-level quantization but no per-layer overrides.
        // Walk every `.scales` key and infer (bits, gs) from shapes.
        // This catches bundles whose `config.json` says e.g.
        // `bits: 8` uniformly while individual modules are actually
        // mixed (8-bit attention + 2-bit routed MoE). The algorithm
        // is idempotent: when the config matches reality the inferred
        // map adds no per-layer overrides.
        let inferred =
            JangLoader.inferPerLayerQuantizationFromShapes(
                weights: weights,
                defaultBits: quantization.bits,
                defaultGroupSize: quantization.groupSize)
        if let inferred, !inferred.perLayerQuantization.isEmpty {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] non-JANG shape walk produced \(inferred.perLayerQuantization.count) per-layer quant override(s) over default (bits=\(b), gs=\(g))\n".utf8))
        }
        effectivePerLayerQuantization = inferred
            ?? BaseConfiguration.PerLayerQuantization(
                quantization: quantization, perLayerQuantization: [:])
    } else {
        // No quantization signal in config.json at all — but the
        // bundle may STILL be quantized (e.g., a stripped config).
        // If `.scales` keys exist, infer fully from shapes.
        let inferred =
            JangLoader.inferPerLayerQuantizationFromShapes(weights: weights)
        if let inferred {
            let b = inferred.quantization?.bits ?? -1
            let g = inferred.quantization?.groupSize ?? -1
            FileHandle.standardError.write(
                Data("[Load] config has no quant block — shape walk inferred default (bits=\(b), gs=\(g)) plus \(inferred.perLayerQuantization.count) override(s)\n".utf8))
        }
        effectivePerLayerQuantization = inferred
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

            // MXFP4/MXFP8: quantizeSingle creates QuantizedLinear with dummy biases
            // from MLX.quantized(), but MX formats don't use biases. Create the module
            // directly with nil biases to avoid "biases must be null" at inference time.
            if (mode == .mxfp4 || mode == .mxfp8), m is Linear {
                let linear = m as! Linear
                let (qW, scales, _) = MLX.quantized(linear.weight, groupSize: gs, bits: b)
                return (path, QuantizedLinear(
                    weight: qW, bias: linear.bias, scales: scales, biases: nil,
                    groupSize: gs, bits: b, mode: mode))
            }

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
    // Use .noUnusedKeys instead of .all — MXFP4/MXFP8 quantized layers don't have .biases
    // in the weight files, but QuantizedLinear's optional .biases property gets initialized
    // by the quantize step. Strict .all verification would fail on the missing keys.
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Convert all float16/float32 parameters to bfloat16 to prevent AsType cascades.
    // float16 causes AsType when mixed with internal float32 ops (softmax, RMSNorm).
    // bfloat16 shares float32's exponent range, so promotion is cheaper/eliminated.
    //
    // JANGTQ bypass: Python baseline runs with fp16 TurboQuant norms, and the
    // JANGTQ Metal kernels infer their signature from the norm dtype. Casting
    // those norms to bf16 breaks the gate/up/down projections (verified on
    // MiniMax M2.7 JANGTQ_2L). JANGTQ dispatches are already fp32 internally,
    // so there's no fp16↔fp32 ping-pong to collapse. Skip the cast entirely.
    if !isJANGTQNative {
        let allParams = model.parameters().flattened().map { $0.1 }
        let hasNonBFloat16 = allParams.contains { (arr: MLXArray) in
            arr.dtype == .float16 || arr.dtype == .float32
        }
        if hasNonBFloat16 {
            convertToBFloat16(model: model)
        }
    }

    eval(model)
}

/// Convert float16/float32 model parameters to bfloat16 for MoE performance.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing ~50% speed regression for MoE models where gate routing runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
/// Quantization scales/biases are ALSO converted — QuantizedMatmul uses scales dtype to
/// determine output dtype, so float16 scales → float16 output → AsType when multiplied
/// with bfloat16 norms. Converting scales to bfloat16 eliminates this cascade.
private func convertToBFloat16(model: Module) {
    var converted = [String: MLXArray]()
    for (key, array) in model.parameters().flattened() {
        if array.dtype == .float16 || array.dtype == .float32 {
            converted[key] = array.asType(.bfloat16)
        }
    }
    if !converted.isEmpty {
        let params = ModuleParameters.unflattened(converted)
        do {
            try model.update(parameters: params, verify: [])
        } catch {
            print("[convertToBFloat16] model.update failed: \(error)")
        }
    }
}
