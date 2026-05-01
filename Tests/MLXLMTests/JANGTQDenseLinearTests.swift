// Copyright © 2026 osaurus.
//
// Compile-time + structural tests for `JANGTQDenseLinear` — the dense
// (non-MoE) JANGTQ Linear shim. Exercises:
//
//   - Construction with various (in, out, bits, bias) tuples
//   - Parameter shape contract (.tq_packed 2D, .tq_norms 1D, .biases optional)
//   - Module pathing matches the .safetensors keys produced by
//     `jang_tools/turboquant/linear.tq_quantize_weight`
//
// End-to-end forward verification requires a real Mistral 3 / Mistral
// 3.5 / Laguna JANGTQ bundle on disk + a loaded sidecar via
// `JANGTQRuntimeCache.shared.loadSidecar(...)`. Those bundles are not
// available in CI today, so this test stays at the structural level
// — actual decode quality is gated on a separate verification PR
// once a bundle becomes available.

import Foundation
import MLX
@testable import MLXLMCommon
import XCTest

final class JANGTQDenseLinearTests: XCTestCase {

    // MARK: - Construction + parameter shapes

    func testConstruction_2bit_noBias_correctShapes() {
        let layer = JANGTQDenseLinear(
            inFeatures: 4096, outFeatures: 12288, bits: 2, seed: 42, bias: false)

        XCTAssertEqual(layer.inFeatures, 4096)
        XCTAssertEqual(layer.outFeatures, 12288)
        XCTAssertEqual(layer.bits, 2)
        XCTAssertEqual(layer.mxtqSeed, 42)
        XCTAssertFalse(layer.hasBias)

        // tq_packed: (out, packed_in) where packed_in = ceil(in / (32/bits))
        // For bits=2: vals_per_u32 = 16, packed_in = 4096/16 = 256.
        XCTAssertEqual(layer.packed.shape, [12288, 256])
        XCTAssertEqual(layer.packed.dtype, .uint32)

        // tq_norms: (out,)
        XCTAssertEqual(layer.norms.shape, [12288])
        XCTAssertEqual(layer.norms.dtype, .float16)

        XCTAssertNil(layer.biases)
    }

    func testConstruction_4bit_withBias_correctShapes() {
        let layer = JANGTQDenseLinear(
            inFeatures: 12288, outFeatures: 12288, bits: 4, seed: 42, bias: true)

        // bits=4: vals_per_u32 = 8, packed_in = 12288/8 = 1536.
        XCTAssertEqual(layer.packed.shape, [12288, 1536])
        XCTAssertEqual(layer.norms.shape, [12288])
        XCTAssertNotNil(layer.biases)
        XCTAssertEqual(layer.biases?.shape, [12288])
        XCTAssertTrue(layer.hasBias)
    }

    /// Mistral 3.5 inner ministral3 attention head dim 128 × 96 q heads
    /// → q_proj output is 12288. Verifies the shim can hold typical
    /// Mistral 3.5 attention shapes.
    func testMistral35AttentionShapes() {
        // Q projection: hidden=12288 → 96 heads × head_dim=128 = 12288
        let q = JANGTQDenseLinear(inFeatures: 12288, outFeatures: 12288, bits: 2)
        XCTAssertEqual(q.packed.shape, [12288, 768])  // 12288/16

        // KV projections: hidden=12288 → 8 heads × head_dim=128 = 1024
        let k = JANGTQDenseLinear(inFeatures: 12288, outFeatures: 1024, bits: 2)
        XCTAssertEqual(k.packed.shape, [1024, 768])

        // O projection: 12288 → 12288
        let o = JANGTQDenseLinear(inFeatures: 12288, outFeatures: 12288, bits: 2)
        XCTAssertEqual(o.packed.shape, [12288, 768])
    }

    /// MLP gate/up/down projections for Mistral 3.5: hidden=12288,
    /// intermediate=32768 (rough proxy — actual value comes from config).
    func testMistral35MLPShapes() {
        let gate = JANGTQDenseLinear(inFeatures: 12288, outFeatures: 32768, bits: 2)
        XCTAssertEqual(gate.packed.shape, [32768, 768])
        XCTAssertEqual(gate.norms.shape, [32768])

        let down = JANGTQDenseLinear(inFeatures: 32768, outFeatures: 12288, bits: 2)
        // bits=2, in=32768 → packed_in = 32768/16 = 2048
        XCTAssertEqual(down.packed.shape, [12288, 2048])
    }

    // MARK: - Parameter pathing (matches .safetensors keys)

    /// The Python converter emits keys like `model.layers.0.mlp.gate_proj.tq_packed`,
    /// `.tq_norms`, `.biases`. Verify the @ParameterInfo keys resolve
    /// correctly so the safetensors loader binds the right tensors.
    func testParameterKeysMatchSafetensorsLayout() {
        let layer = JANGTQDenseLinear(
            inFeatures: 4096, outFeatures: 4096, bits: 2, bias: true)
        let params = layer.parameters()
        // Expect three keys: tq_packed, tq_norms, biases. (No "weight" key —
        // that's the whole point of the shim.)
        let keys = params.flattened().map { $0.0 }
        XCTAssertTrue(keys.contains("tq_packed"),
            "expected tq_packed key; got \(keys)")
        XCTAssertTrue(keys.contains("tq_norms"),
            "expected tq_norms key; got \(keys)")
        XCTAssertTrue(keys.contains("biases"),
            "expected biases key when bias=true; got \(keys)")
        XCTAssertFalse(keys.contains("weight"),
            "JANGTQDenseLinear must NOT expose a `weight` key; got \(keys)")
    }

    func testParameterKeys_noBias() {
        let layer = JANGTQDenseLinear(
            inFeatures: 4096, outFeatures: 4096, bits: 2, bias: false)
        let keys = layer.parameters().flattened().map { $0.0 }
        XCTAssertTrue(keys.contains("tq_packed"))
        XCTAssertTrue(keys.contains("tq_norms"))
        XCTAssertFalse(keys.contains("biases"),
            "biases key must be absent when bias=false; got \(keys)")
    }

    // MARK: - Bit-width packed_cols arithmetic

    /// The kernel's `packed_cols = ceil(in_features / (32/bits))` and the
    /// converter assumes `in_features % (32/bits) == 0` for the
    /// vectorized pack. Verify the shim agrees with that contract.
    func testPackedColsArithmetic() {
        // 2-bit: vals_per_u32 = 16. in=4096 → packed=256. in=12288 → 768.
        XCTAssertEqual(
            JANGTQDenseLinear(inFeatures: 4096, outFeatures: 1, bits: 2).packed.dim(-1),
            256)
        XCTAssertEqual(
            JANGTQDenseLinear(inFeatures: 12288, outFeatures: 1, bits: 2).packed.dim(-1),
            768)

        // 3-bit (used by some MiniMax variants): vals_per_u32 = 10 (32/3 = 10.67),
        // ceiling = 11. in=4096 → packed = ceil(4096/10) = 410.
        // Note: jang-tools may not actually emit 3-bit packed for these
        // shapes; this test verifies the shim's arithmetic is correct
        // regardless. Verify ceiling division.
        let l3 = JANGTQDenseLinear(inFeatures: 4096, outFeatures: 1, bits: 3)
        XCTAssertEqual(l3.packed.dim(-1), 410)

        // 4-bit: vals_per_u32 = 8. in=12288 → packed = 1536.
        XCTAssertEqual(
            JANGTQDenseLinear(inFeatures: 12288, outFeatures: 1, bits: 4).packed.dim(-1),
            1536)
    }
}
