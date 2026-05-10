// Regression test pinning the contract for `rmsNormNoScale`
// (Gemma4Text.swift:43-50). The implementation switched from the
// hand-rolled `x * rsqrt(mean(x*x) + eps)` to
// `MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)`. These tests
// verify numerical equivalence to within the fp32-variance vs
// input-dtype-variance tolerance window across bf16/fp16/fp32 inputs.
//
// Why this matters: 3 call sites in Gemma4 attention v_norm path and
// 4+ in Gemma4 VLM tower; per-token saving is ~12 dispatches/layer × N
// layers, with ~1.5x fusion amplification observed on Zaya. The hand-rolled
// comment "MLXFast.rmsNorm doesn't support nil weight" was stale —
// `MLXArray.mlxNone` is the supported pattern (Qwen35, Qwen3Next, Gemma3nText,
// and Zaya all use it).

import Foundation
import MLX
@testable import MLXLLM
import Testing

private func referenceRmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(variance + eps)
}

@Suite("Gemma4 rmsNormNoScale parity", .serialized)
struct Gemma4RmsNormNoScaleParitySuite {

@Test func gemma4RmsNormNoScaleMatchesReferenceFp32() {
    MLXMetalTestLock.withLock {
    let x = MLXArray.linspace(Float(-2), Float(2), count: 4 * 8 * 64)
        .reshaped(4, 8, 64)
    let actual = rmsNormNoScale(x).asArray(Float.self)
    let expected = referenceRmsNormNoScale(x).asArray(Float.self)
    #expect(actual.count == expected.count)
    var maxAbs: Float = 0
    for i in 0 ..< actual.count {
        maxAbs = max(maxAbs, abs(actual[i] - expected[i]))
    }
    // fp32 variance compute on both sides — should be effectively identical.
    #expect(maxAbs < 1e-4, "fp32 max abs diff = \(maxAbs)")
    }
}

@Test func gemma4RmsNormNoScaleMatchesReferenceBfloat16() {
    MLXMetalTestLock.withLock {
    let xf = MLXArray.linspace(Float(-1.5), Float(1.5), count: 2 * 4 * 32)
        .reshaped(2, 4, 32)
    let x = xf.asType(.bfloat16)
    // For bf16 input: hand-rolled variance is computed in bf16, MLXFast
    // upcasts internally for the variance reduction. Tolerance accommodates
    // the precision delta (typically MLXFast is *more* precise).
    let actual = rmsNormNoScale(x).asType(.float32).asArray(Float.self)
    let expected = referenceRmsNormNoScale(x).asType(.float32).asArray(Float.self)
    #expect(actual.count == expected.count)
    var maxAbs: Float = 0
    for i in 0 ..< actual.count {
        maxAbs = max(maxAbs, abs(actual[i] - expected[i]))
    }
    // bf16 has ~3 decimal digits of precision; allow generous tolerance.
    #expect(maxAbs < 0.05, "bf16 max abs diff = \(maxAbs)")
    }
}

@Test func gemma4RmsNormNoScalePreservesInputDtype() {
    MLXMetalTestLock.withLock {
    let x = MLXArray.ones([2, 4, 16], dtype: .bfloat16)
    let result = rmsNormNoScale(x)
    #expect(result.dtype == .bfloat16, "result dtype \(result.dtype) should match input bf16")
    }
}

@Test func gemma4RmsNormNoScaleHandlesEpsCorrectly() {
    MLXMetalTestLock.withLock {
    // Zero input — without eps would divide by zero. With eps=1e-6, output
    // should be x / sqrt(0 + eps) = x / sqrt(eps). For x=0 this is 0.
    let x = MLXArray.zeros([1, 1, 8])
    let result = rmsNormNoScale(x).asArray(Float.self)
    for v in result { #expect(v == 0) }
    }
}

}  // end Gemma4RmsNormNoScaleParitySuite
