import Foundation
import MLX
@testable import MLXVLM
import Testing

private func referenceGemma4VLMRmsNormNoScale(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let variance = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(variance + eps)
}

@Suite("Gemma4 VLM rmsNormNoScale parity", .serialized)
struct Gemma4VLMRmsNormNoScaleParitySuite {
    @Test func gemma4VLMRmsNormNoScaleMatchesReferenceFp32() {
        MLXMetalTestLock.withLock {
            let x = MLXArray.linspace(Float(-2), Float(2), count: 2 * 8 * 64)
                .reshaped(2, 8, 64)
            let actual = rmsNormNoScale(x).asArray(Float.self)
            let expected = referenceGemma4VLMRmsNormNoScale(x).asArray(Float.self)
            #expect(actual.count == expected.count)
            var maxAbs: Float = 0
            for i in 0 ..< actual.count {
                maxAbs = max(maxAbs, abs(actual[i] - expected[i]))
            }
            #expect(maxAbs < 1e-4, "fp32 max abs diff = \(maxAbs)")
        }
    }

    @Test func gemma4VLMRmsNormNoScalePreservesInputDtype() {
        MLXMetalTestLock.withLock {
            let x = MLXArray.ones([2, 4, 16], dtype: .bfloat16)
            let result = rmsNormNoScale(x)
            #expect(result.dtype == .bfloat16, "result dtype \(result.dtype) should match input bf16")
        }
    }
}
