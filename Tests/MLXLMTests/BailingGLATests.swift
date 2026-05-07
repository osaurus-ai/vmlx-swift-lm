import MLX
@testable import MLXLLM
import XCTest

final class BailingGLATests: XCTestCase {
    func testRecurrentGLAKernelMatchesReferenceWithPriorState() {
        let batch = 1
        let heads = 2
        let length = 48
        let dim = 32

        let q = shapedValues(
            count: batch * heads * length * dim,
            scale: 0.007,
            period: 19
        ).reshaped(batch, heads, length, dim)
        let k = shapedValues(
            count: batch * heads * length * dim,
            scale: 0.005,
            period: 23
        ).reshaped(batch, heads, length, dim)
        let v = shapedValues(
            count: batch * heads * length * dim,
            scale: 0.006,
            period: 29
        ).reshaped(batch, heads, length, dim)
        let g = MLXArray([Float(-0.04), Float(-0.08)])
        let state = shapedValues(
            count: batch * heads * dim * dim,
            scale: 0.002,
            period: 31
        ).reshaped(batch, heads, dim, dim)

        let (kernelOut, kernelState) = recurrentGLA(
            q: q, k: k, v: v, g: g, scale: 0.125, h: state)
        let (refOut, refState) = recurrentGLAReference(
            q: q, k: k, v: v, g: g, scale: 0.125, h: state)

        MLX.eval(kernelOut, kernelState, refOut, refState)
        assertClose(kernelOut, refOut, tolerance: 1e-4)
        assertClose(kernelState, refState, tolerance: 1e-4)
    }

    func testRecurrentGLAKernelHandlesLongSyntheticPrefill() {
        let batch = 1
        let heads = 1
        let length = 96
        let dim = 32

        let q = shapedValues(
            count: batch * heads * length * dim,
            scale: 0.004,
            period: 17
        ).reshaped(batch, heads, length, dim)
        let k = shapedValues(
            count: batch * heads * length * dim,
            scale: 0.004,
            period: 13
        ).reshaped(batch, heads, length, dim)
        let v = shapedValues(
            count: batch * heads * length * dim,
            scale: 0.004,
            period: 11
        ).reshaped(batch, heads, length, dim)
        let g = MLXArray([Float(-0.06)])

        let (out, state) = recurrentGLA(q: q, k: k, v: v, g: g, scale: 0.125, h: nil)

        MLX.eval(out, state)
        XCTAssertEqual(out.shape, [batch, heads, length, dim])
        XCTAssertEqual(state.shape, [batch, heads, dim, dim])
        XCTAssertFalse(out.asArray(Float.self).contains { !$0.isFinite })
        XCTAssertFalse(state.asArray(Float.self).contains { !$0.isFinite })
    }

    private func shapedValues(count: Int, scale: Float, period: Int) -> MLXArray {
        let values = (0..<count).map { i -> Float in
            Float((i % period) - (period / 2)) * scale
        }
        return MLXArray(values)
    }

    private func assertClose(
        _ actual: MLXArray,
        _ expected: MLXArray,
        tolerance: Float,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let actualValues = actual.asArray(Float.self)
        let expectedValues = expected.asArray(Float.self)
        XCTAssertEqual(actualValues.count, expectedValues.count, file: file, line: line)

        var maxDiff: Float = 0
        for (a, e) in zip(actualValues, expectedValues) {
            maxDiff = max(maxDiff, abs(a - e))
        }
        XCTAssertLessThanOrEqual(maxDiff, tolerance, file: file, line: line)
    }
}
