import Foundation
import MLX
@testable import MLXVLM
import Testing

@Suite("ZAYA1-VL image-mask LoRA gate", .serialized)
struct Zaya1VLImageMaskLoRATests {
    @Test("Image-mask add only affects image-token rows")
    func maskedAddOnlyTouchesImageRows() throws {
        try MLXMetalTestLock.withLock {
            let base = MLXArray.zeros([1, 4, 3])
            let addon = MLXArray.ones([1, 4, 3])
            let imageMask = MLXArray([false, true, false, true])

            let actual = try Zaya1VLRuntimeSupport.applyImageMaskedAdd(
                base: base,
                addon: addon,
                imageMask: imageMask
            )
            let expected = MLXArray([
                Float(0), 0, 0,
                Float(1), 1, 1,
                Float(0), 0, 0,
                Float(1), 1, 1,
            ]).reshaped(1, 4, 3)

            let maxDelta = (actual.asType(.float32) - expected).abs().max().item(Float.self)
            #expect(maxDelta < 1e-6)
            #expect(actual.shape == base.shape)
        }
    }

    @Test("Image-mask add preserves nil-mask text-only path")
    func nilMaskReturnsBaseUnchanged() throws {
        try MLXMetalTestLock.withLock {
            let base = MLXArray([
                Float(0.25), -0.5, 1.0,
                Float(2.0), -1.0, 0.75,
            ]).reshaped(2, 3)
            let addon = MLXArray.ones([2, 3])

            let actual = try Zaya1VLRuntimeSupport.applyImageMaskedAdd(
                base: base,
                addon: addon,
                imageMask: nil
            )

            let maxDelta = (actual.asType(.float32) - base.asType(.float32))
                .abs().max().item(Float.self)
            #expect(maxDelta < 1e-6)
        }
    }

    @Test("Image-mask add supports routed-token flat shape")
    func flatMaskSupportsRoutedExpertShape() throws {
        try MLXMetalTestLock.withLock {
            let base = MLXArray.zeros([4, 2])
            let addon = MLXArray([
                Float(1), 2,
                Float(3), 4,
                Float(5), 6,
                Float(7), 8,
            ]).reshaped(4, 2)
            let imageMask = MLXArray([true, false, true, false])

            let actual = try Zaya1VLRuntimeSupport.applyImageMaskedAdd(
                base: base,
                addon: addon,
                imageMask: imageMask
            )
            let expected = MLXArray([
                Float(1), 2,
                Float(0), 0,
                Float(5), 6,
                Float(0), 0,
            ]).reshaped(4, 2)

            let maxDelta = (actual.asType(.float32) - expected).abs().max().item(Float.self)
            #expect(maxDelta < 1e-6)
        }
    }

    @Test("Image-mask add rejects incompatible mask shape")
    func incompatibleMaskShapeThrows() {
        MLXMetalTestLock.withLock {
            let base = MLXArray.zeros([1, 4, 3])
            let addon = MLXArray.ones([1, 4, 3])
            let imageMask = MLXArray([true, false, true])

            do {
                _ = try Zaya1VLRuntimeSupport.applyImageMaskedAdd(
                    base: base,
                    addon: addon,
                    imageMask: imageMask
                )
                Issue.record("Expected incompatible image mask to throw")
            } catch {
                #expect(String(describing: error).contains("image mask shape"))
            }
        }
    }
}
