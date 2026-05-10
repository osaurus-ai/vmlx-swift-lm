// Gemma4 image preprocessing edge cases.
//
// Pins guards in `Gemma4Processor.prepare(input:)` (and the underlying
// `MLXVLM/Models/Gemma4.swift` arithmetic that derives target image
// dimensions from the source extent). Without the guards, a zero-area
// CIImage divides by zero in the scale-factor math and traps in
// `Int(floor(.nan))`.
//
// Wrapped in `MLXMetalTestLock.withLock { ... }` to serialize Metal
// command-buffer access with cross-suite GPU tests (the same
// convention `Gemma4VLMTests` follows).

import CoreImage
import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import Testing

@Suite("Gemma4 image preprocessing edge cases", .serialized)
struct Gemma4ImageEdgeCaseTests {

    /// 2026-05-10 audit: a zero-extent CIImage (corrupted source data,
    /// crop intersected with empty rect) flowed into
    /// `Gemma4Processor.prepare` would divide by zero in
    /// `f = sqrt(maxP * ps * ps / (w * h))`, producing infinity; the
    /// next line cast `Int(floor(NaN))` and trapped. The fix at
    /// `Gemma4.swift:990-1001` rejects zero/negative dims explicitly
    /// with `VLMError.imageProcessingFailure`. This test pins the new
    /// behavior.
    @Test("Gemma4Processor rejects zero-area images cleanly instead of trapping")
    func zeroAreaImageThrowsImageProcessingFailure() async throws {
        try await MLXMetalTestLock.withLock {
            let configJSON = """
            {
              "processor_class": "Gemma4Processor",
              "patch_size": 16,
              "max_soft_tokens": 280,
              "pooling_kernel_size": 3,
              "image_seq_length": 280,
              "audio_seq_length": 750
            }
            """.data(using: .utf8)!
            let config = try JSONDecoder.json5().decode(
                Gemma4ProcessorConfiguration.self, from: configJSON)
            let processor = Gemma4Processor(config, tokenizer: TestTokenizer())

            // Build a zero-area CIImage by intersecting an arbitrary
            // CIImage with `.zero`. The resulting extent is `.zero`
            // (width=0, height=0), which is exactly the corrupted-source
            // shape the guard at `Gemma4.swift:996-1001` rejects.
            let zero = CIImage(color: .red).cropped(to: .zero)
            #expect(zero.extent.width == 0)
            #expect(zero.extent.height == 0)

            let userInput = UserInput(
                prompt: "Describe this image.",
                images: [.ciImage(zero)])

            await #expect(throws: VLMError.self) {
                _ = try await processor.prepare(input: userInput)
            }
        }
    }
}
