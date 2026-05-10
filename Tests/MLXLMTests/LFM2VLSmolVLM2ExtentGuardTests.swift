// Pin LFM2VL and SmolVLM2 image-extent guard contracts.
//
// Background (mirrors Qwen-family fix in `QwenVLIntExtentTests`):
//
// `Int(CGFloat.infinity)` traps Swift with "Double value cannot be
// converted to Int because it is either infinite or NaN." `CIImage(color:)`
// returns `extent.size = (.infinity, .infinity)` per Apple docs. LFM2VL
// (`splitIntoPatchesAndPreprocess`) and SmolVLM2 (`tiles(from:)`) both
// had raw `Int(image.extent.width)` / `Int(image.extent.height)` calls
// that would trap before any guard could fire. They now route through
// `QwenVL.intExtent` (the same helper Qwen-family + Gemma + ZAYA1-VL +
// GlmOcr already use), throwing `VLMError.imageProcessingFailure`
// instead of trapping the runtime.
//
// Source-coverage style — no MLX runtime needed. The extent helper
// itself is exercised in `QwenVLIntExtentTests`; this file pins the
// adopter source contracts.

import Foundation
import Testing

@Suite("LFM2VL + SmolVLM2 image-extent guard adopter source coverage")
struct LFM2VLSmolVLM2ExtentGuardTests {

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    @Test("LFM2VL.splitIntoPatchesAndPreprocess uses QwenVL.intExtent")
    func lfm2vlAdopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/LFM2VL.swift")

        // The new extent-helper call is present.
        #expect(
            source.contains("try QwenVL.intExtent(image.extent.size)"),
            "LFM2VL must use QwenVL.intExtent so infinity/NaN/zero/negative extents throw VLMError instead of trapping the runtime.")

        // The prior raw `Int(image.extent.width)` / `Int(image.extent.height)`
        // pair must NOT come back.
        #expect(
            !source.contains("let width = Int(image.extent.width)")
            && !source.contains("let height = Int(image.extent.height)"),
            "LFM2VL must not reintroduce raw `Int(image.extent.width)` / `Int(image.extent.height)` — they trap on infinity (e.g. `CIImage(color:)`).")
    }

    @Test("SmolVLM2.tiles(from:) is throws and validates extent up front")
    func smolVLM2Adopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/SmolVLM2.swift")

        // The function signature is `throws`.
        #expect(
            source.contains(
                "func tiles(from originalImage: CIImage) throws -> (tiles: [CIImage], rows: Int, cols: Int)"),
            "SmolVLM2.tiles(from:) must be throws so the extent guard can surface as VLMError.")

        // The extent is validated at the entry via QwenVL.intExtent.
        #expect(
            source.contains("try QwenVL.intExtent(originalImage.extent.size)"),
            "SmolVLM2.tiles(from:) must validate originalImage.extent up front via QwenVL.intExtent.")

        // The caller in prepare(input:) uses `try`.
        #expect(
            source.contains("try tiles(from: image)"),
            "SmolVLM2's caller must use `try tiles(from: image)`.")
    }
}
