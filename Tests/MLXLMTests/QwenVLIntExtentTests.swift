// Pin `QwenVL.intExtent(_:)` — the centralized helper that converts a
// `CIImage.extent.size` into a finite-positive `(height, width)` tuple,
// rejecting infinity / NaN / zero before they reach `Int(...)` and trap
// the process.
//
// Background: `CIImage(color: .red).extent.size` returns `(.infinity, .infinity)`;
// a procedurally-generated CIImage without `cropped(to:)` will have an
// infinite extent. Direct `Int(extent.height)` then traps with
// "Double value cannot be converted to Int because it is either
// infinite or NaN." The pre-existing zero-area guard in Gemma4
// `Gemma4Processor.prepare` did NOT cover infinity (it ran AFTER the
// `Int(...)` conversion). Same vulnerability existed in Qwen2VL,
// Qwen25VL, Qwen3VL, Zaya1VL, GlmOcr — all of which call
// `QwenVL.targetSize(height: Int(size.height), width: Int(size.width), ...)`
// on raw `extent.size`.
//
// Fix: `QwenVL.intExtent(_:)` is the single guard. All call sites now
// route through it. This test pins (a) the helper's contract and
// (b) source coverage that every adopter uses it.
//
// Source-coverage style — no MLX runtime needed for the source-coverage
// portion. The helper-direct portion exercises pure CoreImage / Foundation
// conversions, no Metal.

import CoreGraphics
import Foundation
import Testing

@testable import MLXVLM
import MLXLMCommon

@Suite("QwenVL.intExtent finite-positive helper + adopter source coverage")
struct QwenVLIntExtentTests {

    // MARK: - Helper contract

    @Test("Finite positive extent decodes to integer pair (rounded)")
    func finitePositiveDecodes() throws {
        let (h, w) = try QwenVL.intExtent(CGSize(width: 1024, height: 768))
        #expect(h == 768)
        #expect(w == 1024)
    }

    @Test("Fractional extent rounds to nearest int")
    func fractionalExtentRounds() throws {
        let (h, w) = try QwenVL.intExtent(CGSize(width: 1024.7, height: 768.3))
        #expect(h == 768)
        #expect(w == 1025)
    }

    @Test("Infinite height rejects with imageProcessingFailure")
    func infiniteHeightRejects() throws {
        #expect(throws: VLMError.self) {
            _ = try QwenVL.intExtent(CGSize(width: 1024, height: CGFloat.infinity))
        }
    }

    @Test("Infinite width rejects with imageProcessingFailure")
    func infiniteWidthRejects() throws {
        #expect(throws: VLMError.self) {
            _ = try QwenVL.intExtent(CGSize(width: CGFloat.infinity, height: 768))
        }
    }

    @Test("NaN height rejects")
    func nanHeightRejects() throws {
        #expect(throws: VLMError.self) {
            _ = try QwenVL.intExtent(CGSize(width: 1024, height: CGFloat.nan))
        }
    }

    @Test("Zero height rejects")
    func zeroHeightRejects() throws {
        #expect(throws: VLMError.self) {
            _ = try QwenVL.intExtent(CGSize(width: 1024, height: 0))
        }
    }

    @Test("Zero width rejects")
    func zeroWidthRejects() throws {
        #expect(throws: VLMError.self) {
            _ = try QwenVL.intExtent(CGSize(width: 0, height: 768))
        }
    }

    @Test("Negative dimensions reject")
    func negativeDimensionsReject() throws {
        #expect(throws: VLMError.self) {
            _ = try QwenVL.intExtent(CGSize(width: -1024, height: 768))
        }
    }

    // MARK: - Adopter source coverage

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// Pin that every Qwen-family VLM (the adopters of `QwenVL.targetSize`)
    /// routes through `QwenVL.intExtent` instead of raw `Int(size.height)`.
    @Test("Qwen2VL.swift uses QwenVL.intExtent")
    func qwen2vlAdopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Qwen2VL.swift")
        #expect(source.contains("try QwenVL.intExtent("))
        #expect(!source.contains("height: Int(size.height), width: Int(size.width)"))
    }

    @Test("Qwen25VL.swift uses QwenVL.intExtent")
    func qwen25vlAdopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Qwen25VL.swift")
        #expect(source.contains("try QwenVL.intExtent("))
        #expect(!source.contains("height: Int(size.height), width: Int(size.width)"))
    }

    @Test("Qwen3VL.swift uses QwenVL.intExtent")
    func qwen3vlAdopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Qwen3VL.swift")
        #expect(source.contains("try QwenVL.intExtent("))
        #expect(!source.contains("height: Int(extent.height)"))
        #expect(!source.contains("height: Int(size.height)"))
    }

    @Test("Zaya1VL.swift uses QwenVL.intExtent")
    func zaya1vlAdopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Zaya1VL.swift")
        #expect(source.contains("try QwenVL.intExtent("))
        #expect(!source.contains("height: Int(size.height), width: Int(size.width)"))
    }

    @Test("GlmOcr.swift uses QwenVL.intExtent")
    func glmOcrAdopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/GlmOcr.swift")
        #expect(source.contains("try QwenVL.intExtent("))
        #expect(!source.contains("height: Int(size.height), width: Int(size.width)"))
    }

    @Test("Gemma4.swift uses QwenVL.intExtent for its image preprocessing too")
    func gemma4Adopts() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Gemma4.swift")
        #expect(source.contains("try QwenVL.intExtent(ci.extent.size)"))
        // The prior raw `Int(ci.extent.width)` form (which traps on infinity) must NOT come back.
        #expect(!source.contains("(Int(ci.extent.width), Int(ci.extent.height))"))
    }
}
