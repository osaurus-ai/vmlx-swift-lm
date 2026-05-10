// Pin the Gemma3 + Gemma4 maskedScatter throwing contract.
//
// Background (see `docs/GEMMA4-DEEP-TRACE-2026-05-10.md` §7.3):
// Both Gemma3 and Gemma4 had a private `maskedScatter` helper that
// `fatalError`'d when `imageSeqLength` (preprocessor_config) didn't match
// the vision tower's output token count. This is a config/processor-stamp
// drift the caller should be able to surface — a process abort on first
// image is the wrong failure mode for a server runtime.
//
// 2026-05-10 fix: both helpers now throw `VLMError.processing`, and the
// error cascades through `prepareInputsForMultimodal` /
// `getInputEmbeddings` (Gemma3) / `prepare` (Gemma4) so the call site
// can recover.
//
// This test pins the source contract via static-coverage (no MLX
// runtime required, mirroring the source-coverage test pattern used elsewhere.

import Foundation
import Testing

@Suite("Gemma3 + Gemma4 maskedScatter throwing-contract source coverage")
struct GemmaMaskedScatterThrowsTests {

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// Gemma4: `private func maskedScatter(...) throws -> MLXArray` AND its
    /// caller in `Gemma4.prepare` uses `try maskedScatter(...)`. The
    /// previous `fatalError("Gemma4 maskedScatter: size mismatch...")`
    /// must NOT come back.
    @Test("Gemma4 maskedScatter throws VLMError instead of fatalError on size mismatch")
    func gemma4MaskedScatterThrows() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Gemma4.swift")

        // Helper signature is `throws`.
        #expect(
            source.contains("private func maskedScatter(input: MLXArray, mask: MLXArray, source: MLXArray) throws -> MLXArray"),
            "Gemma4 maskedScatter must throw — see deep-trace §7.3.")

        // Mismatch path throws VLMError.processing.
        #expect(
            source.contains("throw VLMError.processing(") &&
            source.contains("Gemma4 maskedScatter: size mismatch"),
            "Mismatch path must throw VLMError.processing with the diagnostic message.")

        // The prior fatalError is gone.
        #expect(
            !source.contains(#"fatalError("#) || !source.contains("Gemma4 maskedScatter: size mismatch"),
            "Gemma4 maskedScatter must not reintroduce fatalError on size mismatch.")

        // Caller uses `try`.
        #expect(
            source.contains("try maskedScatter(input: emb, mask: imgMaskExp, source: imgFeatures)"),
            "Gemma4.prepare's maskedScatter call site must use `try`.")
    }

    /// Gemma3: same contract — `maskedScatter` throws,
    /// `prepareInputsForMultimodal` throws, `getInputEmbeddings` throws,
    /// and `Gemma3.prepare`'s caller uses `try`.
    @Test("Gemma3 maskedScatter cascade throws cleanly through prepareInputsForMultimodal + getInputEmbeddings")
    func gemma3MaskedScatterThrowsCascade() throws {
        let source = try Self.source("Libraries/MLXVLM/Models/Gemma3.swift")

        // maskedScatter signature has `throws`.
        #expect(
            source.contains("private func maskedScatter(") &&
            source.contains("scaledImageFeatures: MLXArray\n) throws -> MLXArray"),
            "Gemma3 maskedScatter must throw — see deep-trace §7.3.")

        // Mismatch path throws VLMError.processing with diagnostic.
        #expect(
            source.contains("throw VLMError.processing(") &&
            source.contains("Gemma3 maskedScatter: size mismatch"),
            "Gemma3 maskedScatter mismatch path must throw VLMError.processing.")

        // The prior fatalError is gone.
        #expect(
            !source.contains("Critical error in maskedScatter"),
            "Gemma3 maskedScatter must not reintroduce the prior `Critical error in maskedScatter` fatalError block.")

        // prepareInputsForMultimodal cascades the throw.
        #expect(
            source.contains("private func prepareInputsForMultimodal(") &&
            source.contains("attentionMask: MLXArray?\n    ) throws -> (MLXArray, MLXArray?)"),
            "Gemma3 prepareInputsForMultimodal must throw to cascade the maskedScatter throw.")

        // getInputEmbeddings cascades the throw.
        #expect(
            source.contains("private func getInputEmbeddings(") &&
            source.contains("mask: MLXArray? = nil\n    ) throws -> (MLXArray, MLXArray?)"),
            "Gemma3 getInputEmbeddings must throw to cascade the maskedScatter throw.")

        // Caller uses `try`.
        #expect(
            source.contains("let (inputEmbeddings, _) = try getInputEmbeddings("),
            "Gemma3.prepare's getInputEmbeddings call must use `try`.")
        #expect(
            source.contains("let (finalEmbedding, finalAttentionMask4d) = try prepareInputsForMultimodal("),
            "Gemma3 getInputEmbeddings's prepareInputsForMultimodal call must use `try`.")
        #expect(
            source.contains("finalEmbedding = try maskedScatter("),
            "Gemma3 prepareInputsForMultimodal's maskedScatter call must use `try`.")
    }
}
