// Pin Gemma4 audio-input guard + image-token-id resolution contracts via
// source-coverage tests (no MLX runtime required).
//
// Background (see `docs/GEMMA4-DEEP-TRACE-2026-05-10.md` §7.4 + §7.5):
//
// §7.5 Audio guard. `Gemma4.sanitize` drops `audio_tower.*` and
// `embed_audio.*` weights silently and there's no audio module wired
// in `init`, so a caller passing `LMInput.audio` to `Gemma4.prepare`
// would have its waveform silently ignored. The audio guard at the
// top of `Gemma4.prepare` surfaces this as a clean
// `VLMError.processing` so the call site sees the failure
// immediately. This test pins the guard's presence + message contract.
//
// §7.4 Image-token-id resolution. The processor used to call
// `tokenizer.encode("<|image|>").last ?? 258880` — fragile because
// `encode()` defaults to `addSpecialTokens: true`, which can prepend
// BOS on some tokenizers and silently change the encoded length.
// `convertTokenToId("<|image|>")` skips that special-token machinery
// entirely and looks the special token up directly in the vocab.
// This test pins the new lookup form so a future regression that
// reintroduces `encode().last` is caught at test time.
//
// Source-coverage instead of full forward: `Gemma4.init` triggers
// MLX Metal allocations (Embedding + layers + vision tower), and the
// SwiftPM test runner here cannot load the mlx-swift `default.metallib`
// resource. The guard contract is purely structural — pinning the
// source pattern is sufficient to catch regressions.

import Foundation
import Testing

@Suite("Gemma4 audio guard + image-token-id resolution source coverage")
struct Gemma4AudioGuardTests {

    /// Resolves the absolute path of `Libraries/MLXVLM/Models/Gemma4.swift`
    /// relative to the test source file.
    private static func gemma4Source() throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // .../Tests/MLXLMTests
            .deletingLastPathComponent()  // .../Tests
            .deletingLastPathComponent()  // repo root
        let url = repo
            .appendingPathComponent("Libraries")
            .appendingPathComponent("MLXVLM")
            .appendingPathComponent("Models")
            .appendingPathComponent("Gemma4.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// Pins the audio guard at the top of `Gemma4.prepare`.
    /// The guard MUST throw a `VLMError.processing` that mentions
    /// `LMInput.audio` (so the caller can grep for the failure cause)
    /// and explains why audio is rejected (so they don't loop on
    /// retry).
    @Test("Gemma4.prepare contains the LMInput.audio guard")
    func audioGuardIsPresent() throws {
        let source = try Self.gemma4Source()

        // The guard fires before any forward op.
        #expect(source.contains("if input.audio != nil {"),
            "Gemma4.prepare must check `input.audio != nil` before proceeding.")

        // The throw is a VLMError.processing for graceful caller handling.
        #expect(source.contains("throw VLMError.processing("),
            "Audio guard must surface as VLMError.processing, not fatalError or silent fallthrough.")

        // The error message names the field that's rejected so the
        // caller sees a specific diagnostic.
        #expect(source.contains("LMInput.audio must be nil"),
            "Audio guard message must name `LMInput.audio` so the call site sees what to fix.")

        // The message explains WHY (audio_tower weights are sanitized
        // away) so the caller doesn't loop on retry.
        #expect(
            source.contains("audio_tower.*") || source.contains("audio_tower.\\*"),
            "Audio guard rationale should reference `audio_tower.*` so a future maintainer sees the connection to sanitize.")
    }

    /// Pins the image-token-id resolution path in `Gemma4Processor.prepare`.
    @Test("Gemma4Processor uses convertTokenToId for <|image|> instead of encode().last")
    func imageTokenIdUsesConvertTokenToId() throws {
        let source = try Self.gemma4Source()

        // The lookup must use the special-token map directly.
        #expect(
            source.contains("tokenizer.convertTokenToId(\"<|image|>\")"),
            "Gemma4Processor must look up `<|image|>` via `convertTokenToId` so the result is independent of encode()'s addSpecialTokens default.")

        // The fragile `encode("<|image|>").last` form must NOT come back.
        #expect(
            !source.contains("tokenizer.encode(text: \"<|image|>\").last"),
            "Gemma4Processor must not reintroduce `encode(text: \"<|image|>\").last` — see deep-trace §7.4 for why.")

        // Sanity: the 258880 fallback is preserved (covers tokenizers
        // that don't expose `<|image|>` as an addable special token).
        #expect(source.contains("?? 258880"),
            "Gemma4Processor must keep the 258880 fallback for tokenizers without an `<|image|>` special token.")
    }
}
