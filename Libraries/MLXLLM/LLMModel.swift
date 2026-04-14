// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512

        // Work on a flat 1D view of the tokens internally so the slicing math
        // below is dimension-independent. Callers may pass tokens either as
        // 1D `[T]` (legacy) or 2D `[B=1, T]` (TokenIterator / Bench / Osaurus
        // path). The old code used `y[.newAxis, ..<step]` and `y[step...]`
        // which silently sliced the WRONG axis when the input was 2D — the
        // `..<step` would apply to the batch dim (size 1) leaving the chunk
        // shape unchanged, and `y[step...]` would produce an empty `[0, T]`
        // tensor which then crashed the next forward pass with
        // `[reshape] Cannot infer the shape of an empty array`.
        let originalShape = input.text.tokens.shape
        var flatTokens = input.text.tokens.reshaped([-1])
        var flatMask: MLXArray? = nil
        if let m = input.text.mask {
            flatMask = m.ndim >= 2 ? m.reshaped([-1]) : m
        }

        // Prepare the prompt in chunks if larger than the prefill size.
        // Clear Metal cache between chunks to reduce memory pressure,
        // matching Python mlx-lm behavior. Critical for MoE models.
        while flatTokens.size > prefillStepSize {
            // Build a [1, prefillStepSize] chunk for the model forward pass.
            let chunkTokens = flatTokens[..<prefillStepSize][.newAxis, 0...]
            let chunkMask = flatMask.map { $0[..<prefillStepSize] }
            let chunkText = LMInput.Text(tokens: chunkTokens, mask: chunkMask)
            _ = self(chunkText, cache: cache.isEmpty ? nil : cache, state: nil)
            MLX.eval(cache)
            flatTokens = flatTokens[prefillStepSize...]
            if let m = flatMask { flatMask = m[prefillStepSize...] }
            Memory.clearCache()
        }

        // Return the remaining tokens in the original shape (1D or 2D),
        // so downstream code that inspects `.ndim` sees what it expects.
        let remaining: MLXArray
        if originalShape.count >= 2 {
            remaining = flatTokens[.newAxis, 0...]
        } else {
            remaining = flatTokens
        }
        return .tokens(LMInput.Text(tokens: remaining, mask: flatMask))
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
