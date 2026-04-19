// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 3 probe — does the existing `RotatingKVCache` survive being
// captured by `MLX.compile(...)`, or does its buffer-growth path
// (`self.keys = concatenated(...)`) or its ring-buffer rotation break the
// trace?
//
// Structured like the Stage 2 probe (`TurboQuantCompileProbeTests`): first
// check whether any compile attempt works at all, then compare compiled vs
// uncompiled logits for numerical equivalence.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

class RotatingKVCacheCompileProbeTests: XCTestCase {

    private func skipIfCompileUnsafe() throws {
        guard HardwareInfo.isCompiledDecodeSupported else {
            throw XCTSkip("Compiled decode not supported on this hardware")
        }
    }

    private func makeModelAndPrompt(promptLen: Int = 8) -> (any LanguageModel, MLXArray) {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 200, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)
        let prompt = MLXArray(Int32(1) ..< Int32(1 + Int32(promptLen)))
        return (model, prompt)
    }

    private func makeRotatingCache(
        model: any LanguageModel, prompt: MLXArray, maxSize: Int, keep: Int = 0
    ) -> [KVCache] {
        let cache: [KVCache] = (0 ..< 4).map { _ in
            RotatingKVCache(maxSize: maxSize, keep: keep, step: 256) as KVCache
        }
        let _ = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: cache, state: nil)
        MLX.eval(cache)
        return cache
    }

    /// Probe 1: compiled RotatingKVCache logits match uncompiled reference
    /// on the same fresh prefill state. Linear (pre-wrap) segment only.
    func testRotatingCompileLinearSegment() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 8)

        // maxSize=256 is large enough that short decode stays pre-wrap.
        let refCache = makeRotatingCache(model: model, prompt: prompt, maxSize: 256)
        let compiledCache = makeRotatingCache(model: model, prompt: prompt, maxSize: 256)

        let decodeToken = MLXArray([Int32(42)])
        let uncompiled = model(
            LMInput.Text(tokens: decodeToken)[text: .newAxis],
            cache: refCache, state: nil)
        MLX.eval(uncompiled.logits)
        let uncompiledLogits = uncompiled.logits[0 ..< 1, 0, 0...]
        MLX.eval(uncompiledLogits)

        let capturedModel = model
        let captured = compiledCache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }
        let compiledResult = forward([decodeToken])
        XCTAssertEqual(compiledResult.count, 1,
            "Compiled forward over RotatingKVCache should return one output")
        MLX.eval(compiledResult[0])
        XCTAssertEqual(compiledResult[0].shape.count, 3,
            "Shape should be [1, 1, V]; got \(compiledResult[0].shape)")

        let compiledLogits = compiledResult[0][0 ..< 1, 0, 0...]
        MLX.eval(compiledLogits)

        let compiledMax = compiledLogits.abs().max().item(Float.self)
        XCTAssertTrue(compiledMax.isFinite, "Compiled logits must be finite")

        let diff = (compiledLogits - uncompiledLogits).abs().max().item(Float.self)
        let refMax = uncompiledLogits.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? diff / refMax : diff

        print("""

            === Stage 3 probe (linear): RotatingKVCache compile ===
              uncompiled max abs: \(refMax)
              compiled max abs: \(compiledMax)
              abs diff: \(diff)
              relative diff: \(relativeDiff)
              tolerance: 0.05 (5%)
            ======================================================

            """)

        XCTAssertLessThan(relativeDiff, 0.05,
            "Compiled Rotating logits diverge from uncompiled by \(relativeDiff). "
            + "If this fails, Stage 3 needs CompilableRotatingKVCache.")
    }

    /// Probe 3 (DIAGNOSTIC — EXPECTED TO SHOW DRIFT): decode that crosses
    /// the `step`-chunk growth boundary. `RotatingKVCache.updateInPlace`
    /// grows the buffer via `self.keys = concatenated([currentKeys, newK],
    /// axis: 2)` which REBINDS the property. Compile traces capture the
    /// old object; if growth triggers mid-trace, the trace goes stale.
    ///
    /// This test currently documents the drift rather than asserting a
    /// threshold — Stage 3 is not shipped yet. Once `CompilableRotatingKVCache`
    /// lands, the `XCTAssertLessThan` at the bottom should be re-enabled
    /// to enforce the closeness contract.
    func testRotatingCompileBufferGrowth() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 4)

        // step=8 + prompt_len=4 means prefill allocates 8-token chunks.
        // After prefill at offset 4, buffer.dim(2)=8. Decode position 4..7
        // fits. Decode position 8 requires a new chunk → self.keys =
        // concatenated(...) → property rebind.
        let refCache: [KVCache] = (0 ..< 4).map { _ in
            RotatingKVCache(maxSize: 256, keep: 0, step: 8) as KVCache
        }
        let compiledCache: [KVCache] = (0 ..< 4).map { _ in
            RotatingKVCache(maxSize: 256, keep: 0, step: 8) as KVCache
        }
        let _ = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: refCache, state: nil)
        MLX.eval(refCache)
        let _ = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: compiledCache, state: nil)
        MLX.eval(compiledCache)

        // Uncompiled: run 10 decode steps, collect last-token logits.
        var uncompiledLastLogits: MLXArray?
        var currentToken = MLXArray([Int32(42)])
        for step in 0 ..< 10 {
            let result = model(
                LMInput.Text(tokens: currentToken)[text: .newAxis],
                cache: refCache, state: nil)
            MLX.eval(result.logits)
            let logits = result.logits[0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == 9 {
                uncompiledLastLogits = logits
            }
            // Feed the greedy next token
            currentToken = argMax(logits, axis: -1)
            MLX.eval(currentToken)
        }

        // Compiled: same 10 decode steps through the compiled closure.
        let capturedModel = model
        let captured = compiledCache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }

        var compiledLastLogits: MLXArray?
        var compiledToken = MLXArray([Int32(42)])
        for step in 0 ..< 10 {
            let result = forward([compiledToken])
            MLX.eval(result[0])
            let logits = result[0][0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == 9 {
                compiledLastLogits = logits
            }
            compiledToken = argMax(logits, axis: -1)
            MLX.eval(compiledToken)
        }

        guard let unc = uncompiledLastLogits, let cmp = compiledLastLogits else {
            XCTFail("Expected both paths to produce last-step logits")
            return
        }

        let diff = (cmp - unc).abs().max().item(Float.self)
        let refMax = unc.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? diff / refMax : diff

        print("""

            === Stage 3 probe (growth): RotatingKVCache after buffer-growth ===
              10 decode steps crossing step-boundary (step=8, prompt=4)
              uncompiled last-step max abs: \(refMax)
              compiled last-step max abs: \(cmp.abs().max().item(Float.self))
              abs diff: \(diff)
              relative diff: \(relativeDiff)
              tolerance: 0.05 (5%)

            If this fails, the `self.keys = concatenated(...)` rebind in
            updateInPlace breaks the compile trace under long decodes.
            Stage 3 must then build CompilableRotatingKVCache.
            ===================================================================

            """)

        // Stage 3 not yet shipped: diagnostic-only assertion that still
        // tests something useful — the compiled path shouldn't explode
        // into NaN / inf. Numerical equivalence comes back once Stage 3
        // ships. Current observed drift on tiny test model: ~30% relative.
        let compiledMax = cmp.abs().max().item(Float.self)
        XCTAssertTrue(compiledMax.isFinite,
            "Compiled Rotating logits must at least be finite even with drift")
        XCTAssertGreaterThan(relativeDiff, 0.01,
            "Expected drift >1% (Stage 3 blocker); got \(relativeDiff). "
            + "If this unexpectedly passes, the buffer-growth blocker may have "
            + "resolved on its own — re-enable the < 0.05 assertion.")
    }

    /// Probe 4 (DIAGNOSTIC — EXPECTED TO SHOW DRIFT): decode that crosses
    /// the ring-buffer wrap point
    /// (offset >= maxCacheSize → `idx = keep`). This is the trickiest
    /// semantic — after wrap, the buffer is no longer in logical temporal
    /// order, and the model's attention mask has to handle the ring layout.
    func testRotatingCompileWrapAround() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 4)

        // maxSize=8 + prompt=4 means decode position 4..7 stays pre-wrap.
        // Decode position 8 hits the wrap: idx resets to keep (0 here)
        // and overwrites the oldest position.
        let refCache: [KVCache] = (0 ..< 4).map { _ in
            RotatingKVCache(maxSize: 8, keep: 0, step: 256) as KVCache
        }
        let compiledCache: [KVCache] = (0 ..< 4).map { _ in
            RotatingKVCache(maxSize: 8, keep: 0, step: 256) as KVCache
        }
        let _ = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: refCache, state: nil)
        MLX.eval(refCache)
        let _ = model(
            LMInput.Text(tokens: prompt)[text: .newAxis],
            cache: compiledCache, state: nil)
        MLX.eval(compiledCache)

        // 10 decode steps — crosses the wrap point at decode step 5
        // (offset goes 4→5→...→13, wraps at 8→idx=0).
        var uncompiledLastLogits: MLXArray?
        var currentToken = MLXArray([Int32(42)])
        for step in 0 ..< 10 {
            let result = model(
                LMInput.Text(tokens: currentToken)[text: .newAxis],
                cache: refCache, state: nil)
            MLX.eval(result.logits)
            let logits = result.logits[0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == 9 { uncompiledLastLogits = logits }
            currentToken = argMax(logits, axis: -1)
            MLX.eval(currentToken)
        }

        let capturedModel = model
        let captured = compiledCache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }

        var compiledLastLogits: MLXArray?
        var compiledToken = MLXArray([Int32(42)])
        for step in 0 ..< 10 {
            let result = forward([compiledToken])
            MLX.eval(result[0])
            let logits = result[0][0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == 9 { compiledLastLogits = logits }
            compiledToken = argMax(logits, axis: -1)
            MLX.eval(compiledToken)
        }

        guard let unc = uncompiledLastLogits, let cmp = compiledLastLogits else {
            XCTFail("Expected both paths to produce last-step logits")
            return
        }

        let diff = (cmp - unc).abs().max().item(Float.self)
        let refMax = unc.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? diff / refMax : diff

        print("""

            === Stage 3 probe (wrap): RotatingKVCache after wrap-around ===
              10 decode steps crossing maxCacheSize=8 (prompt=4)
              uncompiled last-step max abs: \(refMax)
              compiled last-step max abs: \(cmp.abs().max().item(Float.self))
              abs diff: \(diff)
              relative diff: \(relativeDiff)
              tolerance: 0.05 (5%)

            If this fails, ring-buffer rotation (idx = keep when idx ==
            maxCacheSize) breaks the compile trace. Stage 3 must build
            CompilableRotatingKVCache with traceable idx (MLXArray) and
            traceable wrap-around logic.
            ===============================================================

            """)

        // Stage 3 not yet shipped: diagnostic-only assertion. Wrap-around
        // drift is consistently much larger than growth drift — observed
        // ~68% relative on tiny test model.
        let compiledMax = cmp.abs().max().item(Float.self)
        XCTAssertTrue(compiledMax.isFinite,
            "Compiled Rotating logits must at least be finite post-wrap")
        XCTAssertGreaterThan(relativeDiff, 0.05,
            "Expected substantial drift post-wrap (Stage 3 blocker); got \(relativeDiff). "
            + "If this unexpectedly passes, re-examine the wrap semantics.")
    }

    /// Probe 2: cache advances between compiled calls (non-idempotent).
    /// Distinguishes "trace captured stale state" from "trace works".
    func testRotatingCompileCacheAdvances() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 8)
        let cache = makeRotatingCache(model: model, prompt: prompt, maxSize: 256)

        let capturedModel = model
        let captured = cache
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured,
                state: nil)
            return [result.logits]
        }

        let tokenA = MLXArray([Int32(7)])
        let tokenB = MLXArray([Int32(99)])

        let r1 = forward([tokenA])
        MLX.eval(r1[0])
        let logits1 = r1[0][0 ..< 1, 0, 0...]
        MLX.eval(logits1)

        let r2 = forward([tokenB])
        MLX.eval(r2[0])
        let logits2 = r2[0][0 ..< 1, 0, 0...]
        MLX.eval(logits2)

        let absDiff = (logits2 - logits1).abs().max().item(Float.self)
        print("""

            === Stage 3 probe (advancement): RotatingKVCache ===
              max abs diff between calls: \(absDiff)
            Non-zero diff means cache state advances through the trace.
            ====================================================

            """)

        XCTAssertGreaterThan(absDiff, 1e-6,
            "RotatingKVCache under compile: cache state did not advance "
            + "between successive compiled calls.")
    }
}
