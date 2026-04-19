// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 3 tests for `CompilableRotatingKVCache`.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

class CompilableRotatingKVCacheTests: XCTestCase {

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

    private func makeMatchedCaches(
        model: any LanguageModel, prompt: MLXArray,
        maxSize: Int, keep: Int = 0, step: Int = 256
    ) -> (ref: [KVCache], compiled: [KVCache]) {
        func prefill() -> [KVCache] {
            let cache: [KVCache] = (0 ..< 4).map { _ in
                RotatingKVCache(maxSize: maxSize, keep: keep, step: step) as KVCache
            }
            let _ = model(
                LMInput.Text(tokens: prompt)[text: .newAxis],
                cache: cache, state: nil)
            MLX.eval(cache)
            return cache
        }

        let refCache = prefill()
        let rawBase = prefill()
        let compiledCache: [KVCache] = rawBase.map { layer in
            CompilableRotatingKVCache(from: layer as! RotatingKVCache) as KVCache
        }
        MLX.eval(compiledCache)
        return (refCache, compiledCache)
    }

    /// Sanity: promote from a populated RotatingKVCache and verify the
    /// compiled variant has the right identity + shape.
    func testPromotionSanity() throws {
        let src = RotatingKVCache(maxSize: 64, keep: 0, step: 256)
        for _ in 0 ..< 10 {
            _ = src.update(
                keys: MLXArray.ones([1, 4, 1, 16]),
                values: MLXArray.ones([1, 4, 1, 16]))
        }
        XCTAssertEqual(src.offset, 10)

        let cmp = CompilableRotatingKVCache(from: src)
        XCTAssertEqual(cmp.offset, 10, "Offset carries over")
        XCTAssertEqual(cmp.idxArray.shape, [1])
        XCTAssertEqual(cmp.offsetArray.shape, [1])

        // Inspect keys buffer via innerState — `keys` itself is module-
        // private. The first entry of CompilableRotatingKVCache.innerState
        // is the keys buffer (see the override).
        let state = cmp.innerState()
        XCTAssertGreaterThanOrEqual(state.count, 2,
            "CompilableRotatingKVCache.innerState should include keys + values")
        XCTAssertEqual(state[0].dim(2), 64,
            "Buffer should be pre-allocated to maxCacheSize")
    }

    /// Linear phase, single-step.
    func testLinearSingleStepMatchesUncompiled() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 8)
        let caches = makeMatchedCaches(
            model: model, prompt: prompt, maxSize: 256, keep: 0, step: 256)

        let decodeToken = MLXArray([Int32(42)])

        let uncompiled = model(
            LMInput.Text(tokens: decodeToken)[text: .newAxis],
            cache: caches.ref, state: nil)
        MLX.eval(uncompiled.logits)
        let refLogits = uncompiled.logits[0 ..< 1, 0, 0...]
        MLX.eval(refLogits)

        let capturedModel = model
        let captured = caches.compiled
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured, state: nil)
            return [result.logits]
        }
        let compiled = forward([decodeToken])
        MLX.eval(compiled[0])
        let cmpLogits = compiled[0][0 ..< 1, 0, 0...]
        MLX.eval(cmpLogits)

        let diff = (cmpLogits - refLogits).abs().max().item(Float.self)
        let refMax = refLogits.abs().max().item(Float.self)
        let rel = refMax > 0 ? diff / refMax : diff
        print("Stage 3 linear single-step: refMax=\(refMax) diff=\(diff) rel=\(rel)")
        XCTAssertLessThan(rel, 0.05,
            "Stage 3 linear single-step should match uncompiled within 5%, got \(rel)")
    }

    /// Growth-boundary crossing: decode past the initial `step`-chunk.
    /// Probe measured ~30% drift on un-promoted cache; Stage 3 should close it.
    func testGrowthBoundaryMatchesUncompiled() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 4)

        let caches = makeMatchedCaches(
            model: model, prompt: prompt, maxSize: 256, keep: 0, step: 8)

        let fixedToken = MLXArray([Int32(42)])
        let steps = 10

        var refLogits: MLXArray?
        for step in 0 ..< steps {
            let result = model(
                LMInput.Text(tokens: fixedToken)[text: .newAxis],
                cache: caches.ref, state: nil)
            MLX.eval(result.logits)
            let l = result.logits[0 ..< 1, 0, 0...]
            MLX.eval(l)
            if step == steps - 1 { refLogits = l }
        }

        let capturedModel = model
        let captured = caches.compiled
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured, state: nil)
            return [result.logits]
        }

        var cmpLogits: MLXArray?
        for step in 0 ..< steps {
            let r = forward([fixedToken])
            MLX.eval(r[0])
            let l = r[0][0 ..< 1, 0, 0...]
            MLX.eval(l)
            if step == steps - 1 { cmpLogits = l }
        }

        guard let ref = refLogits, let cmp = cmpLogits else {
            XCTFail("Both paths should produce last-step logits")
            return
        }

        let diff = (cmp - ref).abs().max().item(Float.self)
        let refMax = ref.abs().max().item(Float.self)
        let rel = refMax > 0 ? diff / refMax : diff
        print("Stage 3 growth-boundary (10 steps, step=8): refMax=\(refMax) diff=\(diff) rel=\(rel)")

        // ITER 21 FIX (RoPE routing): drift collapsed from ~30% pre-iter-13,
        // to ~8% iter-13/20, to ~5e-7 iter-21. The `applyRotaryPosition`
        // fix in iter 21 applies to CompilableRotatingKVCache too —
        // both fell through to Int `cache.offset` before.
        XCTAssertLessThan(rel, 0.05,
            "Stage 3 growth-boundary should match uncompiled within 5%: got \(rel)")
    }

    /// Wrap-around: decode past `maxCacheSize` so the ring wraps.
    func testWrapAroundMatchesUncompiled() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt(promptLen: 4)

        let caches = makeMatchedCaches(
            model: model, prompt: prompt, maxSize: 16, keep: 0, step: 256)

        let fixedToken = MLXArray([Int32(42)])
        let steps = 20

        var refLogits: MLXArray?
        for step in 0 ..< steps {
            let result = model(
                LMInput.Text(tokens: fixedToken)[text: .newAxis],
                cache: caches.ref, state: nil)
            MLX.eval(result.logits)
            let l = result.logits[0 ..< 1, 0, 0...]
            MLX.eval(l)
            if step == steps - 1 { refLogits = l }
        }

        let capturedModel = model
        let captured = caches.compiled
        let forward: @Sendable ([MLXArray]) -> [MLXArray] = compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0])[text: .newAxis],
                cache: captured, state: nil)
            return [result.logits]
        }

        var cmpLogits: MLXArray?
        for step in 0 ..< steps {
            let r = forward([fixedToken])
            MLX.eval(r[0])
            let l = r[0][0 ..< 1, 0, 0...]
            MLX.eval(l)
            if step == steps - 1 { cmpLogits = l }
        }

        guard let ref = refLogits, let cmp = cmpLogits else {
            XCTFail("Both paths should produce last-step logits")
            return
        }

        let diff = (cmp - ref).abs().max().item(Float.self)
        let refMax = ref.abs().max().item(Float.self)
        let rel = refMax > 0 ? diff / refMax : diff
        print("Stage 3 wrap (20 steps, maxSize=16, prompt=4): refMax=\(refMax) diff=\(diff) rel=\(rel)")

        // ITER 21 FIX: wrap-around drift closed from ~68% pre-fix to ~5e-7
        // post-iter-21 (RoPE routing fix in RoPEApplication.swift).
        XCTAssertLessThan(rel, 0.05,
            "Stage 3 wrap-around should match uncompiled within 5%: got \(rel)")
    }
}
