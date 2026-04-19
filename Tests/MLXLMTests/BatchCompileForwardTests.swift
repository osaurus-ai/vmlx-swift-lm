// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 1B.1 + 1B.2 tests for `BatchCompile.compileForward(model:cacheRef:)`.
//
// Stage 1B.1 scope (already green):
//  - Precondition validation and closure-construction smoke test.
//  - Family-gate alignment sanity.
//
// Stage 1B.2 scope (this file's additions below):
//  - End-to-end invocation: prefill → clone cache → compile → invoke.
//  - Logit correctness vs uncompiled reference within FP tolerance.
//  - Shape + finiteness checks.
//
// The invocation tests walk the full model compile path that historically
// tripped MLX#3329 (`compiledState.callsToFill[0]` out of range) on some
// macOS Tahoe Metal driver builds. The issue was confirmed fixed on M4 Max
// per the HardwareInfo flag history (2026-04-13 re-enable). If a new
// regression appears, these tests will catch it here first.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import XCTest

// MARK: - Precondition tests

@Suite("BatchCompile.compileForward preconditions", .serialized)
struct BatchCompileForwardPreconditionTests {

    /// Build a tiny Llama-shaped model just to have a `LanguageModel` handle.
    /// We never actually invoke the returned closure — we only call
    /// `compileForward` to confirm it constructs.
    private func makeTinyModel() -> (any LanguageModel, [KVCache]) {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)
        let cache = model.newCache(parameters: nil)
        return (model, cache)
    }

    @Test("compileForward returns a closure when all layers are CompilableKVCache")
    func testValidInputReturnsClosure() {
        let (model, _) = makeTinyModel()
        // Create a fresh CompilableKVCache per layer — shape inferred on
        // first invocation.
        let cacheRef: [KVCache] = (0 ..< 2).map { _ in CompilableKVCache(maxLength: 128) }

        // Just constructs the closure; does not invoke it. This path is
        // safe on all hardware — the compile call returns immediately
        // without tracing until the closure is first called.
        let forward = BatchCompile.compileForward(model: model, cacheRef: cacheRef)

        _ = forward  // silence "unused" warning on unused-binding
        #expect(true, "compileForward constructed a closure without crashing")
    }
}

// MARK: - Family-gate sanity (XCTest)

/// Precondition failures are fatal in release builds. These tests validate
/// structure we can inspect without tripping them — but they do confirm the
/// helpers behave consistently with the preconditions' stated contracts by
/// inspecting type-family utilities that share the same gate.
class BatchCompileForwardSanityTests: XCTestCase {

    /// All-CompilableKVCache arrays pass the family gate documented in
    /// `compileForward`'s precondition. Keeps precondition and classifier
    /// in agreement.
    func testFamilyGateAlignment() {
        let compilables: [KVCache] = (0 ..< 2).map { _ in CompilableKVCache(maxLength: 128) }
        XCTAssertEqual(CacheFamily.classify(compilables), .simple,
            "All-Compilable arrays classify as .simple — the family compileForward wants")
        XCTAssertTrue(compilables.allSatisfy { $0 is CompilableKVCache },
            "Matches the stricter compileForward precondition")
    }

    /// Mixed CompilableKVCache + KVCacheSimple still classifies as `.simple`
    /// but would fail the stricter `allSatisfy CompilableKVCache` precondition
    /// that `compileForward` uses. Keeps the two gates' semantics documented.
    func testFamilyGateVsPrecondition() {
        let mixed: [KVCache] = [
            CompilableKVCache(maxLength: 128),
            KVCacheSimple(),
        ]
        XCTAssertEqual(CacheFamily.classify(mixed), .simple,
            "CacheFamily.classify treats CompilableKVCache and KVCacheSimple as one family")
        XCTAssertFalse(mixed.allSatisfy { $0 is CompilableKVCache },
            "compileForward's stricter precondition would trap on this mix — caller must"
            + " promote ALL layers, not just some")
    }

    /// Confirms the family enum advertises compile-eligibility correctly for
    /// the current stage. A regression that flipped `.simple.isCompileEligibleAtCurrentStage`
    /// to false would silently bypass every compile site.
    func testSimpleFamilyIsEligible() {
        XCTAssertTrue(
            CacheFamily.simple.isCompileEligibleAtCurrentStage,
            "Stage 1 requires .simple to be compile-eligible. If this flips,"
            + " BatchCompile.compileForward has nothing to fire on."
        )
    }
}

// MARK: - Stage 1B.2: End-to-end compile invocation + logit compare

/// Integration-level verification that `BatchCompile.compileForward` produces
/// output numerically equivalent to the uncompiled forward path.
///
/// ## Test pattern (all tests in this class follow this structure)
///
/// 1. Build a tiny quantised Llama model (4 layers / hiddenSize 64 / 4 kv heads).
/// 2. Run a prefill on the uncompiled path with `KVCacheSimple` layers →
///    capture the post-prefill cache state.
/// 3. Clone the cache for two parallel paths:
///    - **Reference path:** keep as `KVCacheSimple`, run one more decode step
///      through `model(...)` uncompiled → logits_ref.
///    - **Compiled path:** convert each layer to `CompilableKVCache(from:)`
///      and call `BatchCompile.compileForward(model: model, cacheRef:)` to
///      build the compiled closure. Invoke with the same decode token →
///      logits_compiled.
/// 4. Assert `logits_compiled ≈ logits_ref` within FP tolerance.
///
/// ## Hardware gating
///
/// `HardwareInfo.isCompiledDecodeSupported` controls whether the underlying
/// MLX fused-op path is safe. The tests skip (`XCTSkip`) cleanly when that
/// flag is false so CI on affected hardware stays green.
class BatchCompileForwardInvocationTests: XCTestCase {

    // MARK: Model + prompt fixture

    /// Build a small deterministic test model. Same shape across tests so
    /// any cross-test interaction is visible.
    private func makeModelAndPrompt() -> (any LanguageModel, MLXArray) {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)
        // Short prompt so prefill is cheap but the cache has real state.
        let prompt = MLXArray(Int32(1) ..< Int32(9))  // 8 tokens
        return (model, prompt)
    }

    /// Full prefill into a fresh cache. Returns the cache (for cloning) and
    /// the decode token drawn greedily from the last prompt position.
    private func prefillAndPickDecodeToken(
        model: any LanguageModel, prompt: MLXArray
    ) -> (cache: [KVCache], decodeToken: MLXArray) {
        let cache = model.newCache(parameters: nil)
        let prefill = model(
            LMInput.Text(tokens: prompt)[text: .newAxis], cache: cache, state: nil)
        MLX.eval(prefill.logits)
        MLX.eval(cache)

        // Greedy token for the next position.
        let lastLogits = prefill.logits[0 ..< 1, -1, 0...]  // [1, V]
        let token = argMax(lastLogits, axis: -1)  // [1]
        MLX.eval(token)
        return (cache, token)
    }

    /// Skip early if the hardware or driver combo is not safe for compiled
    /// decode. Better to skip loudly than crash the whole test process.
    private func skipIfCompileUnsafe() throws {
        guard HardwareInfo.isCompiledDecodeSupported else {
            throw XCTSkip("""
                Compiled decode not supported on this hardware/driver combo. \
                See HardwareInfo.isCompiledDecodeSupported for rationale.
                """)
        }
    }

    // MARK: - Tests

    /// Compiled logits match uncompiled logits within tolerance for one
    /// decode step after a short prefill.
    func testCompiledLogitsMatchUncompiledLogits() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt()
        let (prefillCache, decodeToken) = prefillAndPickDecodeToken(
            model: model, prompt: prompt)

        // Reference (uncompiled) path: one decode step through the original
        // post-prefill cache. This mutates the cache in place — fine, we
        // made a fresh one above and won't reuse for the compiled path.
        let uncompiledResult = model(
            LMInput.Text(tokens: decodeToken)[text: .newAxis],
            cache: prefillCache, state: nil)
        MLX.eval(uncompiledResult.logits)
        let uncompiledLogits = uncompiledResult.logits[0 ..< 1, 0, 0...]  // [1, V]
        MLX.eval(uncompiledLogits)

        // Compiled path: start from a fresh prefill to get a cache in the
        // same state as BEFORE the uncompiled decode step.
        let (freshCache, _) = prefillAndPickDecodeToken(model: model, prompt: prompt)
        let compilableCache: [KVCache] = freshCache.map { layer in
            CompilableKVCache(from: layer, maxLength: 256)
        }
        MLX.eval(compilableCache)

        let forward = BatchCompile.compileForward(
            model: model, cacheRef: compilableCache)
        let compiledResult = forward([decodeToken])
        XCTAssertEqual(compiledResult.count, 1,
            "compileForward closure returns [logits]")
        let compiledLogits3D = compiledResult[0]  // [1, 1, V]
        MLX.eval(compiledLogits3D)
        XCTAssertEqual(compiledLogits3D.shape.count, 3,
            "Compiled logits must be 3D [B, L, V]; got shape \(compiledLogits3D.shape)")
        let compiledLogits = compiledLogits3D[0 ..< 1, 0, 0...]  // [1, V]
        MLX.eval(compiledLogits)

        // Shape match
        XCTAssertEqual(uncompiledLogits.shape, compiledLogits.shape,
            "Compiled and uncompiled logits must share shape")

        // Finite check
        let compiledMax = compiledLogits.abs().max().item(Float.self)
        XCTAssertTrue(compiledMax.isFinite,
            "Compiled logits must be finite; got max abs \(compiledMax)")

        // Numerical closeness: compile may reorder ops → small FP drift.
        // 4-bit quantised linear ops tolerate rtol ~1e-2 in practice.
        let diff = (compiledLogits - uncompiledLogits).abs().max().item(Float.self)
        let refMax = uncompiledLogits.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? diff / refMax : diff
        XCTAssertLessThan(relativeDiff, 0.05,
            "Compiled-vs-uncompiled logit drift must be <5% relative. "
            + "abs diff=\(diff), refMax=\(refMax), relative=\(relativeDiff)")
    }

    /// The compiled closure advances cache state correctly: a second decode
    /// step through the compiled closure produces different logits than the
    /// first (cache effectively mutated between calls).
    func testCompiledForwardAdvancesCache() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt()
        let (freshCache, decodeToken) = prefillAndPickDecodeToken(
            model: model, prompt: prompt)
        let compilableCache: [KVCache] = freshCache.map { layer in
            CompilableKVCache(from: layer, maxLength: 256)
        }
        MLX.eval(compilableCache)

        let forward = BatchCompile.compileForward(
            model: model, cacheRef: compilableCache)

        // Two sequential invocations mutate state.
        let r1 = forward([decodeToken])[0]  // [1, 1, V]
        MLX.eval(r1)

        // Pick the next token greedily from r1
        let r1Flat = r1[0 ..< 1, 0, 0...]  // [1, V]
        let nextToken = argMax(r1Flat, axis: -1)  // [1]
        MLX.eval(nextToken)

        let r2 = forward([nextToken])[0]  // [1, 1, V]
        MLX.eval(r2)

        XCTAssertEqual(r1.shape, r2.shape,
            "Two compiled decode steps must share output shape")

        // They should differ — the cache advanced between calls, so logits
        // for a different input token + different cache position will
        // differ. If they match, the cache isn't advancing through the
        // compiled trace.
        let diff = (r2 - r1).abs().max().item(Float.self)
        XCTAssertGreaterThan(diff, 1e-6,
            "Cache state did not advance between compiled decode steps. "
            + "Max abs diff between consecutive decode logits was \(diff).")
    }

    /// ### Multi-step correctness — the iter-8 lesson applied to CompilableKVCache
    ///
    /// Iteration 8 caught a silent Stage 2 bug by running 50-step compiled
    /// decode with **fixed inputs** and comparing to uncompiled. That probe
    /// revealed TurboQuantKVCache's Swift-Int windowOffset was captured at
    /// trace-build time, so every compiled call wrote at the same position.
    ///
    /// This test applies the same shape to `CompilableKVCache`. If it fails,
    /// Stage 1B.3 (shipped) has the same class of bug — and every downstream
    /// stage built on CompilableKVCache is suspect.
    ///
    /// **Acceptance:** 50 compiled decode steps with a fixed token produce
    /// logits that match 50 uncompiled decode steps within 5% relative FP.
    func testCompiledForwardMultiStepWithFixedTokens() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt()

        // Build two matched caches.
        let (refCache, _) = prefillAndPickDecodeToken(model: model, prompt: prompt)
        let (compilableRaw, _) = prefillAndPickDecodeToken(model: model, prompt: prompt)
        let compilableCache: [KVCache] = compilableRaw.map { layer in
            CompilableKVCache(from: layer, maxLength: 256)
        }
        MLX.eval(compilableCache)

        let fixedToken = MLXArray([Int32(42)])
        let steps = 50

        // Uncompiled reference — 50 sequential decode steps, same token.
        var uncompiledLogits: MLXArray?
        for step in 0 ..< steps {
            let result = model(
                LMInput.Text(tokens: fixedToken)[text: .newAxis],
                cache: refCache, state: nil)
            MLX.eval(result.logits)
            let logits = result.logits[0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == steps - 1 { uncompiledLogits = logits }
        }

        // Compiled — same 50 sequential steps.
        let forward = BatchCompile.compileForward(
            model: model, cacheRef: compilableCache)

        var compiledLogits: MLXArray?
        for step in 0 ..< steps {
            let result = forward([fixedToken])
            MLX.eval(result[0])
            let logits = result[0][0 ..< 1, 0, 0...]
            MLX.eval(logits)
            if step == steps - 1 { compiledLogits = logits }
        }

        guard let unc = uncompiledLogits, let cmp = compiledLogits else {
            XCTFail("Both paths should produce last-step logits")
            return
        }

        let diff = (cmp - unc).abs().max().item(Float.self)
        let refMax = unc.abs().max().item(Float.self)
        let relativeDiff = refMax > 0 ? diff / refMax : diff

        print("""

            === CompilableKVCache multi-step correctness (iter 9) ===
              \(steps) fixed-token decode steps, compiled vs uncompiled
              uncompiled last-step max abs: \(refMax)
              compiled last-step max abs: \(cmp.abs().max().item(Float.self))
              abs diff: \(diff)
              relative diff: \(relativeDiff)
              tolerance: 0.05 (5%)
            ==========================================================

            """)

        XCTAssertLessThan(relativeDiff, 0.05,
            "CompilableKVCache multi-step compiled diverges from uncompiled by "
            + "\(relativeDiff). If this fails, Stage 1B.3 (shipped) has the "
            + "same class of silent bug that Stage 2 had. Investigate.")
    }

    /// Compiled closure works when invoked repeatedly with the same token
    /// (cache still advances). Catches regressions where the closure
    /// accidentally becomes idempotent after the first call.
    func testCompiledForwardIdempotentlyAdvances() async throws {
        try skipIfCompileUnsafe()

        let (model, prompt) = makeModelAndPrompt()
        let (freshCache, decodeToken) = prefillAndPickDecodeToken(
            model: model, prompt: prompt)
        let compilableCache: [KVCache] = freshCache.map { layer in
            CompilableKVCache(from: layer, maxLength: 256)
        }
        MLX.eval(compilableCache)

        let forward = BatchCompile.compileForward(
            model: model, cacheRef: compilableCache)

        // Read the starting offset from the first layer's offsetArray.
        let firstLayer = compilableCache[0] as! CompilableKVCache
        let initialOffset = firstLayer.offset
        XCTAssertGreaterThan(initialOffset, 0,
            "After prefill the cache should have offset > 0")

        // Three compiled decode calls with the SAME decode token. The cache
        // should advance by one each call — offsetArray grows by 1 per call.
        _ = forward([decodeToken])
        _ = forward([decodeToken])
        _ = forward([decodeToken])
        MLX.eval(compilableCache)

        let finalOffset = firstLayer.offset
        XCTAssertEqual(finalOffset, initialOffset + 3,
            "Three compiled invocations must advance offset by 3; "
            + "initial=\(initialOffset), final=\(finalOffset)")
    }
}
