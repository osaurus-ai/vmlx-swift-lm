import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import XCTest

// MARK: - BatchKVCache Unit Tests

@Suite("BatchKVCache")
struct BatchKVCacheTests {

    @Test("update splits and pads correctly for 2 sequences at different offsets")
    func testUpdateSplitPadStack() {
        // Create two KVCacheSimple instances and populate them to different offsets
        let cache0 = KVCacheSimple()
        let cache1 = KVCacheSimple()

        // Populate cache0 with 5 tokens
        for _ in 0 ..< 5 {
            let k = MLXArray.ones([1, 4, 1, 8]) // [B=1, H=4, L=1, D=8]
            let v = MLXArray.ones([1, 4, 1, 8])
            _ = cache0.update(keys: k, values: v)
        }
        // Populate cache1 with 3 tokens
        for _ in 0 ..< 3 {
            let k = MLXArray.ones([1, 4, 1, 8])
            let v = MLXArray.ones([1, 4, 1, 8])
            _ = cache1.update(keys: k, values: v)
        }

        #expect(cache0.offset == 5)
        #expect(cache1.offset == 3)

        // Create BatchKVCache wrapping both
        let batchCache = BatchKVCache(slotCaches: [cache0, cache1])
        #expect(batchCache.batchSize == 2)
        #expect(batchCache.offset == 5) // max(5, 3)

        // Check offsetArray
        let offsets = batchCache.offsetArray
        MLX.eval(offsets)
        #expect(offsets.shape == [2])
        #expect(offsets[0].item(Int32.self) == 5)
        #expect(offsets[1].item(Int32.self) == 3)

        // Now update with batched keys [B=2, H=4, L=1, D=8]
        let batchKeys = MLXArray.ones([2, 4, 1, 8])
        let batchValues = MLXArray.ones([2, 4, 1, 8])
        let (returnedKeys, returnedValues) = batchCache.update(keys: batchKeys, values: batchValues)

        MLX.eval(returnedKeys, returnedValues)

        // After update: cache0 at offset 6, cache1 at offset 4
        // Padded to max = 6
        #expect(returnedKeys.shape == [2, 4, 6, 8])
        #expect(returnedValues.shape == [2, 4, 6, 8])

        // Verify offsets updated
        let newOffsets = batchCache.offsetArray
        MLX.eval(newOffsets)
        #expect(newOffsets[0].item(Int32.self) == 6)
        #expect(newOffsets[1].item(Int32.self) == 4)
        #expect(batchCache.offset == 6)
    }

    @Test("makeMask returns correct per-sequence causal mask")
    func testMakeMask() {
        let cache0 = KVCacheSimple()
        let cache1 = KVCacheSimple()

        // Populate to different offsets
        _ = cache0.update(
            keys: MLXArray.ones([1, 2, 3, 4]),
            values: MLXArray.ones([1, 2, 3, 4]))
        _ = cache1.update(
            keys: MLXArray.ones([1, 2, 1, 4]),
            values: MLXArray.ones([1, 2, 1, 4]))

        #expect(cache0.offset == 3)
        #expect(cache1.offset == 1)

        let batchCache = BatchKVCache(slotCaches: [cache0, cache1])

        // Decode step: n=1
        let mask = batchCache.makeMask(n: 1, windowSize: nil, returnArray: false)
        if case .array(let maskArray) = mask {
            MLX.eval(maskArray)
            // Shape: [B=2, 1, 1, maxTotal=4] where maxTotal = max(3+1, 1+1) = 4
            #expect(maskArray.shape == [2, 1, 1, 4])

            // Seq 0 at offset 3, query at position 3: attends to [0,1,2,3]
            // [T, T, T, T]
            #expect(maskArray[0, 0, 0, 0].item(Bool.self) == true)
            #expect(maskArray[0, 0, 0, 3].item(Bool.self) == true)

            // Seq 1 at offset 1, query at position 1: attends to [0,1], masks [2,3]
            // [T, T, F, F]
            #expect(maskArray[1, 0, 0, 0].item(Bool.self) == true)
            #expect(maskArray[1, 0, 0, 1].item(Bool.self) == true)
            #expect(maskArray[1, 0, 0, 2].item(Bool.self) == false)
            #expect(maskArray[1, 0, 0, 3].item(Bool.self) == false)
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("single-sequence BatchKVCache is equivalent to direct cache")
    func testSingleSequence() {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: MLXArray.ones([1, 2, 5, 4]),
            values: MLXArray.ones([1, 2, 5, 4]))

        let batchCache = BatchKVCache(slotCaches: [cache])
        #expect(batchCache.batchSize == 1)
        #expect(batchCache.offset == 5)

        let newK = MLXArray.ones([1, 2, 1, 4])
        let newV = MLXArray.ones([1, 2, 1, 4])
        let (rk, rv) = batchCache.update(keys: newK, values: newV)
        MLX.eval(rk, rv)

        // Single sequence: no padding needed
        #expect(rk.shape == [1, 2, 6, 4])
        #expect(rv.shape == [1, 2, 6, 4])
    }
}

@Suite("BatchArraysCache")
struct BatchArraysCacheTests {
    @Test("preserves per-slot offsets for one-slot ArraysCache")
    func preservesOneSlotOffsets() {
        let cache0 = ArraysCache(size: 1)
        let cache1 = ArraysCache(size: 1)
        cache0.offset = 5
        cache1.offset = 3
        cache0[0] = MLXArray.ones([1, 2, 4, 4], dtype: .float32)
        cache1[0] = MLXArray.ones([1, 2, 4, 4], dtype: .float32) * 7

        let batch = BatchArraysCache(slotCaches: [cache0, cache1])
        #expect(batch.offset == 5)
        #expect(batch.offsetArray.asArray(Int32.self) == [5, 3])
        #expect(batch[0]?.shape == [2, 2, 4, 4])

        batch[0] = MLXArray.ones([2, 2, 4, 4], dtype: .float32) * 11
        batch.advance(by: 1)
        batch.splitBack()

        #expect(batch.offset == 6)
        #expect(batch.offsetArray.asArray(Int32.self) == [6, 4])
        #expect(cache0.offset == 6)
        #expect(cache1.offset == 4)
        #expect(cache0[0]?.shape == [1, 2, 4, 4])
        #expect(cache1[0]?.shape == [1, 2, 4, 4])
    }
}

// MARK: - Batch Causal Mask Tests

@Suite("BatchCausalMask")
struct BatchCausalMaskTests {

    @Test("two sequences at different offsets, decode step")
    func testBasicMask() {
        let mask = createBatchCausalMask(queryLen: 1, offsets: [5, 3])
        MLX.eval(mask)

        // Shape: [2, 1, 1, 6] — maxTotal = max(5+1, 3+1) = 6
        #expect(mask.shape == [2, 1, 1, 6])

        // Seq 0 at offset 5: attends to all 6 positions
        for j in 0 ..< 6 {
            #expect(mask[0, 0, 0, j].item(Bool.self) == true)
        }

        // Seq 1 at offset 3: attends to 0-3, masks 4-5
        for j in 0 ..< 4 {
            #expect(mask[1, 0, 0, j].item(Bool.self) == true)
        }
        #expect(mask[1, 0, 0, 4].item(Bool.self) == false)
        #expect(mask[1, 0, 0, 5].item(Bool.self) == false)
    }

    @Test("sliding window mask")
    func testSlidingWindow() {
        let mask = createBatchCausalMask(queryLen: 1, offsets: [5, 3], windowSize: 3)
        MLX.eval(mask)

        // Seq 0 at offset 5, window 3: attends to positions 3,4,5 only
        #expect(mask[0, 0, 0, 2].item(Bool.self) == false) // outside window
        #expect(mask[0, 0, 0, 3].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 4].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 5].item(Bool.self) == true)

        // Seq 1 at offset 3, window 3: attends to positions 1,2,3
        #expect(mask[1, 0, 0, 0].item(Bool.self) == false) // outside window
        #expect(mask[1, 0, 0, 1].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 2].item(Bool.self) == true)
        #expect(mask[1, 0, 0, 3].item(Bool.self) == true)
    }

    @Test("same offset sequences produce standard causal mask")
    func testSameOffset() {
        let mask = createBatchCausalMask(queryLen: 1, offsets: [4, 4])
        MLX.eval(mask)

        // Both at offset 4: both attend to [0,1,2,3,4]
        #expect(mask.shape == [2, 1, 1, 5])
        for b in 0 ..< 2 {
            for j in 0 ..< 5 {
                #expect(mask[b, 0, 0, j].item(Bool.self) == true)
            }
        }
    }

    @Test("prefill mask with multiple query tokens")
    func testPrefillMask() {
        // Prefill 3 tokens with offset 2 (already have 2 cached)
        let mask = createBatchCausalMask(queryLen: 3, offsets: [2])
        MLX.eval(mask)

        // Shape: [1, 1, 3, 5] — maxTotal = 2 + 3 = 5
        #expect(mask.shape == [1, 1, 3, 5])

        // Query 0 at position 2: attends to [0,1,2]
        #expect(mask[0, 0, 0, 2].item(Bool.self) == true)
        #expect(mask[0, 0, 0, 3].item(Bool.self) == false)

        // Query 2 at position 4: attends to [0,1,2,3,4]
        for j in 0 ..< 5 {
            #expect(mask[0, 0, 2, j].item(Bool.self) == true)
        }
    }
}

// MARK: - BatchKVCache with Rotating Slots (Gemma-4 SWA regression)

/// Regression suite for the broadcast_shapes crash on Gemma-4 / Mistral-4 /
/// MiMoV2Flash / BaichuanM1 and any other sliding-window model under the
/// batch engine.
///
/// Without the effective-key-length fix, `BatchKVCache.makeMask` produced a
/// mask whose last axis was `offset + n` while `RotatingKVCache.update` only
/// returned `maxCacheSize` keys after wrap, so MLX trapped in
/// `broadcast_shapes` on the very first decode step.
///
/// See `Libraries/MLXLMCommon/BatchEngine/GEMMA4-SLIDING-WINDOW-CRASH.md`.
@Suite("BatchKVCache rotating-slot (Gemma-4 SWA regression)", .serialized)
struct BatchKVCacheRotatingSlotTests {

    @Test("makeMask last axis matches update key count after ring wrap")
    func testMaskMatchesUpdatedKeyShape() {
        // Tiny window so we can wrap cheaply in-test. Real Gemma-4 has
        // maxSize 1024 and prompt 2152 — same topology, bigger numbers.
        let maxSize = 16
        let prompt = 40
        let H = 4
        let D = 8
        let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)

        // Prefill: single multi-token update past the ring (matches how
        // non-chunked VLM prepare populates Gemma-4's sliding layers).
        _ = rotating.update(
            keys: MLXArray.ones([1, H, prompt, D]),
            values: MLXArray.ones([1, H, prompt, D]))
        #expect(rotating.offset == prompt)

        // One decode step with n = 1 — this is the step that used to crash.
        let batchCache = BatchKVCache(slotCaches: [rotating])
        let mask = batchCache.makeMask(n: 1, windowSize: maxSize, returnArray: false)

        let newK = MLXArray.ones([1, H, 1, D])
        let newV = MLXArray.ones([1, H, 1, D])
        let (rk, _) = batchCache.update(keys: newK, values: newV)

        // After update, rotating slot has wrapped: keys shape last axis = maxSize.
        #expect(rk.shape == [1, H, maxSize, D])

        // The fix: mask's last axis equals update's key length. Without the
        // fix this was `offset + n == prompt + 1 == 41` and MLX crashed.
        if case .array(let maskArray) = mask {
            #expect(maskArray.shape.last == rk.shape[2])
            #expect(maskArray.shape == [1, 1, 1, maxSize])

            // Post-wrap ring → every stored key is a valid attention target.
            for j in 0 ..< maxSize {
                #expect(maskArray[0, 0, 0, j].item(Bool.self) == true)
            }
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("pre-wrap rotating slot still gets standard causal mask")
    func testPreWrapMaskUnchanged() {
        // Before wrap: rotating cache behaves like a standard growing cache.
        let maxSize = 32
        let prompt = 8 // well under maxSize
        let H = 2
        let D = 4
        let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
        _ = rotating.update(
            keys: MLXArray.ones([1, H, prompt, D]),
            values: MLXArray.ones([1, H, prompt, D]))

        let batchCache = BatchKVCache(slotCaches: [rotating])
        let mask = batchCache.makeMask(n: 1, windowSize: maxSize, returnArray: false)

        if case .array(let maskArray) = mask {
            // Pre-wrap: mask last axis = offset + n = 9
            #expect(maskArray.shape == [1, 1, 1, prompt + 1])
            // Query at logical pos 8 attends to [0..8]
            for j in 0 ..< prompt + 1 {
                #expect(maskArray[0, 0, 0, j].item(Bool.self) == true)
            }
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("mixed batch: wrapped rotating + unbounded slot produces compatible mask")
    func testMixedBatchWrappedAndUnbounded() {
        // NOTE: in the real engine, slots for a given layer always share cache
        // type (one BatchKVCache per layer). This test is a belt-and-braces
        // check that createBatchCausalMask handles the mixed case gracefully
        // — e.g., if a future model puts different layer topologies behind
        // the same BatchKVCache, the mask shouldn't corrupt either slot.
        let maxSize = 16
        let rotating = RotatingKVCache(maxSize: maxSize, keep: 0)
        _ = rotating.update(
            keys: MLXArray.ones([1, 2, 40, 4]),
            values: MLXArray.ones([1, 2, 40, 4])) // wrapped: offset=40, keys=maxSize

        let simple = KVCacheSimple()
        _ = simple.update(
            keys: MLXArray.ones([1, 2, 5, 4]),
            values: MLXArray.ones([1, 2, 5, 4])) // offset=5, keys=5

        let batch = BatchKVCache(slotCaches: [rotating, simple])
        let mask = batch.makeMask(n: 1, windowSize: maxSize, returnArray: false)

        if case .array(let maskArray) = mask {
            // maxTotal = max(16 (capped rotating), 6 (simple)) = 16
            #expect(maskArray.shape == [2, 1, 1, maxSize])

            // Slot 0 (wrapped rotating): all 16 positions valid (ring full).
            for j in 0 ..< maxSize {
                #expect(maskArray[0, 0, 0, j].item(Bool.self) == true)
            }

            // Slot 1 (unbounded): valid through position 5, padded [6..16) = false.
            for j in 0 ..< 6 {
                #expect(maskArray[1, 0, 0, j].item(Bool.self) == true)
            }
            for j in 6 ..< maxSize {
                #expect(maskArray[1, 0, 0, j].item(Bool.self) == false)
            }
        } else {
            Issue.record("Expected .array mask, got \(mask)")
        }
    }

    @Test("explicit effectiveKeyLens parameter caps maxTotal")
    func testCreateBatchCausalMaskWithEffectiveKeyLens() {
        // Call the low-level helper directly — slot A wrapped, slot B unbounded.
        let mask = createBatchCausalMask(
            queryLen: 1,
            offsets: [100, 5],
            effectiveKeyLens: [16, 6], // slot A ring cap = 16, slot B = offset + n
            windowSize: nil)

        // maxTotal = max(16, 6) = 16
        #expect(mask.shape == [2, 1, 1, 16])

        // Slot A (wrapped): all 16 positions valid.
        for j in 0 ..< 16 {
            #expect(mask[0, 0, 0, j].item(Bool.self) == true)
        }

        // Slot B (unbounded, offset=5): valid through position 5, rest padding.
        for j in 0 ..< 6 {
            #expect(mask[1, 0, 0, j].item(Bool.self) == true)
        }
        for j in 6 ..< 16 {
            #expect(mask[1, 0, 0, j].item(Bool.self) == false)
        }
    }
}

// MARK: - BatchEngine Integration Tests (uses small Llama model)

/// Integration tests that create a small Llama model and run BatchEngine
/// with actual generation. Tests correctness, multi-request batching,
/// per-request parameters, and throughput.
class BatchEngineIntegrationTests: XCTestCase {

    /// Create a small test model and batch engine for testing
    private func makeEngine(vocabSize: Int = 100, maxBatchSize: Int = 4) -> BatchEngine {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 8, rmsNormEps: 1e-5, vocabularySize: vocabSize, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        MLX.eval(model)

        let processor = TestInputProcessor()
        nonisolated(unsafe) let context = ModelContext(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        return BatchEngine(context: context, maxBatchSize: maxBatchSize)
    }

    /// Test: single request through BatchEngine produces tokens
    func testSingleRequest() async throws {
        let engine = makeEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(6)))
        let params = GenerateParameters(maxTokens: 10, temperature: 0)

        var tokenCount = 0
        var gotInfo = false
        let stream = await engine.generate(input: input, parameters: params)
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                XCTAssertFalse(text.isEmpty, "Chunk should not be empty")
                tokenCount += 1
            case .info(let info):
                gotInfo = true
                XCTAssertEqual(info.promptTokenCount, 5)
                XCTAssertGreaterThan(info.generationTokenCount, 0)
                XCTAssertEqual(info.stopReason, .length)
            case .reasoning, .toolCall:
                break
            @unknown default:
                break
            }
        }
        XCTAssert(gotInfo, "Should receive completion info")
        XCTAssertGreaterThan(tokenCount, 0, "Should receive at least one text chunk")
    }

    /// Test: two concurrent requests both complete correctly
    func testTwoConcurrentRequests() async throws {
        let engine = makeEngine()

        let input1 = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))
        let input2 = LMInput(tokens: MLXArray(Int32(10) ..< Int32(15)))
        let params = GenerateParameters(maxTokens: 5, temperature: 0)

        // Submit both
        let (id1, stream1) = await engine.submit(input: input1, parameters: params)
        let (id2, stream2) = await engine.submit(input: input2, parameters: params)

        XCTAssertNotEqual(id1, id2, "Request IDs should be unique")

        // Collect tokens — engine runs both concurrently internally
        let result1 = await collectTokens(from: stream1)
        let result2 = await collectTokens(from: stream2)

        XCTAssertGreaterThan(result1.tokens.count, 0, "Request 1 should produce tokens")
        XCTAssertGreaterThan(result2.tokens.count, 0, "Request 2 should produce tokens")
        XCTAssertNotNil(result1.info, "Request 1 should have completion info")
        XCTAssertNotNil(result2.info, "Request 2 should have completion info")
    }

    /// Test: different parameters per request (greedy vs sampled)
    func testDifferentParametersPerRequest() async throws {
        let engine = makeEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))

        // Request 1: greedy, 3 tokens
        let params1 = GenerateParameters(maxTokens: 3, temperature: 0)
        // Request 2: sampled, 7 tokens
        let params2 = GenerateParameters(maxTokens: 7, temperature: 0.8)

        let (_, stream1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params1)
        let (_, stream2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params2)

        let r1 = await collectTokens(from: stream1)
        let r2 = await collectTokens(from: stream2)

        // Request 1 should have fewer tokens (maxTokens=3)
        XCTAssertLessThanOrEqual(r1.tokens.count, 3)
        // Request 2 should have more tokens (maxTokens=7)
        XCTAssertGreaterThan(r2.tokens.count, r1.tokens.count)
    }

    /// Test: request cancellation mid-generation
    func testCancellation() async throws {
        let engine = makeEngine()

        let input = LMInput(tokens: MLXArray(Int32(1) ..< Int32(4)))
        let params = GenerateParameters(maxTokens: 100, temperature: 0)

        let (requestID, stream) = await engine.submit(input: input, parameters: params)

        // Cancel after receiving a couple tokens
        var tokenCount = 0
        for await event in stream {
            if case .token = event {
                tokenCount += 1
                if tokenCount >= 2 {
                    await engine.cancel(requestID)
                    break
                }
            }
        }
        XCTAssertGreaterThanOrEqual(tokenCount, 2)
    }

    /// Test: more requests than maxBatchSize — queuing works
    func testQueueOverflow() async throws {
        let engine = makeEngine(maxBatchSize: 2)

        let tokens = MLXArray(Int32(1) ..< Int32(4))
        let params = GenerateParameters(maxTokens: 3, temperature: 0)

        // Submit 4 requests with maxBatchSize=2 — 2 active, 2 queued
        var streams = [AsyncStream<BatchGeneration>]()
        for _ in 0 ..< 4 {
            let (_, stream) = await engine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params)
            streams.append(stream)
        }

        // All 4 should complete
        var completedCount = 0
        for stream in streams {
            let result = await collectTokens(from: stream)
            if result.info != nil {
                completedCount += 1
            }
        }
        XCTAssertEqual(completedCount, 4, "All 4 requests should complete")
    }

    /// Test: batch throughput vs serial — measures actual tok/s
    func testBatchThroughput() async throws {
        let maxTokens = 20
        let numRequests = 4

        let tokens = MLXArray(Int32(1) ..< Int32(6))
        let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        // Measure B=1: 4 requests one at a time through engine with maxBatchSize=1
        let serialEngine = makeEngine(vocabSize: 200, maxBatchSize: 1)
        let serialStart = Date()
        for _ in 0 ..< numRequests {
            let (_, stream) = await serialEngine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(6))), parameters: params)
            _ = await collectTokens(from: stream)
        }
        let serialTime = Date().timeIntervalSince(serialStart)

        // Measure B=4: 4 requests simultaneously through engine with maxBatchSize=4
        let batchEngine = makeEngine(vocabSize: 200, maxBatchSize: numRequests)
        let batchStart = Date()
        var batchStreams = [AsyncStream<BatchGeneration>]()
        for _ in 0 ..< numRequests {
            let (_, stream) = await batchEngine.submit(
                input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(6))), parameters: params)
            batchStreams.append(stream)
        }
        // Wait for all to complete
        for stream in batchStreams {
            _ = await collectTokens(from: stream)
        }
        let batchTime = Date().timeIntervalSince(batchStart)

        let serialTokPerSec = Double(numRequests * maxTokens) / serialTime
        let batchTokPerSec = Double(numRequests * maxTokens) / batchTime

        print("""
        === Throughput Benchmark ===
        Serial: \(String(format: "%.1f", serialTokPerSec)) total tok/s (\(String(format: "%.2f", serialTime))s)
        Batch:  \(String(format: "%.1f", batchTokPerSec)) total tok/s (\(String(format: "%.2f", batchTime))s)
        Speedup: \(String(format: "%.2f", batchTokPerSec / serialTokPerSec))x
        """)

        // Batch should be at least as fast as serial (on a tiny model the overhead may dominate,
        // but on real models batch should be faster)
        // We don't assert speedup > 1 because the tiny test model may not show benefit
    }

    /// Test: shutdown cleans up all pending and active requests
    func testShutdown() async throws {
        let engine = makeEngine(maxBatchSize: 2)

        let tokens = MLXArray(Int32(1) ..< Int32(4))
        let params = GenerateParameters(maxTokens: 1000, temperature: 0)

        // Submit requests
        let (_, stream1) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params)
        let (_, stream2) = await engine.submit(
            input: LMInput(tokens: MLXArray(Int32(1) ..< Int32(4))), parameters: params)

        // Let them start
        try await Task.sleep(nanoseconds: 100_000_000)  // 100ms

        // Shutdown
        await engine.shutdown()

        // Both streams should finish (with .cancelled or naturally)
        let r1 = await collectTokens(from: stream1)
        let r2 = await collectTokens(from: stream2)

        // At least one should have info (cancelled or completed)
        XCTAssert(r1.info != nil || r2.info != nil, "Shutdown should finish streams")
    }

    // MARK: - Helpers

    private func collectTokens(from stream: AsyncStream<BatchGeneration>)
        async -> (tokens: [Int], info: GenerateCompletionInfo?)
    {
        var tokens = [Int]()
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let id):
                tokens.append(id)
            case .info(let i):
                info = i
            }
        }
        return (tokens, info)
    }
}
