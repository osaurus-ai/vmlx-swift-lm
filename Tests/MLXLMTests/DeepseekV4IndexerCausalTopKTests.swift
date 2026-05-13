// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
@testable import MLXLLM
import Testing

@Suite("DSV4 indexer causal top-k", .serialized)
struct DeepseekV4IndexerCausalTopKTests {

    @Test("prefill indexer scores mask future compressed chunks before top-k")
    func prefillMasksFutureCompressedChunksBeforeTopK() {
        // Query position 3 can only see compressed chunk 0 when ratio=4.
        // Chunk 5 has a much larger raw score; if top-k runs before the
        // causal mask, argpartition picks chunk 5 and the later attention
        // visibility mask filters it out, leaving the query starved.
        let scores = MLXArray([
            Float(10), 20, 30, 40, 50, 60,
            Float(10), 20, 30, 40, 50, 60,
            Float(10), 20, 30, 40, 50, 60,
            Float(1), 2, 3, 4, 5, 1000,
        ]).reshaped(1, 4, 6)

        let masked = DeepseekV4Math.causalMaskedIndexerScores(
            scores, offset: 0, ratio: 4)
        let top1 = MLX.argPartition(-masked, kth: 0, axis: -1)[
            .ellipsis, 0..<1
        ]
        MLX.eval(masked, top1)

        #expect(top1[0, 3, 0].item(Int32.self) == 0)
        #expect(masked[0, 3, 0].item(Float.self) > 0)
        #expect(masked[0, 3, 5].item(Float.self) < -1.0e20)
    }

    @Test("ratio-4 overlap cache preserves previous complete window across decode calls")
    func overlapDecodeKeepsPreviousWindowLeftHalf() {
        var cfg = DeepseekV4Configuration()
        cfg.hiddenSize = 8
        cfg.headDim = 4
        cfg.qkRopeHeadDim = 2
        cfg.rmsNormEps = 1e-6
        let compressor = DeepseekV4Compressor(config: cfg, compressRatio: 4, headDim: 2)
        let cache = DeepseekV4Cache(slidingWindow: 16, compressRatio: 4)

        func tensor(_ start: Int, _ count: Int) -> MLXArray {
            var values: [Float] = []
            for token in start..<(start + count) {
                values.append(Float(token))
                values.append(Float(token) + 0.25)
                values.append(Float(token) + 100.0)
                values.append(Float(token) + 100.25)
            }
            return MLXArray(values).reshaped(1, count, 4)
        }

        // Initial prefill rows 0...7 leave the last complete window
        // (tokens 4...7) in the overlap buffer.
        let prefill = compressor.accumulateOverlapWindows(
            kv: tensor(0, 8),
            gate: tensor(0, 8),
            cache: cache,
            branch: .compressor,
            ratio: 4,
            startPos: 0)
        MLX.eval(prefill.kvRows)
        #expect(prefill.kvRows.shape == [1, 2, 8, 2])
        #expect(prefill.poolBase == 0)

        // Feed decode tokens 8, 9, 10, 11 one at a time. The completed row
        // at token 11 must use tokens 4...7 as its left half and 8...11 as
        // its right half. The old plain remainder-buffer path produced
        // zeros for the left half here.
        for pos in 8..<11 {
            let row = compressor.accumulateOverlapWindows(
                kv: tensor(pos, 1),
                gate: tensor(pos, 1),
                cache: cache,
                branch: .compressor,
                ratio: 4,
                startPos: pos)
            MLX.eval(row.kvRows)
            #expect(row.kvRows.dim(1) == 0)
        }

        let completed = compressor.accumulateOverlapWindows(
            kv: tensor(11, 1),
            gate: tensor(11, 1),
            cache: cache,
            branch: .compressor,
            ratio: 4,
            startPos: 11)
        MLX.eval(completed.kvRows)

        #expect(completed.poolBase == 8)
        #expect(completed.kvRows.shape == [1, 1, 8, 2])
        #expect(completed.kvRows[0, 0, 0, 0].item(Float.self) == 4.0)
        #expect(completed.kvRows[0, 0, 3, 0].item(Float.self) == 7.0)
        #expect(completed.kvRows[0, 0, 4, 0].item(Float.self) == 108.0)
        #expect(completed.kvRows[0, 0, 7, 0].item(Float.self) == 111.0)
    }
}
