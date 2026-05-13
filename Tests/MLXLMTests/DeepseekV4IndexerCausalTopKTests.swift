// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXLLM
import Testing

@Suite("DSV4 indexer causal top-k")
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
}
