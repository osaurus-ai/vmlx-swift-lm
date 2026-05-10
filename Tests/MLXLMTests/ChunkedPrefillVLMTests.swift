// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import MLX
import MLXLMCommon
import XCTest

final class ChunkedPrefillVLMTests: XCTestCase {

    func testShortEmbeddingCallsStepOnceWithFullInput() {
        let embedding = MLXArray.zeros([1, 4, 3])
        var chunkShapes: [[Int]] = []

        let result = chunkedPrefillEmbedding(
            inputEmbedding: embedding,
            cache: [],
            prefillStepSize: 8
        ) { chunk in
            chunkShapes.append(chunk.shape)
            return chunk.dim(1)
        }

        XCTAssertEqual(result, 4)
        XCTAssertEqual(chunkShapes, [[1, 4, 3]])
    }

    func testDisabledChunkingCallsStepOnceWithFullInput() {
        let embedding = MLXArray.zeros([1, 6, 3])
        var callCount = 0

        let result = chunkedPrefillEmbedding(
            inputEmbedding: embedding,
            cache: [],
            prefillStepSize: 0
        ) { chunk in
            callCount += 1
            return chunk.shape
        }

        XCTAssertEqual(callCount, 1)
        XCTAssertEqual(result, [1, 6, 3])
    }

    func testExactChunkBoundaryCallsStepOnceWithFullInput() {
        let embedding = MLXArray.zeros([1, 5, 3])
        var chunkShapes: [[Int]] = []

        let result = chunkedPrefillEmbedding(
            inputEmbedding: embedding,
            cache: [],
            prefillStepSize: 5
        ) { chunk in
            chunkShapes.append(chunk.shape)
            return chunk.dim(1)
        }

        XCTAssertEqual(result, 5)
        XCTAssertEqual(chunkShapes, [[1, 5, 3]])
    }

    func testLongEmbeddingChunksAlongSequenceAxisAndReturnsFinalChunkResult() {
        let embedding = MLXArray.zeros([1, 10, 3])
        var chunkShapes: [[Int]] = []

        let result = chunkedPrefillEmbedding(
            inputEmbedding: embedding,
            cache: [KVCacheSimple()],
            prefillStepSize: 4
        ) { chunk in
            chunkShapes.append(chunk.shape)
            return "final:\(chunk.dim(1))"
        }

        XCTAssertEqual(result, "final:2")
        XCTAssertEqual(chunkShapes, [
            [1, 4, 3],
            [1, 4, 3],
            [1, 2, 3],
        ])
    }
}
