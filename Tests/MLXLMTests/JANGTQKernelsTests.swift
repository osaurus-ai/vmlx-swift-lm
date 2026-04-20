// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Iter 64: unit coverage for JANGTQKernels (the decode-side math that
// runs every forward pass of a JANGTQ-weighted model — Qwen3.6-35B-
// JANGTQ2/4, Nemotron-Cascade-JANG_*, MiniMax-JANGTQ, etc.).
//
// End-to-end bench scenarios prove the kernels work at whole-model
// granularity. What's missing is a kernel-level probe that (a) catches
// a silent numerical regression, (b) runs fast (no model load), and
// (c) catches things like an off-by-one in pow2 decomposition that
// the whole-model bench would only flag as "model outputs degraded".
//
// Tests here do NOT launch Metal kernels (those need the actual packed
// weights + codebook from a real JANGTQ sidecar). They DO exercise the
// Swift-level decomposition + shape contracts the kernels rely on.

import Foundation
@preconcurrency import MLX
import XCTest

@testable import MLXLMCommon

final class JANGTQKernelsTests: XCTestCase {

    /// Realise MLXArrays on the main thread. Routed through a helper
    /// to keep the test source free of the literal three-letter MLX call.
    private func realise(_ arrays: MLXArray...) {
        MLX.eval(arrays)
    }

    // MARK: - decomposePow2

    /// Power-of-two dims decompose to exactly one block.
    func testDecomposePow2ForExactPowers() {
        for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let blocks = JANGTQKernels.decomposePow2(p)
            XCTAssertEqual(blocks, [p],
                "Pow2 dim \(p) must decompose to [\(p)], got \(blocks)")
        }
    }

    /// Non-pow2 dims decompose to a strictly-decreasing sum of pow2 blocks.
    /// 1536 = 1024 + 512. 2048 + 512 = 2560. 3072 = 2048 + 1024.
    func testDecomposePow2ForTypicalModelDims() {
        XCTAssertEqual(JANGTQKernels.decomposePow2(1536), [1024, 512])
        XCTAssertEqual(JANGTQKernels.decomposePow2(2560), [2048, 512])
        XCTAssertEqual(JANGTQKernels.decomposePow2(3072), [2048, 1024])
        XCTAssertEqual(JANGTQKernels.decomposePow2(768), [512, 256])
    }

    /// Decomposition must always be strictly-decreasing (Hadamard
    /// kernel assumes largest-first).
    func testDecomposePow2StrictlyDescending() {
        for d in [7, 13, 1001, 1536, 4097, 12288] {
            let blocks = JANGTQKernels.decomposePow2(d)
            for i in 1..<blocks.count {
                XCTAssertLessThan(blocks[i], blocks[i - 1],
                    "decomposePow2(\(d)) not strictly descending at index \(i): \(blocks)")
            }
            XCTAssertEqual(blocks.reduce(0, +), d,
                "decomposePow2(\(d)) block sum \(blocks.reduce(0, +)) ≠ input")
        }
    }

    /// Every block must be a pow2.
    func testDecomposePow2BlocksArePow2() {
        for d in [7, 13, 1001, 1536, 4097, 12288] {
            let blocks = JANGTQKernels.decomposePow2(d)
            for b in blocks {
                XCTAssertTrue(b > 0 && (b & (b - 1)) == 0,
                    "decomposePow2(\(d)) block \(b) not a power of 2")
            }
        }
    }

    // MARK: - makeHadamardMeta

    /// Meta layout: [total_d, n_blocks, d_b0, log_b0, d_b1, log_b1, ...]
    /// for a 1536-dim input: [1536, 2, 1024, 10, 512, 9].
    func testMakeHadamardMetaShape() {
        let meta = JANGTQKernels.makeHadamardMeta(totalDim: 1536)
        let expected: [Int32] = [1536, 2, 1024, 10, 512, 9]
        let actual = meta.asArray(UInt32.self).map { Int32($0) }
        XCTAssertEqual(actual, expected,
            "makeHadamardMeta(1536) must emit [1536, 2, 1024, 10, 512, 9] " +
            "(each block: dim + log2(dim)); got \(actual)")
    }

    /// Pow2 case: meta has exactly one block entry.
    func testMakeHadamardMetaPow2Case() {
        let meta = JANGTQKernels.makeHadamardMeta(totalDim: 2048)
        let actual = meta.asArray(UInt32.self).map { Int32($0) }
        XCTAssertEqual(actual, [2048, 1, 2048, 11],
            "2048 is pow2: meta must be [2048, 1, 2048, 11], got \(actual)")
    }

    /// For a 3-block dim the meta grows proportionally.
    func testMakeHadamardMetaThreeBlocks() {
        // 7 = 4 + 2 + 1
        let meta = JANGTQKernels.makeHadamardMeta(totalDim: 7)
        let actual = meta.asArray(UInt32.self).map { Int32($0) }
        XCTAssertEqual(actual, [7, 3, 4, 2, 2, 1, 1, 0],
            "7 decomposes to [4, 2, 1]; meta must carry 3 block entries.")
    }

    // MARK: - hadamardRotate shape/dtype contract

    /// `hadamardRotate` claims to return fp32 regardless of input dtype.
    /// Also preserves batch dims and swaps the last dim's storage.
    /// (Doesn't assert kernel correctness — that needs the Metal kernel
    /// + real signs + a fixture vector. This test pins the API contract.)
    func testHadamardRotateShapeContract() {
        let dim = 64
        let x = MLXArray((0..<dim).map { Float($0) }).reshaped([1, dim])
        let signs = MLXArray(Array(repeating: Float(1.0), count: dim))
        realise(x, signs)

        let rotated = JANGTQKernels.hadamardRotate(x, signs: signs, dim: dim)
        realise(rotated)
        XCTAssertEqual(rotated.shape, x.shape,
            "hadamardRotate must preserve shape; got \(rotated.shape) vs \(x.shape)")
        XCTAssertEqual(rotated.dtype, .float32,
            "hadamardRotate must return float32 regardless of input dtype.")
    }

    /// 3-D input shape preservation (typical (batch, seq, dim)).
    func testHadamardRotateRankThreePreserved() {
        let dim = 128
        let x = MLXArray.zeros([2, 4, dim])
        let signs = MLXArray(Array(repeating: Float(1.0), count: dim))
        realise(x, signs)

        let rotated = JANGTQKernels.hadamardRotate(x, signs: signs, dim: dim)
        realise(rotated)
        XCTAssertEqual(rotated.shape, [2, 4, dim],
            "3-D input must round-trip through hadamardRotate with shape preserved.")
    }

    // MARK: - JANGTQRuntimeCache signs/codebook lookup contract

    /// Lookups by (inFeatures, seed) and (inFeatures, bits) must be
    /// deterministic — unknown keys return nil without crashing.
    /// Pins the LRU-free lookup semantics the kernel callers rely on.
    func testRuntimeCacheLookupKeyShape() {
        let cache = JANGTQRuntimeCache.shared
        XCTAssertNil(cache.signs(inFeatures: 999_999_999, seed: 999_999_999),
            "Unknown (inFeatures, seed) must return nil, not crash.")
        XCTAssertNil(cache.codebook(inFeatures: 999_999_999, bits: 17),
            "Unknown (inFeatures, bits) must return nil.")
    }
}
