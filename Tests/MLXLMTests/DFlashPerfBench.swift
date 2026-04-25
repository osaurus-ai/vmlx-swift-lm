// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Real-model perf bench for DFlash speculative decoding.
//
// Runs OUTSIDE the normal unit-test flow — gated on env `DFLASH_PERF=1`
// so it doesn't spin up 8GB of model weights in the routine swift test
// run. When enabled, loads a target + DFlash drafter pair and measures
// cached AR tok/s vs DFlash tok/s on N new tokens for the prompt
// "The capital of France is".
//
// Usage:
//
//   DFLASH_PERF=1 \
//     DFLASH_PERF_TARGET=/tmp/ddtree-downloads/Qwen3-8B-target \
//     DFLASH_PERF_DRAFTER=/tmp/ddtree-downloads/Qwen3-8B-DFlash \
//     DFLASH_PERF_TOKENS=128 \
//     swift test -c release --filter "DFlashPerfBench"

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXLLM

@inline(__always)
private func materializeArray(_ a: MLXArray) { MLX.eval(a) }

/// Minimal tokenizer loader — we don't actually use the tokenizer in this
/// perf bench (we pass raw token IDs), but `loadModel(from:using:)` requires
/// a loader to satisfy the API. Returns a stub tokenizer that refuses real
/// work (which is fine since this bench doesn't tokenize).
private struct MinimalTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        NullTokenizer()
    }
}

private struct NullTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("DFlash perf bench — real models (gated on DFLASH_PERF=1)",
       .serialized)
struct DFlashPerfBench {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["DFLASH_PERF"] == "1"
    }

    @Test("cached-AR vs DFlash linear tok/s (gated)")
    func testPerfParity() async throws {
        guard Self.enabled else {
            #expect(Bool(true), "DFLASH_PERF != 1 — skipping")
            return
        }
        let env = ProcessInfo.processInfo.environment
        guard let targetPath = env["DFLASH_PERF_TARGET"],
              let drafterPath = env["DFLASH_PERF_DRAFTER"]
        else {
            #expect(Bool(false),
                "DFLASH_PERF_TARGET and DFLASH_PERF_DRAFTER must be set")
            return
        }
        let maxNew = Int(env["DFLASH_PERF_TOKENS"] ?? "128") ?? 128

        let targetURL = URL(fileURLWithPath: targetPath)
        let drafterURL = URL(fileURLWithPath: drafterPath)
        guard DFlashDrafterLoader.looksLikeDrafter(at: drafterURL) else {
            #expect(Bool(false), "not a drafter: \(drafterPath)")
            return
        }
        let drafter = try DFlashDrafterLoader.load(from: drafterURL)
        let targetLayerIDs = drafter.config.dflashConfig.targetLayerIds

        print("\n=== DFlash perf (\(maxNew) tokens) ===")
        print("  target:  \(targetURL.lastPathComponent)")
        print("  drafter: \(drafterURL.lastPathComponent)")

        // Load target via the container API (bypasses the macro-based
        // tokenizer loaders that require MLXHuggingFace).
        let loader = MinimalTokenizerLoader()
        let ctx = try await MLXLMCommon.loadModel(
            from: targetURL, using: loader)
        guard let target = ctx.model
            as? any (HiddenStateCaptureModel & TokenEmbedderModel)
        else {
            print("  [skip] \(type(of: ctx.model)) not spec-dec capable")
            return
        }

        // Raw token IDs — we're measuring tok/s, not tokenization accuracy.
        // These are a reasonable "fairly diverse" sequence for either
        // Qwen3 or Qwen3.5 vocab (ids 1000-1015 are definitely in range).
        let promptInts: [Int32] = Array(1000..<1015).map { Int32($0) }
        let promptIds = MLXArray(promptInts).reshaped(1, promptInts.count)

        // --- Cached AR (honest baseline) ---
        let arT0 = CFAbsoluteTimeGetCurrent()
        let arTokens = cachedGreedyAR(
            target: target, promptInts: promptInts, maxNew: maxNew)
        let arDt = CFAbsoluteTimeGetCurrent() - arT0
        let arTps = Double(maxNew) / arDt
        print(String(format: "  cached AR:     %.2fs / %.1f tok/s",
                     arDt, arTps))

        // --- DFlash linear with drafter KV cache (iter 2) ---
        let dfArgs = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: targetLayerIDs,
            maskTokenID: Int32(drafter.config.dflashConfig.maskTokenId),
            inputIds: promptIds, maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0)
        let dfT0 = CFAbsoluteTimeGetCurrent()
        let dfResult = try SpecDecRuntimeLinear.run(dfArgs)
        let dfDt = CFAbsoluteTimeGetCurrent() - dfT0
        let dfGenerated = max(0, dfResult.tokenIds.count - promptInts.count)
        let dfTps = dfDt > 0 ? Double(dfGenerated) / dfDt : 0
        let meanAccept = dfResult.meanAcceptanceLength
        let bs = drafter.config.blockSize
        print(String(
            format: "  DFlash linear: %.2fs / %.1f tok/s "
                + "(acc %.2f/%d, %d rounds)",
            dfDt, dfTps, meanAccept, bs - 1,
            dfResult.acceptanceLengths.count))

        // --- Ratio + byte-parity ---
        let ratio = arTps > 0 ? dfTps / arTps : 0
        print(String(format: "  ratio DFlash/AR: %.2fx %@",
                     ratio, ratio >= 1.0 ? "WIN" : "loss"))

        let n = min(dfResult.tokenIds.count, arTokens.count)
        var match = 0
        for i in 0..<n where dfResult.tokenIds[i] == arTokens[i] { match += 1 }
        let pct = n > 0 ? 100.0 * Double(match) / Double(n) : 100.0
        print(String(format: "  byte-match vs AR: %d/%d (%.1f%%)",
                     match, n, pct))

        #expect(dfTps > 0)
    }

    /// Proper cached AR — prefill once, step-by-step decode with persistent
    /// KV cache. Byte-match reference for other paths.
    private func cachedGreedyAR(
        target: any (HiddenStateCaptureModel & TokenEmbedderModel),
        promptInts: [Int32],
        maxNew: Int
    ) -> [Int32] {
        var out = promptInts
        let cache = target.newCache(parameters: nil)
        let promptArr = MLXArray(promptInts).reshaped(1, promptInts.count)
        var (logits, _) = target(
            promptArr, cache: cache, captureLayerIDs: [])
        materializeArray(logits)
        var nextTok = argMax(
            logits[0, logits.dim(1) - 1, 0...], axis: -1
        ).asType(.int32).item(Int32.self)
        out.append(nextTok)
        for _ in 1..<maxNew {
            let stepIn = MLXArray([nextTok]).reshaped(1, 1)
            (logits, _) = target(stepIn, cache: cache, captureLayerIDs: [])
            materializeArray(logits)
            nextTok = argMax(
                logits[0, logits.dim(1) - 1, 0...], axis: -1
            ).asType(.int32).item(Int32.self)
            out.append(nextTok)
        }
        return out
    }
}
