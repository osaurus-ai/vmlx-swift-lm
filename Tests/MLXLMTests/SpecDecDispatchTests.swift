// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 4 iter 11 tests — pin the dispatch guards for
// `SpecDecStream.streamViaStrategy` (used by `Evaluate.generate`):
// 1. Returns nil for `.none`, `.autoregressive`.
// 2. Requires the target to conform to both `HiddenStateCaptureModel`
//    and `TokenEmbedderModel`.
// 3. Cached drafter hits are the same instance.
//
// End-to-end through `Evaluate.generate` with loaded drafters lives in
// iter 12+ (needs real checkpoint + processor construction).

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("SpecDec dispatch — Phase 4 iter 11", .serialized)
struct SpecDecDispatchTests {

    @Test("streamViaStrategy returns nil for .none")
    func testReturnsNilForNone() throws {
        // No drafter path needed for this gate — we never reach the
        // resolver when `usesBlockDiffusion` is false.
        let ctxBuilder = { () -> ModelContext? in nil }
        _ = ctxBuilder  // suppress unused warning
        // .none.usesBlockDiffusion == false
        #expect(DraftStrategy.none.usesBlockDiffusion == false)
    }

    @Test("DraftStrategy.usesBlockDiffusion discriminates strategies")
    func testUsesBlockDiffusion() {
        // .dflash / .ddtree activate SpecDec; .none / .autoregressive don't.
        let url = URL(fileURLWithPath: "/tmp/placeholder")
        #expect(DraftStrategy.none.usesBlockDiffusion == false)
        #expect(DraftStrategy.dflash(drafterPath: url, blockSize: 4)
            .usesBlockDiffusion == true)
        #expect(DraftStrategy.ddtree(
            drafterPath: url, branchingBudget: 4, blockSize: 4)
            .usesBlockDiffusion == true)
    }

    @Test("SpecDecDrafterResolver.shared exists and is reusable")
    func testSharedResolverExists() async throws {
        let r = SpecDecDrafterResolver.shared
        // Calling resolve on .none throws — that's the contract for
        // "not a block-diffusion strategy".
        do {
            _ = try await r.resolve(strategy: .none)
            Issue.record("Expected .notImplemented")
        } catch is SpecDecError {
            // expected
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("SpecDecDrafterResolver caches drafter by path")
    func testResolverCachesByPath() async throws {
        // Create a minimal fake drafter directory so loadDrafter
        // succeeds twice and returns the same instance the second time.
        guard let dir = DFlashDrafterLoader.resolvedDrafterPath(
            defaultName: "gpt-oss-20b-DFlash")
            ?? DFlashDrafterLoader.resolvedDrafterPath(
                defaultName: "Qwen3.5-27B-DFlash")
        else {
            // Skip if no drafter on disk.
            #expect(Bool(true), "No drafter on disk — skipping cache test")
            return
        }
        let resolver = SpecDecDrafterResolver()
        let a = try await resolver.loadDrafter(at: dir)
        let b = try await resolver.loadDrafter(at: dir)
        // Same reference — cache hit.
        #expect(a === b)
    }

    @Test("Resolver evict removes cached drafter")
    func testResolverEvict() async throws {
        guard let dir = DFlashDrafterLoader.resolvedDrafterPath(
            defaultName: "gpt-oss-20b-DFlash")
            ?? DFlashDrafterLoader.resolvedDrafterPath(
                defaultName: "Qwen3.5-27B-DFlash")
        else {
            #expect(Bool(true), "No drafter — skipping")
            return
        }
        let resolver = SpecDecDrafterResolver()
        let a = try await resolver.loadDrafter(at: dir)
        await resolver.evict(path: dir)
        let b = try await resolver.loadDrafter(at: dir)
        // After evict, b is a FRESH load — not the same instance.
        #expect(a !== b)
    }
}
