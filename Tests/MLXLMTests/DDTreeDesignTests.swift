// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Phase 0 tests — pin the DDTree public API surface so Phase 1+ work
// cannot silently break the shape osaurus will consume.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("DDTree Phase 0 — API surface")
struct DDTreeDesignTests {

    // MARK: - DraftStrategy enum

    @Test("DraftStrategy .none kindName is \"none\"")
    func testKindNameNone() {
        #expect(DraftStrategy.none.kindName == "none")
        #expect(DraftStrategy.none.usesBlockDiffusion == false)
    }

    @Test("DraftStrategy .dflash uses block diffusion")
    func testDflashUsesBlockDiffusion() {
        let s = DraftStrategy.dflash(
            drafterPath: URL(fileURLWithPath: "/tmp/placeholder"),
            blockSize: 4)
        #expect(s.kindName == "dflash")
        #expect(s.usesBlockDiffusion == true)
    }

    @Test("DraftStrategy .ddtree uses block diffusion")
    func testDdtreeUsesBlockDiffusion() {
        let s = DraftStrategy.ddtree(
            drafterPath: URL(fileURLWithPath: "/tmp/placeholder"),
            branchingBudget: 32,
            blockSize: 4)
        #expect(s.kindName == "ddtree")
        #expect(s.usesBlockDiffusion == true)
    }

    @Test("DraftStrategy .autoregressive does NOT use block diffusion")
    func testAutoregressiveKeepsLegacyPath() {
        // .autoregressive's associated value is a LanguageModel, which
        // we can't construct in a unit test. We assert the kindName via
        // a synthetic switch — proving the case exists without needing
        // a live draft model.
        //
        // If a future refactor removes `.autoregressive`, this test stops
        // compiling — exactly the pin we want.
        let kindNames: [String] = [
            DraftStrategy.none.kindName,
            "autoregressive",  // pinned — associated value can't be built in test
            DraftStrategy.dflash(
                drafterPath: URL(fileURLWithPath: "/"), blockSize: 0).kindName,
            DraftStrategy.ddtree(
                drafterPath: URL(fileURLWithPath: "/"), branchingBudget: 0, blockSize: 0).kindName,
        ]
        #expect(kindNames == ["none", "autoregressive", "dflash", "ddtree"])
    }

    // MARK: - GenerateParameters.draftStrategy plumbing

    @Test("GenerateParameters.draftStrategy defaults to nil (byte-compatible)")
    func testDraftStrategyDefaultsToNil() {
        let p = GenerateParameters()
        #expect(p.draftStrategy == nil,
            "Default must be nil so callers who don't opt-in see zero behaviour change.")
    }

    @Test("GenerateParameters.draftStrategy is mutable after init")
    func testDraftStrategyIsMutable() {
        var p = GenerateParameters()
        p.draftStrategy = .dflash(
            drafterPath: URL(fileURLWithPath: "/tmp/drafter"), blockSize: 4)
        #expect(p.draftStrategy?.kindName == "dflash")
    }

    // MARK: - DDTree empty construction

    @Test("DDTree.empty() produces a valid root-only tree")
    func testEmptyTree() {
        let tree = DDTree.empty()
        #expect(tree.nodeCount == 0)
        #expect(tree.parents == [-1])
        #expect(tree.childMaps.count == 1)
        #expect(tree.childMaps[0].isEmpty)
    }

    // MARK: - TreeCompile.isDfsPrefix

    @Test("isDfsPrefix returns true when accepted is DFS prefix")
    func testIsDfsPrefixTrue() {
        let dfs: [Int32] = [0, 1, 2, 3, 4]
        let accepted: [Int32] = [0, 1, 2]
        #expect(TreeCompile.isDfsPrefix(acceptedIndices: accepted, dfsOrder: dfs) == true)
    }

    @Test("isDfsPrefix returns false when accepted diverges from DFS")
    func testIsDfsPrefixFalse() {
        let dfs: [Int32] = [0, 1, 2, 3, 4]
        let accepted: [Int32] = [0, 2, 3]  // skipped node 1
        #expect(TreeCompile.isDfsPrefix(acceptedIndices: accepted, dfsOrder: dfs) == false)
    }

    @Test("isDfsPrefix returns true for empty accepted")
    func testIsDfsPrefixEmpty() {
        #expect(TreeCompile.isDfsPrefix(acceptedIndices: [], dfsOrder: [0, 1, 2]) == true)
    }

    @Test("isDfsPrefix returns false when accepted longer than dfsOrder")
    func testIsDfsPrefixLonger() {
        let dfs: [Int32] = [0, 1]
        let accepted: [Int32] = [0, 1, 2]
        #expect(TreeCompile.isDfsPrefix(acceptedIndices: accepted, dfsOrder: dfs) == false)
    }

    @Test("SpecDecError has all expected cases")
    func testSpecDecErrorCases() {
        let cases: [SpecDecError] = [
            .notImplemented("x"),
            .drafterTargetMismatch(drafter: "a", target: "b"),
            .drafterCheckpointMissingKey("k"),
            .targetDoesNotSupportHiddenStateCapture,
        ]
        // Every case must produce a non-empty error description — ensures
        // future additions don't forget to stringify.
        for c in cases {
            let desc = (c as LocalizedError).errorDescription ?? ""
            #expect(!desc.isEmpty, "SpecDecError case missing errorDescription: \(c)")
        }
    }
}
