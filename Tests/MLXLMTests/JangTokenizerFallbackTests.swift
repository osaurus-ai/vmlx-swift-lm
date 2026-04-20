// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Unit tests for `JangLoader.resolveTokenizerDirectory(for:)`.
//
// Motivation: `jangq-ai/MiniMax-M2.7-JANGTQ` (and many other JANGTQ / JANG
// bundles) ship **weights-only** — the snapshot directory contains
// `model.safetensors`, `config.json`, and `jang_config.json` but no
// `tokenizer.json`, `tokenizer_config.json`, or `chat_template.jinja`.
// Loaders are expected to resolve those from the cached source model.
//
// These tests use a fake HuggingFace cache laid out under a temp directory
// so they are fully self-contained — no network, no dependency on what
// happens to be cached on the developer machine.

import Foundation
import XCTest

@testable import MLXLMCommon

final class JangTokenizerFallbackTests: XCTestCase {

    // MARK: - Fixtures

    /// Temporary workspace root. Cleared between tests.
    private var tmpRoot: URL!

    /// Simulated HuggingFace cache root (`<tmpRoot>/hub`).
    private var hfCacheRoot: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("JangTokenizerFallbackTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tmpRoot, withIntermediateDirectories: true)
        hfCacheRoot = tmpRoot.appendingPathComponent("hub")
        try FileManager.default.createDirectory(
            at: hfCacheRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tmpRoot {
            try? FileManager.default.removeItem(at: tmpRoot)
        }
        try super.tearDownWithError()
    }

    // MARK: - Helpers

    /// Compare URLs for equality by their symlink-resolved absolute paths.
    /// macOS returns `/private/var/...` from `contentsOfDirectory(at:)` even
    /// when the input URL uses the symlinked `/var/...`, so a raw
    /// `XCTAssertEqual` on URL values flakes on matched directories.
    private func assertSamePath(
        _ lhs: URL, _ rhs: URL,
        _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let l = lhs.resolvingSymlinksInPath().standardizedFileURL.path
        let r = rhs.resolvingSymlinksInPath().standardizedFileURL.path
        XCTAssertEqual(l, r, message, file: file, line: line)
    }

    /// Write a (possibly-empty) file at `path` with `contents`.
    private func writeFile(at path: URL, contents: String = "{}") throws {
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try contents.data(using: .utf8)!.write(to: path)
    }

    /// Write a minimal `jang_config.json` with the given source_model block.
    private func writeJangConfig(
        at modelDir: URL,
        sourceModel: [String: Any]? = nil
    ) throws {
        var payload: [String: Any] = [
            "format": "JANG",
            "format_version": 2,
            "quantization": ["method": "jang-importance", "profile": "JANG_2S"]
        ]
        if let sourceModel {
            payload["source_model"] = sourceModel
        }
        let data = try JSONSerialization.data(withJSONObject: payload)
        try FileManager.default.createDirectory(
            at: modelDir, withIntermediateDirectories: true)
        try data.write(to: modelDir.appendingPathComponent("jang_config.json"))
    }

    /// Materialise a fake HuggingFace snapshot at
    /// `<hfCacheRoot>/models--<org>--<name>/snapshots/<hash>/` with
    /// `tokenizer.json` + `tokenizer_config.json` inside.
    @discardableResult
    private func fakeHFSnapshot(
        org: String,
        name: String,
        includingTokenizer: Bool = true,
        hash: String = "0123456789abcdef0123456789abcdef01234567"
    ) throws -> URL {
        let snapshotDir = hfCacheRoot
            .appendingPathComponent("models--\(org)--\(name)")
            .appendingPathComponent("snapshots")
            .appendingPathComponent(hash)
        try FileManager.default.createDirectory(
            at: snapshotDir, withIntermediateDirectories: true)
        if includingTokenizer {
            try writeFile(at: snapshotDir.appendingPathComponent("tokenizer.json"))
            try writeFile(
                at: snapshotDir.appendingPathComponent("tokenizer_config.json"),
                contents: #"{"chat_template":""}"#)
        }
        return snapshotDir
    }

    // MARK: - hasTokenizerFiles

    func testHasTokenizerFilesReturnsFalseForEmptyDirectory() throws {
        let dir = tmpRoot.appendingPathComponent("empty")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        XCTAssertFalse(JangLoader.hasTokenizerFiles(at: dir))
    }

    func testHasTokenizerFilesDetectsTokenizerJson() throws {
        let dir = tmpRoot.appendingPathComponent("tok_json_only")
        try writeFile(at: dir.appendingPathComponent("tokenizer.json"))
        XCTAssertTrue(JangLoader.hasTokenizerFiles(at: dir))
    }

    func testHasTokenizerFilesDetectsTokenizerConfigJson() throws {
        let dir = tmpRoot.appendingPathComponent("tok_config_only")
        try writeFile(at: dir.appendingPathComponent("tokenizer_config.json"))
        XCTAssertTrue(JangLoader.hasTokenizerFiles(at: dir))
    }

    // MARK: - resolveTokenizerDirectory — identity path

    func testResolveReturnsSameDirectoryWhenTokenizerFilesPresent() throws {
        let dir = tmpRoot.appendingPathComponent("standard")
        try writeFile(at: dir.appendingPathComponent("tokenizer.json"))
        try writeJangConfig(
            at: dir,
            sourceModel: ["name": "MiniMax-M2.7", "org": "MiniMaxAI"])

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: dir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, dir,
            "Model dirs that already have tokenizer files must be returned unchanged.")
    }

    func testResolveReturnsSameDirectoryForNonJangModelMissingTokenizer() throws {
        // A standard (non-JANG) model that happens to be missing tokenizer
        // files: the loader should still return the directory unchanged so
        // the downstream tokenizer-load error is surfaced to the caller.
        let dir = tmpRoot.appendingPathComponent("broken_nonjang")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: dir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, dir)
    }

    // MARK: - resolveTokenizerDirectory — JANG fallback path

    func testResolveFallsBackToCachedSourceModelTokenizer() throws {
        // A JANGTQ bundle (weights-only) with a source_model pointer that
        // IS cached locally with a populated tokenizer.
        let jangDir = tmpRoot.appendingPathComponent("MiniMax-M2.7-JANGTQ")
        try writeJangConfig(
            at: jangDir,
            sourceModel: ["name": "MiniMax-M2.7", "org": "MiniMaxAI"])
        let expectedSnapshot = try fakeHFSnapshot(
            org: "MiniMaxAI", name: "MiniMax-M2.7")

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)

        assertSamePath(resolved, expectedSnapshot,
            "JANG dir with no tokenizer should resolve to the cached source-model snapshot.")
    }

    func testResolveFallsBackEvenWithMultipleSnapshots() throws {
        // Real HF caches often have multiple snapshot hashes. Any snapshot
        // with tokenizer files is equally good — we just need to pick one.
        let jangDir = tmpRoot.appendingPathComponent("JANGTQ_multi_snap")
        try writeJangConfig(
            at: jangDir,
            sourceModel: ["name": "M", "org": "org"])
        // First snapshot: no tokenizer files (partial download).
        _ = try fakeHFSnapshot(
            org: "org", name: "M", includingTokenizer: false,
            hash: "aaaa111122223333444455556666777788889999")
        // Second snapshot: has tokenizer files. Should win.
        let good = try fakeHFSnapshot(
            org: "org", name: "M", includingTokenizer: true,
            hash: "bbbb111122223333444455556666777788889999")

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)

        assertSamePath(resolved, good,
            "Resolver must skip snapshots without tokenizer files.")
    }

    // MARK: - resolveTokenizerDirectory — defensive fallbacks

    func testResolveReturnsSelfWhenSourceModelMissingOrg() throws {
        // source_model present but incomplete → no fallback possible.
        let jangDir = tmpRoot.appendingPathComponent("no_org")
        try writeJangConfig(
            at: jangDir,
            sourceModel: ["name": "Some-Model"])  // no org

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, jangDir,
            "Incomplete source_model (missing org) should not trigger fallback.")
    }

    func testResolveReturnsSelfWhenSourceModelMissingName() throws {
        let jangDir = tmpRoot.appendingPathComponent("no_name")
        try writeJangConfig(
            at: jangDir,
            sourceModel: ["org": "OrgOnly"])

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, jangDir)
    }

    func testResolveReturnsSelfWhenSourceBlockAbsent() throws {
        let jangDir = tmpRoot.appendingPathComponent("no_source_model")
        try writeJangConfig(at: jangDir, sourceModel: nil)

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, jangDir)
    }

    func testResolveReturnsSelfWhenSourceSnapshotLacksTokenizer() throws {
        // Source repo is cached but is a partial download (e.g. only the
        // chat template has been fetched). We should not incorrectly route
        // callers to a directory that doesn't actually have the tokenizer.
        let jangDir = tmpRoot.appendingPathComponent("partial_source_cache")
        try writeJangConfig(
            at: jangDir,
            sourceModel: ["name": "Partial", "org": "SomeOrg"])
        _ = try fakeHFSnapshot(
            org: "SomeOrg", name: "Partial", includingTokenizer: false)

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, jangDir,
            "No snapshot has tokenizer files → must not falsely promote one.")
    }

    func testResolveReturnsSelfWhenSourceRepoNotCached() throws {
        let jangDir = tmpRoot.appendingPathComponent("uncached_source")
        try writeJangConfig(
            at: jangDir,
            sourceModel: ["name": "NeverDownloaded", "org": "Unknown"])

        let resolved = JangLoader.resolveTokenizerDirectory(
            for: jangDir, huggingFaceCacheRoot: hfCacheRoot)
        assertSamePath(resolved, jangDir)
    }

    // MARK: - JangSourceModel helpers

    func testHuggingFaceRepoIDFromPopulatedSource() {
        let src = JangSourceModel(name: "MiniMax-M2.7", org: "MiniMaxAI")
        XCTAssertEqual(src.huggingFaceRepoID, "MiniMaxAI/MiniMax-M2.7")
    }

    func testHuggingFaceRepoIDEmptyWhenMissingOrg() {
        let src = JangSourceModel(name: "Model")
        XCTAssertEqual(src.huggingFaceRepoID, "",
            "An empty org (default) should yield an empty repo id.")
    }

    func testHuggingFaceRepoIDEmptyWhenMissingName() {
        let src = JangSourceModel(org: "Org")
        XCTAssertEqual(src.huggingFaceRepoID, "")
    }

    // MARK: - jang_config.json round-trip of new fields

    func testLoadConfigParsesOrgAndArchitectureFromSourceModel() throws {
        let dir = tmpRoot.appendingPathComponent("full_source_block")
        try writeJangConfig(at: dir, sourceModel: [
            "name": "MiniMax-M2.7",
            "org": "MiniMaxAI",
            "architecture": "minimax_m2",
            "dtype": "bfloat16",
            "parameters": "230000000000"
        ])
        let config = try JangLoader.loadConfig(at: dir)
        XCTAssertEqual(config.sourceModel.name, "MiniMax-M2.7")
        XCTAssertEqual(config.sourceModel.org, "MiniMaxAI")
        XCTAssertEqual(config.sourceModel.architecture, "minimax_m2")
        XCTAssertEqual(config.sourceModel.dtype, "bfloat16")
        XCTAssertEqual(config.sourceModel.parameters, "230000000000")
    }
}
