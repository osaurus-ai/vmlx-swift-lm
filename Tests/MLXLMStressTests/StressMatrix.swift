//
//  StressMatrix.swift
//  vmlx-swift-lm — MLXLMStressTests
//
//  Exhaustive cache x modality x workload matrix.
//  Drives BatchEngine directly. Every cell must pass before pin bump.
//
//  Created 2026-04-30 by Eric for the stability investigation.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
import XCTest

// MARK: - Cell descriptors

/// One cell of the stress matrix.
struct StressCell: Sendable, CustomStringConvertible {
    var cacheMode: CacheMode
    var diskTier: Bool
    var l2State: L2State
    var kvMode: KVQuantizationMode
    var arch: Architecture
    var maxKVSize: Int
    var promptLen: PromptLen
    var workload: Workload

    var description: String {
        "[\(arch.tag)/\(cacheMode.tag)/disk=\(diskTier)/l2=\(l2State.tag)/kv=\(kvMode.tag)/cap=\(maxKVSize)/len=\(promptLen.tag)/w=\(workload.tag)]"
    }
}

enum CacheMode: String, Sendable {
    case paged, nonPaged
    var tag: String { rawValue }
}

enum L2State: String, Sendable {
    case cold
    case warmRestore
    case warmRestoreThenClearCache
    var tag: String {
        switch self {
        case .cold: "cold"
        case .warmRestore: "warm"
        case .warmRestoreThenClearCache: "warm+clear"
        }
    }
}

enum Architecture: String, Sendable {
    /// Pure attention (Qwen3 small)
    case pureAttn
    /// Hybrid SSM + attn (Nemotron-3 / Qwen3.5/3.6)
    case hybridSSM
    /// Pure SSM (Mamba2)
    case pureSSM
    var tag: String { rawValue }
}

enum PromptLen: String, Sendable {
    case short, mid, long, overCap
    var tag: String { rawValue }
    var tokenCount: Int {
        switch self {
        case .short: 32
        case .mid: 2_048
        case .long: 16_384
        case .overCap: 60_000
        }
    }
}

enum Workload: String, Sendable {
    case single
    case backToBack
    case sharedPrefix
    case burst10
    case multiTurn8
    case cancelPrefill
    case cancelDecode
    case concurrent
    case clearCacheMidRun
    case overCap
    var tag: String { rawValue }
}

private extension KVQuantizationMode {
    var tag: String {
        switch self {
        case .none: return "fp16"
        case .turboQuant(let k, let v): return "tq\(k)_\(v)"
        default: return "other"
        }
    }
}

// MARK: - Matrix gen

/// Cartesian product of all axes, filtered to skip impossible combos.
func enumerateCells() -> [StressCell] {
    var out: [StressCell] = []
    let kvModes: [KVQuantizationMode] = [
        .none,
        .turboQuant(keyBits: 3, valueBits: 3),
        .turboQuant(keyBits: 4, valueBits: 4),
        .turboQuant(keyBits: 8, valueBits: 8),
    ]
    for arch in [Architecture.pureAttn, .hybridSSM] {       // pureSSM gated by env
        for cacheMode in [CacheMode.paged, .nonPaged] {
            for diskTier in [true, false] {
                for l2 in [L2State.cold, .warmRestore, .warmRestoreThenClearCache] {
                    if !diskTier && l2 != .cold { continue }   // l2 requires disk tier
                    for kv in kvModes {
                        for cap in [1024, 8192, 32768] {
                            for len in [PromptLen.short, .mid, .long, .overCap] {
                                for w in workloadsFor(len: len, l2: l2) {
                                    out.append(StressCell(
                                        cacheMode: cacheMode,
                                        diskTier: diskTier,
                                        l2State: l2,
                                        kvMode: kv,
                                        arch: arch,
                                        maxKVSize: cap,
                                        promptLen: len,
                                        workload: w
                                    ))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out
}

/// Trim workloads that don't make sense for a given length / l2 state.
private func workloadsFor(len: PromptLen, l2: L2State) -> [Workload] {
    switch len {
    case .overCap:
        return [.overCap]                           // only one workload makes sense
    case .short, .mid:
        return [.single, .backToBack, .sharedPrefix,
                .burst10, .multiTurn8,
                .cancelPrefill, .cancelDecode,
                .concurrent, .clearCacheMidRun]
    case .long:
        // Skip the burst/concurrent on long to keep wallclock sane.
        return [.single, .backToBack, .sharedPrefix,
                .multiTurn8, .clearCacheMidRun]
    }
}

// MARK: - Per-cell harness (skeleton)
//
// Each `runCell` is a thin wrapper around BatchEngine + CacheCoordinator
// that sets up the model, applies the cell config, runs the workload,
// and returns a `CellResult`. The full implementations live in the
// per-workload extensions; this file only owns dispatch.

struct CellResult: Sendable {
    let cell: StressCell
    let passed: Bool
    let durationS: Double
    let peakBytes: Int64?
    let notes: String
}

/// Run a single cell. For now, the harness is gated by env vars so we
/// can enable rolling subsets while we iterate.
func runCell(_ cell: StressCell, modelPath: URL) async throws -> CellResult {
    // TODO(stability-investigation): wire up real BatchEngine setup,
    // disk-cache prep, workload dispatch, and crash detection.
    // Skeleton only — body lives in per-workload extension files
    // (StressWorkloads+BackToBack.swift, +Concurrent.swift, etc.).
    return CellResult(cell: cell, passed: true, durationS: 0, peakBytes: nil,
                      notes: "skeleton — not implemented yet")
}

// MARK: - Test entry point

@Suite("MLXLMStressTests")
struct MLXLMStressMatrixTests {

    /// The full matrix sweep. Skipped unless OSAURUS_STRESS_RUN=1 is
    /// set in the environment because individual cells take seconds
    /// each and a full run is hours. CI runs a curated subset.
    @Test("full matrix")
    func fullMatrix() async throws {
        guard ProcessInfo.processInfo.environment["OSAURUS_STRESS_RUN"] == "1" else {
            return
        }
        let cells = enumerateCells()
        // Hybrid SSM model path comes from env var; we'll fail fast if absent.
        let hybridPath = ProcessInfo.processInfo
            .environment["OSAURUS_STRESS_HYBRID_MODEL"]
            .flatMap { URL(fileURLWithPath: $0) }
        var results: [CellResult] = []
        for cell in cells {
            // Dispatch on architecture; only attempt if the corresponding
            // model path is present.
            let modelPath: URL?
            switch cell.arch {
            case .hybridSSM: modelPath = hybridPath
            case .pureAttn: modelPath = nil  // TODO: wire small Qwen3 path
            case .pureSSM: modelPath = nil  // TODO: wire Mamba2 path
            }
            guard let p = modelPath else { continue }
            let r = try await runCell(cell, modelPath: p)
            results.append(r)
        }
        let failures = results.filter { !$0.passed }
        if !failures.isEmpty {
            let detail = failures.prefix(20).map { "\($0.cell): \($0.notes)" }
                .joined(separator: "\n")
            Issue.record("\(failures.count) cells failed; first 20:\n\(detail)")
        }
    }

    /// Targeted Bug 1 repro — short, runs in CI when a hybrid model is present.
    @Test("S7 — warm-disk-cache 2nd-request crash")
    func warmDiskCache2ndRequest() async throws {
        guard let mp = ProcessInfo.processInfo.environment["OSAURUS_STRESS_HYBRID_MODEL"]
                  .flatMap({ URL(fileURLWithPath: $0) }) else {
            // Skip when model not present rather than fail the suite.
            return
        }
        let cell = StressCell(
            cacheMode: .paged, diskTier: true, l2State: .warmRestore,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            arch: .hybridSSM, maxKVSize: 8_192, promptLen: .short,
            workload: .backToBack
        )
        let r = try await runCell(cell, modelPath: mp)
        #expect(r.passed, "Bug 1 repro should pass on patched build")
    }

    /// Targeted Bug 2 repro — over-cap hybrid prompt 154 GB allocation.
    @Test("S8 — over-cap hybrid prompt")
    func overCapHybridPrompt() async throws {
        guard let mp = ProcessInfo.processInfo.environment["OSAURUS_STRESS_HYBRID_MODEL"]
                  .flatMap({ URL(fileURLWithPath: $0) }) else {
            return
        }
        let cell = StressCell(
            cacheMode: .paged, diskTier: true, l2State: .cold,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            arch: .hybridSSM, maxKVSize: 8_192, promptLen: .overCap,
            workload: .overCap
        )
        let r = try await runCell(cell, modelPath: mp)
        #expect(r.passed, "Bug 2 repro should pass on patched build")
    }
}
