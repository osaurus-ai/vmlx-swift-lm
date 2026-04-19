// Copyright 2025 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Stage 4 probe — can the existing `MambaCache` / `ArraysCache` be captured
// by `MLX.compile(...)` at all? Hybrid SSM models (Qwen3.5 Mamba,
// Qwen3Next/GDN, LFM2, LFM2MoE, Jamba, GraniteMoeHybrid, NemotronH) depend
// on this.
//
// ## Observed behaviour (2026-04-18)
//
// Attempting to compile a closure that mutates a `MambaCache` via its
// `subscript(index:)` setter triggers a **fatal** MLX error:
//
//   ```
//   MLX/ErrorHandler.swift:343: Fatal error: [compile]
//     Attempting to compile a function with uncaptured inputs is not
//     allowed. at mlx-c/mlx/c/closure.cpp:104
//   ```
//
// Unlike the Stage 3 probe (which returned stale logits silently), the
// Stage 4 probe **crashes the test process** — MLX's tracer refuses to
// accept the closure outright. This is a stronger signal that
// `ArraysCache` state can't be treated as compile input without
// modification.
//
// ## Root cause (hypothesis)
//
// `ArraysCache` stores state as `[MLXArray?]` (optional elements) rather
// than direct MLXArray properties. `innerState()` returns
// `cache.compactMap { $0 }` which creates a new Swift array each call.
// The compile tracer likely can't flatten this indirection into stable
// state inputs, so it flags the closure's reads/writes as referring to
// uncaptured state.
//
// `CompilableKVCache` sidesteps this by exposing `keys` / `values` /
// `offsetArray` as direct MLXArray properties. `TurboQuantKVCache` has
// direct `unifiedKeys` / `unifiedValues` properties as well.
//
// ## Stage 4 implications
//
// `CompilableMambaCache` must do one of:
//
//   1. Expose its conv state + hidden state as direct MLXArray properties
//      (not `[MLXArray?]`).
//   2. Write a custom compile adapter that provides explicit inputs/outputs
//      for each state slot.
//
// The spec §9 design anticipates option (1): a dedicated class with
// traceable-in-place updates and a layout compile can introspect.
//
// ## What this file ships
//
// The probe tests are marked `XCTSkip` — invoking them currently crashes
// the process. Keeping the tests + skip reason in the repo gives the
// next iteration a ready-to-flip place to verify a `CompilableMambaCache`
// fix once it ships.

import Foundation
import MLX
import MLXLMCommon
import XCTest

class MambaCacheCompileProbeTests: XCTestCase {

    private let stage4Blocker = """
        Stage 4 blocker confirmed: MLX compile rejects closures that read/write
        MambaCache via `subscript(index:)`. Fatal error at
        `closure.cpp:104`: "Attempting to compile a function with uncaptured
        inputs is not allowed." Running this test crashes the process — a
        `CompilableMambaCache` with direct MLXArray state properties is
        required. See Stage 4 probe notes in BATCH_ENGINE.md.
        """

    /// Probe 1 (skipped): confirmed to crash the test process. See the
    /// class-level doc comment for the root cause and the `stage4Blocker`
    /// message describing the exact failure mode.
    func testMambaCacheSurvivesCompileTrace() async throws {
        throw XCTSkip(stage4Blocker)
    }

    /// Probe 2 (skipped): same failure mode as probe 1.
    func testMambaCacheCompiledVsUncompiled() async throws {
        throw XCTSkip(stage4Blocker)
    }
}
