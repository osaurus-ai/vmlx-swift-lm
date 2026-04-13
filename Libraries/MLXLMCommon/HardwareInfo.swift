// Copyright © 2026 Osaurus AI

import Foundation

/// Hardware capability detection for Apple Silicon chip generations.
///
/// This module provides runtime detection of Apple Silicon GPU families to gate
/// Metal JIT features that behave differently across chip generations.
///
/// ## Background
///
/// `compile(shapeless: true)` in MLX wraps closures in a `CompiledFunction` that
/// calls the C++ `Compiled::eval_gpu`. On certain macOS Tahoe GPU drivers (particularly
/// M1/M2 with A14/A15 GPU), this compiled kernel path returns zero results instead
/// of the expected array. This causes an `Index out of range` crash when Swift code
/// accesses `compileState.call([a])[0]` on an empty array.
///
/// - M1/M2: A14/A15 GPU (g7x family) — **Metal JIT bug present**
/// - M3+: A16/A17/A18 GPU (g8x family) — Metal JIT works correctly
///
/// See: MLX issues #3329, #3201, #3256
public enum HardwareInfo {
    /// Returns `true` when `compile(shapeless: true)` is safe to use.
    ///
    /// Currently returns `false` unconditionally. The macOS Tahoe Metal JIT bug
    /// (MLX #3329, #3201, #3256) causes `Compiled::eval_gpu` to return zero results,
    /// crashing at `compiledState.callsToFill[0]` (Index out of range).
    ///
    /// Originally gated by chip generation (M1/M2 = false, M3+ = true), but the
    /// crash was confirmed on M4 Pro (Mac16,x) as well — the bug is a macOS Tahoe
    /// Metal shader compiler issue, not hardware-specific.
    ///
    /// Performance impact of disabling: minimal. These are small activation function
    /// fusions (GELU, SwiGLU, softcap). The individual Metal ops work correctly on
    /// all hardware and the per-op overhead is negligible vs. the model forward pass.
    ///
    /// Re-enable when Apple fixes the Metal JIT in a future macOS update.
    public static var isCompiledDecodeSupported: Bool {
        // Re-enabled 2026-04-13 after direct measurement on M4 Max + mlx-swift osaurus-0.31.3:
        //
        // The broad kill-switch was set in commit a8a6a6f citing MLX#3329 "compiledState
        // callsToFill[0] index out of range" on M4 Pro. That crash was empirically tied to
        // the whole-model compile path (setupCompiledDecode + CompilableKVCache), not to
        // the small fixed-shape micro-fusions this flag now controls.
        //
        // Commit cf55f6d re-enabled compile(shapeless: true) for compute_g on M4 Max with
        // no crash and a measurable +1.9 tok/s on Qwen 3.5-35B decode. Python mlx_lm 0.31.2
        // on the same machine runs ~9 tok/s faster with its compile islands vs without
        // (120/117/114/106/104 w/ compile vs 110/105/103/100/98 with mx.disable_compile()),
        // which is where Swift's decode gap comes from.
        //
        // This flag now gates only the per-op micro-fusion helpers (swiglu, precise_swiglu,
        // geglu, softcap, sigmoid-gate etc.). The whole-model compile path in
        // setupCompiledDecode is still gated separately by `GenerateParameters.enableCompiledDecode`
        // and remains off by default, so flipping this flag cannot revive the MLX#3329 crash.
        //
        // If the crash is observed again on any specific site, opt that site out individually
        // instead of flipping this flag back to false globally.
        return true
    }

    /// Returns `true` if running on Apple Silicon hardware.
    private static var isAppleSilicon: Bool {
        #if os(macOS) && arch(arm64)
        return true
        #else
        return false
        #endif
    }

    /// Returns the hardware machine identifier string (e.g., "Mac15,10", "Mac14,7").
    ///
    /// Queried via `sysctl hw.machine` at runtime.
    private static var machineIdentifier: String {
        var size: Int = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        guard size > 0 else { return "" }
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        let count = machine.firstIndex(of: 0) ?? machine.count
        let bytes = machine.prefix(count).map { UInt8(bitPattern: $0) }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }
}
