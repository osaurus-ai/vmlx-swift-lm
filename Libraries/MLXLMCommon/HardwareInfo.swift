// Copyright © 2026 Apple Inc. and JANG Research

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
    /// Returns `true` on Apple Silicon M3 or later where `compile(shapeless: true)`
    /// works correctly with the Metal JIT.
    ///
    /// Returns `false` on:
    /// - M1/M2 chips (A14/A15 GPU) — Metal JIT crashes with compile(shapeless: true)
    /// - Intel Macs — no Metal GPU or older GPU
    /// - Unknown hardware — safe default
    ///
    /// This property is intended to gate `compile(shapeless: true)` usage in
    /// activation functions and other compiled closures. When `false`, fall back
    /// to non-compiled closure evaluation (individual Metal ops, which work fine
    /// on all hardware).
    public static var isCompiledDecodeSupported: Bool {
        guard isAppleSilicon else { return false }

        let identifier = machineIdentifier

        // M3+ machine identifiers (A16/A17/A18 GPU — g8x family):
        // Mac15,10+ = M3 Pro/Max (MacBook Pro)
        // Mac15,11+ = M3 Air
        // Mac15,12  = M3 Pro 14"
        // Mac15,13  = M3 Max 16"
        // Mac16,3+  = M4 family (MacBook Pro, Mac mini, iMac)
        // Mac16,8   = M4 Pro Mac mini
        // Mac16,5   = M3 Max MacBook Pro
        // Mac16,6   = M3 Pro MacBook Pro
        // Mac16,7   = M3 Max MacBook Pro
        // Mac16,8   = M4 Pro Mac mini
        // Mac16,9   = M4 Max Mac Pro
        // Mac16,10  = M4 Pro MacBook Pro
        // Mac16,11  = M4 Max MacBook Pro
        // Mac16,12  = M4 Air
        if identifier.hasPrefix("Mac15,10") ||
           identifier.hasPrefix("Mac15,11") ||
           identifier.hasPrefix("Mac15,12") ||
           identifier.hasPrefix("Mac15,13") ||
           identifier.hasPrefix("Mac16,3") ||
           identifier.hasPrefix("Mac16,5") ||
           identifier.hasPrefix("Mac16,6") ||
           identifier.hasPrefix("Mac16,7") ||
           identifier.hasPrefix("Mac16,8") ||
           identifier.hasPrefix("Mac16,9") ||
           identifier.hasPrefix("Mac16,10") ||
           identifier.hasPrefix("Mac16,11") ||
           identifier.hasPrefix("Mac16,12") {
            return true
        }

        // M1 (Mac13,*, Mac14,*) and M2 (Mac15,7, Mac15,8) — A14/A15 GPU (g7x family)
        // These have the Metal JIT bug with compile(shapeless: true)
        if identifier.hasPrefix("Mac13") ||
           identifier.hasPrefix("Mac14") ||
           identifier.hasPrefix("Mac15,7") ||
           identifier.hasPrefix("Mac15,8") {
            return false
        }

        // Unknown Apple Silicon — conservative false
        return false
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
