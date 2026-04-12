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

        // Parse "MacNN,MM" to get generation number.
        // M1 = Mac13/Mac14, M2 = Mac15,3-Mac15,8, M3 = Mac15,10+, M4 = Mac16+
        // Mac17+ = future chips (M5+), always supported.
        guard let genStr = identifier.split(separator: ",").first,
              let gen = Int(genStr.dropFirst(3)) else {
            return false
        }

        // Mac16+ (M4 and later) — always supported
        if gen >= 16 { return true }

        // Mac15,10+ = M3 family (A16/A17 GPU, g8x family — Metal JIT works)
        // Mac15,3-Mac15,8 = M2 family (A15 GPU, g7x — Metal JIT bug)
        if gen == 15 {
            if let suffix = identifier.split(separator: ",").last,
               let model = Int(suffix) {
                return model >= 10  // Mac15,10+ = M3
            }
            return false
        }

        // Mac13, Mac14 = M1 family — Metal JIT bug
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
