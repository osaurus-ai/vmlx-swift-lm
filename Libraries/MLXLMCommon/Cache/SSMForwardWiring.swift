// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Sendable wrapper around a non-Sendable ``LanguageModel`` reference and
/// the ``GenerateParameters`` used to allocate caches for the re-derive
/// path.
///
/// ## Why this exists
///
/// ``SSMReDeriver`` consumes two `@Sendable` closures: one to run a
/// forward chunk, one to allocate a fresh cache array. Both need to call
/// through the loaded ``LanguageModel``, which is declared as `any
/// LanguageModel` — a protocol existential that is not `Sendable` under
/// strict concurrency because protocol authors cannot guarantee the
/// property across every conforming type.
///
/// In practice the model is safe to call from any actor because:
///   - MLX serializes Metal command-buffer submissions through a global
///     `evalLock` at the C++ layer.
///   - `LanguageModel.callAsFunction` reads model parameters that are
///     loaded once and never mutated afterwards.
///   - `LanguageModel.newCache` produces a fresh per-call cache — no
///     shared mutable state.
///
/// Capturing the model directly in a `@Sendable` closure would require
/// `nonisolated(unsafe)` at the capture site. Using the closure as a
/// `sending` parameter then trips a strict-concurrency data-race warning
/// because the compiler can't see the capture's safety. This wrapper
/// hides the capture in a type explicitly marked
/// `@unchecked Sendable` — the unsafety is declared once, in one
/// auditable place, instead of scattered across every call site that
/// wires a re-deriver.
///
/// ## Usage
///
/// ```swift
/// let wiring = SSMForwardWiring(model: context.model, parameters: params)
/// await reDeriver.wireModel(
///     forward:  wiring.makeForward(),
///     newCache: wiring.makeNewCache()
/// )
/// ```
public struct SSMForwardWiring: @unchecked Sendable {

    /// The loaded model instance. Held by reference — not copied per
    /// call. Safe because MLX forward passes serialize through the
    /// global `evalLock` in the mlx core.
    private let model: any LanguageModel

    /// Parameters that control cache allocation (e.g. `maxKVSize`,
    /// `kvMode`). Value-type; captured once at construction.
    private let parameters: GenerateParameters

    public init(model: any LanguageModel, parameters: GenerateParameters) {
        self.model = model
        self.parameters = parameters
    }

    /// Build the forward-chunk closure. The capture is marked
    /// `nonisolated(unsafe)` here because the enclosing
    /// ``SSMForwardWiring`` is already `@unchecked Sendable` and
    /// encapsulates the data-race audit — see the type-level doc.
    public func makeForward() -> SSMForwardChunk {
        nonisolated(unsafe) let model = self.model
        return { tokens, cache in
            _ = model(
                LMInput.Text(tokens: tokens),
                cache: cache,
                state: nil
            )
        }
    }

    /// Build the cache-allocator closure. Same `nonisolated(unsafe)`
    /// justification as ``makeForward()``.
    public func makeNewCache() -> SSMCacheAllocator {
        nonisolated(unsafe) let model = self.model
        let parameters = self.parameters
        return { model.newCache(parameters: parameters) }
    }
}
