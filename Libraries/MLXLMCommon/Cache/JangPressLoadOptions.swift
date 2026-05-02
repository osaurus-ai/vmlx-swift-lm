// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressLoadOptions — opt-in surface for JangPress (axis E,
// cold-weight tier). Default-off; consumers (osaurus, JANG Studio,
// CLI tools) explicitly construct an instance and pass it through
// `loadWeights(...)` to enable the feature.
//
// See JANGPRESS-VMLX-SWIFT-LM-INTEGRATION-2026-05-02.md for the full
// SDK-side wiring and JANGPRESS-AGENTS.md for the per-host
// recommendation table.

import Foundation

public struct JangPressLoadOptions: Sendable, Equatable {
    /// Master switch. Default `false` — JangPress is opt-in.
    public var enabled: Bool

    /// 0..100 — % of routed-MoE weight mass open to compaction during
    /// quiesce. `0` arms the failsafe controller without compacting
    /// anything (kernel reclaim under pressure still works);
    /// `70` is the production-recommended value for tight hosts;
    /// `100` keeps only the top-k hot expert set pinned.
    public var compressPct: Int

    /// Backend selection. `.mmap` is the production default
    /// (file-backed, page-cache shared with MLX, zero RAM doubling).
    /// `.mach` is gated on the MLX-swift fork; doubles RAM at load
    /// because it allocates fresh purgeable VM regions. `.none`
    /// disables the routed-expert tier even if `enabled == true`
    /// (still arms the embed/lm_head Zipfian tier).
    public var backend: Backend

    /// Eviction aggressiveness for the `.mmap` backend.
    /// `.soft` issues `madvise(MADV_DONTNEED)` — kernel HINTS, ignored
    /// when free RAM is plentiful. **Failsafe default.**
    /// `.force` issues `msync(MS_INVALIDATE)` — kernel drops pages
    /// immediately. Use only on memory-constrained hosts where eager
    /// reclaim is required and cold-fault latency is acceptable.
    public var forceMode: ForceMode

    /// Pre-fault top-`hotPercent` of tiles at arm time. Defaults to
    /// `true`. Disabling adds within-process drift at temperature 0
    /// (see JANGPRESS-DEEP-TRACE Issue 5 for the full analysis).
    public var enablePrefetch: Bool

    public enum Backend: String, Sendable, Equatable {
        case mmap, mach, none
    }

    public enum ForceMode: String, Sendable, Equatable {
        case soft, force
    }

    /// Sane production-grade default — feature off, ready to be turned
    /// on per-call via the constructor parameters below.
    public static let disabled = JangPressLoadOptions()

    public init(
        enabled: Bool = false,
        compressPct: Int = 70,
        backend: Backend = .mmap,
        forceMode: ForceMode = .soft,
        enablePrefetch: Bool = true
    ) {
        self.enabled = enabled
        self.compressPct = max(0, min(100, compressPct))
        self.backend = backend
        self.forceMode = forceMode
        self.enablePrefetch = enablePrefetch
    }
}

/// JangPress runtime handles attached to a `ModelContext` after
/// `loadWeights(...)` instantiates the tiers. All four are nil when
/// `JangPressLoadOptions.enabled == false` (the default). When
/// enabled, exactly one of `mmap` / `mach` is non-nil per the
/// configured backend; `controller` is always non-nil; `embed` is
/// co-instantiated.
///
/// `BatchEngine` and `TokenIterator` use `controller` for the
/// `willStartInference()` / `didFinishInference()` brackets that
/// keep the failsafe state machine ticking.
public struct JangPressRuntime: Sendable {
    public weak var mmap: JangPressMmapTier?
    public weak var mach: JangPressMachCache?
    public weak var embed: JangPressEmbedTier?
    public weak var controller: JangPressController?

    public init(
        mmap: JangPressMmapTier? = nil,
        mach: JangPressMachCache? = nil,
        embed: JangPressEmbedTier? = nil,
        controller: JangPressController? = nil
    ) {
        self.mmap = mmap
        self.mach = mach
        self.embed = embed
        self.controller = controller
    }

    public static let none = JangPressRuntime()

    public var isActive: Bool {
        controller != nil
    }
}
