// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressActivation — single-call helper that instantiates the four
// JangPress runtime objects per `JangPressLoadOptions` and returns
// them as a `JangPressRuntime` for the caller to attach to whatever
// per-model context they hold.
//
// Why a separate function (and not a method on `ModelContext`)? The
// `ModelContext` struct lives in `MLXLMCommon` but is value-typed —
// extending it with `var jangPress: JangPressRuntime?` would change
// its memberwise init signature and break source-compat for every
// existing consumer. The helper here is purely additive: callers that
// don't use JangPress are unaffected; callers that do call
// `JangPressActivation.activate(...)` once after `loadModel` and
// stash the returned `JangPressRuntime` next to their `ModelContext`.
//
// See JANGPRESS-VMLX-SWIFT-LM-INTEGRATION-2026-05-02.md for the full
// integration spec.

import Foundation

public enum JangPressActivation {

    /// Instantiate the JangPress tiers per `options` and arm the
    /// controller. Returns `JangPressRuntime.none` when
    /// `options.enabled == false` so callers can call this
    /// unconditionally and let the option toggle gate the work.
    ///
    /// Failure modes are non-fatal: any thrown error from the tier
    /// constructors is logged to stderr and the function returns
    /// `JangPressRuntime.none`. The caller's inference path keeps
    /// working with full-resident weights (the failsafe contract).
    ///
    /// - Parameters:
    ///   - bundleURL: the model bundle directory (same path passed to
    ///     `loadModel`).
    ///   - options: `JangPressLoadOptions(enabled: ..., compressPct:
    ///     ..., backend: ..., forceMode: ..., enablePrefetch: ...)`.
    ///     Use `.disabled` for the no-op default.
    /// - Returns: `JangPressRuntime` with the instantiated tiers
    ///   (controller + mmap-or-mach + embed). All weak references
    ///   so the caller controls the lifetime by holding strong
    ///   references inline.
    public static func activate(
        bundleURL: URL,
        options: JangPressLoadOptions
    ) -> JangPressRuntime {
        guard options.enabled else { return .none }

        // Routed-expert tier — pick backend per `options.backend`.
        var mmapTier: JangPressMmapTier?
        var machCache: JangPressMachCache?
        switch options.backend {
        case .mmap:
            do {
                // `startCold: !enablePrefetch` is the prefetch knob —
                // when prefetch is on (default), all routed tiles
                // start resident (the mmap region is faulted in on
                // first access; the kernel pre-reads with
                // MADV_WILLNEED via the JangPressShard ctor). When
                // prefetch is off, every tile starts MADV_DONTNEED
                // and the first inference's `acquire()` calls fault
                // them in — useful for memory-tight startups.
                // Semantics flip: caller-facing `compressPct=70` means
                // "70% of routed mass is open to compression",
                // so the tier's `hotPercent = 100 - compressPct = 30`
                // (the fraction kept MADV_WILLNEED at all times).
                let cfg = JangPressMmapConfig(
                    bundleURL: bundleURL,
                    hotPercent: 100 - options.compressPct,
                    startCold: !options.enablePrefetch)
                mmapTier = try JangPressMmapTier(config: cfg)
            } catch {
                FileHandle.standardError.write(Data(
                    "[JangPressActivation] mmap-tier init failed: \(error)\n".utf8))
                return .none
            }
        case .mach:
            // .mach allocates fresh purgeable VM regions and would
            // double RAM at load. Gated on the MLX-swift fork. For
            // now we surface a clear error and fall through to
            // disabled — when the fork lands, the MachCache.register
            // path will be the right place to populate it.
            let msg = "[JangPressActivation] backend=.mach not yet wired in vmlx-swift-lm; falls back to disabled. Use .mmap for production.\n"
            FileHandle.standardError.write(Data(msg.utf8))
            return .none
        case .none:
            break
        }

        // Embed/lm_head Zipfian tier — co-instantiated regardless of
        // routed-expert backend choice. Independent failure surface;
        // returns nil on missing keys (e.g. tied embeddings, no lm_head).
        var embedTier: JangPressEmbedTier?
        do {
            // Map compressPct → hot percent for embed/lm_head: at
            // pct=70 we keep ~5% hot rows; at pct=0 we keep ~30%.
            let embedHot = max(1, min(50, 30 - (options.compressPct / 4)))
            let cfg = JangPressEmbedConfig(
                bundleURL: bundleURL,
                hotPercent: embedHot,
                skipLMHead: false)
            embedTier = try JangPressEmbedTier(config: cfg)
        } catch {
            FileHandle.standardError.write(Data(
                "[JangPressActivation] embed-tier init failed (non-fatal): \(error)\n".utf8))
            // Continue — routed-expert tier still useful even if embed
            // discovery missed the bundle's keys.
        }

        // Controller — wire the routed-expert backend and arm.
        let controller: JangPressController
        if let mmapTier {
            controller = JangPressController(
                mmapTier: mmapTier,
                keepHotFraction: 1.0 - Double(options.compressPct) / 100.0,
                useForceRelease: options.forceMode == .force)
        } else if let machCache {
            controller = JangPressController(
                cache: machCache,
                keepHotFraction: 1.0 - Double(options.compressPct) / 100.0)
        } else {
            // Backend == .none and embed-only — return runtime with
            // just the embed tier. No controller bracket needed.
            return JangPressRuntime(
                mmap: nil, mach: nil, embed: embedTier, controller: nil)
        }
        controller.arm()

        return JangPressRuntime(
            mmap: mmapTier,
            mach: machCache,
            embed: embedTier,
            controller: controller)
    }

    /// Disarm the controller and release the tiers. Safe to call on a
    /// `JangPressRuntime.none` (no-op).
    ///
    /// Call from `unloadModel` paths so the memory-pressure listener
    /// + quiesce timer don't leak past the model's lifetime.
    public static func deactivate(_ runtime: JangPressRuntime) {
        runtime.controller?.disarm()
    }
}
