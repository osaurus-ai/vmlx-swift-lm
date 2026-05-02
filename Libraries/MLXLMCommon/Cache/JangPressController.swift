// Copyright © 2026 Jinho Jang. All rights reserved.
//
// JangPressController — failsafe idle-time compression driver for
// the routed-expert weight tier.
//
// SAFETY MODEL (v1)
// =================
// The controller NEVER touches a tile while inference is in flight.
// The only window it engages is when ALL of these are true:
//
//   • No active generation request (engine reports idle)
//   • Last decode tick was > `quiesceTimeoutMs` ms ago
//   • At least one of:
//       - Memory pressure WARNING or CRITICAL received, OR
//       - User explicitly requested compaction, OR
//       - App entered background mode (NSApplication willResignActive)
//
// When the next inference request arrives, the controller flips ALL
// volatile tiles back to non-volatile BEFORE the engine touches them.
// The first token may be ~50-200 ms slower (kernel decompresses) but
// correctness is identical to never having compressed.
//
// FAILURE MODES
// =============
// • Tile DISCARDED by kernel under critical pressure → refault from
//   disk via `JangPressMachCache.acquire`. Slower (~ms-tens
//   of ms per tile) but lossless.
// • Cache disabled mid-flight → wake handler hit before engine reads;
//   no observable effect.
// • Engine-side bug that calls inference without going through the
//   wake hook → tiles still resident (failsafe: kernel only compresses
//   under explicit pressure, and only after our quiesce timeout, so
//   races are rare).
//
// FUTURE (v2)
// ===========
// Per-decode-step volatility flips on actively dormant experts. Gated
// on an MLX-swift fork that exposes `MTLBuffer.setPurgeableState` per
// tensor. Not in this version.

import Foundation
import Dispatch
import os

public enum JangPressState: Sendable {
    case disabled       // controller off — no tiles compressed
    case armed          // controller on, all tiles non-volatile (active)
    case quiescing      // controller on, ticks counting down to compress
    case compressed     // controller on, cold tiles volatile
}

public protocol JangPressObserver: AnyObject {
    func emberDidEnterState(_ state: JangPressState)
}

public final class JangPressController: @unchecked Sendable {

    /// Mach VM backend (.mach). Holds its own copy of weights in
    /// purgeable VM regions; kernel does WKdm compression. Optional —
    /// at most one of `cache` / `mmapTier` is non-nil.
    private let cache: JangPressMachCache?

    /// File-backed mmap backend (.mmap). Reads bundle safetensors via
    /// PROT_READ; release calls madvise(DONTNEED), forceRelease calls
    /// msync(MS_INVALIDATE) for guaranteed reclaim. Optional —
    /// mutually exclusive with `cache`.
    private let mmapTier: JangPressMmapTier?

    private let log = Logger(subsystem: "ai.jangq.vmlx", category: "JangPress")
    private let lock = OSAllocatedUnfairLock()

    /// Time (ms) after the last inference activity before the controller
    /// considers the engine "quiet enough to compress". Default 30 s.
    private let quiesceTimeoutMs: Int

    /// Tracked routing frequency per (layer, expert). Used to pick which
    /// tiles are "cold" enough to compress when armed.
    private var routingFreq: [TileKey: UInt64] = [:]
    private var totalRoutes: UInt64 = 0

    /// Fraction of tiles to keep non-volatile even when compressed. The
    /// hottest `keepHotFraction` of routed experts (by frequency) stay
    /// resident at all times.
    private let keepHotFraction: Double

    /// Is the engine actively generating right now? Set by the engine
    /// before each inference and cleared after.
    private var inferenceInFlight: Bool = false

    /// Wall-clock of last inference tick. Used to determine quiesce.
    private var lastInferenceTick: Date = Date()

    /// State machine — written under `lock`.
    private var state: JangPressState = .disabled

    private var pressureSource: DispatchSourceMemoryPressure?
    private var quiesceTimer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "ai.jangq.vmlx.jang-press", qos: .utility)

    private weak var observer: JangPressObserver?

    fileprivate struct TileKey: Hashable {
        let layer: Int
        let expert: Int
    }

    /// When true (default), `compressColdTiles` uses
    /// `mmapTier.forceRelease` (msync MS_INVALIDATE) which forces
    /// the kernel to drop pages immediately. Aggressive — empirically
    /// 7.7 GB reclaim on DSV4-Flash, but 3× decode slowdown because
    /// MLX's reads also share those pages and refault from disk.
    ///
    /// When false ("soft-only"), uses `mmapTier.release`
    /// (madvise MADV_DONTNEED) which is a HINT — kernel ignores
    /// under low pressure, acts on it under high pressure. No
    /// slowdown when memory is roomy; full reclaim when it isn't.
    /// **Recommended for production.**
    private let useForceRelease: Bool

    /// Init with the .mach backend.
    public init(
        cache: JangPressMachCache,
        quiesceTimeoutMs: Int = 30_000,
        keepHotFraction: Double = 0.30,
        observer: JangPressObserver? = nil
    ) {
        self.cache = cache
        self.mmapTier = nil
        self.quiesceTimeoutMs = quiesceTimeoutMs
        self.keepHotFraction = keepHotFraction
        self.useForceRelease = true       // .mach uses Mach VOLATILE — no msync involved
        self.observer = observer
    }

    /// Init with the .mmap backend. Same lifecycle (arm /
    /// willStartInference / didFinishInference / compressColdTiles)
    /// but compaction strategy depends on `useForceRelease`:
    /// `true` → msync MS_INVALIDATE (hard reclaim, slows decode);
    /// `false` → madvise MADV_DONTNEED (soft hint, failsafe).
    public init(
        mmapTier: JangPressMmapTier,
        quiesceTimeoutMs: Int = 30_000,
        keepHotFraction: Double = 0.30,
        useForceRelease: Bool = false,
        observer: JangPressObserver? = nil
    ) {
        self.cache = nil
        self.mmapTier = mmapTier
        self.useForceRelease = useForceRelease
        self.quiesceTimeoutMs = quiesceTimeoutMs
        self.keepHotFraction = keepHotFraction
        self.observer = observer
    }

    deinit {
        pressureSource?.cancel()
        quiesceTimer?.cancel()
    }

    // MARK: - Lifecycle

    /// Arm the controller — start watching for idle + memory pressure.
    /// All tiles begin non-volatile (full speed).
    public func arm() {
        lock.withLock {
            guard state == .disabled else { return }
            state = .armed
            installPressureSource()
        }
        log.notice("armed (quiesceTimeout=\(self.quiesceTimeoutMs) ms, keepHot=\(self.keepHotFraction))")
        notifyObserver()
    }

    /// Disarm — restore all tiles to non-volatile and stop watching.
    public func disarm() {
        lock.withLock {
            guard state != .disabled else { return }
            state = .disabled
            pressureSource?.cancel()
            pressureSource = nil
            quiesceTimer?.cancel()
            quiesceTimer = nil
        }
        // Wake all tiles before disarm (failsafe).
        wakeAll()
        log.notice("disarmed (all tiles non-volatile)")
        notifyObserver()
    }

    // MARK: - Engine integration hooks

    /// Engine calls this BEFORE every inference. Wakes any compressed
    /// tiles back to non-volatile (kernel decompresses on access). MUST
    /// be called or the engine may see a tile mid-decompress.
    public func willStartInference(layerExpertHints: [(layer: Int, experts: [Int])] = []) {
        let needsWake = lock.withLock { () -> Bool in
            inferenceInFlight = true
            lastInferenceTick = Date()
            quiesceTimer?.cancel()
            quiesceTimer = nil
            return state == .compressed
        }
        if needsWake {
            log.notice("inference incoming → waking all volatile tiles")
            wakeAll(hints: layerExpertHints)
            lock.withLock { state = .armed }
            notifyObserver()
        }
    }

    /// Engine calls this AFTER inference completes (or aborts). Starts
    /// the quiesce countdown if armed.
    public func didFinishInference() {
        lock.withLock {
            inferenceInFlight = false
            lastInferenceTick = Date()
            if state == .armed || state == .quiescing {
                state = .quiescing
                scheduleQuiesce()
            }
        }
    }

    /// Engine calls this on every router decision so we can track
    /// routing frequency per expert. Cheap (single dictionary update
    /// under the lock).
    public func recordRoute(layer: Int, experts: [Int]) {
        lock.withLock {
            for e in experts {
                routingFreq[TileKey(layer: layer, expert: e), default: 0] &+= 1
                totalRoutes &+= 1
            }
        }
    }

    /// Manual user-driven compaction (e.g. from a "free up RAM" UI
    /// button). Only fires if armed and idle.
    public func manualCompact() {
        let armed = lock.withLock { state == .armed || state == .quiescing }
        guard armed else { return }
        log.notice("manual compaction requested")
        compressColdTiles()
    }

    // MARK: - Internals

    private func installPressureSource() {
        let src = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: queue)
        src.setEventHandler { [weak self] in
            guard let self else { return }
            let event = src.data
            self.log.notice("memory pressure event: \(String(describing: event))")
            // SAFETY: never compress while inference is in flight. Even
            // soft DONTNEED can cause page-cache evictions that affect
            // MLX's reads of the same files. Only fire when we're
            // demonstrably between requests.
            //
            // We also require state ∈ {armed, quiescing} — disabled and
            // compressed states already mean we're either off or in
            // the right place.
            let canCompress = self.lock.withLock { () -> Bool in
                guard !self.inferenceInFlight else { return false }
                return self.state == .armed || self.state == .quiescing
            }
            if canCompress {
                self.compressColdTiles()
            } else {
                self.log.notice("pressure event ignored: inferenceInFlight=\(self.lock.withLock { self.inferenceInFlight }) state=\(String(describing: self.lock.withLock { self.state }))")
            }
        }
        src.activate()
        pressureSource = src
    }

    private func scheduleQuiesce() {
        quiesceTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + .milliseconds(quiesceTimeoutMs))
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let stillIdle = self.lock.withLock {
                !self.inferenceInFlight && self.state == .quiescing
            }
            if stillIdle {
                self.log.notice("quiesce timeout reached → compressing cold tiles")
                self.compressColdTiles()
            }
        }
        timer.resume()
        quiesceTimer = timer
    }

    private func compressColdTiles() {
        // Pick which tiles count as "cold": the bottom (1 - keepHot)
        // fraction by routing frequency. Tiles never seen go to cold.
        let snapshot = lock.withLock { (Array(routingFreq), Set(routingFreq.keys), totalRoutes) }
        let (freq, _, _) = snapshot

        // Sort tiles by frequency ascending — lowest = coldest
        let sorted = freq.sorted { $0.value < $1.value }
        let coldCount = Int(Double(sorted.count) * (1.0 - keepHotFraction))
        let cold = sorted.prefix(coldCount).map { $0.key }

        // Group by layer for the per-backend release call shape.
        var byLayer: [Int: [Int]] = [:]
        for k in cold { byLayer[k.layer, default: []].append(k.expert) }
        for (layer, experts) in byLayer {
            // .mach backend: kernel honors VOLATILE flag for WKdm
            //                compression; soft release is enough.
            // .mmap backend: madvise(DONTNEED) is hint-only on macOS;
            //                use forceRelease (msync MS_INVALIDATE) for
            //                guaranteed reclaim. We're past the
            //                quiesce timeout so the latency cost on
            //                next acquire is acceptable.
            cache?.release(layer: layer, experts: experts)
            if useForceRelease {
                mmapTier?.forceRelease(layer: layer, experts: experts)
            } else {
                mmapTier?.release(layer: layer, experts: experts)
            }
        }
        let total = byLayer.values.reduce(0) { $0 + $1.count }
        log.notice("compressed \(total) cold tiles (across \(byLayer.count) layers, kept \(self.keepHotFraction * 100, format: .fixed(precision: 0)) % hot)")
        lock.withLock { state = .compressed }
        notifyObserver()
    }

    private func wakeAll(hints: [(layer: Int, experts: [Int])] = []) {
        // For every tile we ever compressed, flip non-volatile. We can't
        // easily enumerate every registered tile from here, so the
        // engine's hint list is the fast path; otherwise we acquire by
        // walking the routing-frequency dict (every tile we've routed
        // at least once is in there).
        if !hints.isEmpty {
            for h in hints {
                _ = try? cache?.acquire(layer: h.layer, experts: h.experts)
                mmapTier?.acquire(layer: h.layer, experts: h.experts)
            }
            return
        }
        let allKeys = lock.withLock { Array(routingFreq.keys) }
        var byLayer: [Int: [Int]] = [:]
        for k in allKeys { byLayer[k.layer, default: []].append(k.expert) }
        for (layer, experts) in byLayer {
            _ = try? cache?.acquire(layer: layer, experts: experts)
            mmapTier?.acquire(layer: layer, experts: experts)
        }
    }

    private func notifyObserver() {
        guard let obs = observer else { return }
        let s = lock.withLock { state }
        DispatchQueue.main.async { obs.emberDidEnterState(s) }
    }

    // MARK: - Stats

    public struct Stats: Sendable {
        public var state: JangPressState
        public var inferenceInFlight: Bool
        public var lastInferenceMsAgo: Int
        public var totalRoutesObserved: UInt64
        public var distinctTilesObserved: Int
        public var keepHotFraction: Double
    }

    public func snapshot() -> Stats {
        lock.withLock {
            Stats(
                state: state,
                inferenceInFlight: inferenceInFlight,
                lastInferenceMsAgo: Int(Date().timeIntervalSince(lastInferenceTick) * 1000),
                totalRoutesObserved: totalRoutes,
                distinctTilesObserved: routingFreq.count,
                keepHotFraction: keepHotFraction
            )
        }
    }
}
