// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Router-aware JangPress advisory coordinator.
//
// The osaurus mlx-swift fork can expose safetensors tensor storage as
// canonical mmap-backed MLX arrays. This coordinator connects routed-MoE
// gate decisions to that storage: model code reports the `(layer, expert)`
// ids it is about to use, and the coordinator asks the C++ mmap registry to
// prefetch those expert pages while advising older per-layer hot-set entries
// cold. The OS still owns actual page reclaim; this is precise page advice,
// not a custom compressed blob format.

import Foundation
import MLX
#if canImport(Cmlx)
import Cmlx
#endif

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct JangPressCanonicalExpertAdvisorStatus: Sendable, Equatable {
    public var enabled: Bool
    public var asyncReadback: Bool
    public var warmAdvice: Bool
    public var symbolAvailable: Bool
    public var hotPerLayer: Int
    public var hotExpertCount: Int
    public var warmCalls: Int
    public var coldCalls: Int
    public var warmBytes: Int64
    public var coldBytes: Int64
    public var pendingObservations: Int
    public var droppedQueueFull: Int
    public var skippedLargeIndexTensors: Int
    public var skippedTracerArrays: Int
    public var readbacks: Int
    public var rewarms: Int
    public var distinctColdAdvisedPairs: Int
}

private typealias SafetensorsMmapAdviseExpertsFn = @convention(c) (
    Int32,
    UnsafePointer<Int32>?,
    UnsafePointer<Int32>?,
    Int64
) -> Int64

public final class JangPressCanonicalExpertAdvisor: @unchecked Sendable {
    public static let shared = JangPressCanonicalExpertAdvisor()

    private struct Config {
        var enabled: Bool = false
        var asyncReadback: Bool = true
        var warmAdvice: Bool = false
        var hotPerLayer: Int = 32
        var maxIndicesPerReadback: Int = 32
        var maxPendingObservations: Int = 512
        var drainBatchSize: Int = 64
        var debug: Bool = false
    }

    private final class PendingObservation: @unchecked Sendable {
        let configID: UInt64
        let layer: Int
        let expertIDs: [Int32]

        init(configID: UInt64, layer: Int, expertIDs: [Int32]) {
            self.configID = configID
            self.layer = layer
            self.expertIDs = expertIDs
        }
    }

    private struct MutableState {
        var config = Config()
        var configID: UInt64 = 0
        var generation: UInt64 = 0
        var hotByLayer: [Int: [Int: UInt64]] = [:]
        var coldHistoryByLayer: [Int: Set<Int>] = [:]
        var pendingObservations: [PendingObservation] = []
        var workerScheduled = false
        var symbolResolved = false
        var symbolAvailable = false
        var warmCalls = 0
        var coldCalls = 0
        var warmBytes: Int64 = 0
        var coldBytes: Int64 = 0
        var droppedQueueFull = 0
        var skippedLargeIndexTensors = 0
        var skippedTracerArrays = 0
        var readbacks = 0
        var rewarms = 0
    }

    private let lock = NSLock()
    private let workerQueue = DispatchQueue(
        label: "org.osaurus.jangpress.router-advisor",
        qos: .utility)
    private var state = MutableState()
    private var adviseExperts: SafetensorsMmapAdviseExpertsFn?
    private nonisolated(unsafe) var fastEnabled = false

    private init() {}

    public func configure(
        options: JangPressLoadOptions,
        mmapEnabled: Bool,
        numRoutedExperts: Int? = nil,
        topK: Int? = nil
    ) {
        let env = ProcessInfo.processInfo.environment
        let routerEnv = env["JANGPRESS_ROUTER_ADVICE"]?.lowercased()
        let envEnabled = routerEnv == "1" || routerEnv == "true"
            || routerEnv == "on" || routerEnv == "yes"
        let envDisabled = routerEnv == "0" || routerEnv == "false"
            || routerEnv == "off" || routerEnv == "no"
        let enabled = mmapEnabled
            && options.enabled
            && options.backend == .mmap
            && (options.enableRouterAdvice || envEnabled)
            && !envDisabled
        let hotPerLayer = parsePositiveEnv(
            "JANGPRESS_ROUTER_HOT_PER_LAYER",
            env: env,
            default: hotPerLayerDefault(
                compressPct: options.compressPct,
                numRoutedExperts: numRoutedExperts,
                topK: topK))
        let maxIndices = parsePositiveEnv(
            "JANGPRESS_ROUTER_MAX_INDICES",
            env: env,
            default: 32)
        let asyncReadback = parseBoolEnv(
            "JANGPRESS_ROUTER_ASYNC_READBACK",
            env: env,
            default: true)
        let warmAdvice = parseBoolEnv(
            "JANGPRESS_ROUTER_WARM_ADVICE",
            env: env,
            default: false)
        let maxPending = parsePositiveEnv(
            "JANGPRESS_ROUTER_MAX_PENDING",
            env: env,
            default: 512)
        let drainBatch = parsePositiveEnv(
            "JANGPRESS_ROUTER_DRAIN_BATCH",
            env: env,
            default: 64)
        let debug = env["JANGPRESS_DEBUG"] == "1"
            || env["MLX_SAFETENSORS_MMAP_DEBUG"] == "1"
            || env["JANGPRESS_ROUTER_DEBUG"] == "1"

        lock.lock()
        fastEnabled = enabled
        state.configID &+= 1
        state.config = Config(
            enabled: enabled,
            asyncReadback: asyncReadback,
            warmAdvice: warmAdvice,
            hotPerLayer: max(1, hotPerLayer),
            maxIndicesPerReadback: max(1, maxIndices),
            maxPendingObservations: max(1, maxPending),
            drainBatchSize: max(1, drainBatch),
            debug: debug)
        state.hotByLayer.removeAll(keepingCapacity: true)
        state.coldHistoryByLayer.removeAll(keepingCapacity: true)
        state.pendingObservations.removeAll(keepingCapacity: true)
        state.workerScheduled = false
        state.warmCalls = 0
        state.coldCalls = 0
        state.warmBytes = 0
        state.coldBytes = 0
        state.droppedQueueFull = 0
        state.skippedLargeIndexTensors = 0
        state.skippedTracerArrays = 0
        state.readbacks = 0
        state.rewarms = 0
        if enabled {
            _ = resolveAdviseExpertsSymbolLocked()
        }
        let symbolAvailable = state.symbolAvailable
        lock.unlock()

        if debug {
            let availability = symbolAvailable ? "available" : "missing"
            FileHandle.standardError.write(Data(
                "[JangPressRouter] enabled=\(enabled) async=\(asyncReadback) warmAdvice=\(warmAdvice) hotPerLayer=\(hotPerLayer) maxIndices=\(maxIndices) maxPending=\(maxPending) symbol=\(availability)\n".utf8))
        }
    }

    public func observe(layer: Int, indices: MLXArray) {
        if !fastEnabled { return }
        lock.lock()
        let config = state.config
        let configID = state.configID
        if !config.enabled {
            lock.unlock()
            return
        }
        if !resolveAdviseExpertsSymbolLocked() {
            lock.unlock()
            return
        }
        if indices.size > config.maxIndicesPerReadback {
            state.skippedLargeIndexTensors += 1
            lock.unlock()
            return
        }
        lock.unlock()

        if isTracerArray(indices) {
            lock.lock()
            state.skippedTracerArrays += 1
            lock.unlock()
            return
        }

        // Keep MLXArray use on the caller's execution path. Passing
        // unevaluated arrays into a background Dispatch queue is not a
        // stable MLX contract and can crash during concurrent decode.
        //
        // This readback is also a deliberate speed tradeoff: it forces
        // CPU visibility of a tiny top-k router tensor. Router advice stays
        // default-off until this path is proven tok/s-neutral on real MoE
        // bundles; JangPress's production win is canonical mmap residency,
        // not per-token readback.
        let uniqueExperts = readUniqueExperts(indices)
        guard !uniqueExperts.isEmpty else { return }

        if config.asyncReadback {
            enqueue(configID: configID, layer: layer, experts: uniqueExperts)
            return
        }

        processExperts(configID: configID, layer: layer, uniqueExperts: uniqueExperts)
    }

    public func observe(layer: Int, experts: [Int32]) {
        if !fastEnabled { return }
        lock.lock()
        let config = state.config
        let configID = state.configID
        if !config.enabled {
            lock.unlock()
            return
        }
        if !resolveAdviseExpertsSymbolLocked() {
            lock.unlock()
            return
        }
        if experts.count > config.maxIndicesPerReadback {
            state.skippedLargeIndexTensors += 1
            lock.unlock()
            return
        }
        lock.unlock()

        let unique = Array(Set(experts.filter { $0 >= 0 })).sorted()
        guard !unique.isEmpty else { return }

        if config.asyncReadback {
            enqueue(configID: configID, layer: layer, experts: unique)
        } else {
            processExperts(configID: configID, layer: layer, uniqueExperts: unique)
        }
    }

    private func enqueue(
        configID: UInt64,
        layer: Int,
        experts: [Int32]
    ) {
        lock.lock()
        guard state.config.enabled, state.configID == configID else {
            lock.unlock()
            return
        }
        if state.pendingObservations.count >= state.config.maxPendingObservations {
            state.droppedQueueFull += 1
            lock.unlock()
            return
        }
        state.pendingObservations.append(PendingObservation(
            configID: configID,
            layer: layer,
            expertIDs: experts))
        let shouldSchedule = !state.workerScheduled
        if shouldSchedule {
            state.workerScheduled = true
        }
        lock.unlock()

        if shouldSchedule {
            workerQueue.async { [weak self] in
                self?.drainPendingObservations()
            }
        }
    }

    private func drainPendingObservations() {
        while true {
            lock.lock()
            let batchSize = max(1, state.config.drainBatchSize)
            if state.pendingObservations.isEmpty {
                state.workerScheduled = false
                lock.unlock()
                return
            }
            let count = min(batchSize, state.pendingObservations.count)
            let batch = Array(state.pendingObservations.prefix(count))
            state.pendingObservations.removeFirst(count)
            lock.unlock()

            for observation in batch {
                processExperts(
                    configID: observation.configID,
                    layer: observation.layer,
                    uniqueExperts: observation.expertIDs)
            }
        }
    }

    private func readUniqueExperts(_ indices: MLXArray) -> [Int32] {
        let routed = indices.reshaped([-1]).asType(.int32).asArray(Int32.self)
        return Array(Set(routed.filter { $0 >= 0 })).sorted()
    }

    private func isTracerArray(_ indices: MLXArray) -> Bool {
        // The standalone vmlx tree exposes a patched C ABI for tracer
        // detection. This vmlx-swift-lm checkout does not, so keep the
        // guard as a conservative no-op until the MLX pin grows that
        // surface. Size guards above still keep normal decode telemetry
        // bounded.
        false
    }

    private func processExperts(
        configID: UInt64,
        layer: Int,
        uniqueExperts: [Int32]
    ) {
        var warm: [(Int32, Int32)] = []
        var cold: [(Int32, Int32)] = []
        var debug = false
        var warmAdvice = false

        lock.lock()
        let config = state.config
        guard state.config.enabled, state.configID == configID else {
            lock.unlock()
            return
        }
        state.readbacks += 1
        state.generation &+= 1
        let generation = state.generation
        var hot = state.hotByLayer[layer] ?? [:]
        var coldHistory = state.coldHistoryByLayer[layer] ?? Set<Int>()
        var layerRewarms = 0
        for expert in uniqueExperts {
            let e = Int(expert)
            if hot[e] == nil {
                warm.append((Int32(layer), expert))
                if coldHistory.remove(e) != nil {
                    layerRewarms &+= 1
                }
            }
            hot[e] = generation
        }

        if hot.count > config.hotPerLayer {
            let overflow = hot.count - config.hotPerLayer
            let evicted = hot
                .sorted { lhs, rhs in
                    if lhs.value == rhs.value { return lhs.key < rhs.key }
                    return lhs.value < rhs.value
                }
                .prefix(overflow)
            for (expert, _) in evicted {
                hot.removeValue(forKey: expert)
                cold.append((Int32(layer), Int32(expert)))
                coldHistory.insert(expert)
            }
        }
        state.hotByLayer[layer] = hot
        state.coldHistoryByLayer[layer] = coldHistory
        state.rewarms &+= layerRewarms
        debug = config.debug
        warmAdvice = config.warmAdvice
        lock.unlock()

        if warmAdvice, !warm.isEmpty {
            let bytes = advise(pairs: warm, advice: 1)
            lock.lock()
            state.warmCalls += 1
            state.warmBytes += bytes
            lock.unlock()
        }
        if !cold.isEmpty {
            let bytes = advise(pairs: cold, advice: 0)
            lock.lock()
            state.coldCalls += 1
            state.coldBytes += bytes
            lock.unlock()
        }

        if debug, !warm.isEmpty || !cold.isEmpty {
            FileHandle.standardError.write(Data(
                "[JangPressRouter] layer=\(layer) warm=\(warm.count) cold=\(cold.count)\n".utf8))
        }
    }

    public func snapshot() -> JangPressCanonicalExpertAdvisorStatus {
        lock.lock()
        defer { lock.unlock() }
        let hotCount = state.hotByLayer.values.reduce(0) { $0 + $1.count }
        let coldHistoryCount = state.coldHistoryByLayer.values.reduce(0) {
            $0 + $1.count
        }
        return JangPressCanonicalExpertAdvisorStatus(
            enabled: state.config.enabled,
            asyncReadback: state.config.asyncReadback,
            warmAdvice: state.config.warmAdvice,
            symbolAvailable: state.symbolAvailable,
            hotPerLayer: state.config.hotPerLayer,
            hotExpertCount: hotCount,
            warmCalls: state.warmCalls,
            coldCalls: state.coldCalls,
            warmBytes: state.warmBytes,
            coldBytes: state.coldBytes,
            pendingObservations: state.pendingObservations.count,
            droppedQueueFull: state.droppedQueueFull,
            skippedLargeIndexTensors: state.skippedLargeIndexTensors,
            skippedTracerArrays: state.skippedTracerArrays,
            readbacks: state.readbacks,
            rewarms: state.rewarms,
            distinctColdAdvisedPairs: coldHistoryCount)
    }

    public func waitUntilIdle(timeoutSeconds: TimeInterval = 2.0) {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            lock.lock()
            let idle = state.pendingObservations.isEmpty
                && !state.workerScheduled
            lock.unlock()
            if idle { return }
            Thread.sleep(forTimeInterval: 0.005)
        }
    }

    private func advise(pairs: [(Int32, Int32)], advice: Int32) -> Int64 {
#if canImport(Cmlx)
        lock.lock()
        let adviseExperts = resolveAdviseExpertsSymbolLocked()
            ? self.adviseExperts
            : nil
        lock.unlock()

        guard let adviseExperts else {
            return 0
        }

        let layers = pairs.map { $0.0 }
        let experts = pairs.map { $0.1 }
        return layers.withUnsafeBufferPointer { layerBuffer in
            experts.withUnsafeBufferPointer { expertBuffer in
                adviseExperts(
                    advice,
                    layerBuffer.baseAddress,
                    expertBuffer.baseAddress,
                    Int64(pairs.count))
            }
        }
#else
        return 0
#endif
    }

    private func resolveAdviseExpertsSymbolLocked() -> Bool {
        if state.symbolResolved {
            return state.symbolAvailable
        }
        state.symbolResolved = true
#if canImport(Cmlx)
        guard let handle = dlopen(nil, RTLD_NOW),
              let symbol = dlsym(handle, "mlx_safetensors_mmap_advise_experts")
        else {
            adviseExperts = nil
            state.symbolAvailable = false
            return false
        }
        adviseExperts = unsafeBitCast(
            symbol,
            to: SafetensorsMmapAdviseExpertsFn.self)
        state.symbolAvailable = true
        return true
#else
        adviseExperts = nil
        state.symbolAvailable = false
        return false
#endif
    }

    private func parsePositiveEnv(
        _ key: String,
        env: [String: String],
        default defaultValue: Int
    ) -> Int {
        guard let value = env[key], let parsed = Int(value), parsed > 0 else {
            return defaultValue
        }
        return parsed
    }

    private func parseBoolEnv(
        _ key: String,
        env: [String: String],
        default defaultValue: Bool
    ) -> Bool {
        guard let raw = env[key]?.lowercased() else {
            return defaultValue
        }
        if raw == "1" || raw == "true" || raw == "on" || raw == "yes" {
            return true
        }
        if raw == "0" || raw == "false" || raw == "off" || raw == "no" {
            return false
        }
        return defaultValue
    }

    private func defaultHotPerLayer(compressPct: Int) -> Int {
        let clamped = max(0, min(100, compressPct))
        let hotFraction = Double(100 - clamped) / 100.0
        return max(8, min(64, Int((hotFraction * 128.0).rounded(.up))))
    }

    private func hotPerLayerDefault(
        compressPct: Int,
        numRoutedExperts: Int?,
        topK: Int?
    ) -> Int {
        guard let n = numRoutedExperts, n > 0 else {
            return defaultHotPerLayer(compressPct: compressPct)
        }

        let pct = max(0, min(100, compressPct))
        let pctBudget = Int(ceil(Double(n) * Double(100 - pct) / 100.0))
        let k = max(1, topK ?? 4)
        var hot = max(k * 4, pctBudget)
        let lowerBound = max(8, k * 2)
        if hot < lowerBound { hot = lowerBound }
        if hot > n { hot = n }
        return hot
    }
}
