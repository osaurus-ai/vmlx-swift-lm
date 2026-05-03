import Foundation

/// Default `DiscoveryProvider` for standalone consumers (RunBench, SDK).
///
/// Uses Foundation `NetService` / `NetServiceBrowser` rather than
/// `Network.framework`'s newer `NWBrowser` / `NWListener` to match the
/// pattern used by osaurus's existing `BonjourAdvertiser` /
/// `BonjourBrowser`. Keeps the `OsaurusBonjourBridge` (Plan 1B) trivial.
///
/// Concurrency:
/// - Public surface is an actor.
/// - The `NetServiceBrowser` delegate callbacks need a RunLoop; we pin
///   the bridge to `@MainActor` so callbacks land on the main RunLoop
///   (XCTest pumps main RunLoop during async tests, matching osaurus's
///   production setup) and hop back into the actor via `Task`.
public actor BonjourDiscoveryProvider: DiscoveryProvider {
    public static let defaultServiceType = "_vmlx._tcp."

    private let serviceType: String
    private var advertiser: NetService?
    private var advertisedTXT: [String: String] = [:]

    private var bridge: BonjourBrowserBridge?
    private var browseLifetimeTask: Task<Void, Never>?
    private var continuations: [UUID: AsyncStream<[Peer]>.Continuation] = [:]
    private var knownPeers: [UUID: Peer] = [:]

    public init(serviceType: String = BonjourDiscoveryProvider.defaultServiceType) {
        self.serviceType = serviceType
    }

    // MARK: - DiscoveryProvider

    nonisolated public func peerStream() -> AsyncStream<[Peer]> {
        AsyncStream { continuation in
            let token = UUID()
            Task {
                await self._registerStream(token: token, continuation: continuation)
            }
            continuation.onTermination = { @Sendable _ in
                Task { await self._dropStream(token: token) }
            }
        }
    }

    public func advertise(_ peer: Peer) async throws {
        let txt = try TXTSchema.encode(peer)
        advertisedTXT = txt
        advertiser?.stop()

        let port = peer.endpoints.compactMap { ep -> Int32? in
            if case .tls(_, let p, _) = ep { return Int32(p) }
            return nil
        }.first ?? 0

        let service = NetService(
            domain: "",
            type: serviceType,
            name: peer.id.uuidString.lowercased(),
            port: port
        )
        let asData = txt.mapValues { $0.data(using: .utf8) ?? Data() }
        service.setTXTRecord(NetService.data(fromTXTRecord: asData))
        service.publish()
        advertiser = service
    }

    public func stopAdvertising() async {
        advertiser?.stop()
        advertiser = nil
        advertisedTXT.removeAll()
    }

    /// Test hook — exposes the last TXT we asked NetService to publish.
    /// Not part of the public DiscoveryProvider surface.
    public func _advertisedTXTForTesting() -> [String: String] {
        advertisedTXT
    }

    // MARK: - Internal browse-stream bookkeeping

    private func _registerStream(
        token: UUID,
        continuation: AsyncStream<[Peer]>.Continuation
    ) {
        continuations[token] = continuation
        // Replay current state to a fresh subscriber.
        continuation.yield(Array(knownPeers.values))
        startBrowsingIfNeeded()
    }

    private func _dropStream(token: UUID) {
        continuations.removeValue(forKey: token)
        if continuations.isEmpty {
            browseLifetimeTask?.cancel()
            browseLifetimeTask = nil
            let bridge = self.bridge
            self.bridge = nil
            if let bridge {
                Task { @MainActor in bridge.stop() }
            }
        }
    }

    private func emitSnapshot() {
        let peers = Array(knownPeers.values)
        for c in continuations.values { c.yield(peers) }
    }

    fileprivate func _peerSeen(_ peer: Peer) {
        knownPeers[peer.id] = peer
        emitSnapshot()
    }

    fileprivate func _peerLost(_ id: UUID) {
        if knownPeers.removeValue(forKey: id) != nil { emitSnapshot() }
    }

    private func startBrowsingIfNeeded() {
        guard bridge == nil else { return }
        let weakSelf = WeakProvider(actor: self)
        let bridge = BonjourBrowserBridge(serviceType: serviceType, sink: weakSelf)
        self.bridge = bridge
        Task { @MainActor in bridge.start() }
        browseLifetimeTask = Task {
            // Sentinel — keeps a strong reference to the bridge while
            // any subscriber is active. _dropStream cancels us when the
            // last subscriber unsubscribes.
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }
}

// MARK: - NetServiceBrowser bridge (lives on @MainActor)

private final class WeakProvider: @unchecked Sendable {
    weak var actor: BonjourDiscoveryProvider?
    init(actor: BonjourDiscoveryProvider) { self.actor = actor }
}

@MainActor
private final class BonjourBrowserBridge: NSObject {
    private let serviceType: String
    private let sink: WeakProvider
    private var browser: NetServiceBrowser?
    private var resolving: [String: NetService] = [:]

    nonisolated init(serviceType: String, sink: WeakProvider) {
        self.serviceType = serviceType
        self.sink = sink
        super.init()
    }

    func start() {
        let b = NetServiceBrowser()
        b.delegate = self
        b.searchForServices(ofType: serviceType, inDomain: "")
        browser = b
    }

    func stop() {
        browser?.stop()
        for s in resolving.values { s.stop() }
        resolving.removeAll()
        browser = nil
    }
}

extension BonjourBrowserBridge: @preconcurrency NetServiceBrowserDelegate {
    func netServiceBrowser(
        _ browser: NetServiceBrowser,
        didFind service: NetService,
        moreComing: Bool
    ) {
        service.delegate = self
        resolving[service.name] = service
        service.resolve(withTimeout: 5)
    }

    func netServiceBrowser(
        _ browser: NetServiceBrowser,
        didRemove service: NetService,
        moreComing: Bool
    ) {
        resolving.removeValue(forKey: service.name)
        if let id = UUID(uuidString: service.name), let actor = sink.actor {
            Task { await actor._peerLost(id) }
        }
    }
}

extension BonjourBrowserBridge: @preconcurrency NetServiceDelegate {
    func netServiceDidResolveAddress(_ sender: NetService) {
        guard let txtData = sender.txtRecordData() else { return }
        let dataDict = NetService.dictionary(fromTXTRecord: txtData)
        let strDict = dataDict.compactMapValues { String(data: $0, encoding: .utf8) }
        guard let actor = sink.actor else { return }
        do {
            let peer = try TXTSchema.decode(
                strDict,
                hostname: sender.hostName ?? sender.name
            )
            Task { await actor._peerSeen(peer) }
        } catch {
            // Malformed TXT: silently skip in Phase 1A.
            // Phase 7 wires this to a metric / structured event.
        }
    }
}
