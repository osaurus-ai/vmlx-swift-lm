// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Network
import os

/// Zero-config peer discovery over Bonjour / mDNS on the local network.
///
/// Matches the UX used by apps like Inferencer — a worker machine
/// advertises itself on the network; a driver machine browses and sees
/// it show up with no configuration. For vmlx-swift-lm distributed
/// inference this means a user can plug a Mac Studio into the network
/// (or into a Thunderbolt Bridge), start the worker, and run the driver
/// on any other Mac on the same network without ever knowing the IP.
///
/// ## Service name
///
/// `_vmlx-pipeline._tcp` — the service type under which a pipeline
/// worker advertises. Distinct from Inferencer's `_inferencer._tcp`
/// so the two can coexist on the same subnet without confusion.
///
/// Service instances are named by hostname + port, e.g.
/// `mac-studio.local:7437`. The Bonjour name can be customized by the
/// worker if multiple instances run on the same host.
///
/// ## Usage
///
/// Worker side (Mac Studio):
/// ```swift
/// let advertiser = BonjourAdvertiser(serviceName: nil, port: 7437)
/// try await advertiser.start()
/// // … run the pipeline worker loop …
/// await advertiser.stop()
/// ```
///
/// Driver side (this Mac):
/// ```swift
/// let browser = BonjourBrowser()
/// let peers = try await browser.discover(timeout: 5.0)
/// // peers: [DiscoveredPeer(name: "mac-studio", host: "192.168.1.42", port: 7437)]
/// ```
///
/// ## What this does NOT do
///
/// - No authentication / TLS. Phase 1 target is local trusted network
///   (typical Mac Studio at home / lab). Phase 4+ will add a
///   `TLSCredentials`-backed variant.
/// - No NAT traversal. Bonjour is multicast-DNS over the local link
///   subnet only.
/// - No dynamic handoff if the worker moves. Browser snapshots at
///   discovery time; re-browse for a refresh.

// MARK: - DiscoveredPeer

/// One result of a Bonjour browse.
public struct DiscoveredPeer: Sendable, Hashable, CustomStringConvertible {
    /// The Bonjour service instance name (typically `Hostname` or
    /// `Hostname (2)` when multiple instances share a host).
    public let name: String
    /// Resolved hostname or IP string suitable for ``MLXDistributed``
    /// / ring backend host list (`"mac-studio.local"` or `"192.168.1.42"`).
    public let host: String
    /// TCP port the worker is listening on for pipeline traffic.
    public let port: Int

    public init(name: String, host: String, port: Int) {
        self.name = name
        self.host = host
        self.port = port
    }

    public var description: String {
        "\(name) @ \(host):\(port)"
    }
}

// MARK: - BonjourAdvertiser (worker side)

/// Advertises this process as a `_vmlx-pipeline._tcp` worker on the
/// local network. Call ``start()`` once before beginning the
/// pipeline worker loop; call ``stop()`` on shutdown.
///
/// Uses Apple's `Network.framework` ``NWListener`` underneath so the
/// advertisement integrates with the system's mDNS responder — no
/// custom socket handling required on the worker's side.
public actor BonjourAdvertiser {

    public static let serviceType = "_vmlx-pipeline._tcp"

    private static let logger = Logger(
        subsystem: "vmlx", category: "BonjourAdvertiser")

    /// Bonjour instance name. `nil` uses the system hostname — the
    /// usual case for a single-worker-per-Mac setup.
    public let serviceName: String?

    /// TCP port to advertise. The actual bind happens in ``start()``.
    public let port: Int

    private var listener: NWListener?

    public init(serviceName: String? = nil, port: Int = 7437) {
        self.serviceName = serviceName
        self.port = port
    }

    /// Start advertising. Returns once the listener has transitioned
    /// to `.ready` (confirmed listening). Throws on bind failure —
    /// typically "port already in use" if another worker is running.
    public func start() async throws {
        guard listener == nil else { return }

        let params = NWParameters.tcp
        params.includePeerToPeer = true

        guard let nwPort = NWEndpoint.Port(rawValue: UInt16(port)) else {
            throw BonjourAdvertiserError.invalidPort(port)
        }

        let newListener = try NWListener(using: params, on: nwPort)
        newListener.service = NWListener.Service(
            name: serviceName,
            type: Self.serviceType
        )

        // Accept incoming connections with a no-op close — the real
        // data plane (mlx-core ring) uses its own sockets. Bonjour
        // serves only as a discovery bootstrap; the worker's actual
        // ring backend binds on the same port via env-var config.
        newListener.newConnectionHandler = { connection in
            connection.cancel()
        }

        let readyBox = ReadyBox()
        newListener.stateUpdateHandler = { state in
            Task { await readyBox.update(state: state) }
        }
        newListener.start(queue: .global(qos: .utility))

        // Wait for .ready — throw on .failed / .cancelled.
        try await readyBox.awaitReady()
        self.listener = newListener
        Self.logger.info(
            "Bonjour advertising \(Self.serviceType, privacy: .public) on port \(self.port)"
        )
    }

    /// Stop advertising and release the listener.
    public func stop() {
        listener?.cancel()
        listener = nil
    }

    /// Internal helper to bridge NWListener's state callback
    /// to an awaitable condition.
    private actor ReadyBox {
        private var continuation: CheckedContinuation<Void, Error>?
        private var finished = false

        func awaitReady() async throws {
            try await withCheckedThrowingContinuation { cont in
                if finished { cont.resume(returning: ()); return }
                continuation = cont
            }
        }

        func update(state: NWListener.State) {
            guard !finished else { return }
            switch state {
            case .ready:
                finished = true
                continuation?.resume(returning: ())
                continuation = nil
            case .failed(let error):
                finished = true
                continuation?.resume(throwing: error)
                continuation = nil
            case .cancelled:
                finished = true
                continuation?.resume(
                    throwing: BonjourAdvertiserError.cancelled)
                continuation = nil
            default:
                break
            }
        }
    }
}

public enum BonjourAdvertiserError: Error, CustomStringConvertible {
    case invalidPort(Int)
    case cancelled

    public var description: String {
        switch self {
        case .invalidPort(let port):
            return "invalid port \(port) — valid range is 1…65535"
        case .cancelled:
            return "advertiser cancelled before becoming ready"
        }
    }
}

// MARK: - BonjourBrowser (driver side)

/// Browses the local network for `_vmlx-pipeline._tcp` advertisers.
/// Returns after a caller-supplied timeout with whatever peers have
/// been discovered so far — typical workflow is "browse for a few
/// seconds, pick the matching peers, start the pipeline".
public actor BonjourBrowser {

    private static let logger = Logger(
        subsystem: "vmlx", category: "BonjourBrowser")

    public init() {}

    /// Discover advertised pipeline workers on the local network.
    ///
    /// - Parameter timeout: how long to keep the browser open before
    ///   returning. Most networks resolve within 500 ms; 5 seconds is
    ///   a safe default for a cold start.
    /// - Returns: the peers found before the timeout elapsed, sorted
    ///   by name for stable ordering across runs.
    public func discover(timeout: TimeInterval = 5.0) async throws -> [DiscoveredPeer] {
        let bucket = PeerBucket()

        let params = NWParameters()
        params.includePeerToPeer = true

        let browser = NWBrowser(
            for: .bonjourWithTXTRecord(
                type: BonjourAdvertiser.serviceType, domain: nil),
            using: params
        )

        browser.browseResultsChangedHandler = { results, _ in
            for result in results {
                guard case .service(let name, _, _, _) = result.endpoint else {
                    continue
                }
                let endpoint = result.endpoint
                // Resolve each hit to host:port via a short-lived
                // NWConnection so we capture the actual IPv4 address.
                // The connection is immediately cancelled.
                Task.detached {
                    if let resolved = await Self.resolveEndpoint(
                        endpoint, serviceName: name)
                    {
                        await bucket.insert(resolved)
                    }
                }
            }
        }

        browser.start(queue: .global(qos: .utility))

        // Yield the whole timeout window; caller just wants a snapshot.
        try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
        browser.cancel()

        let found = await bucket.all()
        Self.logger.info(
            "Bonjour browse complete: \(found.count) peer(s) found")
        return found.sorted { $0.name < $1.name }
    }

    /// Resolve a Bonjour service endpoint into host + port. Uses a
    /// disposable NWConnection whose pathUpdateHandler surfaces the
    /// resolved IPv4 endpoint. The connection is cancelled as soon as
    /// resolution succeeds or times out.
    private static func resolveEndpoint(
        _ endpoint: NWEndpoint,
        serviceName: String
    ) async -> DiscoveredPeer? {
        let params = NWParameters.tcp
        params.includePeerToPeer = true

        let connection = NWConnection(to: endpoint, using: params)

        return await withCheckedContinuation { cont in
            let resolved = ResolutionBox()

            connection.pathUpdateHandler = { path in
                guard let ep = path.gateways.first ?? path.availableInterfaces.first.map({ _ in endpoint }) else {
                    return
                }
                _ = ep  // unused unless we want to inspect gateway
            }

            connection.betterPathUpdateHandler = { _ in }

            connection.stateUpdateHandler = { state in
                Task {
                    if case .ready = state {
                        if let remote = connection.currentPath?.remoteEndpoint {
                            switch remote {
                            case .hostPort(let host, let port):
                                let hostStr = Self.hostToString(host)
                                let peer = DiscoveredPeer(
                                    name: serviceName,
                                    host: hostStr,
                                    port: Int(port.rawValue)
                                )
                                await resolved.resolve(peer: peer, cont: cont)
                            default:
                                break
                            }
                        }
                        connection.cancel()
                    } else if case .failed = state {
                        await resolved.resolveNil(cont: cont)
                        connection.cancel()
                    }
                }
            }
            connection.start(queue: .global(qos: .utility))

            // Safety timeout — don't hang forever on one unresolvable peer.
            Task {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                await resolved.resolveNil(cont: cont)
                connection.cancel()
            }
        }
    }

    private static func hostToString(_ host: NWEndpoint.Host) -> String {
        switch host {
        case .name(let s, _): return s
        case .ipv4(let addr):
            return addr.rawValue.map { "\($0)" }.joined(separator: ".")
        case .ipv6(let addr):
            // Return the textual form for use in host lists — ring
            // backend handles both v4 and v6.
            return "\(addr)"
        @unknown default: return ""
        }
    }

    private actor PeerBucket {
        private var peers: [DiscoveredPeer] = []
        func insert(_ peer: DiscoveredPeer) {
            if !peers.contains(where: { $0.host == peer.host && $0.port == peer.port }) {
                peers.append(peer)
            }
        }
        func all() -> [DiscoveredPeer] { peers }
    }

    private actor ResolutionBox {
        private var resolved = false
        func resolve(
            peer: DiscoveredPeer,
            cont: CheckedContinuation<DiscoveredPeer?, Never>
        ) {
            guard !resolved else { return }
            resolved = true
            cont.resume(returning: peer)
        }
        func resolveNil(
            cont: CheckedContinuation<DiscoveredPeer?, Never>
        ) {
            guard !resolved else { return }
            resolved = true
            cont.resume(returning: nil)
        }
    }
}
