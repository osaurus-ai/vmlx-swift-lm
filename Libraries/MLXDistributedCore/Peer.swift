import Foundation

/// Stable identity for a discovered peer. Equality and hashing are id-only;
/// hostname / endpoints / capabilities can change across re-resolution
/// without the peer becoming a "different" peer.
public struct Peer: Sendable, Equatable, Hashable {
    public let id: UUID
    public let hostname: String
    public let capabilities: PeerCapabilities
    public let endpoints: [Endpoint]
    public let modelHashes: ModelHashSet
    public let memFreeMiB: UInt64?
    public let willingToBeCoordinator: Bool

    public init(
        id: UUID,
        hostname: String,
        capabilities: PeerCapabilities,
        endpoints: [Endpoint],
        modelHashes: ModelHashSet,
        memFreeMiB: UInt64? = nil,
        willingToBeCoordinator: Bool = false
    ) {
        self.id = id
        self.hostname = hostname
        self.capabilities = capabilities
        self.endpoints = endpoints
        self.modelHashes = modelHashes
        self.memFreeMiB = memFreeMiB
        self.willingToBeCoordinator = willingToBeCoordinator
    }

    public static func == (lhs: Peer, rhs: Peer) -> Bool { lhs.id == rhs.id }
    public func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

/// What a peer is willing to participate in.
public struct PeerCapabilities: Sendable, Equatable, Hashable {
    public let modes: Set<Mode>

    public init(modes: Set<Mode>) {
        self.modes = modes
    }

    public static let empty = PeerCapabilities(modes: [])

    public func supports(_ m: Mode) -> Bool { modes.contains(m) }
}

/// A reachable transport endpoint advertised by a peer.
public enum Endpoint: Sendable, Equatable, Hashable {
    case tls(host: String, port: UInt16, fingerprintSHA256: String)
    case rdma(gid: String, devices: [String])
}

/// Set of locally-available model bundle hashes, or the overflow sentinel.
/// The sentinel means the peer has too many models to fit in the TXT record;
/// callers must fetch the full list via `/v1/dist/models` over TLS (engine
/// spec §10).
public enum ModelHashSet: Sendable, Equatable, Hashable {
    case explicit([String])
    case overflow
}
