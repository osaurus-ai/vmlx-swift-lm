import Foundation

public enum TLSState: Codable, Sendable, Equatable, Hashable {
    case unknown
    case handshakeFailed(reason: String)
    case trusted(fingerprintSHA256: String)

    public var isTrusted: Bool {
        if case .trusted = self { return true }
        return false
    }
}

public struct BackendBuildIdentity: Codable, Sendable, Equatable, Hashable {
    public let ringBackend: BackendState
    public let jacclBackend: BackendState
    public let mlxSwiftCommit: String?
    public let sdkVersion: String?

    public init(
        ringBackend: BackendState,
        jacclBackend: BackendState,
        mlxSwiftCommit: String? = nil,
        sdkVersion: String? = nil
    ) {
        self.ringBackend = ringBackend
        self.jacclBackend = jacclBackend
        self.mlxSwiftCommit = mlxSwiftCommit
        self.sdkVersion = sdkVersion
    }
}

public enum BackendState: String, Codable, Sendable, Equatable, Hashable {
    case unavailable
    case stub
    case real
}

public struct SocketEdge: Codable, Sendable, Equatable, Hashable {
    public let host: String
    public let port: UInt16
    public let interfaceName: String?
    public let linkClass: LinkClass
    public let latencyMilliseconds: Double?
    public let bandwidthMegabitsPerSecond: Double?
    public let tlsState: TLSState
    public let observedAt: Date

    public init(
        host: String,
        port: UInt16,
        interfaceName: String? = nil,
        linkClass: LinkClass,
        latencyMilliseconds: Double? = nil,
        bandwidthMegabitsPerSecond: Double? = nil,
        tlsState: TLSState,
        observedAt: Date
    ) {
        self.host = host
        self.port = port
        self.interfaceName = interfaceName
        self.linkClass = linkClass
        self.latencyMilliseconds = latencyMilliseconds
        self.bandwidthMegabitsPerSecond = bandwidthMegabitsPerSecond
        self.tlsState = tlsState
        self.observedAt = observedAt
    }

    public var isTrusted: Bool {
        tlsState.isTrusted
    }
}

public struct RdmaEdge: Codable, Sendable, Equatable, Hashable {
    public let sourceDevice: String
    public let sinkDevice: String
    public let rdmaCtlEnabled: Bool
    public let backendBuild: BackendBuildIdentity
    public let validatedAt: Date

    public init(
        sourceDevice: String,
        sinkDevice: String,
        rdmaCtlEnabled: Bool,
        backendBuild: BackendBuildIdentity,
        validatedAt: Date
    ) {
        self.sourceDevice = sourceDevice
        self.sinkDevice = sinkDevice
        self.rdmaCtlEnabled = rdmaCtlEnabled
        self.backendBuild = backendBuild
        self.validatedAt = validatedAt
    }

    public var isJACCLUsable: Bool {
        rdmaCtlEnabled
            && !sourceDevice.isEmpty
            && !sinkDevice.isEmpty
            && backendBuild.jacclBackend == .real
    }
}

public struct RelayEdge: Codable, Sendable, Equatable, Hashable {
    public let rendezvousID: String
    public let region: String?
    public let latencyMilliseconds: Double?
    public let bandwidthMegabitsPerSecond: Double?
    public let authenticated: Bool
    public let observedAt: Date

    public init(
        rendezvousID: String,
        region: String? = nil,
        latencyMilliseconds: Double? = nil,
        bandwidthMegabitsPerSecond: Double? = nil,
        authenticated: Bool,
        observedAt: Date
    ) {
        self.rendezvousID = rendezvousID
        self.region = region
        self.latencyMilliseconds = latencyMilliseconds
        self.bandwidthMegabitsPerSecond = bandwidthMegabitsPerSecond
        self.authenticated = authenticated
        self.observedAt = observedAt
    }
}

public enum TopologyEdge: Codable, Sendable, Equatable, Hashable {
    case socket(SocketEdge)
    case rdma(RdmaEdge)
    case relay(RelayEdge)

    public var kind: PeerTopology.EdgeKind {
        switch self {
        case .socket: return .socket
        case .rdma: return .rdma
        case .relay: return .relay
        }
    }
}

public struct PeerTopology: Codable, Sendable, Equatable {
    public enum EdgeKind: String, Codable, Sendable, Hashable {
        case socket
        case rdma
        case relay
    }

    public struct EdgeKey: Codable, Sendable, Equatable, Hashable {
        public let localPeerID: UUID
        public let remotePeerID: UUID
        public let kind: EdgeKind
        public let label: String

        public init(
            localPeerID: UUID,
            remotePeerID: UUID,
            kind: EdgeKind,
            label: String
        ) {
            self.localPeerID = localPeerID
            self.remotePeerID = remotePeerID
            self.kind = kind
            self.label = label
        }
    }

    private var edgesByKey: [EdgeKey: TopologyEdge]

    public init(edges: [EdgeKey: TopologyEdge] = [:]) {
        self.edgesByKey = edges
    }

    public var edges: [EdgeKey: TopologyEdge] {
        edgesByKey
    }

    public mutating func upsert(
        _ edge: TopologyEdge,
        localPeerID: UUID,
        remotePeerID: UUID,
        label: String
    ) {
        let key = EdgeKey(
            localPeerID: localPeerID,
            remotePeerID: remotePeerID,
            kind: edge.kind,
            label: label)
        edgesByKey[key] = edge
    }

    public func edges(
        from localPeerID: UUID,
        to remotePeerID: UUID,
        kind: EdgeKind? = nil
    ) -> [TopologyEdge] {
        edgesByKey
            .filter { key, _ in
                key.localPeerID == localPeerID
                    && key.remotePeerID == remotePeerID
                    && (kind == nil || key.kind == kind)
            }
            .sorted { lhs, rhs in
                if lhs.key.kind.rawValue != rhs.key.kind.rawValue {
                    return lhs.key.kind.rawValue < rhs.key.kind.rawValue
                }
                return lhs.key.label < rhs.key.label
            }
            .map(\.value)
    }

    public func hasUsableJACCLEdge(
        from localPeerID: UUID,
        to remotePeerID: UUID
    ) -> Bool {
        edges(from: localPeerID, to: remotePeerID, kind: .rdma).contains {
            guard case .rdma(let edge) = $0 else { return false }
            return edge.isJACCLUsable
        }
    }
}
