import Foundation

public enum DiscoverySource: String, CaseIterable, Codable, Sendable, Hashable {
    case bonjour
    case interfaceBonjour
    case manual
    case sshBootstrap
    case tailnet
    case relay
    case subnetProbe
    case peerCard
}

public enum LinkClass: String, CaseIterable, Codable, Sendable, Hashable {
    case wifi
    case lan
    case thunderboltBridge
    case tailnet
    case relay
    case rdma
    case unknown
}

public enum PeerReadinessState: String, CaseIterable, Codable, Sendable, Hashable {
    case discovered
    case reachable
    case trusted
    case inventoried
    case replicaReady
    case pipelineCandidate
    case wiredCandidate
    case rdmaConfigured
    case rdmaReady
    case tpReady

    public var rank: Int {
        switch self {
        case .discovered: return 0
        case .reachable: return 1
        case .trusted: return 2
        case .inventoried: return 3
        case .replicaReady: return 4
        case .pipelineCandidate: return 5
        case .wiredCandidate: return 6
        case .rdmaConfigured: return 7
        case .rdmaReady: return 8
        case .tpReady: return 9
        }
    }
}

public struct ModeBlocker: Codable, Sendable, Equatable, Hashable {
    public let mode: Mode
    public let reason: String

    public init(mode: Mode, reason: String) {
        self.mode = mode
        self.reason = reason
    }
}

public struct PeerEvidence: Codable, Sendable, Equatable, Hashable {
    public let peer: Peer
    public let source: DiscoverySource
    public let state: PeerReadinessState
    public let linkClass: LinkClass
    public let observedAt: Date
    public let latencyMilliseconds: Double?
    public let bandwidthMegabitsPerSecond: Double?
    public let blockers: [ModeBlocker]

    public init(
        peer: Peer,
        source: DiscoverySource,
        state: PeerReadinessState,
        linkClass: LinkClass,
        observedAt: Date,
        latencyMilliseconds: Double? = nil,
        bandwidthMegabitsPerSecond: Double? = nil,
        blockers: [ModeBlocker] = []
    ) {
        self.peer = peer
        self.source = source
        self.state = state
        self.linkClass = linkClass
        self.observedAt = observedAt
        self.latencyMilliseconds = latencyMilliseconds
        self.bandwidthMegabitsPerSecond = bandwidthMegabitsPerSecond
        self.blockers = blockers
    }
}

public struct PeerSnapshot: Sendable, Equatable {
    public let peer: Peer
    public let evidence: [PeerEvidence]
    public let highestState: PeerReadinessState
    public let sources: [DiscoverySource]
    public let linkClasses: [LinkClass]
    public let bestLatencyMilliseconds: Double?
    public let bestBandwidthMegabitsPerSecond: Double?
    public let blockers: [ModeBlocker]
}

public struct DistributedPeerRegistry: Sendable, Equatable {
    private var evidenceByPeerID: [UUID: [PeerEvidence]]

    public init(evidence: [PeerEvidence] = []) {
        self.evidenceByPeerID = [:]
        for item in evidence {
            merge(item)
        }
    }

    public mutating func merge(_ evidence: PeerEvidence) {
        evidenceByPeerID[evidence.peer.id, default: []].append(evidence)
    }

    public mutating func removePeer(id: UUID) {
        evidenceByPeerID.removeValue(forKey: id)
    }

    public func snapshots(
        now: Date = Date(),
        staleAfter: TimeInterval? = nil
    ) -> [PeerSnapshot] {
        evidenceByPeerID.keys.sorted { $0.uuidString < $1.uuidString }.compactMap { id in
            let active = activeEvidence(
                from: evidenceByPeerID[id] ?? [],
                now: now,
                staleAfter: staleAfter)
            guard !active.isEmpty else { return nil }
            return Self.snapshot(from: active)
        }
    }

    private func activeEvidence(
        from evidence: [PeerEvidence],
        now: Date,
        staleAfter: TimeInterval?
    ) -> [PeerEvidence] {
        guard let staleAfter else { return evidence }
        return evidence.filter { now.timeIntervalSince($0.observedAt) <= staleAfter }
    }

    private static func snapshot(from evidence: [PeerEvidence]) -> PeerSnapshot {
        let best = evidence.max { lhs, rhs in
            if lhs.state.rank != rhs.state.rank {
                return lhs.state.rank < rhs.state.rank
            }
            return lhs.observedAt < rhs.observedAt
        }!
        let highestState = best.state
        let sources = Set(evidence.map(\.source)).sorted { $0.rawValue < $1.rawValue }
        let linkClasses = Set(evidence.map(\.linkClass)).sorted { $0.rawValue < $1.rawValue }
        let latencies = evidence.compactMap(\.latencyMilliseconds)
        let bandwidths = evidence.compactMap(\.bandwidthMegabitsPerSecond)
        let blockers = Array(Set(evidence.flatMap(\.blockers)))
            .sorted {
                if $0.mode.rawValue != $1.mode.rawValue {
                    return $0.mode.rawValue < $1.mode.rawValue
                }
                return $0.reason < $1.reason
            }

        return PeerSnapshot(
            peer: best.peer,
            evidence: evidence.sorted {
                if $0.observedAt != $1.observedAt {
                    return $0.observedAt < $1.observedAt
                }
                return $0.source.rawValue < $1.source.rawValue
            },
            highestState: highestState,
            sources: sources,
            linkClasses: linkClasses,
            bestLatencyMilliseconds: latencies.min(),
            bestBandwidthMegabitsPerSecond: bandwidths.max(),
            blockers: blockers)
    }
}
