import Foundation

public struct DistributedModeCandidate: Sendable, Equatable {
    public let mode: Mode
    public let peers: [PeerSnapshot]
    public let readiness: PeerReadinessState
    public let score: Int

    public init(
        mode: Mode,
        peers: [PeerSnapshot],
        readiness: PeerReadinessState,
        score: Int
    ) {
        self.mode = mode
        self.peers = peers
        self.readiness = readiness
        self.score = score
    }
}

public struct DistributedModePlanner: Sendable, Equatable {
    public init() {}

    public func candidates(
        for model: ModelHandle,
        snapshots: [PeerSnapshot]
    ) -> [DistributedModeCandidate] {
        [
            replicaCandidate(for: model, snapshots: snapshots),
            pipelinedCandidate(for: model, snapshots: snapshots),
            wiredCandidate(for: model, snapshots: snapshots),
        ]
        .compactMap { $0 }
        .sorted {
            if $0.score != $1.score {
                return $0.score > $1.score
            }
            return $0.mode.rawValue < $1.mode.rawValue
        }
    }

    public func bestCandidate(
        for model: ModelHandle,
        snapshots: [PeerSnapshot],
        preferredMode: Mode = .auto
    ) -> DistributedModeCandidate? {
        let all = candidates(for: model, snapshots: snapshots)
        switch preferredMode {
        case .auto:
            return all.first
        case .localOnly:
            return nil
        case .replica, .pipelined, .wired:
            return all.first { $0.mode == preferredMode }
        }
    }

    private func replicaCandidate(
        for model: ModelHandle,
        snapshots: [PeerSnapshot]
    ) -> DistributedModeCandidate? {
        guard let best = snapshots
            .filter({ isRunnable($0, for: model, mode: .replica, minimumState: .replicaReady) })
            .sorted(by: preferFastestThenNewest)
            .first else {
            return nil
        }
        return DistributedModeCandidate(
            mode: .replica,
            peers: [best],
            readiness: best.highestState,
            score: 100 + readinessScore(best))
    }

    private func pipelinedCandidate(
        for model: ModelHandle,
        snapshots: [PeerSnapshot]
    ) -> DistributedModeCandidate? {
        let stages = snapshots
            .filter { isRunnable($0, for: model, mode: .pipelined, minimumState: .pipelineCandidate) }
            .sorted(by: preferFastestThenNewest)
        guard !stages.isEmpty else { return nil }
        return DistributedModeCandidate(
            mode: .pipelined,
            peers: stages,
            readiness: stages.map(\.highestState).min(by: { $0.rank < $1.rank }) ?? .pipelineCandidate,
            score: 200 + stages.count + stages.map(readinessScore).reduce(0, +))
    }

    private func wiredCandidate(
        for model: ModelHandle,
        snapshots: [PeerSnapshot]
    ) -> DistributedModeCandidate? {
        let peers = snapshots
            .filter { isRunnable($0, for: model, mode: .wired, minimumState: .rdmaReady) }
            .filter { $0.linkClasses.contains(.rdma) || $0.linkClasses.contains(.thunderboltBridge) }
            .sorted(by: preferFastestThenNewest)
        guard !peers.isEmpty else { return nil }
        return DistributedModeCandidate(
            mode: .wired,
            peers: peers,
            readiness: peers.map(\.highestState).min(by: { $0.rank < $1.rank }) ?? .rdmaReady,
            score: 300 + peers.count + peers.map(readinessScore).reduce(0, +))
    }

    private func isRunnable(
        _ snapshot: PeerSnapshot,
        for model: ModelHandle,
        mode: Mode,
        minimumState: PeerReadinessState
    ) -> Bool {
        guard snapshot.highestState.rank >= minimumState.rank,
              !snapshot.blockers.contains(where: { $0.mode == mode }) else {
            return false
        }

        switch mode {
        case .replica:
            return snapshot.peer.isEligibleForReplica(model: model)
        case .pipelined:
            return snapshot.peer.isEligibleForPipelined(model: model)
        case .wired:
            return snapshot.peer.isEligibleForWired(model: model)
        case .auto, .localOnly:
            return false
        }
    }

    private func readinessScore(_ snapshot: PeerSnapshot) -> Int {
        snapshot.highestState.rank * 10
    }

    private func preferFastestThenNewest(_ lhs: PeerSnapshot, _ rhs: PeerSnapshot) -> Bool {
        switch (lhs.bestLatencyMilliseconds, rhs.bestLatencyMilliseconds) {
        case let (left?, right?) where left != right:
            return left < right
        case (.some, nil):
            return true
        case (nil, .some):
            return false
        default:
            return lhs.peer.id.uuidString < rhs.peer.id.uuidString
        }
    }
}
