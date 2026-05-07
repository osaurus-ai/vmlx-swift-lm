import Foundation

/// User-facing distributed inference session. The default path remains
/// single-node. `Mode.pipelined` is explicit opt-in and only plans against
/// peers that advertise a matching model plus a TLS pipeline endpoint.
/// `Mode.replica` is request-level fan-out through an injected transport.
/// `Mode.wired` still throws `notImplementedYet`.
public struct ClusterSession: Sendable {
    private let discovery: any DiscoveryProvider
    private let localGenerator: any LocalGenerator
    private let replicaTransport: (any ReplicaTransport)?
    private let pipelinedTransport: (any PipelinedTransport)?
    private let mode: Mode
    private let trust: TrustPolicy
    private let staticPeers: [Peer]

    public init(
        discovery: any DiscoveryProvider,
        localGenerator: any LocalGenerator,
        replicaTransport: (any ReplicaTransport)? = nil,
        pipelinedTransport: (any PipelinedTransport)? = nil,
        mode: Mode = .auto,
        trust: TrustPolicy = .tofu,
        staticPeers: [Peer] = []
    ) async throws {
        self.discovery = discovery
        self.localGenerator = localGenerator
        self.replicaTransport = replicaTransport
        self.pipelinedTransport = pipelinedTransport
        self.mode = mode
        self.trust = trust
        self.staticPeers = staticPeers
    }

    /// Live peer-set updates from the injected provider. Each emission is
    /// the full current peer set; consumers diff against their previous view.
    public var peers: AsyncStream<[Peer]> { discovery.peerStream() }

    /// Decide how to run a model. `.localOnly` and `.auto` are intentionally
    /// local until dynamic discovery, health, and replan policy are wired.
    /// `.replica` selects one request-level peer. `.pipelined` currently
    /// selects a single remote stage because
    /// `TLSPipelinedTransport` is a two-rank prompt/token path, not a
    /// full multi-stage activation runtime yet.
    public func plan(model: ModelHandle) async throws -> ParallelPlan {
        switch mode {
        case .localOnly, .auto:
            return ParallelPlan(placement: .local, model: model)

        case .pipelined:
            guard pipelinedTransport != nil else {
                throw DistributionError.notImplementedYet(.pipelined)
            }
            guard let stage = staticPeers.first(where: {
                $0.isEligibleForPipelined(model: model)
            }) else {
                throw DistributionError.noEligiblePeers
            }
            return ParallelPlan(
                placement: .pipelinedOver([stage.id]),
                model: model)

        case .replica:
            guard replicaTransport != nil else {
                throw DistributionError.notImplementedYet(.replica)
            }
            guard let peer = staticPeers.first(where: {
                $0.isEligibleForReplica(model: model)
            }) else {
                throw DistributionError.noEligiblePeers
            }
            return ParallelPlan(
                placement: .replicaOnPeer(peer.id),
                model: model)

        case .wired:
            throw DistributionError.notImplementedYet(.wired)
        }
    }

    /// Stream tokens for a request. Phase 2 routes `.pipelinedOver` through
    /// the supplied `PipelinedTransport`; other remote placements still
    /// emit a structured error.
    public func generate(
        _ request: GenerateRequest,
        plan: ParallelPlan
    ) -> AsyncStream<Token> {
        switch plan.placement {
        case .local:
            return localGenerator.generate(request)

        case .pipelinedOver(let ids):
            guard let transport = pipelinedTransport else {
                return Self.endStream(.error(
                    "pipelinedOver placement requires a PipelinedTransport"))
            }
            var peersByID: [UUID: Peer] = [:]
            for peer in staticPeers where peersByID[peer.id] == nil {
                peersByID[peer.id] = peer
            }
            let stages = ids.compactMap { peersByID[$0] }
            guard stages.count == ids.count else {
                return Self.endStream(.error(
                    "plan referenced peers not in current peer set"))
            }
            return transport.generate(request, stages: stages)

        case .replicaOnPeer(let id):
            guard let transport = replicaTransport else {
                return Self.endStream(.error(
                    "replicaOnPeer placement requires a ReplicaTransport"))
            }
            guard let peer = staticPeers.first(where: { $0.id == id }) else {
                return Self.endStream(.error(
                    "plan referenced peer not in current peer set"))
            }
            return transport.generate(request, peer: peer)

        case .wiredOver:
            return Self.endStream(.error(
                "remote placement not implemented in this engine version"))
        }
    }

    private static func endStream(_ reason: Token.EndReason) -> AsyncStream<Token> {
        AsyncStream { continuation in
            continuation.yield(.end(reason: reason))
            continuation.finish()
        }
    }
}
