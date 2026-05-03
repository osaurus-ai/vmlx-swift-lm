import Foundation

/// User-facing distributed inference session. Phase 1A implemented the
/// single-node path; Phase 2 wires `Mode.pipelined` to a caller-supplied
/// `PipelinedTransport`. `Mode.replica` (Phase 1B in osaurus) and
/// `Mode.wired` (Phase 6 JACCL) still throw `notImplementedYet`.
public struct ClusterSession: Sendable {
    private let discovery: any DiscoveryProvider
    private let localGenerator: any LocalGenerator
    private let pipelinedTransport: (any PipelinedTransport)?
    private let mode: Mode
    private let trust: TrustPolicy
    private let staticPeers: [Peer]

    public init(
        discovery: any DiscoveryProvider,
        localGenerator: any LocalGenerator,
        pipelinedTransport: (any PipelinedTransport)? = nil,
        mode: Mode = .auto,
        trust: TrustPolicy = .tofu,
        staticPeers: [Peer] = []
    ) async throws {
        self.discovery = discovery
        self.localGenerator = localGenerator
        self.pipelinedTransport = pipelinedTransport
        self.mode = mode
        self.trust = trust
        self.staticPeers = staticPeers
    }

    /// Live peer-set updates from the injected provider. Each emission is
    /// the full current peer set; consumers diff against their previous view.
    public var peers: AsyncStream<[Peer]> { discovery.peerStream() }

    /// Decide how to run a model. Phase 1A: local for `.localOnly` and for
    /// `.auto` (since no other transport exists yet). Phase 2: `.pipelined`
    /// returns `.pipelinedOver(stagePeerIDs)` when a `PipelinedTransport`
    /// is configured AND there is at least one peer; otherwise throws so
    /// the caller can surface a "no eligible peers" error.
    public func plan(model: ModelHandle) async throws -> ParallelPlan {
        switch mode {
        case .localOnly, .auto:
            return ParallelPlan(placement: .local, model: model)

        case .pipelined:
            guard pipelinedTransport != nil else {
                throw DistributionError.notImplementedYet(.pipelined)
            }
            guard !staticPeers.isEmpty else {
                throw DistributionError.noEligiblePeers
            }
            return ParallelPlan(
                placement: .pipelinedOver(staticPeers.map(\.id)),
                model: model)

        case .replica:
            throw DistributionError.notImplementedYet(.replica)
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
            let stages = staticPeers.filter { ids.contains($0.id) }
            guard stages.count == ids.count else {
                return Self.endStream(.error(
                    "plan referenced peers not in current peer set"))
            }
            return transport.generate(request, stages: stages)

        case .replicaOnPeer, .wiredOver:
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
