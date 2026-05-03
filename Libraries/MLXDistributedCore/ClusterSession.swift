import Foundation

/// User-facing distributed inference session. Phase 1A implements only the
/// local single-node path through `Mode.localOnly` and `Mode.auto`-with-no-
/// peers; networked / wired modes throw `DistributionError.notImplementedYet`
/// and are filled in by Phases 1B / 2 / 3 / 6.
public struct ClusterSession: Sendable {
    private let discovery: any DiscoveryProvider
    private let localGenerator: any LocalGenerator
    private let mode: Mode
    private let trust: TrustPolicy

    public init(
        discovery: any DiscoveryProvider,
        localGenerator: any LocalGenerator,
        mode: Mode = .auto,
        trust: TrustPolicy = .tofu
    ) async throws {
        self.discovery = discovery
        self.localGenerator = localGenerator
        self.mode = mode
        self.trust = trust
    }

    /// Live peer-set updates from the injected provider. Each emission is
    /// the full current peer set; consumers diff against their previous view.
    public var peers: AsyncStream<[Peer]> { discovery.peerStream() }

    /// Decide how to run a model. Phase 1A: local for `.localOnly` and for
    /// `.auto` (since no other transport exists yet); throws for the other modes.
    public func plan(model: ModelHandle) async throws -> ParallelPlan {
        switch mode {
        case .localOnly, .auto:
            return ParallelPlan(placement: .local, model: model)
        case .replica:
            throw DistributionError.notImplementedYet(.replica)
        case .pipelined:
            throw DistributionError.notImplementedYet(.pipelined)
        case .wired:
            throw DistributionError.notImplementedYet(.wired)
        }
    }

    /// Stream tokens for a request. Phase 1A: only `.local` placement
    /// is implemented; remote placements emit a structured error and end.
    public func generate(
        _ request: GenerateRequest,
        plan: ParallelPlan
    ) -> AsyncStream<Token> {
        switch plan.placement {
        case .local:
            return localGenerator.generate(request)
        case .replicaOnPeer, .pipelinedOver, .wiredOver:
            return AsyncStream { continuation in
                continuation.yield(.end(reason: .error(
                    "remote placement not implemented in Phase 1A")))
                continuation.finish()
            }
        }
    }
}
