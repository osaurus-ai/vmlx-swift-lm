import XCTest
@testable import MLXDistributedCore

private struct StubGenerator: LocalGenerator {
    let tokens: [String]

    func generate(_ request: GenerateRequest) -> AsyncStream<Token> {
        let tokens = self.tokens
        return AsyncStream { continuation in
            Task {
                for tok in tokens { continuation.yield(.text(tok)) }
                continuation.yield(.end(reason: .completed))
                continuation.finish()
            }
        }
    }
}

private struct StubPipelinedTransport: PipelinedTransport {
    func generate(_ request: GenerateRequest, stages: [Peer]) -> AsyncStream<Token> {
        AsyncStream { continuation in
            for stage in stages {
                continuation.yield(.text(stage.hostname))
            }
            continuation.yield(.end(reason: .completed))
            continuation.finish()
        }
    }
}

private struct StubReplicaTransport: ReplicaTransport {
    func generate(_ request: GenerateRequest, peer: Peer) -> AsyncStream<Token> {
        AsyncStream { continuation in
            continuation.yield(.text(peer.hostname))
            continuation.yield(.end(reason: .completed))
            continuation.finish()
        }
    }
}

private actor StubDiscovery: DiscoveryProvider {
    private(set) var advertised: Peer?
    private let initialPeers: [Peer]

    init(peers: [Peer] = []) { self.initialPeers = peers }

    nonisolated func peerStream() -> AsyncStream<[Peer]> {
        let peers = self.initialPeers
        return AsyncStream { continuation in
            continuation.yield(peers)
            continuation.finish()
        }
    }

    func advertise(_ peer: Peer) async throws { advertised = peer }
    func stopAdvertising() async { advertised = nil }
}

final class ClusterSessionTests: XCTestCase {
    private func handle() -> ModelHandle {
        ModelHandle(bundleHash: "abcd000000000001", displayName: "test")
    }

    private func peer(
        id: UUID = UUID(),
        hostname: String = "stage.local",
        modes: Set<Mode> = [.pipelined],
        endpoints: [Endpoint]? = nil,
        modelHashes: ModelHashSet? = nil
    ) -> Peer {
        Peer(
            id: id,
            hostname: hostname,
            capabilities: PeerCapabilities(modes: modes),
            endpoints: endpoints ?? [
                .tls(
                    host: hostname,
                    port: 9443,
                    fingerprintSHA256: String(repeating: "a", count: 64))
            ],
            modelHashes: modelHashes ?? .explicit([handle().bundleHash]),
            memFreeMiB: 1024,
            willingToBeCoordinator: false
        )
    }

    func testLocalOnlyDelegatesToLocalGenerator() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: ["hello", " world"]),
            mode: .localOnly
        )
        let plan = try await session.plan(model: handle())
        XCTAssertEqual(plan.placement, .local)

        var collected: [Token] = []
        for await t in session.generate(
            GenerateRequest(model: handle(), prompt: "hi", maxTokens: 10),
            plan: plan
        ) {
            collected.append(t)
        }
        XCTAssertEqual(collected.count, 3)
        XCTAssertEqual(collected[0], .text("hello"))
        XCTAssertEqual(collected[1], .text(" world"))
        XCTAssertEqual(collected[2], .end(reason: .completed))
    }

    func testAutoModeFallsBackToLocalWhenNoPeers() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(peers: []),
            localGenerator: StubGenerator(tokens: ["x"]),
            mode: .auto
        )
        let plan = try await session.plan(model: handle())
        XCTAssertEqual(plan.placement, .local)
    }

    func testReplicaThrowsNotImplementedYet() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            mode: .replica
        )
        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected notImplementedYet")
        } catch DistributionError.notImplementedYet(.replica) {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testReplicaPlansFirstEligiblePeer() async throws {
        let ineligible = peer(
            hostname: "pp-only.local",
            modes: [.pipelined])
        let eligible = peer(
            hostname: "replica.local",
            modes: [.replica])
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            replicaTransport: StubReplicaTransport(),
            mode: .replica,
            staticPeers: [ineligible, eligible]
        )

        let plan = try await session.plan(model: handle())
        XCTAssertEqual(plan.placement, .replicaOnPeer(eligible.id))
    }

    func testReplicaRejectsPeerWithoutMatchingModel() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            replicaTransport: StubReplicaTransport(),
            mode: .replica,
            staticPeers: [
                peer(
                    modes: [.replica],
                    modelHashes: .explicit(["ffff000000000001"]))
            ]
        )

        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected noEligiblePeers")
        } catch DistributionError.noEligiblePeers {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testReplicaGenerateUsesSelectedPeer() async throws {
        let selected = peer(
            hostname: "selected-replica.local",
            modes: [.replica])
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            replicaTransport: StubReplicaTransport(),
            mode: .replica,
            staticPeers: [selected]
        )
        let plan = try await session.plan(model: handle())
        let request = GenerateRequest(model: handle(), prompt: "hi", maxTokens: 8)

        var collected: [Token] = []
        for await token in session.generate(request, plan: plan) {
            collected.append(token)
        }

        XCTAssertEqual(collected, [
            .text("selected-replica.local"),
            .end(reason: .completed)
        ])
    }

    func testPipelinedThrowsNotImplementedYet() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            mode: .pipelined
        )
        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected notImplementedYet")
        } catch DistributionError.notImplementedYet(.pipelined) {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testPipelinedPlansFirstEligiblePeerOnly() async throws {
        let ineligible = peer(
            hostname: "replica.local",
            modes: [.replica])
        let eligible = peer(hostname: "stage-a.local")
        let laterEligible = peer(hostname: "stage-b.local")
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            pipelinedTransport: StubPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [ineligible, eligible, laterEligible]
        )

        let plan = try await session.plan(model: handle())
        XCTAssertEqual(plan.placement, .pipelinedOver([eligible.id]))
    }

    func testPipelinedRejectsPeerWithoutPipelinedMode() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            pipelinedTransport: StubPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [peer(modes: [.replica])]
        )

        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected noEligiblePeers")
        } catch DistributionError.noEligiblePeers {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testPipelinedRejectsPeerWithoutMatchingModel() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            pipelinedTransport: StubPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [peer(modelHashes: .explicit(["ffff000000000001"]))]
        )

        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected noEligiblePeers")
        } catch DistributionError.noEligiblePeers {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testPipelinedAcceptsOverflowModelSet() async throws {
        let stage = peer(modelHashes: .overflow)
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            pipelinedTransport: StubPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [stage]
        )

        let plan = try await session.plan(model: handle())
        XCTAssertEqual(plan.placement, .pipelinedOver([stage.id]))
    }

    func testPipelinedRejectsPeerWithoutTLSEndpoint() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            pipelinedTransport: StubPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [
                peer(endpoints: [.rdma(gid: "abc", devices: ["rdma0"])])
            ]
        )

        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected noEligiblePeers")
        } catch DistributionError.noEligiblePeers {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testPipelinedGeneratePreservesPlanStageOrder() async throws {
        let first = peer(hostname: "first.local")
        let second = peer(hostname: "second.local")
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            pipelinedTransport: StubPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [second, first]
        )
        let plan = ParallelPlan(
            placement: .pipelinedOver([first.id, second.id]),
            model: handle())
        let request = GenerateRequest(model: handle(), prompt: "hi", maxTokens: 8)

        var collected: [Token] = []
        for await token in session.generate(request, plan: plan) {
            collected.append(token)
        }

        XCTAssertEqual(collected, [
            .text("first.local"),
            .text("second.local"),
            .end(reason: .completed)
        ])
    }

    func testWiredThrowsNotImplementedYet() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: StubGenerator(tokens: []),
            mode: .wired
        )
        do {
            _ = try await session.plan(model: handle())
            XCTFail("expected notImplementedYet")
        } catch DistributionError.notImplementedYet(.wired) {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }
}
