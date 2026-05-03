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
