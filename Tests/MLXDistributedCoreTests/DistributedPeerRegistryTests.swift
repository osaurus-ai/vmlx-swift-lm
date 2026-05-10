import XCTest
@testable import MLXDistributedCore

final class DistributedPeerRegistryTests: XCTestCase {
    func testMergesEvidenceByPeerAndKeepsHighestState() {
        let id = UUID()
        let peer = makePeer(id: id, hostname: "m5.local")
        let now = Date(timeIntervalSince1970: 1_000)
        var registry = DistributedPeerRegistry()

        registry.merge(PeerEvidence(
            peer: peer,
            source: .bonjour,
            state: .discovered,
            linkClass: .wifi,
            observedAt: now,
            blockers: []))
        registry.merge(PeerEvidence(
            peer: makePeer(id: id, hostname: "m5-bridge.local"),
            source: .manual,
            state: .replicaReady,
            linkClass: .thunderboltBridge,
            observedAt: now.addingTimeInterval(1),
            latencyMilliseconds: 0.42,
            blockers: []))

        let snapshot = registry.snapshots(now: now.addingTimeInterval(2))

        XCTAssertEqual(snapshot.count, 1)
        XCTAssertEqual(snapshot[0].peer.id, id)
        XCTAssertEqual(snapshot[0].peer.hostname, "m5-bridge.local")
        XCTAssertEqual(snapshot[0].highestState, .replicaReady)
        XCTAssertEqual(snapshot[0].linkClasses, [.thunderboltBridge, .wifi])
        XCTAssertEqual(snapshot[0].sources, [.bonjour, .manual])
        XCTAssertEqual(snapshot[0].bestLatencyMilliseconds, 0.42)
    }

    func testExpiresStaleEvidence() {
        let id = UUID()
        let peer = makePeer(id: id, hostname: "old.local")
        var registry = DistributedPeerRegistry()
        registry.merge(PeerEvidence(
            peer: peer,
            source: .bonjour,
            state: .reachable,
            linkClass: .wifi,
            observedAt: Date(timeIntervalSince1970: 10),
            blockers: []))

        let snapshot = registry.snapshots(
            now: Date(timeIntervalSince1970: 200),
            staleAfter: 30)

        XCTAssertTrue(snapshot.isEmpty)
    }

    func testPreservesModeBlockersForFalsePositiveGuards() {
        let id = UUID()
        var registry = DistributedPeerRegistry()
        registry.merge(PeerEvidence(
            peer: makePeer(id: id, hostname: "wired.local"),
            source: .interfaceBonjour,
            state: .wiredCandidate,
            linkClass: .thunderboltBridge,
            observedAt: Date(timeIntervalSince1970: 20),
            blockers: [
                ModeBlocker(mode: .wired, reason: "JACCL backend is unavailable")
            ]))

        let snapshot = registry.snapshots(now: Date(timeIntervalSince1970: 21))

        XCTAssertEqual(snapshot.count, 1)
        XCTAssertEqual(snapshot[0].highestState, .wiredCandidate)
        XCTAssertEqual(snapshot[0].blockers, [
            ModeBlocker(mode: .wired, reason: "JACCL backend is unavailable")
        ])
    }

    private func makePeer(id: UUID, hostname: String) -> Peer {
        Peer(
            id: id,
            hostname: hostname,
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [
                .tls(
                    host: hostname,
                    port: 7901,
                    fingerprintSHA256: String(repeating: "a", count: 64))
            ],
            modelHashes: .explicit(["abcd000000000001"]),
            memFreeMiB: 1024,
            willingToBeCoordinator: false)
    }
}
