import XCTest
@testable import MLXDistributedCore

final class DistributedModePlannerTests: XCTestCase {
    func testAutoPrefersRDMAReadyWiredCandidateOverReplica() {
        let model = ModelHandle(bundleHash: "abcd000000000001", displayName: "test")
        let planner = DistributedModePlanner()
        let snapshots = [
            makeSnapshot(
                id: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
                mode: .replica,
                state: .replicaReady,
                linkClass: .wifi,
                endpoint: .tls(host: "wifi.local", port: 7901, fingerprintSHA256: fp("a")),
                latencyMilliseconds: 5.0),
            makeSnapshot(
                id: UUID(uuidString: "00000000-0000-0000-0000-000000000002")!,
                mode: .wired,
                state: .rdmaReady,
                linkClass: .rdma,
                endpoint: .rdma(gid: "fe80::2", devices: ["rdma_en5"]),
                latencyMilliseconds: 0.4),
        ]

        let best = planner.bestCandidate(for: model, snapshots: snapshots)

        XCTAssertEqual(best?.mode, .wired)
        XCTAssertEqual(best?.peers.map(\.peer.id), [snapshots[1].peer.id])
    }

    func testModeBlockersPreventFalsePositiveWiredSelection() {
        let model = ModelHandle(bundleHash: "abcd000000000001", displayName: "test")
        let planner = DistributedModePlanner()
        let snapshots = [
            makeSnapshot(
                id: UUID(),
                mode: .wired,
                state: .rdmaReady,
                linkClass: .rdma,
                endpoint: .rdma(gid: "fe80::2", devices: ["rdma_en5"]),
                blockers: [
                    ModeBlocker(mode: .wired, reason: "JACCL backend is unavailable")
                ]),
        ]

        XCTAssertNil(planner.bestCandidate(
            for: model,
            snapshots: snapshots,
            preferredMode: .wired))
    }

    func testMissingModelHashSuppressesRemoteCandidate() {
        let model = ModelHandle(bundleHash: "missing", displayName: "test")
        let planner = DistributedModePlanner()
        let snapshots = [
            makeSnapshot(
                id: UUID(),
                mode: .replica,
                state: .replicaReady,
                linkClass: .wifi,
                endpoint: .tls(host: "wifi.local", port: 7901, fingerprintSHA256: fp("a"))),
        ]

        XCTAssertTrue(planner.candidates(for: model, snapshots: snapshots).isEmpty)
    }

    private func makeSnapshot(
        id: UUID,
        mode: Mode,
        state: PeerReadinessState,
        linkClass: LinkClass,
        endpoint: Endpoint,
        latencyMilliseconds: Double? = nil,
        blockers: [ModeBlocker] = []
    ) -> PeerSnapshot {
        let peer = Peer(
            id: id,
            hostname: "\(id.uuidString).local",
            capabilities: PeerCapabilities(modes: [mode]),
            endpoints: [endpoint],
            modelHashes: .explicit(["abcd000000000001"]),
            memFreeMiB: 128 * 1024,
            willingToBeCoordinator: false)
        let evidence = PeerEvidence(
            peer: peer,
            source: .manual,
            state: state,
            linkClass: linkClass,
            observedAt: Date(timeIntervalSince1970: 100),
            latencyMilliseconds: latencyMilliseconds,
            blockers: blockers)
        var registry = DistributedPeerRegistry()
        registry.merge(evidence)
        return registry.snapshots(now: Date(timeIntervalSince1970: 101))[0]
    }

    private func fp(_ value: Character) -> String {
        String(repeating: String(value), count: 64)
    }
}
