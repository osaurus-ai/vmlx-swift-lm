import XCTest
@testable import MLXDistributedCore

final class TopologyEdgeTests: XCTestCase {
    func testSocketAndRdmaEdgesBetweenSamePeersCoexist() {
        let local = UUID()
        let remote = UUID()
        var topology = PeerTopology()

        topology.upsert(
            .socket(SocketEdge(
                host: "fe80::2%bridge0",
                port: 7901,
                interfaceName: "bridge0",
                linkClass: .thunderboltBridge,
                tlsState: .trusted(fingerprintSHA256: fp("a")),
                observedAt: Date(timeIntervalSince1970: 1))),
            localPeerID: local,
            remotePeerID: remote,
            label: "bridge0-tls")
        topology.upsert(
            .rdma(RdmaEdge(
                sourceDevice: "rdma_en5",
                sinkDevice: "rdma_en6",
                rdmaCtlEnabled: true,
                backendBuild: BackendBuildIdentity(
                    ringBackend: .real,
                    jacclBackend: .real),
                validatedAt: Date(timeIntervalSince1970: 2))),
            localPeerID: local,
            remotePeerID: remote,
            label: "jaccl-rdma")

        XCTAssertEqual(topology.edges(from: local, to: remote).count, 2)
        XCTAssertEqual(topology.edges(from: local, to: remote, kind: .socket).count, 1)
        XCTAssertTrue(topology.hasUsableJACCLEdge(from: local, to: remote))
    }

    func testThunderboltSocketDoesNotImplyUsableRdma() {
        let local = UUID()
        let remote = UUID()
        var topology = PeerTopology()
        topology.upsert(
            .socket(SocketEdge(
                host: "fe80::2%bridge0",
                port: 7901,
                interfaceName: "bridge0",
                linkClass: .thunderboltBridge,
                tlsState: .trusted(fingerprintSHA256: fp("b")),
                observedAt: Date(timeIntervalSince1970: 1))),
            localPeerID: local,
            remotePeerID: remote,
            label: "bridge0-tls")

        XCTAssertFalse(topology.hasUsableJACCLEdge(from: local, to: remote))
    }

    func testStubJACCLBackendBlocksRdmaUsability() {
        let edge = RdmaEdge(
            sourceDevice: "rdma_en5",
            sinkDevice: "rdma_en6",
            rdmaCtlEnabled: true,
            backendBuild: BackendBuildIdentity(
                ringBackend: .real,
                jacclBackend: .stub),
            validatedAt: Date(timeIntervalSince1970: 1))

        XCTAssertFalse(edge.isJACCLUsable)
    }

    func testHandshakeFailureIsNotTrustedSocket() {
        let edge = SocketEdge(
            host: "node.local",
            port: 7901,
            linkClass: .wifi,
            tlsState: .handshakeFailed(reason: "fingerprint mismatch"),
            observedAt: Date(timeIntervalSince1970: 1))

        XCTAssertFalse(edge.isTrusted)
    }

    private func fp(_ value: Character) -> String {
        String(repeating: String(value), count: 64)
    }
}
