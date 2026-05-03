import XCTest
@testable import MLXDistributedCore

final class PeerTests: XCTestCase {
    func testPeerEqualityIsDrivenByID() {
        let id = UUID()
        let a = Peer(
            id: id, hostname: "a.local", capabilities: .empty,
            endpoints: [], modelHashes: .explicit([]))
        let b = Peer(
            id: id, hostname: "b.local", capabilities: .empty,
            endpoints: [], modelHashes: .overflow)
        XCTAssertEqual(a, b, "Peer equality must use only the stable id")
    }

    func testCapabilitiesContainsCheck() {
        let caps = PeerCapabilities(modes: [.replica, .pipelined])
        XCTAssertTrue(caps.supports(.replica))
        XCTAssertTrue(caps.supports(.pipelined))
        XCTAssertFalse(caps.supports(.wired))
    }

    func testEndpointTagging() {
        let tls = Endpoint.tls(host: "192.168.1.5", port: 7901,
                               fingerprintSHA256: "ab")
        let rdma = Endpoint.rdma(gid: "fe80...", devices: ["mlx5_0"])
        if case .tls(let h, let p, _) = tls {
            XCTAssertEqual(h, "192.168.1.5")
            XCTAssertEqual(p, 7901)
        } else {
            XCTFail("Expected .tls")
        }
        if case .rdma(_, let devs) = rdma {
            XCTAssertEqual(devs, ["mlx5_0"])
        } else {
            XCTFail("Expected .rdma")
        }
    }
}
