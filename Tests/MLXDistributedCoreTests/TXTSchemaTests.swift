import XCTest
@testable import MLXDistributedCore

final class TXTSchemaEncodeTests: XCTestCase {
    func testEncodesMinimalReplicaPeer() throws {
        let id = UUID(uuidString: "9F9E1F44-0000-0000-0000-000000000001")!
        let peer = Peer(
            id: id,
            hostname: "host.local",
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [.tls(host: "192.168.1.5", port: 7901,
                             fingerprintSHA256: String(repeating: "a", count: 64))],
            modelHashes: .explicit(["ab12cd34ef561234"]),
            memFreeMiB: 87432,
            willingToBeCoordinator: false
        )

        let txt = try TXTSchema.encode(peer)

        XCTAssertEqual(txt["dist.v"], "1")
        XCTAssertEqual(txt["dist.modes"], "replica")
        XCTAssertEqual(txt["dist.tls.port"], "7901")
        XCTAssertEqual(txt["dist.tls.fp"], String(repeating: "a", count: 64))
        XCTAssertEqual(txt["dist.models"], "ab12cd34ef561234")
        XCTAssertEqual(txt["dist.mem.free"], "87432")
        XCTAssertEqual(txt["dist.peer.id"], id.uuidString.lowercased())
        XCTAssertEqual(txt["dist.coord"], "0")
        XCTAssertNil(txt["dist.rdma.gid"], "no RDMA endpoint -> no rdma keys")
    }

    func testEncodesRDMAKeysWhenPresent() throws {
        let peer = Peer(
            id: UUID(),
            hostname: "rdma.local",
            capabilities: PeerCapabilities(modes: [.wired, .pipelined, .replica]),
            endpoints: [
                .tls(host: "10.0.0.2", port: 7901,
                     fingerprintSHA256: String(repeating: "b", count: 64)),
                .rdma(gid: "fe80000000000000", devices: ["mlx5_0", "mlx5_1"])
            ],
            modelHashes: .explicit([]),
            memFreeMiB: nil,
            willingToBeCoordinator: true
        )

        let txt = try TXTSchema.encode(peer)
        XCTAssertEqual(txt["dist.rdma.gid"], "fe80000000000000")
        XCTAssertEqual(txt["dist.rdma.devs"], "mlx5_0,mlx5_1")
        XCTAssertEqual(txt["dist.coord"], "1")
        XCTAssertEqual(
            txt["dist.modes"]?.split(separator: ",").map(String.init).sorted(),
            ["pp", "replica", "tp"])
        XCTAssertNil(txt["dist.mem.free"], "nil memFreeMiB omits the key")
    }

    func testOverflowSentinel() throws {
        let peer = Peer(
            id: UUID(), hostname: "h",
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [.tls(host: "x", port: 1,
                             fingerprintSHA256: String(repeating: "c", count: 64))],
            modelHashes: .overflow,
            memFreeMiB: 1, willingToBeCoordinator: false
        )

        let txt = try TXTSchema.encode(peer)
        XCTAssertEqual(txt["dist.models"], "*")
    }
}
