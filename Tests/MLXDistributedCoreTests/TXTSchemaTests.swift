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

final class TXTSchemaDecodeTests: XCTestCase {
    func testDecodesMinimalReplica() throws {
        let id = UUID(uuidString: "9F9E1F44-0000-0000-0000-000000000001")!
        let txt: [String: String] = [
            "dist.v": "1",
            "dist.peer.id": id.uuidString.lowercased(),
            "dist.modes": "replica",
            "dist.tls.port": "7901",
            "dist.tls.fp": String(repeating: "a", count: 64),
            "dist.models": "ab12cd34ef561234",
            "dist.coord": "0"
        ]
        let peer = try TXTSchema.decode(txt, hostname: "host.local")
        XCTAssertEqual(peer.id, id)
        XCTAssertEqual(peer.hostname, "host.local")
        XCTAssertEqual(peer.capabilities.modes, [.replica])
        XCTAssertEqual(peer.modelHashes, .explicit(["ab12cd34ef561234"]))
        XCTAssertFalse(peer.willingToBeCoordinator)
    }

    func testRejectsUnknownSchemaVersion() {
        let txt: [String: String] = [
            "dist.v": "99",
            "dist.peer.id": UUID().uuidString,
            "dist.modes": "replica",
            "dist.tls.port": "1",
            "dist.tls.fp": String(repeating: "a", count: 64),
            "dist.models": "*",
            "dist.coord": "0"
        ]
        XCTAssertThrowsError(try TXTSchema.decode(txt, hostname: "h"))
    }

    func testRejectsMissingRequiredKeys() {
        let txt: [String: String] = ["dist.v": "1"]
        XCTAssertThrowsError(try TXTSchema.decode(txt, hostname: "h"))
    }

    func testDecodesOverflowSentinel() throws {
        let txt: [String: String] = [
            "dist.v": "1", "dist.peer.id": UUID().uuidString,
            "dist.modes": "replica", "dist.tls.port": "1",
            "dist.tls.fp": String(repeating: "a", count: 64),
            "dist.models": "*", "dist.coord": "0"
        ]
        let peer = try TXTSchema.decode(txt, hostname: "h")
        XCTAssertEqual(peer.modelHashes, .overflow)
    }

    func testRoundTripsAllFields() throws {
        let id = UUID()
        let original = Peer(
            id: id,
            hostname: "rt.local",
            capabilities: PeerCapabilities(modes: [.wired, .pipelined, .replica]),
            endpoints: [
                .tls(host: "10.0.0.2", port: 9999,
                     fingerprintSHA256: String(repeating: "f", count: 64)),
                .rdma(gid: "fe80000000000005", devices: ["mlx5_0"])
            ],
            modelHashes: .explicit(["abcd000000000001", "abcd000000000002"]),
            memFreeMiB: 42,
            willingToBeCoordinator: true
        )

        let txt = try TXTSchema.encode(original)
        let decoded = try TXTSchema.decode(txt, hostname: "rt.local")

        XCTAssertEqual(decoded.id, id)
        XCTAssertEqual(decoded.capabilities.modes, original.capabilities.modes)
        XCTAssertEqual(decoded.modelHashes, original.modelHashes)
        XCTAssertEqual(decoded.memFreeMiB, 42)
        XCTAssertTrue(decoded.willingToBeCoordinator)

        let hasRDMA = decoded.endpoints.contains {
            if case .rdma = $0 { return true } else { return false }
        }
        let hasTLS = decoded.endpoints.contains {
            if case .tls = $0 { return true } else { return false }
        }
        XCTAssertTrue(hasRDMA)
        XCTAssertTrue(hasTLS)
    }
}
