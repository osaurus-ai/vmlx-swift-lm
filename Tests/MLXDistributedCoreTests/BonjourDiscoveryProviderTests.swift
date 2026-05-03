import XCTest
@testable import MLXDistributedCore

final class BonjourAdvertiseTests: XCTestCase {
    func testAdvertiseSetsTXTRecordFromPeer() async throws {
        let provider = BonjourDiscoveryProvider(serviceType: "_vmlxtest._tcp.")
        let peer = Peer(
            id: UUID(),
            hostname: "self.local",
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [.tls(host: "self.local", port: 7901,
                             fingerprintSHA256: String(repeating: "a", count: 64))],
            modelHashes: .explicit(["abcd000000000001"]),
            memFreeMiB: 1024,
            willingToBeCoordinator: false
        )
        try await provider.advertise(peer)

        let advertisedTXT = await provider._advertisedTXTForTesting()
        XCTAssertEqual(advertisedTXT["dist.v"], "1")
        XCTAssertEqual(advertisedTXT["dist.modes"], "replica")
        XCTAssertEqual(advertisedTXT["dist.tls.port"], "7901")
        XCTAssertEqual(advertisedTXT["dist.coord"], "0")

        await provider.stopAdvertising()
        let after = await provider._advertisedTXTForTesting()
        XCTAssertNil(after["dist.v"], "stopAdvertising clears the cache")
    }

    func testAdvertiseTwiceReplacesPriorTXT() async throws {
        let provider = BonjourDiscoveryProvider(serviceType: "_vmlxtest._tcp.")
        let id = UUID()
        let p1 = Peer(
            id: id, hostname: "h",
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [.tls(host: "h", port: 1,
                             fingerprintSHA256: String(repeating: "a", count: 64))],
            modelHashes: .explicit(["one"]),
            memFreeMiB: 1, willingToBeCoordinator: false
        )
        let p2 = Peer(
            id: id, hostname: "h",
            capabilities: PeerCapabilities(modes: [.replica, .pipelined]),
            endpoints: [.tls(host: "h", port: 1,
                             fingerprintSHA256: String(repeating: "b", count: 64))],
            modelHashes: .explicit(["two"]),
            memFreeMiB: 2, willingToBeCoordinator: true
        )
        try await provider.advertise(p1)
        try await provider.advertise(p2)
        let txt = await provider._advertisedTXTForTesting()
        XCTAssertEqual(txt["dist.coord"], "1", "second advertise replaces the first")
        XCTAssertEqual(txt["dist.models"], "two")
    }
}

final class BonjourRoundTripTests: XCTestCase {
    /// End-to-end: two providers in the same process, one advertises, the
    /// other browses, and we verify the second one sees the first.
    /// Bonjour over loopback in unit tests can be flaky on sandboxed CI;
    /// gate on env var so developers can opt in for local validation.
    func testAdvertiseAndBrowseRoundTrip() async throws {
        guard ProcessInfo.processInfo.environment["VMLX_RUN_BONJOUR_TESTS"] == "1"
        else { throw XCTSkip("set VMLX_RUN_BONJOUR_TESTS=1 to run") }

        let serviceType = "_vmlxbjrt._tcp."
        let advertiser = BonjourDiscoveryProvider(serviceType: serviceType)
        let browser = BonjourDiscoveryProvider(serviceType: serviceType)

        let id = UUID()
        let peer = Peer(
            id: id,
            hostname: ProcessInfo.processInfo.hostName,
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [.tls(host: "localhost", port: 17901,
                             fingerprintSHA256: String(repeating: "9", count: 64))],
            modelHashes: .explicit(["abcd000000000099"]),
            memFreeMiB: 64, willingToBeCoordinator: false
        )

        try await advertiser.advertise(peer)
        defer { Task { await advertiser.stopAdvertising() } }

        let stream = browser.peerStream()
        let saw = Task {
            for await peers in stream {
                if peers.contains(where: { $0.id == id }) { return true }
            }
            return false
        }
        let timeout = Task<Bool, Never> {
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            return false
        }
        let result = await withTaskGroup(of: Bool.self) { group -> Bool in
            group.addTask { await saw.value }
            group.addTask { await timeout.value }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        XCTAssertTrue(result, "browse did not see advertise within 5s")
    }
}
