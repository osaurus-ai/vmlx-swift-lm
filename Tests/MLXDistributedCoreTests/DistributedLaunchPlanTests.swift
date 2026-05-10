import XCTest
@testable import MLXDistributedCore

final class DistributedLaunchPlanTests: XCTestCase {
    func testRingPlanBuildsPerRankHostfileWithSelfBoundToAnyAddress() throws {
        let first = UUID()
        let second = UUID()
        let plan = try DistributedRingLaunchPlan(
            ranks: [
                RingRank(rank: 1, peerID: second, host: "fe80::2%bridge0", port: 9000),
                RingRank(rank: 0, peerID: first, host: "fe80::1%bridge0", port: 9000),
            ])

        XCTAssertEqual(try plan.hosts(forRank: 0), [
            "0.0.0.0:9000",
            "[fe80::2%bridge0]:9000",
        ])
        XCTAssertEqual(try plan.environment(forRank: 1, hostfilePath: "/tmp/hosts.json"), [
            "MLX_HOSTFILE": "/tmp/hosts.json",
            "MLX_RANK": "1",
            "MLX_RING_VERBOSE": "1",
        ])
    }

    func testRingPlanRejectsNonContiguousRanks() {
        XCTAssertThrowsError(try DistributedRingLaunchPlan(
            ranks: [
                RingRank(rank: 0, peerID: UUID(), host: "a.local", port: 9000),
                RingRank(rank: 2, peerID: UUID(), host: "b.local", port: 9000),
            ])) { error in
                XCTAssertEqual(
                    error as? DistributedLaunchPlanError,
                    .nonContiguousRanks(expected: [0, 1], actual: [0, 2]))
            }
    }

    func testJACCLPlanBuildsRankEnvironment() throws {
        let plan = try DistributedJACCLLaunchPlan(
            deviceMatrix: [
                [nil, "rdma_en5"],
                ["rdma_en6", nil],
            ],
            coordinators: [
                0: "0.0.0.0:9001",
                1: "10.0.0.10:9001",
            ])

        XCTAssertEqual(try plan.environment(forRank: 1, devicesFilePath: "/tmp/ibv.json"), [
            "MLX_IBV_DEVICES": "/tmp/ibv.json",
            "MLX_JACCL_COORDINATOR": "10.0.0.10:9001",
            "MLX_RANK": "1",
        ])
    }

    func testJACCLPlanRejectsMissingOffDiagonalDevice() {
        XCTAssertThrowsError(try DistributedJACCLLaunchPlan(
            deviceMatrix: [
                [nil, nil],
                ["rdma_en6", nil],
            ],
            coordinators: [
                0: "0.0.0.0:9001",
                1: "10.0.0.10:9001",
            ])) { error in
                XCTAssertEqual(
                    error as? DistributedLaunchPlanError,
                    .missingJACCLDevice(sourceRank: 0, sinkRank: 1))
            }
    }

    func testJACCLPlanRejectsDiagonalDevice() {
        XCTAssertThrowsError(try DistributedJACCLLaunchPlan(
            deviceMatrix: [
                ["rdma_self", "rdma_en5"],
                ["rdma_en6", nil],
            ],
            coordinators: [
                0: "0.0.0.0:9001",
                1: "10.0.0.10:9001",
            ])) { error in
                XCTAssertEqual(
                    error as? DistributedLaunchPlanError,
                    .diagonalJACCLDevice(rank: 0))
            }
    }
}
