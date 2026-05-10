import Foundation

public enum DistributedLaunchPlanError: Error, Equatable, CustomStringConvertible {
    case emptyRanks
    case singleRankRequiresLocalPath
    case duplicateRank(Int)
    case duplicatePeer(UUID)
    case nonContiguousRanks(expected: [Int], actual: [Int])
    case invalidHost(rank: Int)
    case invalidPort(rank: Int)
    case unknownRank(Int)
    case invalidJACCLMatrix
    case diagonalJACCLDevice(rank: Int)
    case missingJACCLDevice(sourceRank: Int, sinkRank: Int)
    case missingCoordinator(rank: Int)

    public var description: String {
        switch self {
        case .emptyRanks:
            return "distributed launch plan has no ranks"
        case .singleRankRequiresLocalPath:
            return "distributed launch plan requires at least two ranks"
        case .duplicateRank(let rank):
            return "duplicate rank \(rank)"
        case .duplicatePeer(let id):
            return "duplicate peer \(id.uuidString)"
        case .nonContiguousRanks(let expected, let actual):
            return "ranks must be contiguous; expected \(expected), got \(actual)"
        case .invalidHost(let rank):
            return "rank \(rank) has an empty host"
        case .invalidPort(let rank):
            return "rank \(rank) has an invalid port"
        case .unknownRank(let rank):
            return "unknown rank \(rank)"
        case .invalidJACCLMatrix:
            return "JACCL device matrix must be square and contain at least two ranks"
        case .diagonalJACCLDevice(let rank):
            return "JACCL device matrix diagonal for rank \(rank) must be nil"
        case .missingJACCLDevice(let sourceRank, let sinkRank):
            return "missing JACCL device from rank \(sourceRank) to rank \(sinkRank)"
        case .missingCoordinator(let rank):
            return "missing JACCL coordinator for rank \(rank)"
        }
    }
}

public struct RingRank: Codable, Sendable, Equatable, Hashable {
    public let rank: Int
    public let peerID: UUID
    public let host: String
    public let port: UInt16

    public init(rank: Int, peerID: UUID, host: String, port: UInt16) {
        self.rank = rank
        self.peerID = peerID
        self.host = host
        self.port = port
    }
}

public struct DistributedRingLaunchPlan: Codable, Sendable, Equatable {
    public let ranks: [RingRank]

    public init(ranks: [RingRank]) throws {
        try Self.validate(ranks)
        self.ranks = ranks.sorted { $0.rank < $1.rank }
    }

    public var worldSize: Int { ranks.count }

    public func hosts(forRank rank: Int) throws -> [String] {
        guard ranks.contains(where: { $0.rank == rank }) else {
            throw DistributedLaunchPlanError.unknownRank(rank)
        }
        return ranks.map { item in
            let host = item.rank == rank ? "0.0.0.0" : Self.hostfileHost(item.host)
            return "\(host):\(item.port)"
        }
    }

    public func environment(
        forRank rank: Int,
        hostfilePath: String
    ) throws -> [String: String] {
        guard ranks.contains(where: { $0.rank == rank }) else {
            throw DistributedLaunchPlanError.unknownRank(rank)
        }
        return [
            "MLX_HOSTFILE": hostfilePath,
            "MLX_RANK": String(rank),
            "MLX_RING_VERBOSE": "1",
        ]
    }

    private static func validate(_ ranks: [RingRank]) throws {
        guard !ranks.isEmpty else {
            throw DistributedLaunchPlanError.emptyRanks
        }
        guard ranks.count > 1 else {
            throw DistributedLaunchPlanError.singleRankRequiresLocalPath
        }

        var seenRanks = Set<Int>()
        var seenPeers = Set<UUID>()
        for item in ranks {
            guard !item.host.isEmpty else {
                throw DistributedLaunchPlanError.invalidHost(rank: item.rank)
            }
            guard item.port > 0 else {
                throw DistributedLaunchPlanError.invalidPort(rank: item.rank)
            }
            guard seenRanks.insert(item.rank).inserted else {
                throw DistributedLaunchPlanError.duplicateRank(item.rank)
            }
            guard seenPeers.insert(item.peerID).inserted else {
                throw DistributedLaunchPlanError.duplicatePeer(item.peerID)
            }
        }

        let actual = ranks.map(\.rank).sorted()
        let expected = Array(0..<ranks.count)
        guard actual == expected else {
            throw DistributedLaunchPlanError.nonContiguousRanks(
                expected: expected,
                actual: actual)
        }
    }

    private static func hostfileHost(_ host: String) -> String {
        guard host.contains(":"),
              !host.hasPrefix("["),
              !host.hasSuffix("]") else {
            return host
        }
        return "[\(host)]"
    }
}

public struct DistributedJACCLLaunchPlan: Codable, Sendable, Equatable {
    public let deviceMatrix: [[String?]]
    public let coordinators: [Int: String]

    public init(
        deviceMatrix: [[String?]],
        coordinators: [Int: String]
    ) throws {
        try Self.validate(deviceMatrix: deviceMatrix, coordinators: coordinators)
        self.deviceMatrix = deviceMatrix
        self.coordinators = coordinators
    }

    public var worldSize: Int { deviceMatrix.count }

    public func environment(
        forRank rank: Int,
        devicesFilePath: String
    ) throws -> [String: String] {
        guard (0..<worldSize).contains(rank) else {
            throw DistributedLaunchPlanError.unknownRank(rank)
        }
        guard let coordinator = coordinators[rank] else {
            throw DistributedLaunchPlanError.missingCoordinator(rank: rank)
        }
        return [
            "MLX_IBV_DEVICES": devicesFilePath,
            "MLX_JACCL_COORDINATOR": coordinator,
            "MLX_RANK": String(rank),
        ]
    }

    public func encodedDeviceMatrix() throws -> Data {
        try JSONEncoder().encode(deviceMatrix)
    }

    private static func validate(
        deviceMatrix: [[String?]],
        coordinators: [Int: String]
    ) throws {
        let count = deviceMatrix.count
        guard count > 1, deviceMatrix.allSatisfy({ $0.count == count }) else {
            throw DistributedLaunchPlanError.invalidJACCLMatrix
        }

        for source in 0..<count {
            for sink in 0..<count {
                let device = deviceMatrix[source][sink]
                if source == sink {
                    if device != nil {
                        throw DistributedLaunchPlanError.diagonalJACCLDevice(rank: source)
                    }
                } else if (device ?? "").isEmpty {
                    throw DistributedLaunchPlanError.missingJACCLDevice(
                        sourceRank: source,
                        sinkRank: sink)
                }
            }
        }

        for rank in 0..<count {
            guard let coordinator = coordinators[rank],
                  !coordinator.isEmpty else {
                throw DistributedLaunchPlanError.missingCoordinator(rank: rank)
            }
        }
    }
}
