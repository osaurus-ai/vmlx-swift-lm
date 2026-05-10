import Foundation

public struct DistributedModelInventoryComparison: Sendable, Codable, Equatable {
    public let replicaMatches: [DistributedModelManifest]
    public let nameHashMismatches: [DistributedModelNameHashMismatch]
    public let localOnly: [DistributedModelManifest]
    public let remoteOnly: [DistributedModelManifest]
    public let hashNameMismatches: [DistributedModelHashNameMismatch]

    public init(local: [DistributedModelManifest], remote: [DistributedModelManifest]) {
        let remoteByHash = Dictionary(grouping: remote, by: \.bundleHash)
        let localByHash = Dictionary(grouping: local, by: \.bundleHash)
        let remoteByName = Dictionary(grouping: remote, by: \.displayName)
        let localByName = Dictionary(grouping: local, by: \.displayName)

        let localHashes = Set(localByHash.keys)
        let remoteHashes = Set(remoteByHash.keys)
        let sharedHashes = localHashes.intersection(remoteHashes)
        let localNames = Set(localByName.keys)
        let remoteNames = Set(remoteByName.keys)
        let sharedNames = localNames.intersection(remoteNames)

        self.replicaMatches = sharedHashes
            .compactMap { hash in
                localByHash[hash]?.first
            }
            .sortedByDisplayName()

        self.localOnly = local
            .filter { !remoteHashes.contains($0.bundleHash) && !remoteNames.contains($0.displayName) }
            .sortedByDisplayName()
        self.remoteOnly = remote
            .filter { !localHashes.contains($0.bundleHash) && !localNames.contains($0.displayName) }
            .sortedByDisplayName()

        self.nameHashMismatches = sharedNames
            .compactMap { name -> DistributedModelNameHashMismatch? in
                guard let localModel = localByName[name]?.first,
                      let remoteModel = remoteByName[name]?.first,
                      localModel.bundleHash != remoteModel.bundleHash
                else {
                    return nil
                }
                return DistributedModelNameHashMismatch(
                    displayName: name,
                    local: localModel,
                    remote: remoteModel)
            }
            .sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }

        self.hashNameMismatches = sharedHashes
            .compactMap { hash -> DistributedModelHashNameMismatch? in
                guard let localModel = localByHash[hash]?.first,
                      let remoteModel = remoteByHash[hash]?.first,
                      localModel.displayName != remoteModel.displayName
                else {
                    return nil
                }
                return DistributedModelHashNameMismatch(
                    bundleHash: hash,
                    local: localModel,
                    remote: remoteModel)
            }
            .sorted { $0.bundleHash < $1.bundleHash }
    }
}

public struct DistributedModelNameHashMismatch: Sendable, Codable, Equatable {
    public let displayName: String
    public let local: DistributedModelManifest
    public let remote: DistributedModelManifest
}

public struct DistributedModelHashNameMismatch: Sendable, Codable, Equatable {
    public let bundleHash: String
    public let local: DistributedModelManifest
    public let remote: DistributedModelManifest
}

private extension Array where Element == DistributedModelManifest {
    func sortedByDisplayName() -> [DistributedModelManifest] {
        sorted {
            let nameOrder = $0.displayName.localizedStandardCompare($1.displayName)
            if nameOrder != .orderedSame {
                return nameOrder == .orderedAscending
            }
            return $0.path.localizedStandardCompare($1.path) == .orderedAscending
        }
    }
}
