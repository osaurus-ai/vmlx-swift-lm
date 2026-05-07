import Foundation

public extension Endpoint {
    var isTLSEndpoint: Bool {
        if case .tls = self { return true }
        return false
    }

    var isRDMAEndpoint: Bool {
        if case .rdma = self { return true }
        return false
    }
}

public extension ModelHashSet {
    func contains(_ model: ModelHandle) -> Bool {
        contains(hash: model.bundleHash)
    }

    func contains(hash: String) -> Bool {
        switch self {
        case .overflow:
            return true
        case .explicit(let hashes):
            return hashes.contains {
                $0.caseInsensitiveCompare(hash) == .orderedSame
            }
        }
    }
}

public extension Peer {
    var firstTLSEndpoint: Endpoint? {
        endpoints.first(where: \.isTLSEndpoint)
    }

    var firstRDMAEndpoint: Endpoint? {
        endpoints.first(where: \.isRDMAEndpoint)
    }

    func isEligibleForReplica(model: ModelHandle) -> Bool {
        capabilities.supports(.replica)
            && modelHashes.contains(model)
            && firstTLSEndpoint != nil
    }

    func isEligibleForPipelined(model: ModelHandle) -> Bool {
        capabilities.supports(.pipelined)
            && modelHashes.contains(model)
            && firstTLSEndpoint != nil
    }

    func isEligibleForWired(model: ModelHandle) -> Bool {
        capabilities.supports(.wired)
            && modelHashes.contains(model)
            && firstRDMAEndpoint != nil
    }
}
