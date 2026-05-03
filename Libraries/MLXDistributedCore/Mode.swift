import Foundation

/// User-facing distribution mode. Only the three TXT-advertisable modes
/// (`replica`, `pipelined`, `wired`) round-trip through the wire schema;
/// `auto` and `localOnly` are caller-side selections that never appear
/// in `dist.modes` TXT keys.
public enum Mode: String, Sendable, Hashable, CaseIterable {
    case auto
    case localOnly
    case replica
    case pipelined
    case wired

    /// CSV token used in `dist.modes` TXT keys per the engine spec §10.
    /// Returns nil for caller-side modes (auto / localOnly).
    public var rawCSV: String? {
        switch self {
        case .replica: return "replica"
        case .pipelined: return "pp"
        case .wired: return "tp"
        case .auto, .localOnly: return nil
        }
    }

    public init?(rawCSV: String) {
        switch rawCSV {
        case "replica": self = .replica
        case "pp": self = .pipelined
        case "tp": self = .wired
        default: return nil
        }
    }
}

/// How peers are trusted on first sight.
public enum TrustPolicy: Sendable, Equatable {
    /// Trust on first use — pin the cert fingerprint after the first
    /// successful handshake, reject mismatches afterwards.
    case tofu
    /// Only trust peers whose fingerprints are in this allowlist.
    case allowlist(Set<String>)
    /// Reject every peer (testing).
    case denyAll
}

public enum DistributionError: Error, Equatable {
    case notImplementedYet(Mode)
    case noEligiblePeers
    case discoveryFailed(String)
    case malformedTXT(String)
    case trustRejected(peerID: UUID, reason: String)
}
