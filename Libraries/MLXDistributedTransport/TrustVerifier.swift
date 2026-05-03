import Foundation
import Crypto
import MLXDistributedCore

/// Decides whether to accept a peer's TLS certificate. Phase 2 supports
/// the three `TrustPolicy` cases defined by `MLXDistributedCore`:
/// TOFU (pin on first sight), allowlist (pre-shared), and denyAll
/// (testing). Returning a fingerprint mismatch always rejects.
public actor TrustVerifier {
    public private(set) var policy: TrustPolicy
    private var pinned: [UUID: String] = [:]  // peerID -> fingerprint

    public init(policy: TrustPolicy) {
        self.policy = policy
    }

    /// Compute the canonical fingerprint of a DER-encoded cert (lowercase
    /// hex SHA-256). Matches the `dist.tls.fp` TXT-record encoding.
    public nonisolated static func fingerprint(of derBytes: Data) -> String {
        SHA256.hash(data: derBytes).map { String(format: "%02x", $0) }.joined()
    }

    /// Decide whether a peer's presented cert is trustworthy. Records
    /// the fingerprint on first sight under TOFU.
    @discardableResult
    public func verify(
        peerID: UUID,
        presentedFingerprint: String,
        advertisedFingerprint: String?
    ) async throws -> Bool {
        let presented = presentedFingerprint.lowercased()
        let advertised = advertisedFingerprint?.lowercased()

        // Spec invariant: the cert presented in TLS must match the
        // fingerprint advertised in TXT, regardless of policy.
        if let adv = advertised, adv != presented {
            throw DistributionError.trustRejected(
                peerID: peerID,
                reason: "TLS fingerprint \(presented) does not match advertised \(adv)"
            )
        }

        switch policy {
        case .tofu:
            if let known = pinned[peerID] {
                guard known == presented else {
                    throw DistributionError.trustRejected(
                        peerID: peerID,
                        reason: "TOFU pin \(known) replaced by \(presented)"
                    )
                }
                return true
            }
            pinned[peerID] = presented
            return true

        case .allowlist(let allowed):
            guard allowed.contains(presented) else {
                throw DistributionError.trustRejected(
                    peerID: peerID,
                    reason: "fingerprint \(presented) not in allowlist"
                )
            }
            return true

        case .denyAll:
            throw DistributionError.trustRejected(
                peerID: peerID,
                reason: "TrustPolicy.denyAll rejects every peer"
            )
        }
    }

    /// Drop a TOFU pin (e.g. when the user revokes a peer in settings).
    public func unpin(_ peerID: UUID) {
        pinned.removeValue(forKey: peerID)
    }

    /// Inspection hook for tests / debug UIs.
    public func currentPins() -> [UUID: String] { pinned }
}
