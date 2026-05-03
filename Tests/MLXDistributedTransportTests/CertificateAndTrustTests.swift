import XCTest
import Crypto
import MLXDistributedCore
@testable import MLXDistributedTransport

final class CertificateAuthorityTests: XCTestCase {
    func testGeneratesPEMWithStableFingerprint() throws {
        let bundle = try CertificateAuthority.generateSelfSigned(commonName: "test.local")
        XCTAssertTrue(bundle.certificatePEM.contains("-----BEGIN CERTIFICATE-----"))
        XCTAssertTrue(bundle.certificatePEM.contains("-----END CERTIFICATE-----"))
        XCTAssertTrue(bundle.privateKeyPEM.contains("PRIVATE KEY"))
        XCTAssertEqual(bundle.fingerprintSHA256.count, 64)
        XCTAssertEqual(bundle.fingerprintSHA256,
                       bundle.fingerprintSHA256.lowercased(),
                       "fingerprint must be lowercase hex per spec §10")
    }

    func testTwoGenerationsProduceDifferentCerts() throws {
        let a = try CertificateAuthority.generateSelfSigned(commonName: "a.local")
        let b = try CertificateAuthority.generateSelfSigned(commonName: "b.local")
        XCTAssertNotEqual(a.fingerprintSHA256, b.fingerprintSHA256,
                          "fresh cert per call -> distinct fingerprints")
    }
}

final class TrustVerifierTests: XCTestCase {

    private func fp(_ tag: UInt8) -> String {
        String(repeating: String(format: "%02x", tag), count: 32)
    }

    func testTOFUPinsOnFirstSight() async throws {
        let verifier = TrustVerifier(policy: .tofu)
        let id = UUID()
        let fingerprint = fp(0xab)
        let ok = try await verifier.verify(
            peerID: id,
            presentedFingerprint: fingerprint,
            advertisedFingerprint: fingerprint
        )
        XCTAssertTrue(ok)
        let pins = await verifier.currentPins()
        XCTAssertEqual(pins[id], fingerprint)
    }

    func testTOFUPinReplacementRejected() async throws {
        let verifier = TrustVerifier(policy: .tofu)
        let id = UUID()
        _ = try await verifier.verify(
            peerID: id, presentedFingerprint: fp(0x01),
            advertisedFingerprint: fp(0x01))

        do {
            _ = try await verifier.verify(
                peerID: id, presentedFingerprint: fp(0x02),
                advertisedFingerprint: fp(0x02))
            XCTFail("expected trustRejected on pin replacement")
        } catch DistributionError.trustRejected {
            // expected
        }
    }

    func testFingerprintMismatchAlwaysRejects() async throws {
        let verifier = TrustVerifier(policy: .tofu)
        let id = UUID()
        do {
            _ = try await verifier.verify(
                peerID: id,
                presentedFingerprint: fp(0xaa),
                advertisedFingerprint: fp(0xbb))
            XCTFail("expected trustRejected on advertise/present mismatch")
        } catch DistributionError.trustRejected {
            // expected
        }
    }

    func testAllowlistAccepts() async throws {
        let allowed = fp(0xcc)
        let verifier = TrustVerifier(policy: .allowlist([allowed]))
        let ok = try await verifier.verify(
            peerID: UUID(),
            presentedFingerprint: allowed,
            advertisedFingerprint: allowed)
        XCTAssertTrue(ok)
    }

    func testAllowlistRejectsUnknown() async throws {
        let verifier = TrustVerifier(policy: .allowlist([fp(0xcc)]))
        do {
            _ = try await verifier.verify(
                peerID: UUID(),
                presentedFingerprint: fp(0xdd),
                advertisedFingerprint: nil)
            XCTFail("expected trustRejected")
        } catch DistributionError.trustRejected {
            // expected
        }
    }

    func testDenyAllAlwaysRejects() async throws {
        let verifier = TrustVerifier(policy: .denyAll)
        do {
            _ = try await verifier.verify(
                peerID: UUID(),
                presentedFingerprint: fp(0x99),
                advertisedFingerprint: fp(0x99))
            XCTFail("expected trustRejected")
        } catch DistributionError.trustRejected {
            // expected
        }
    }

    func testFingerprintHelperMatchesCertificateAuthority() throws {
        let bundle = try CertificateAuthority.generateSelfSigned(commonName: "x")
        // Convert PEM back to DER for the helper
        let stripped = bundle.certificatePEM
            .replacingOccurrences(of: "-----BEGIN CERTIFICATE-----", with: "")
            .replacingOccurrences(of: "-----END CERTIFICATE-----", with: "")
            .replacingOccurrences(of: "\n", with: "")
        let der = Data(base64Encoded: stripped)!
        let computed = TrustVerifier.fingerprint(of: der)
        XCTAssertEqual(computed, bundle.fingerprintSHA256)
    }
}
