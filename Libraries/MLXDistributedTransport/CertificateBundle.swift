import Foundation
import Crypto
import X509
import SwiftASN1

/// PEM-encoded TLS materials a peer presents to others. Generated
/// fresh per peer on first run; the SHA-256 fingerprint is what gets
/// pinned via `dist.tls.fp` in the Bonjour TXT record (engine spec §10).
public struct CertificateBundle: Sendable, Equatable {
    public let certificatePEM: String
    public let privateKeyPEM: String
    /// Lowercase hex SHA-256 of the DER-encoded certificate body, matching
    /// the wire schema used in `dist.tls.fp`.
    public let fingerprintSHA256: String

    public init(certificatePEM: String, privateKeyPEM: String, fingerprintSHA256: String) {
        self.certificatePEM = certificatePEM
        self.privateKeyPEM = privateKeyPEM
        self.fingerprintSHA256 = fingerprintSHA256
    }
}

public enum CertificateAuthorityError: Error, Equatable {
    case generationFailed(String)
}

/// Generates self-signed certs for use by `PipelineStageServer`.
public enum CertificateAuthority {

    /// Generate a fresh self-signed certificate + key pair. Suitable for
    /// pinned-fingerprint TLS only — these certs are NOT chained to a
    /// trusted root and rely on TOFU pinning by the consumer.
    ///
    /// - Parameters:
    ///   - commonName: Subject common name; surfaces in TLS error messages.
    ///   - validity: How long the cert should be valid. Default 90 days.
    public static func generateSelfSigned(
        commonName: String,
        validity: TimeInterval = 90 * 24 * 60 * 60
    ) throws -> CertificateBundle {
        let swiftKey = P256.Signing.PrivateKey()
        let key = Certificate.PrivateKey(swiftKey)

        let subject = try DistinguishedName {
            CommonName(commonName)
        }
        let now = Date()
        let cert = try Certificate(
            version: .v3,
            serialNumber: .init(),
            publicKey: key.publicKey,
            notValidBefore: now.addingTimeInterval(-60),
            notValidAfter: now.addingTimeInterval(validity),
            issuer: subject,
            subject: subject,
            signatureAlgorithm: .ecdsaWithSHA256,
            extensions: try Certificate.Extensions {
                Critical(BasicConstraints.notCertificateAuthority)
                KeyUsage(digitalSignature: true, keyEncipherment: true)
                try ExtendedKeyUsage([.serverAuth, .clientAuth])
                SubjectAlternativeNames([.dnsName(commonName)])
            },
            issuerPrivateKey: key
        )

        let certPEM = try cert.serializeAsPEM().pemString
        let keyDER = swiftKey.derRepresentation
        let keyPEM = pemEncode(label: "EC PRIVATE KEY", body: keyDER)

        var serializer = DER.Serializer()
        try cert.serialize(into: &serializer)
        let derBytes = Data(serializer.serializedBytes)
        let digest = SHA256.hash(data: derBytes)
        let fingerprint = digest.map { String(format: "%02x", $0) }.joined()

        return CertificateBundle(
            certificatePEM: certPEM,
            privateKeyPEM: keyPEM,
            fingerprintSHA256: fingerprint
        )
    }

    private static func pemEncode(label: String, body: Data) -> String {
        let base64 = body.base64EncodedString()
        var out = "-----BEGIN \(label)-----\n"
        for chunk in base64.chunked(into: 64) { out += chunk + "\n" }
        out += "-----END \(label)-----\n"
        return out
    }
}

private extension String {
    func chunked(into size: Int) -> [String] {
        var pieces: [String] = []
        var idx = startIndex
        while idx < endIndex {
            let end = self.index(idx, offsetBy: size, limitedBy: endIndex) ?? endIndex
            pieces.append(String(self[idx..<end]))
            idx = end
        }
        return pieces
    }
}
