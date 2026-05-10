import XCTest
@testable import MLXDistributedCore

final class DistributedModelInventoryTests: XCTestCase {
    func testCompareFindsHashMatchesAndNameMismatches() {
        let localA = manifest(name: "A", hash: "aaaa")
        let remoteA = manifest(name: "A", hash: "aaaa")
        let localB = manifest(name: "B", hash: "bbbb-local")
        let remoteB = manifest(name: "B", hash: "bbbb-remote")
        let localOnly = manifest(name: "LocalOnly", hash: "local-only")
        let remoteOnly = manifest(name: "RemoteOnly", hash: "remote-only")

        let comparison = DistributedModelInventoryComparison(
            local: [localA, localB, localOnly],
            remote: [remoteA, remoteB, remoteOnly])

        XCTAssertEqual(comparison.replicaMatches.map(\.bundleHash), ["aaaa"])
        XCTAssertEqual(comparison.nameHashMismatches.map(\.displayName), ["B"])
        XCTAssertEqual(comparison.localOnly.map(\.displayName), ["LocalOnly"])
        XCTAssertEqual(comparison.remoteOnly.map(\.displayName), ["RemoteOnly"])
    }

    private func manifest(name: String, hash: String) -> DistributedModelManifest {
        DistributedModelManifest(
            path: "/tmp/\(name)",
            displayName: name,
            fullBundleHash: String(String(repeating: hash, count: max(1, 64 / max(1, hash.count))).prefix(64)),
            bundleHash: hash,
            identityMode: .identityFilesOnly,
            files: [],
            metadata: DistributedModelMetadata(
                modelType: nil,
                architectures: [],
                layerCount: nil,
                hiddenSize: nil,
                attentionHeads: nil,
                keyValueHeads: nil,
                quantizationBits: nil,
                quantizationGroupSize: nil,
                hasJangConfig: false,
                hasSafetensorsIndex: false,
                weightFormat: nil,
                hasStateSpaceHints: false),
            cacheClass: .standardKV,
            compatibleModes: [.replica],
            compatibilityWarnings: [])
    }
}
