import XCTest
import NIOCore
import MLXDistributedCore
@testable import MLXDistributedTransport

/// Drives `ClusterSession.Mode.pipelined` through `TLSPipelinedTransport`
/// against a real `PipelineStageServer`. Proves the full Phase 2 path:
/// caller → ClusterSession → transport → TLS → stage handler → tokens.
final class ClusterSessionPipelinedTests: XCTestCase {

    private actor StubDiscovery: DiscoveryProvider {
        nonisolated func peerStream() -> AsyncStream<[Peer]> {
            AsyncStream { c in c.finish() }
        }
        func advertise(_: Peer) async throws {}
        func stopAdvertising() async {}
    }

    private struct UnusedLocalGenerator: LocalGenerator {
        func generate(_ request: GenerateRequest) -> AsyncStream<Token> {
            AsyncStream { c in
                c.yield(.end(reason: .error("local should not run in pipelined")))
                c.finish()
            }
        }
    }

    func testPipelinedModeReturnsTokensFromRemoteStage() async throws {
        // 1. Spin up a server that responds to any prefillRequest with
        //    three token frames spelling "ok!".
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "stage.local")
        let handler = ClosureStageHandler { frame in
            AsyncStream { c in
                guard frame.frameType == .prefillRequest else {
                    c.finish(); return
                }
                for piece in ["o", "k", "!"] {
                    var p = ByteBufferAllocator().buffer(capacity: piece.utf8.count)
                    p.writeString(piece)
                    c.yield(ActivationFrame(frameType: .tokenStream, payload: p))
                }
                c.yield(ActivationFrame(
                    frameType: .tokensComplete,
                    payload: ByteBufferAllocator().buffer(capacity: 0)))
                c.finish()
            }
        }
        let server = PipelineStageServer(
            certificateBundle: cert, host: "127.0.0.1", port: 0, handler: handler)
        let port = try await server.start()
        defer { Task { await server.stop() } }

        // 2. Construct a Peer pointing at the running server.
        let peer = Peer(
            id: UUID(),
            hostname: "127.0.0.1",
            capabilities: PeerCapabilities(modes: [.pipelined]),
            endpoints: [.tls(
                host: "127.0.0.1",
                port: UInt16(port),
                fingerprintSHA256: cert.fingerprintSHA256
            )],
            modelHashes: .explicit(["abcd000000000099"]),
            memFreeMiB: 1, willingToBeCoordinator: false
        )

        // 3. ClusterSession with TLSPipelinedTransport + Mode.pipelined.
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: UnusedLocalGenerator(),
            pipelinedTransport: TLSPipelinedTransport(),
            mode: .pipelined,
            staticPeers: [peer]
        )

        let plan = try await session.plan(
            model: ModelHandle(bundleHash: "abcd000000000099", displayName: "test"))
        if case .pipelinedOver(let ids) = plan.placement {
            XCTAssertEqual(ids, [peer.id])
        } else {
            XCTFail("expected pipelinedOver placement, got \(plan.placement)")
        }

        // 4. Drive generate() and collect tokens.
        let req = GenerateRequest(
            model: plan.model, prompt: "anything", maxTokens: 16)
        var pieces: [String] = []
        var endReason: Token.EndReason?
        for await tok in session.generate(req, plan: plan) {
            switch tok {
            case .text(let s): pieces.append(s)
            case .end(let r): endReason = r
            }
        }
        XCTAssertEqual(pieces, ["o", "k", "!"])
        XCTAssertEqual(endReason, .completed)
    }

    func testPipelinedModeWithNoPeersThrows() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: UnusedLocalGenerator(),
            pipelinedTransport: TLSPipelinedTransport(),
            mode: .pipelined,
            staticPeers: []
        )
        do {
            _ = try await session.plan(
                model: ModelHandle(bundleHash: "x", displayName: "x"))
            XCTFail("expected noEligiblePeers")
        } catch DistributionError.noEligiblePeers {
            // expected
        }
    }

    func testPipelinedModeWithoutTransportStillThrowsNotImplementedYet() async throws {
        let session = try await ClusterSession(
            discovery: StubDiscovery(),
            localGenerator: UnusedLocalGenerator(),
            pipelinedTransport: nil,
            mode: .pipelined,
            staticPeers: []
        )
        do {
            _ = try await session.plan(
                model: ModelHandle(bundleHash: "x", displayName: "x"))
            XCTFail("expected notImplementedYet")
        } catch DistributionError.notImplementedYet(.pipelined) {
            // expected
        }
    }
}
