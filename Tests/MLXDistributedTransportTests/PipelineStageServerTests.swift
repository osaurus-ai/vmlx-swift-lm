import XCTest
import NIOCore
@testable import MLXDistributedTransport

final class PipelineStageServerLifecycleTests: XCTestCase {

    func testStartReturnsBoundPortAndStopsCleanly() async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "stage.local")
        let echo = ClosureStageHandler { frame in
            AsyncStream { continuation in
                continuation.yield(frame)
                continuation.finish()
            }
        }

        let server = PipelineStageServer(
            certificateBundle: cert,
            host: "127.0.0.1",
            port: 0,
            handler: echo
        )

        let bound = try await server.start()
        XCTAssertGreaterThan(bound, 0, "OS-assigned port must be > 0")
        let inspected = await server.boundPort()
        XCTAssertEqual(inspected, bound)

        await server.stop()
        let after = await server.boundPort()
        XCTAssertNil(after, "stop clears the channel")
    }

    func testDoubleStartRejected() async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "x")
        let echo = ClosureStageHandler { frame in
            AsyncStream { c in c.yield(frame); c.finish() }
        }
        let server = PipelineStageServer(certificateBundle: cert, port: 0, handler: echo)
        _ = try await server.start()
        defer { Task { await server.stop() } }
        do {
            _ = try await server.start()
            XCTFail("expected alreadyRunning")
        } catch PipelineStageServer.ServerError.alreadyRunning {
            // expected
        }
    }

    func testCanStartAfterStop() async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "x")
        let echo = ClosureStageHandler { frame in
            AsyncStream { c in c.yield(frame); c.finish() }
        }
        let server = PipelineStageServer(certificateBundle: cert, port: 0, handler: echo)
        let port1 = try await server.start()
        await server.stop()
        let port2 = try await server.start()
        XCTAssertGreaterThan(port2, 0)
        await server.stop()
        XCTAssertNotEqual(port1, 0)
    }
}
