import XCTest
import NIOCore
@testable import MLXDistributedTransport

/// End-to-end test: PipelineStageServer (TLS listener) + PipelineStageClient
/// (TLS dialer) round-trip an ActivationFrame in a single process via
/// loopback. Proves the frame protocol + TLS handshake + reply-streaming
/// composition is healthy.
final class PipelineRoundTripTests: XCTestCase {

    /// Helper: collect up to `count` frames from a stream with timeout.
    private func collect(
        _ stream: AsyncStream<ActivationFrame>,
        count: Int,
        timeout: TimeInterval = 10
    ) async -> [ActivationFrame] {
        await withTaskGroup(of: [ActivationFrame].self) { group -> [ActivationFrame] in
            group.addTask {
                var collected: [ActivationFrame] = []
                for await f in stream {
                    collected.append(f)
                    if collected.count >= count { return collected }
                }
                return collected
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                return []
            }
            let first = await group.next() ?? []
            group.cancelAll()
            return first
        }
    }

    func testEchoOneFrame() async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "echo.local")

        let echo = ClosureStageHandler { frame in
            AsyncStream { continuation in
                continuation.yield(frame)
                continuation.finish()
            }
        }

        let server = PipelineStageServer(
            certificateBundle: cert, host: "127.0.0.1", port: 0, handler: echo)
        let port = try await server.start()
        defer { Task { await server.stop() } }

        let client = PipelineStageClient(
            host: "127.0.0.1",
            port: port,
            expectedFingerprint: cert.fingerprintSHA256
        )
        try await client.connect()
        defer { Task { await client.disconnect() } }

        var payload = ByteBufferAllocator().buffer(capacity: 5)
        payload.writeBytes([0xDE, 0xAD, 0xBE, 0xEF, 0x42])
        let outgoing = ActivationFrame(frameType: .activationsForward, payload: payload)

        let stream = await client.responses
        try await client.send(outgoing)

        let received = await collect(stream, count: 1)
        XCTAssertEqual(received.count, 1, "echo handler should reply with exactly one frame")
        XCTAssertEqual(received[0].frameType, .activationsForward)
        XCTAssertEqual(Array(received[0].payload.readableBytesView),
                       [0xDE, 0xAD, 0xBE, 0xEF, 0x42])
    }

    func testHandlerStreamsMultipleReplies() async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "multi.local")

        let multi = ClosureStageHandler { frame in
            AsyncStream { c in
                // Reply with three token frames after one prefill request.
                for i: UInt8 in 1...3 {
                    var p = ByteBufferAllocator().buffer(capacity: 1)
                    p.writeInteger(i, as: UInt8.self)
                    c.yield(ActivationFrame(frameType: .tokenStream, payload: p))
                }
                c.finish()
            }
        }

        let server = PipelineStageServer(
            certificateBundle: cert, host: "127.0.0.1", port: 0, handler: multi)
        let port = try await server.start()
        defer { Task { await server.stop() } }

        let client = PipelineStageClient(
            host: "127.0.0.1", port: port,
            expectedFingerprint: cert.fingerprintSHA256)
        try await client.connect()
        defer { Task { await client.disconnect() } }

        let stream = await client.responses
        try await client.send(ActivationFrame(
            frameType: .prefillRequest,
            payload: ByteBufferAllocator().buffer(capacity: 0)))

        let received = await collect(stream, count: 3)
        XCTAssertEqual(received.count, 3)
        for (i, f) in received.enumerated() {
            XCTAssertEqual(f.frameType, .tokenStream)
            var p = f.payload
            XCTAssertEqual(p.readInteger(as: UInt8.self), UInt8(i + 1))
        }
    }
}
