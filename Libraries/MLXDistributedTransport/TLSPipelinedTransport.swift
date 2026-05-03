import Foundation
import NIOCore
import MLXDistributedCore

/// `PipelinedTransport` implementation that talks to remote stages over
/// TLS using `PipelineStageClient`. Phase 2 scope: two-rank only — the
/// driver (this process) sends one `prefillRequest` frame containing
/// the prompt to the **first stage** in `stages`, and decodes the reply
/// `tokenStream` frames into `Token`s. Real layer-aware activation
/// streaming + multi-stage chains arrive in Phase 2-Gemma.
public actor TLSPipelinedTransport: PipelinedTransport {

    public init() {}

    public nonisolated func generate(
        _ request: GenerateRequest,
        stages: [Peer]
    ) -> AsyncStream<Token> {
        AsyncStream { continuation in
            Task {
                do {
                    try await self.runPipeline(
                        request: request,
                        stages: stages,
                        continuation: continuation)
                } catch {
                    continuation.yield(.end(reason: .error(
                        "tls pipeline failed: \(String(describing: error))")))
                    continuation.finish()
                }
            }
        }
    }

    private func runPipeline(
        request: GenerateRequest,
        stages: [Peer],
        continuation: AsyncStream<Token>.Continuation
    ) async throws {
        guard let firstStage = stages.first else {
            continuation.yield(.end(reason: .error("no stages provided")))
            continuation.finish()
            return
        }

        // Pull the TLS endpoint + advertised fingerprint from the peer.
        var tlsEndpoint: (host: String, port: UInt16, fp: String)?
        for ep in firstStage.endpoints {
            if case .tls(let host, let port, let fp) = ep {
                tlsEndpoint = (host, port, fp)
                break
            }
        }
        guard let tls = tlsEndpoint else {
            continuation.yield(.end(reason: .error(
                "first stage advertises no TLS endpoint")))
            continuation.finish()
            return
        }

        let client = PipelineStageClient(
            host: tls.host, port: Int(tls.port),
            expectedFingerprint: tls.fp)
        try await client.connect()
        defer { Task { await client.disconnect() } }

        // Encode the prompt as the prefill payload. Phase 2 wire format
        // is just UTF-8 bytes; Phase 2-Gemma replaces this with token
        // ids + sampling params.
        var promptBuf = ByteBufferAllocator().buffer(capacity: request.prompt.utf8.count)
        promptBuf.writeString(request.prompt)
        let prefill = ActivationFrame(frameType: .prefillRequest, payload: promptBuf)

        let replies = await client.responses
        try await client.send(prefill)

        for await frame in replies {
            switch frame.frameType {
            case .tokenStream:
                var p = frame.payload
                if let s = p.readString(length: p.readableBytes) {
                    continuation.yield(.text(s))
                }
            case .tokensComplete:
                continuation.yield(.end(reason: .completed))
                continuation.finish()
                return
            case .error:
                var p = frame.payload
                let msg = p.readString(length: p.readableBytes) ?? "unknown remote error"
                continuation.yield(.end(reason: .error(msg)))
                continuation.finish()
                return
            default:
                continue
            }
        }

        // Connection went away without an explicit completion marker —
        // surface that as an error rather than silent completion.
        continuation.yield(.end(reason: .error("remote stage closed without tokensComplete")))
        continuation.finish()
    }
}
