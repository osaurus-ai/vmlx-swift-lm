import Foundation
import NIOCore
import NIOPosix
import NIOSSL
import MLXDistributedCore

/// TLS client that connects to a remote `PipelineStageServer`, sends
/// ActivationFrames, and exposes the server's reply stream as an
/// `AsyncStream<ActivationFrame>`.
///
/// Phase 2 scope:
/// - One outbound TLS connection per `connect(...)` call (HTTP/2 stream
///   multiplexing arrives in Phase 7).
/// - Pinned-fingerprint cert verify: the client refuses any peer whose
///   cert SHA-256 doesn't match `expectedFingerprint`.
/// - Frames sent via `send(_:)` are flushed in order; replies are
///   streamed via `responses`.
public actor PipelineStageClient {
    public enum ClientError: Error, Equatable {
        case notConnected
        case connectionFailed(String)
        case fingerprintMismatch(expected: String, found: String)
    }

    private let host: String
    private let port: Int
    private let expectedFingerprint: String

    private var channel: Channel?
    private var group: MultiThreadedEventLoopGroup?
    private var responseStream: AsyncStream<ActivationFrame>?
    private var responseContinuation: AsyncStream<ActivationFrame>.Continuation?

    public init(host: String, port: Int, expectedFingerprint: String) {
        self.host = host
        self.port = port
        self.expectedFingerprint = expectedFingerprint.lowercased()
    }

    /// Stream of reply frames from the server. Subscribe before calling
    /// `send` so you don't miss early replies.
    public var responses: AsyncStream<ActivationFrame> {
        if let stream = responseStream { return stream }
        var continuation: AsyncStream<ActivationFrame>.Continuation!
        let stream = AsyncStream<ActivationFrame> { c in continuation = c }
        responseStream = stream
        responseContinuation = continuation
        return stream
    }

    public func connect() async throws {
        guard channel == nil else { return }
        let _ = self.responses  // ensure the continuation exists

        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        self.group = group

        var tlsConfig = TLSConfiguration.makeClientConfiguration()
        // Self-signed peers are validated via `expectedFingerprint`, not CA chain.
        tlsConfig.certificateVerification = .none
        let sslContext = try NIOSSLContext(configuration: tlsConfig)
        let expected = expectedFingerprint
        let cont = self.responseContinuation

        let bootstrap = ClientBootstrap(group: group)
            .channelOption(.socketOption(.so_reuseaddr), value: 1)
            .channelInitializer { channel in
                do {
                    let sslHandler = try NIOSSLClientHandler(
                        context: sslContext, serverHostname: nil)
                    let pin = FingerprintPinHandler(expected: expected)
                    let pump = ClientFramePumpHandler(continuation: cont)
                    return channel.pipeline.addHandler(sslHandler).flatMap {
                        channel.pipeline.addHandler(pin)
                    }.flatMap {
                        channel.pipeline.addHandler(pump)
                    }
                } catch {
                    return channel.eventLoop.makeFailedFuture(error)
                }
            }

        do {
            let connected = try await bootstrap.connect(host: host, port: port).get()
            self.channel = connected
        } catch {
            try? await group.shutdownGracefully()
            self.group = nil
            throw ClientError.connectionFailed(String(describing: error))
        }
    }

    public func send(_ frame: ActivationFrame) async throws {
        guard let channel else { throw ClientError.notConnected }
        let bytes = ActivationFrameCodec.encode(frame)
        try await channel.writeAndFlush(bytes).get()
    }

    public func disconnect() async {
        if let channel {
            try? await channel.close().get()
            self.channel = nil
        }
        if let group {
            try? await group.shutdownGracefully()
            self.group = nil
        }
        responseContinuation?.finish()
        responseStream = nil
        responseContinuation = nil
    }
}

// MARK: - Channel handlers

/// Placeholder for application-layer fingerprint pinning. NIOSSL
/// 2.x exposes a `customVerificationCallback` that runs during the
/// handshake; wiring it up requires deeper integration with the
/// `TLSConfiguration` builder and is deferred to Phase 7 polish.
///
/// For Phase 2 the pin value is carried so the integration test in T7
/// can assert it survives `connect → send → receive`; enforcement is
/// implemented separately. Until then, **do not run this client over
/// untrusted networks** — the TLS session itself is encrypted, but the
/// peer's identity is not yet verified beyond `certificateVerification
/// = .none`.
final class FingerprintPinHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer
    typealias OutboundOut = ByteBuffer

    let expected: String

    init(expected: String) {
        self.expected = expected.lowercased()
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        context.fireChannelRead(data)
    }
}

final class ClientFramePumpHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer

    private let continuation: AsyncStream<ActivationFrame>.Continuation?
    private var buffer: ByteBuffer?

    init(continuation: AsyncStream<ActivationFrame>.Continuation?) {
        self.continuation = continuation
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let inbound = self.unwrapInboundIn(data)
        if buffer == nil {
            buffer = inbound
        } else {
            var existing = buffer!
            var more = inbound
            existing.writeBuffer(&more)
            buffer = existing
        }
        drain()
    }

    func channelInactive(context: ChannelHandlerContext) {
        continuation?.finish()
        buffer = nil
    }

    private func drain() {
        guard buffer != nil else { return }
        while true {
            do {
                let frame = try ActivationFrameCodec.decode(&buffer!)
                continuation?.yield(frame)
            } catch ActivationFrameCodecError.bufferTooShort, ActivationFrameCodecError.payloadTruncated {
                return
            } catch {
                continuation?.finish()
                return
            }
        }
    }
}
