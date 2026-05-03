import Foundation
import NIOCore
import NIOPosix
import NIOSSL

/// TLS listener that accepts inbound pipeline-stage connections. One
/// instance per local rank; when peers connect, each TLS session is
/// fed activation frames via the caller-supplied `StageHandler`.
///
/// Phase 2 scope:
/// - NIOPosix bootstrap (cross-platform; we don't need
///   NetworkTransportServices' Apple-only features yet).
/// - NIOSSL with the PEM cert from `CertificateBundle`.
/// - Length-prefixed frame protocol per `ActivationFrameCodec`.
/// - One stream per connection (HTTP/2 multiplexing arrives in Phase 7).
public actor PipelineStageServer {
    public enum ServerError: Error, Equatable {
        case alreadyRunning
        case notRunning
        case bootstrapFailed(String)
    }

    private let certificateBundle: CertificateBundle
    private let host: String
    private let requestedPort: Int
    private let handler: any StageHandler

    private var channel: Channel?
    private var group: MultiThreadedEventLoopGroup?

    public init(
        certificateBundle: CertificateBundle,
        host: String = "127.0.0.1",
        port: Int = 0,
        handler: any StageHandler
    ) {
        self.certificateBundle = certificateBundle
        self.host = host
        self.requestedPort = port
        self.handler = handler
    }

    /// Start listening; returns the bound port (useful when caller
    /// passed `port: 0` to ask for an OS-assigned port).
    @discardableResult
    public func start() async throws -> Int {
        guard channel == nil else { throw ServerError.alreadyRunning }

        let cert = try NIOSSLCertificate.fromPEMBytes(
            Array(certificateBundle.certificatePEM.utf8))
        let key = try NIOSSLPrivateKey(
            bytes: Array(certificateBundle.privateKeyPEM.utf8),
            format: .pem)

        var tlsConfig = TLSConfiguration.makeServerConfiguration(
            certificateChain: cert.map { .certificate($0) },
            privateKey: .privateKey(key))
        tlsConfig.minimumTLSVersion = .tlsv12

        let sslContext = try NIOSSLContext(configuration: tlsConfig)
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        self.group = group

        let frameHandler = handler

        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 32)
            .serverChannelOption(.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                let sslHandler = NIOSSLServerHandler(context: sslContext)
                return channel.pipeline.addHandler(sslHandler).flatMap {
                    let pump = StageFramePumpHandler(handler: frameHandler)
                    return channel.pipeline.addHandler(pump)
                }
            }
            .childChannelOption(.socketOption(.so_reuseaddr), value: 1)

        do {
            let bound = try await bootstrap.bind(host: host, port: requestedPort).get()
            self.channel = bound
            return bound.localAddress?.port ?? requestedPort
        } catch {
            try? await group.shutdownGracefully()
            self.group = nil
            throw ServerError.bootstrapFailed(String(describing: error))
        }
    }

    public func stop() async {
        if let channel {
            try? await channel.close().get()
            self.channel = nil
        }
        if let group {
            try? await group.shutdownGracefully()
            self.group = nil
        }
    }

    /// Test hook — actual bound port (post-bind) for `port: 0` setups.
    public func boundPort() -> Int? {
        channel?.localAddress?.port
    }
}

// MARK: - Per-connection frame pump

/// Reads bytes off the channel, decodes ActivationFrames, hands each one
/// to the handler, and writes the handler's reply stream back. Channel
/// handlers are not Sendable in current swift-nio; we use @preconcurrency
/// + an internal lock to keep state access scoped to the event loop.
final class StageFramePumpHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer
    typealias OutboundOut = ByteBuffer

    private let handler: any StageHandler
    private var buffer: ByteBuffer?

    init(handler: any StageHandler) {
        self.handler = handler
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
        drainFrames(context: context)
    }

    func channelInactive(context: ChannelHandlerContext) {
        buffer = nil
    }

    private func drainFrames(context: ChannelHandlerContext) {
        guard buffer != nil else { return }
        while true {
            do {
                let frame = try ActivationFrameCodec.decode(&buffer!)
                let handler = self.handler
                let eventLoop = context.eventLoop
                let channel = context.channel
                Task {
                    let stream = handler.handle(frame)
                    for await reply in stream {
                        let bytes = ActivationFrameCodec.encode(reply)
                        try? await eventLoop.submit {
                            channel.writeAndFlush(bytes, promise: nil)
                        }.get()
                    }
                }
            } catch ActivationFrameCodecError.bufferTooShort, ActivationFrameCodecError.payloadTruncated {
                // Wait for more bytes.
                return
            } catch {
                // Malformed frame — close the connection rather than
                // try to resync. Silent drop in Phase 2; Phase 7 wires
                // this to a structured event.
                context.close(promise: nil)
                return
            }
        }
    }

}
