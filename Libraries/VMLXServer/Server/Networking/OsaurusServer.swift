//
//  OsaurusServer.swift
//  osaurus
//
//  Actor-owned NIO server lifecycle (start / stop).
//

import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix

public typealias APIKeyValidatorFactory = @Sendable () -> any APIKeyValidating

/// Host-supplied factory for an extra `ChannelHandler` inserted ahead of
/// the engine `HTTPHandler` in the NIO pipeline. Used by the Mac app to
/// install routes that touch app-only singletons; the CLI leaves it nil.
public typealias PreHandlerFactory = @Sendable (
    _ configuration: ServerConfiguration,
    _ apiKeyValidator: any APIKeyValidating,
    _ trustLoopback: Bool
) -> ChannelHandler

public actor OsaurusServer: Sendable {
    public struct Config: Sendable {
        public var host: String
        public var port: Int
        public var trustLoopback: Bool
        public var validatorFactory: APIKeyValidatorFactory
        public var preHandlerFactory: PreHandlerFactory?

        public init(
            host: String = "127.0.0.1",
            port: Int = 1337,
            trustLoopback: Bool = true,
            validatorFactory: @escaping APIKeyValidatorFactory = { NoOpAPIKeyValidator() },
            preHandlerFactory: PreHandlerFactory? = nil
        ) {
            self.host = host
            self.port = port
            self.trustLoopback = trustLoopback
            self.validatorFactory = validatorFactory
            self.preHandlerFactory = preHandlerFactory
        }
    }

    private var group: MultiThreadedEventLoopGroup?
    private var channel: Channel?

    public init() {}

    public func start(
        _ config: Config = .init(),
        serverConfiguration: ServerConfiguration = .default
    ) async throws {
        guard group == nil, channel == nil else { return }

        let threads = ProcessInfo.processInfo.activeProcessorCount
        let group = MultiThreadedEventLoopGroup(numberOfThreads: threads)

        let validator = config.validatorFactory()
        let trustLoopback = config.trustLoopback
        let preHandlerFactory = config.preHandlerFactory

        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline().flatMap {
                    let mainHandler = HTTPHandler(
                        configuration: serverConfiguration,
                        apiKeyValidator: validator,
                        eventLoop: channel.eventLoop,
                        trustLoopback: trustLoopback
                    )
                    guard let preHandlerFactory else {
                        return channel.pipeline.addHandler(mainHandler)
                    }
                    let preHandler = preHandlerFactory(
                        serverConfiguration, validator, trustLoopback
                    )
                    return channel.pipeline.addHandler(preHandler).flatMap {
                        channel.pipeline.addHandler(mainHandler)
                    }
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.socketOption(.tcp_nodelay), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())

        let ch = try await bootstrap.bind(host: config.host, port: config.port).get()
        self.group = group
        self.channel = ch
        print("[Osaurus] OsaurusServer started on http://\(config.host):\(config.port)")
    }

    public func stop(gracefully: Bool = true) async {
        if let ch = self.channel {
            _ = try? await ch.close()
            self.channel = nil
        }
        if let g = self.group {
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                g.shutdownGracefully { _ in cont.resume() }
            }
            self.group = nil
        }
        print("[Osaurus] OsaurusServer stopped")
    }
}
