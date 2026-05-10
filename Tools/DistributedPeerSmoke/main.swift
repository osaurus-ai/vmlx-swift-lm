import Foundation
import MLXDistributedCore
import MLXDistributedJACCL
import MLXDistributedTransport
import NIOCore

@main
enum DistributedPeerSmoke {
    static func main() async throws {
        do {
            let config = try SmokeConfig.parse(CommandLine.arguments.dropFirst())
            switch config.mode {
            case .listen:
                try await runListener(config)
            case .connect:
                try await runClient(config)
            case .selfTest:
                try await runSelfTest(config)
            }
        } catch SmokeConfig.ParseError.help {
            print(SmokeConfig.usage)
        } catch {
            fputs("error: \(error)\n", stderr)
            fputs(SmokeConfig.usage + "\n", stderr)
            Foundation.exit(2)
        }
    }

    private static func runListener(_ config: SmokeConfig) async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: config.commonName)
        let peerID = config.peerID ?? UUID()
        let handler = SmokeStageHandler(server: makeHandshake(
            peerID: peerID,
            hostname: config.advertiseHost,
            port: UInt16(config.port),
            fingerprint: cert.fingerprintSHA256,
            modelHashes: config.modelHashes,
            modes: config.modes,
            interfaceHint: config.interfaceHint,
            linkClass: classifyLink(host: config.advertiseHost, interfaceHint: config.interfaceHint),
            listenerActive: true
        ))

        let server = PipelineStageServer(
            certificateBundle: cert,
            host: config.host,
            port: config.port,
            handler: handler
        )
        let boundPort = try await server.start()
        handler.update(server: makeHandshake(
            peerID: peerID,
            hostname: config.advertiseHost,
            port: UInt16(boundPort),
            fingerprint: cert.fingerprintSHA256,
            modelHashes: config.modelHashes,
            modes: config.modes,
            interfaceHint: config.interfaceHint,
            linkClass: classifyLink(host: config.advertiseHost, interfaceHint: config.interfaceHint),
            listenerActive: true
        ))

        let peer = try makePeer(
            peerID: peerID,
            hostname: config.advertiseHost,
            port: UInt16(boundPort),
            fingerprint: cert.fingerprintSHA256,
            modelHashes: config.modelHashes,
            modes: config.modes
        )
        let txt = try TXTSchema.encode(peer)

        let ready = ListenerReady(
            listenHost: config.host,
            listenPort: boundPort,
            advertiseHost: config.advertiseHost,
            peerID: peerID.uuidString.lowercased(),
            fingerprintSHA256: cert.fingerprintSHA256,
            txt: txt,
            linkClass: classifyLink(host: config.advertiseHost, interfaceHint: config.interfaceHint),
            jacclAvailable: JACCL.isAvailable(),
            anyDistributedBackendAvailable: JACCL.anyBackendAvailable()
        )
        printJSON(ready)
        fflush(stdout)

        if config.durationSeconds > 0 {
            try await Task.sleep(nanoseconds: UInt64(config.durationSeconds) * 1_000_000_000)
        } else {
            while !Task.isCancelled {
                try await Task.sleep(nanoseconds: 60 * 1_000_000_000)
            }
        }
        await server.stop()
    }

    private static func runClient(_ config: SmokeConfig) async throws {
        guard let remoteHost = config.remoteHost,
              let fingerprint = config.expectedFingerprint else {
            throw SmokeConfig.ParseError.message("--connect requires --fingerprint")
        }
        let remotePort = config.remotePort ?? config.port

        let client = PipelineStageClient(
            host: remoteHost,
            port: remotePort,
            expectedFingerprint: fingerprint
        )
        try await client.connect()
        defer { Task { await client.disconnect() } }

        let local = makeHandshake(
            peerID: config.peerID ?? UUID(),
            hostname: ProcessInfo.processInfo.hostName,
            port: UInt16(config.port),
            fingerprint: fingerprint,
            modelHashes: config.modelHashes,
            modes: config.modes,
            interfaceHint: config.interfaceHint,
            linkClass: classifyLink(host: remoteHost, interfaceHint: config.interfaceHint),
            listenerActive: false
        )

        let responses = await client.responses
        let buffer = try WirePayloadCodec.encode(local)
        try await client.send(ActivationFrame(frameType: .prefillRequest, payload: buffer))

        guard let frame = await firstFrame(from: responses, timeoutSeconds: config.timeoutSeconds) else {
            throw SmokeError.timeout("no response from remote peer")
        }
        guard frame.frameType == .tokenStream else {
            throw SmokeError.protocolMismatch("expected tokenStream response, got \(frame.frameType)")
        }

        let reply = try WirePayloadCodec.decode(SmokeResponse.self, from: frame.payload)
        printJSON(reply)
    }

    private static func runSelfTest(_ config: SmokeConfig) async throws {
        let cert = try CertificateAuthority.generateSelfSigned(commonName: "self-test.local")
        let peerID = UUID()
        let initialServerHandshake = makeHandshake(
            peerID: peerID,
            hostname: "127.0.0.1",
            port: 0,
            fingerprint: cert.fingerprintSHA256,
            modelHashes: config.modelHashes,
            modes: config.modes,
            interfaceHint: "loopback",
            linkClass: .loopback,
            listenerActive: true
        )
        let handler = SmokeStageHandler(server: initialServerHandshake)
        let server = PipelineStageServer(
            certificateBundle: cert,
            host: "127.0.0.1",
            port: 0,
            handler: handler
        )
        let port = try await server.start()
        handler.update(server: makeHandshake(
            peerID: peerID,
            hostname: "127.0.0.1",
            port: UInt16(port),
            fingerprint: cert.fingerprintSHA256,
            modelHashes: config.modelHashes,
            modes: config.modes,
            interfaceHint: "loopback",
            linkClass: .loopback,
            listenerActive: true
        ))
        defer { Task { await server.stop() } }

        let clientConfig = config.withClient(
            host: "127.0.0.1",
            port: port,
            fingerprint: cert.fingerprintSHA256
        )
        try await runClient(clientConfig)
    }

    private static func makePeer(
        peerID: UUID,
        hostname: String,
        port: UInt16,
        fingerprint: String,
        modelHashes: [String],
        modes: Set<Mode>
    ) throws -> Peer {
        Peer(
            id: peerID,
            hostname: hostname,
            capabilities: PeerCapabilities(modes: modes),
            endpoints: [.tls(host: hostname, port: port, fingerprintSHA256: fingerprint)],
            modelHashes: modelHashes.isEmpty ? .overflow : .explicit(modelHashes),
            memFreeMiB: nil,
            willingToBeCoordinator: true
        )
    }

    private static func makeHandshake(
        peerID: UUID,
        hostname: String,
        port: UInt16,
        fingerprint: String,
        modelHashes: [String],
        modes: Set<Mode>,
        interfaceHint: String?,
        linkClass: LinkClass,
        listenerActive: Bool
    ) -> SmokeHandshake {
        let peer = try? makePeer(
            peerID: peerID,
            hostname: hostname,
            port: port,
            fingerprint: fingerprint,
            modelHashes: modelHashes,
            modes: modes
        )
        return SmokeHandshake(
            protocolVersion: 1,
            peerID: peerID.uuidString.lowercased(),
            hostname: hostname,
            interfaceHint: interfaceHint,
            linkClass: linkClass,
            listenerActive: listenerActive,
            tlsFingerprintSHA256: listenerActive ? fingerprint : nil,
            txt: listenerActive ? ((try? peer.map(TXTSchema.encode)) ?? [:]) : [:],
            jacclAvailable: JACCL.isAvailable(),
            anyDistributedBackendAvailable: JACCL.anyBackendAvailable()
        )
    }

    private static func firstFrame(
        from stream: AsyncStream<ActivationFrame>,
        timeoutSeconds: Int
    ) async -> ActivationFrame? {
        await withTaskGroup(of: ActivationFrame?.self) { group in
            group.addTask {
                for await frame in stream { return frame }
                return nil
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(timeoutSeconds) * 1_000_000_000)
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    private static func classifyLink(host: String, interfaceHint: String?) -> LinkClass {
        if interfaceHint == "loopback" || host == "127.0.0.1" || host == "::1" {
            return .loopback
        }
        if interfaceHint == "bridge0" || host.contains("%bridge0") {
            return .thunderboltNetwork
        }
        if interfaceHint?.hasPrefix("utun") == true {
            return .tailnet
        }
        return .network
    }

    private static func printJSON<T: Encodable>(_ value: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(value),
           let text = String(data: data, encoding: .utf8) {
            print(text)
        }
    }
}

private final class SmokeStageHandler: StageHandler, @unchecked Sendable {
    private let lock = NSLock()
    private var server: SmokeHandshake

    init(server: SmokeHandshake) {
        self.server = server
    }

    func update(server: SmokeHandshake) {
        lock.lock()
        self.server = server
        lock.unlock()
    }

    func handle(_ frame: ActivationFrame) -> AsyncStream<ActivationFrame> {
        lock.lock()
        let server = self.server
        lock.unlock()

        return AsyncStream { continuation in
            let request = try? WirePayloadCodec.decode(SmokeHandshake.self, from: frame.payload)
            let response = SmokeResponse(
                ok: request != nil,
                route: RouteReport(
                    linkClass: request?.linkClass ?? .unknown,
                    encryptedTransport: true,
                    peerIdentityPinned: true,
                    rdmaReady: false,
                    rdmaBlockedReason: server.jacclAvailable
                        ? "RDMA devices and two-rank collective smoke are not proven by this tool"
                        : "JACCL backend is unavailable in this package build"
                ),
                server: server,
                client: request,
                message: request == nil ? "request payload was not a SmokeHandshake" : "peer smoke handshake passed"
            )
            let out = (try? WirePayloadCodec.encode(response))
                ?? ByteBufferAllocator().buffer(capacity: 0)
            continuation.yield(ActivationFrame(frameType: .tokenStream, payload: out))
            continuation.finish()
        }
    }
}

private struct SmokeConfig {
    enum ModeSelection {
        case listen
        case connect
        case selfTest
    }

    enum ParseError: Error, CustomStringConvertible {
        case help
        case message(String)

        var description: String {
            switch self {
            case .help:
                return "help"
            case .message(let text):
                return text
            }
        }
    }

    var mode: ModeSelection = .selfTest
    var host = "127.0.0.1"
    var advertiseHost = ProcessInfo.processInfo.hostName
    var port = 7901
    var remoteHost: String?
    var remotePort: Int?
    var expectedFingerprint: String?
    var commonName = "vmlx-peer.local"
    var modelHashes: [String] = ["0000000000000000"]
    var modes: Set<Mode> = [.replica, .pipelined]
    var peerID: UUID?
    var interfaceHint: String?
    var timeoutSeconds = 10
    var durationSeconds = 0

    static let usage = """
    DistributedPeerSmoke:
      --self-test
      --listen [--host ::] [--port 7901] [--advertise-host <host>] [--interface bridge0] [--duration <seconds>]
      --connect <host> --port <port> --fingerprint <64-hex> [--interface bridge0]

    Common:
      --models <hash[,hash]>
      --modes replica,pp
      --peer-id <uuid>
      --timeout <seconds>

    This proves TLS frame reachability and dist.* metadata exchange. It does not prove RDMA.
    """

    static func parse<S: Sequence>(_ arguments: S) throws -> SmokeConfig where S.Element == String {
        var config = SmokeConfig()
        var iterator = arguments.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--help", "-h":
                throw ParseError.help
            case "--self-test":
                config.mode = .selfTest
            case "--listen":
                config.mode = .listen
            case "--connect":
                config.mode = .connect
                config.remoteHost = try nextValue(&iterator, after: arg)
            case "--host":
                config.host = try nextValue(&iterator, after: arg)
            case "--advertise-host":
                config.advertiseHost = try nextValue(&iterator, after: arg)
            case "--port":
                let value = try nextValue(&iterator, after: arg)
                guard let port = Int(value), (1...65535).contains(port) else {
                    throw ParseError.message("invalid --port")
                }
                config.port = port
                config.remotePort = port
            case "--fingerprint":
                config.expectedFingerprint = try nextValue(&iterator, after: arg).lowercased()
            case "--common-name":
                config.commonName = try nextValue(&iterator, after: arg)
            case "--models":
                config.modelHashes = try nextValue(&iterator, after: arg)
                    .split(separator: ",")
                    .map(String.init)
                    .filter { !$0.isEmpty }
            case "--modes":
                let modes = try nextValue(&iterator, after: arg)
                    .split(separator: ",")
                    .compactMap { Mode(rawCSV: String($0)) }
                guard !modes.isEmpty else {
                    throw ParseError.message("invalid --modes")
                }
                config.modes = Set(modes)
            case "--peer-id":
                let value = try nextValue(&iterator, after: arg)
                guard let id = UUID(uuidString: value) else {
                    throw ParseError.message("invalid --peer-id")
                }
                config.peerID = id
            case "--interface":
                config.interfaceHint = try nextValue(&iterator, after: arg)
            case "--timeout":
                let value = try nextValue(&iterator, after: arg)
                guard let timeout = Int(value), timeout > 0 else {
                    throw ParseError.message("invalid --timeout")
                }
                config.timeoutSeconds = timeout
            case "--duration":
                let value = try nextValue(&iterator, after: arg)
                guard let duration = Int(value), duration >= 0 else {
                    throw ParseError.message("invalid --duration")
                }
                config.durationSeconds = duration
            default:
                throw ParseError.message("unknown argument: \(arg)")
            }
        }
        return config
    }

    func withClient(host: String, port: Int, fingerprint: String) -> SmokeConfig {
        var copy = self
        copy.mode = .connect
        copy.remoteHost = host
        copy.remotePort = port
        copy.expectedFingerprint = fingerprint
        copy.interfaceHint = copy.interfaceHint ?? "loopback"
        return copy
    }

    private static func nextValue<I: IteratorProtocol>(
        _ iterator: inout I,
        after arg: String
    ) throws -> String where I.Element == String {
        guard let value = iterator.next() else {
            throw ParseError.message("missing value after \(arg)")
        }
        return value
    }
}

private enum LinkClass: String, Codable {
    case loopback
    case network
    case thunderboltNetwork
    case tailnet
    case unknown
}

private struct SmokeHandshake: Codable {
    let protocolVersion: Int
    let peerID: String
    let hostname: String
    let interfaceHint: String?
    let linkClass: LinkClass
    let listenerActive: Bool
    let tlsFingerprintSHA256: String?
    let txt: [String: String]
    let jacclAvailable: Bool
    let anyDistributedBackendAvailable: Bool
}

private struct RouteReport: Codable {
    let linkClass: LinkClass
    let encryptedTransport: Bool
    let peerIdentityPinned: Bool
    let rdmaReady: Bool
    let rdmaBlockedReason: String
}

private struct SmokeResponse: Codable {
    let ok: Bool
    let route: RouteReport
    let server: SmokeHandshake
    let client: SmokeHandshake?
    let message: String
}

private struct ListenerReady: Codable {
    let listenHost: String
    let listenPort: Int
    let advertiseHost: String
    let peerID: String
    let fingerprintSHA256: String
    let txt: [String: String]
    let linkClass: LinkClass
    let jacclAvailable: Bool
    let anyDistributedBackendAvailable: Bool
}

private enum SmokeError: Error, CustomStringConvertible {
    case timeout(String)
    case protocolMismatch(String)

    var description: String {
        switch self {
        case .timeout(let text), .protocolMismatch(let text):
            return text
        }
    }
}
