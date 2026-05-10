import CryptoKit
import Foundation
import MLXDistributedCore

@main
enum DistributedReplicaSmoke {
    static func main() async throws {
        do {
            let config = try SmokeConfig.parse(CommandLine.arguments.dropFirst())
            let proof = try await run(config)
            printJSON(proof)
        } catch SmokeConfig.ParseError.help {
            print(SmokeConfig.usage)
        } catch {
            fputs("error: \(error)\n", stderr)
            fputs(SmokeConfig.usage + "\n", stderr)
            Foundation.exit(2)
        }
    }

    private static func run(_ config: SmokeConfig) async throws -> ReplicaProof {
        let localManifest = try Manifest(
            DistributedModelManifest.build(modelPath: config.localModelPath))
        let remoteManifest = try manifestHash(
            modelPath: config.remoteModelPath,
            runner: SSHCommandRunner(
                sshExecutable: config.sshExecutable,
                host: config.host,
                timeoutSeconds: config.timeoutSeconds))
        guard localManifest.bundleHash == remoteManifest.bundleHash else {
            throw SmokeError.manifestMismatch(
                local: localManifest.bundleHash,
                remote: remoteManifest.bundleHash)
        }

        let peerID = UUID(uuidString: config.peerID) ?? UUID()
        let peer = Peer(
            id: peerID,
            hostname: config.host,
            capabilities: PeerCapabilities(modes: [.replica]),
            endpoints: [
                .tls(
                    host: config.host,
                    port: 22,
                    fingerprintSHA256: String(repeating: "0", count: 64))
            ],
            modelHashes: .explicit([localManifest.bundleHash]),
            memFreeMiB: nil,
            willingToBeCoordinator: false)

        let transport = SSHRunBenchReplicaTransport(
            sshExecutable: config.sshExecutable,
            host: config.host,
            remoteRunBenchPath: config.remoteRunBenchPath,
            remoteModelPath: config.remoteModelPath,
            timeoutSeconds: config.timeoutSeconds)
        let session = try await ClusterSession(
            discovery: EmptyDiscovery(),
            localGenerator: FailingLocalGenerator(),
            replicaTransport: transport,
            mode: .replica,
            staticPeers: [peer])
        let model = ModelHandle(
            bundleHash: localManifest.bundleHash,
            displayName: config.displayName)
        let plan = try await session.plan(model: model)
        let request = GenerateRequest(
            model: model,
            prompt: config.prompt,
            maxTokens: config.maxTokens)

        var text = ""
        var endReason = "missing"
        for await token in session.generate(request, plan: plan) {
            switch token {
            case .text(let chunk):
                text += chunk
            case .end(let reason):
                endReason = reason.description
            }
        }

        let textMatchesExpectation = config.expectText.isEmpty || text.contains(config.expectText)

        return ReplicaProof(
            ok: endReason == "completed" && textMatchesExpectation,
            mode: "replica",
            placement: plan.placement.description,
            localModelPath: config.localModelPath,
            remoteModelPath: config.remoteModelPath,
            manifestHash: localManifest.bundleHash,
            localManifestLines: localManifest.lines,
            remoteManifestLines: remoteManifest.lines,
            host: config.host,
            prompt: config.prompt,
            expectText: config.expectText,
            maxTokens: config.maxTokens,
            text: text,
            endReason: endReason,
            transport: transport.lastReport())
    }

    private static func manifestHash(
        modelPath: String,
        runner: any CommandRunner
    ) throws -> Manifest {
        let files = DistributedModelManifest.identityFileNames
        let output = try runner.runManifest(modelPath: modelPath, files: files)
        let lines = output
            .split(separator: "\n")
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !lines.isEmpty else {
            throw SmokeError.emptyManifest(modelPath)
        }
        let data = Data(lines.joined(separator: "\n").utf8)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return Manifest(bundleHash: String(digest.prefix(16)), lines: lines)
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

private protocol CommandRunner {
    func runManifest(modelPath: String, files: [String]) throws -> String
}

private struct LocalCommandRunner: CommandRunner {
    func runManifest(modelPath: String, files: [String]) throws -> String {
        var lines: [String] = []
        for file in files {
            let url = URL(fileURLWithPath: modelPath).appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let data = try Data(contentsOf: url)
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            lines.append("\(file) \(digest)")
        }
        return lines.joined(separator: "\n") + "\n"
    }
}

private struct SSHCommandRunner: CommandRunner {
    let sshExecutable: String
    let host: String
    let timeoutSeconds: Int

    func runManifest(modelPath: String, files: [String]) throws -> String {
        let fileArgs = files.map(shellQuote).joined(separator: " ")
        let command = """
        cd \(shellQuote(modelPath)) && for f in \(fileArgs); do if test -f "$f"; then printf "%s " "$f"; shasum -a 256 "$f" | awk '{print $1}'; fi; done
        """
        let result = try runProcess(
            executable: sshExecutable,
            arguments: [
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=\(timeoutSeconds)",
                host,
                command,
            ])
        guard result.exitCode == 0 else {
            throw SmokeError.commandFailed(
                "remote manifest failed: \(result.stderr.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        return result.stdout
    }
}

private final class SSHRunBenchReplicaTransport: ReplicaTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var report = ReplicaTransportReport(
        commandExitCode: nil,
        perfLine: nil,
        textPreviewLine: nil,
        stderrTail: nil)

    let sshExecutable: String
    let host: String
    let remoteRunBenchPath: String
    let remoteModelPath: String
    let timeoutSeconds: Int

    init(
        sshExecutable: String,
        host: String,
        remoteRunBenchPath: String,
        remoteModelPath: String,
        timeoutSeconds: Int
    ) {
        self.sshExecutable = sshExecutable
        self.host = host
        self.remoteRunBenchPath = remoteRunBenchPath
        self.remoteModelPath = remoteModelPath
        self.timeoutSeconds = timeoutSeconds
    }

    func generate(_ request: GenerateRequest, peer: Peer) -> AsyncStream<Token> {
        AsyncStream { continuation in
            Task {
                do {
                    let command = remoteGenerateCommand(
                        runBenchPath: remoteRunBenchPath,
                        modelPath: remoteModelPath,
                        prompt: request.prompt,
                        maxTokens: request.maxTokens)
                    let result = try runProcess(
                        executable: sshExecutable,
                        arguments: [
                            "-o", "BatchMode=yes",
                            "-o", "ConnectTimeout=\(timeoutSeconds)",
                            host,
                            command,
                        ])
                    let parsed = parseRunBench(stdout: result.stdout)
                    setReport(ReplicaTransportReport(
                        commandExitCode: result.exitCode,
                        perfLine: parsed.perfLine,
                        textPreviewLine: parsed.textPreviewLine,
                        stderrTail: tail(result.stderr)))

                    guard result.exitCode == 0 else {
                        continuation.yield(.end(reason: .error(
                            "remote RunBench exited \(result.exitCode)")))
                        continuation.finish()
                        return
                    }
                    if let text = parsed.text {
                        continuation.yield(.text(text))
                    }
                    continuation.yield(.end(reason: .completed))
                    continuation.finish()
                } catch {
                    setReport(ReplicaTransportReport(
                        commandExitCode: nil,
                        perfLine: nil,
                        textPreviewLine: nil,
                        stderrTail: "\(error)"))
                    continuation.yield(.end(reason: .error("\(error)")))
                    continuation.finish()
                }
            }
        }
    }

    func lastReport() -> ReplicaTransportReport {
        lock.lock()
        defer { lock.unlock() }
        return report
    }

    private func setReport(_ value: ReplicaTransportReport) {
        lock.lock()
        report = value
        lock.unlock()
    }
}

private struct EmptyDiscovery: DiscoveryProvider {
    func peerStream() -> AsyncStream<[Peer]> {
        AsyncStream { continuation in
            continuation.yield([])
            continuation.finish()
        }
    }

    func advertise(_ peer: Peer) async throws {}
    func stopAdvertising() async {}
}

private struct FailingLocalGenerator: LocalGenerator {
    func generate(_ request: GenerateRequest) -> AsyncStream<Token> {
        AsyncStream { continuation in
            continuation.yield(.end(reason: .error("local generator should not be used")))
            continuation.finish()
        }
    }
}

private struct SmokeConfig {
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

    var host = "Erics-M5-Max.local"
    var localModelPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("models/JANGQ/Laguna-XS.2-JANGTQ").path
    var remoteModelPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("models/JANGQ/Laguna-XS.2-JANGTQ").path
    var remoteRunBenchPath = "/tmp/vmlx-runbench-smoke"
    var sshExecutable = "/usr/bin/ssh"
    var prompt = "Answer with exactly this text: distributed-ok"
    var expectText = "distributed-ok"
    var displayName = "Laguna-XS.2-JANGTQ"
    var peerID = "11111111-1111-4111-8111-111111111111"
    var maxTokens = 8
    var timeoutSeconds = 10

    static let usage = """
    DistributedReplicaSmoke:
      swift run DistributedReplicaSmoke --host <ssh-host> --local-model <path> --remote-model <path> --remote-runbench <path>

    Options:
      --prompt <text>
      --expect-text <text>   Text that must appear in remote output for ok=true. Empty disables this check.
      --max-tokens <n>
      --timeout <seconds>
      --ssh <path>

    This proves request-level replica routing: local ClusterSession plans a peer
    with a matching manifest hash, then ReplicaTransport runs a real remote
    RunBench generation and streams the text back.
    """

    static func parse<S: Sequence>(_ arguments: S) throws -> SmokeConfig where S.Element == String {
        var config = SmokeConfig()
        var iterator = arguments.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--help", "-h":
                throw ParseError.help
            case "--host":
                config.host = try next(&iterator, after: arg)
            case "--local-model":
                config.localModelPath = try next(&iterator, after: arg)
            case "--remote-model":
                config.remoteModelPath = try next(&iterator, after: arg)
            case "--remote-runbench":
                config.remoteRunBenchPath = try next(&iterator, after: arg)
            case "--prompt":
                config.prompt = try next(&iterator, after: arg)
            case "--expect-text":
                config.expectText = try next(&iterator, after: arg)
            case "--display-name":
                config.displayName = try next(&iterator, after: arg)
            case "--max-tokens":
                config.maxTokens = try parseInt(try next(&iterator, after: arg), name: arg)
            case "--timeout":
                config.timeoutSeconds = try parseInt(try next(&iterator, after: arg), name: arg)
            case "--ssh":
                config.sshExecutable = try next(&iterator, after: arg)
            default:
                throw ParseError.message("unknown argument: \(arg)")
            }
        }
        guard config.maxTokens > 0 else {
            throw ParseError.message("--max-tokens must be positive")
        }
        guard config.timeoutSeconds > 0 else {
            throw ParseError.message("--timeout must be positive")
        }
        return config
    }

    private static func next<I: IteratorProtocol>(
        _ iterator: inout I,
        after name: String
    ) throws -> String where I.Element == String {
        guard let value = iterator.next(), !value.isEmpty else {
            throw ParseError.message("missing value after \(name)")
        }
        return value
    }

    private static func parseInt(_ value: String, name: String) throws -> Int {
        guard let parsed = Int(value) else {
            throw ParseError.message("\(name) expects an integer")
        }
        return parsed
    }
}

private struct Manifest: Encodable {
    let bundleHash: String
    let lines: [String]

    init(bundleHash: String, lines: [String]) {
        self.bundleHash = bundleHash
        self.lines = lines
    }

    init(_ manifest: DistributedModelManifest) {
        self.bundleHash = manifest.bundleHash
        self.lines = manifest.files.map { "\($0.relativePath) \($0.sha256)" }
    }
}

private struct ReplicaProof: Encodable {
    let ok: Bool
    let mode: String
    let placement: String
    let localModelPath: String
    let remoteModelPath: String
    let manifestHash: String
    let localManifestLines: [String]
    let remoteManifestLines: [String]
    let host: String
    let prompt: String
    let expectText: String
    let maxTokens: Int
    let text: String
    let endReason: String
    let transport: ReplicaTransportReport
}

private struct ReplicaTransportReport: Encodable {
    let commandExitCode: Int32?
    let perfLine: String?
    let textPreviewLine: String?
    let stderrTail: String?
}

private enum SmokeError: Error, CustomStringConvertible {
    case commandFailed(String)
    case emptyManifest(String)
    case manifestMismatch(local: String, remote: String)

    var description: String {
        switch self {
        case .commandFailed(let message):
            return message
        case .emptyManifest(let path):
            return "manifest had no known files: \(path)"
        case .manifestMismatch(let local, let remote):
            return "local/remote manifest hash mismatch: local=\(local) remote=\(remote)"
        }
    }
}

private struct ProcessResult {
    let exitCode: Int32
    let stdout: String
    let stderr: String
}

private func runProcess(executable: String, arguments: [String]) throws -> ProcessResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments

    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr

    try process.run()
    process.waitUntilExit()

    let out = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    let err = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    return ProcessResult(exitCode: process.terminationStatus, stdout: out, stderr: err)
}

private func remoteGenerateCommand(
    runBenchPath: String,
    modelPath: String,
    prompt: String,
    maxTokens: Int
) -> String {
    let env: [(String, String)] = [
        ("BENCH_PERF", "1"),
        ("BENCH_PERF_WARMUP", "0"),
        ("BENCH_PERF_RUNS", "1"),
        ("BENCH_PERF_PATH", "batch"),
        ("BENCH_MAX_TOKENS", "\(maxTokens)"),
        ("BENCH_PERF_PROMPT", prompt),
        ("BENCH_MODEL", modelPath),
    ]
    let assignments = env.map { "\($0.0)=\(shellQuote($0.1))" }.joined(separator: " ")
    let runBench = shellQuote(runBenchPath)
    return "cd \(shellQuote(URL(fileURLWithPath: runBenchPath).deletingLastPathComponent().path)) && \(assignments) \(runBench)"
}

private func parseRunBench(stdout: String) -> (text: String?, textPreviewLine: String?, perfLine: String?) {
    let lines = stdout.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    let textPreviewLine = lines.last(where: { $0.contains("TEXT_PREVIEW") })
    let perfLine = lines.last(where: { $0.hasPrefix("PERF ") })
    let text = textPreviewLine.flatMap(extractQuotedPreview)
    return (text, textPreviewLine, perfLine)
}

private func extractQuotedPreview(_ line: String) -> String? {
    guard let first = line.firstIndex(of: "\""),
          let last = line.lastIndex(of: "\""),
          first < last else {
        return nil
    }
    let raw = String(line[line.index(after: first)..<last])
    return raw
        .replacingOccurrences(of: "\\n", with: "\n")
        .replacingOccurrences(of: "\\\"", with: "\"")
        .replacingOccurrences(of: "\\\\", with: "\\")
}

private func tail(_ text: String, maxCharacters: Int = 800) -> String {
    if text.count <= maxCharacters { return text }
    return String(text.suffix(maxCharacters))
}

private func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private extension Token.EndReason {
    var description: String {
        switch self {
        case .completed:
            return "completed"
        case .maxTokens:
            return "maxTokens"
        case .error(let message):
            return "error: \(message)"
        }
    }
}

private extension ParallelPlan.Placement {
    var description: String {
        switch self {
        case .local:
            return "local"
        case .replicaOnPeer(let id):
            return "replicaOnPeer(\(id.uuidString.lowercased()))"
        case .pipelinedOver(let ids):
            return "pipelinedOver(\(ids.map { $0.uuidString.lowercased() }.joined(separator: ",")))"
        case .wiredOver(let ids):
            return "wiredOver(\(ids.map { $0.uuidString.lowercased() }.joined(separator: ",")))"
        }
    }
}
