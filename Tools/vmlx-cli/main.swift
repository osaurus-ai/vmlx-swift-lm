import ArgumentParser
import Foundation
import VMLXServer

@main
struct VMLXCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vmlx-cli",
        abstract: "Headless MLX inference server (OpenAI-compatible).",
        subcommands: [Serve.self, List.self, Version.self],
        defaultSubcommand: Serve.self
    )
}

// MARK: - serve

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Start the HTTP inference server on the configured port."
    )

    @Option(name: .long, help: "Models directory (scanned for installed MLX models).")
    var modelDir: String = NSString(string: "~/.vmlx/models").expandingTildeInPath

    @Option(name: .long, help: "Bind host. Use 0.0.0.0 to expose on LAN.")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Bind port.")
    var port: Int = 1337

    @Flag(name: .long, inversion: .prefixedNo, help: "Trust loopback callers without API keys.")
    var trustLoopback: Bool = true

    mutating func run() async throws {
        let resolved = URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath)
        InferenceServices.register(modelDirectory: CLIModelDirectoryProvider(root: resolved))
        InferenceServices.register(modelLocator: CLIModelLocator(root: resolved))

        let server = OsaurusServer()
        let config = OsaurusServer.Config(
            host: host, port: port,
            trustLoopback: trustLoopback,
            validatorFactory: { NoOpAPIKeyValidator() },
            preHandlerFactory: nil
        )
        try await server.start(config)
        FileHandle.standardError.write(
            Data("[vmlx-cli] serving http://\(host):\(port) (models: \(resolved.path))\n".utf8)
        )

        // Park forever. Signal handling can be added later.
        try await Task.sleep(nanoseconds: .max)
    }
}

// MARK: - list

struct List: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List installed models in the configured models directory."
    )

    @Option(name: .long, help: "Models directory.")
    var modelDir: String = NSString(string: "~/.vmlx/models").expandingTildeInPath

    func run() throws {
        let resolved = URL(fileURLWithPath: (modelDir as NSString).expandingTildeInPath)
        let locator = CLIModelLocator(root: resolved)
        let names = locator.installedModelNames()
        if names.isEmpty {
            print("No models found in \(resolved.path).")
            return
        }
        for name in names { print(name) }
    }
}

// MARK: - version

struct Version: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Print vmlx-cli version."
    )

    func run() {
        print("vmlx-cli 0.1.0")
    }
}

// MARK: - CLI adapters

struct CLIModelDirectoryProvider: ModelDirectoryProvider {
    let root: URL
    func effectiveModelsDirectory() -> URL { root }
}

/// Scans `<root>/<org>/<repo>` two-level layout. Matches the Mac app's
/// on-disk model bundle layout, so locally-downloaded models from osaurus
/// surface here verbatim.
struct CLIModelLocator: ModelLocator {
    let root: URL

    func installedModelNames() -> [String] {
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return [] }
        var names: [String] = []
        for org in orgs where !org.hasPrefix(".") {
            let orgDir = root.appendingPathComponent(org)
            guard let repos = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for repo in repos where !repo.hasPrefix(".") {
                names.append("\(org)/\(repo)")
            }
        }
        return names.sorted()
    }

    func findInstalledModel(named name: String) -> (name: String, id: String)? {
        let lower = name.lowercased()
        for full in installedModelNames() {
            if full.lowercased() == lower { return (name: full, id: full) }
            // Match short repo name (e.g. "Qwen3-7B" matches "OsaurusAI/Qwen3-7B")
            if let slash = full.firstIndex(of: "/"),
                full[full.index(after: slash)...].lowercased() == lower
            {
                return (name: String(full[full.index(after: slash)...]), id: full)
            }
        }
        return nil
    }
}
