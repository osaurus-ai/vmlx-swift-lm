import Foundation
import MLXDistributedCore

@main
enum DistributedModelInventoryTool {
    static func main() {
        do {
            let config = try InventoryConfig.parse(CommandLine.arguments.dropFirst())
            let report = try run(config)
            printJSON(report)
        } catch InventoryConfig.ParseError.help {
            print(InventoryConfig.usage)
        } catch {
            fputs("error: \(error)\n", stderr)
            fputs(InventoryConfig.usage + "\n", stderr)
            Foundation.exit(2)
        }
    }

    private static func run(_ config: InventoryConfig) throws -> InventoryReport {
        let local = DistributedModelManifest.discoverReporting(roots: config.roots)
        var remote: RemoteInventory?
        var comparison: DistributedModelInventoryComparison?

        if let host = config.remoteHost {
            let roots = config.remoteRoots.isEmpty ? config.roots : config.remoteRoots
            let remoteScan = try SSHRemoteInventoryRunner(
                sshExecutable: config.sshExecutable,
                host: host,
                timeoutSeconds: config.timeoutSeconds
            ).scan(roots: roots)
            remote = RemoteInventory(host: host, roots: roots, scan: remoteScan)
            comparison = DistributedModelInventoryComparison(
                local: local.models,
                remote: remoteScan.models)
        }

        return InventoryReport(
            schemaVersion: 1,
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            local: LocalInventory(roots: config.roots, scan: local),
            remote: remote,
            comparison: comparison,
            notes: [
                "bundleHash is a 16-hex advertised prefix of fullBundleHash.",
                "identity_mode=identity_files_only hashes loader/tokenizer/template/media identity files, not full weight shards.",
                "replica matches are candidates until trust, reachability, and a real load/generation proof pass.",
                "pipeline and wired tensor modes must stay blocked without activation/runtime or JACCL/RDMA collective proof."
            ])
    }

    private static func printJSON<T: Encodable>(_ value: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.keyEncodingStrategy = .convertToSnakeCase
        if let data = try? encoder.encode(value),
           let text = String(data: data, encoding: .utf8) {
            print(text)
        }
    }
}

private struct InventoryConfig {
    enum ParseError: Error, CustomStringConvertible {
        case help
        case message(String)

        var description: String {
            switch self {
            case .help:
                return "help"
            case .message(let message):
                return message
            }
        }
    }

    var roots = [FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("models").path]
    var remoteHost: String?
    var remoteRoots: [String] = []
    var sshExecutable = "/usr/bin/ssh"
    var timeoutSeconds = 10

    static let usage = """
    DistributedModelInventory:
      DistributedModelInventory --root ~/models
      DistributedModelInventory --root ~/models --remote-host other-mac.local --remote-root ~/models

    Options:
      --root <path[,path]>          Local model root. Repeatable.
      --remote-host <ssh-host>      Optional SSH host for remote inventory.
      --remote-root <path[,path]>   Remote model root. Defaults to local roots.
      --ssh <path>                  SSH executable. Default: /usr/bin/ssh.
      --timeout <seconds>           SSH connect timeout. Default: 10.

    Output is JSON only. It is an inventory and candidate-matching report, not
    a claim that a model was loaded or generated successfully.
    """

    static func parse<S: Sequence>(_ arguments: S) throws -> InventoryConfig where S.Element == String {
        var config = InventoryConfig()
        var sawRoot = false
        var iterator = arguments.makeIterator()
        while let arg = iterator.next() {
            switch arg {
            case "--help", "-h":
                throw ParseError.help
            case "--root":
                if !sawRoot {
                    config.roots.removeAll()
                    sawRoot = true
                }
                config.roots.append(contentsOf: splitPaths(try next(&iterator, after: arg)))
            case "--remote-host":
                config.remoteHost = try next(&iterator, after: arg)
            case "--remote-root":
                config.remoteRoots.append(contentsOf: splitPaths(try next(&iterator, after: arg)))
            case "--ssh":
                config.sshExecutable = try next(&iterator, after: arg)
            case "--timeout":
                config.timeoutSeconds = try parseInt(try next(&iterator, after: arg), name: arg)
            default:
                throw ParseError.message("unknown argument: \(arg)")
            }
        }
        guard !config.roots.isEmpty else {
            throw ParseError.message("at least one --root is required")
        }
        guard config.timeoutSeconds > 0 else {
            throw ParseError.message("--timeout must be positive")
        }
        return config
    }

    private static func splitPaths(_ value: String) -> [String] {
        value.split(separator: ",")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
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

private struct InventoryReport: Encodable {
    let schemaVersion: Int
    let generatedAt: String
    let local: LocalInventory
    let remote: RemoteInventory?
    let comparison: DistributedModelInventoryComparison?
    let notes: [String]
}

private struct LocalInventory: Encodable {
    let roots: [String]
    let scan: DistributedModelManifestScanResult
}

private struct RemoteInventory: Encodable {
    let host: String
    let roots: [String]
    let scan: DistributedModelManifestScanResult
}

private struct SSHRemoteInventoryRunner {
    let sshExecutable: String
    let host: String
    let timeoutSeconds: Int

    func scan(roots: [String]) throws -> DistributedModelManifestScanResult {
        let command = "python3 -c \(shellQuote(Self.pythonScanner)) \(roots.map(shellQuote).joined(separator: " "))"
        let result = try runProcess(
            executable: sshExecutable,
            arguments: [
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=\(timeoutSeconds)",
                host,
                command,
            ])
        guard result.exitCode == 0 else {
            throw InventoryError.remoteInventoryFailed(
                tail(result.stderr.isEmpty ? result.stdout : result.stderr))
        }
        let data = Data(result.stdout.utf8)
        return try JSONDecoder().decode(DistributedModelManifestScanResult.self, from: data)
    }

    private static let pythonScanner = #"""
import hashlib, json, os, sys

FILES = [
    "config.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "jang_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
]

def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        data = handle.read()
    if not data:
        return {}
    return json.loads(data.decode("utf-8"))

def as_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None

def strings(value):
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str)]
    return []

def state_hints(config, model_type, architectures):
    keys = ["ssm_state_size", "state_size", "conv_kernel", "time_step_rank", "mamba_d_state"]
    if any(k in config for k in keys):
        return True
    text = " ".join([x for x in [model_type] + architectures if x]).lower()
    return "ssm" in text or "mamba" in text

def cache_class(meta):
    family = " ".join([x for x in [meta.get("modelType")] + meta.get("architectures", []) if x]).lower()
    if not meta.get("modelType") and not meta.get("architectures") and meta.get("layerCount") is None and meta.get("hiddenSize") is None:
        return "unknown"
    if meta.get("hasStateSpaceHints") or "ling" in family or "mamba" in family:
        return "hybrid_state"
    if any(x in family for x in ["vl", "vision", "omni", "audio"]):
        return "multimodal"
    return "standard_kv"

def warnings(cache):
    if cache == "standard_kv":
        return []
    if cache == "hybrid_state":
        return ["Hybrid state/cache model: request-level replica can be considered after manifest match; activation pipeline and tensor parallel modes need model-specific state handoff."]
    if cache == "multimodal":
        return ["Multimodal model: request-level replica can be considered after manifest match; media preprocessing and cache salting must be validated before pipeline or tensor modes."]
    return ["Unknown model/cache topology: do not advertise replica, pipeline, or tensor modes without a real loader proof."]

def build(path):
    files = []
    for name in FILES:
        file_path = os.path.join(path, name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        files.append({"relativePath": name, "sha256": digest})
    if not files:
        raise Exception("model identity manifest had no known files: " + path)
    config = read_json(os.path.join(path, "config.json")) or {}
    jang = read_json(os.path.join(path, "jang_config.json"))
    quant = config.get("quantization") or config.get("quantization_config") or {}
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else None
    arch = strings(config.get("architectures"))
    meta = {
        "modelType": model_type,
        "architectures": arch,
        "layerCount": as_int(config.get("num_hidden_layers") or config.get("n_layers") or config.get("num_layers")),
        "hiddenSize": as_int(config.get("hidden_size") or config.get("n_embd") or config.get("model_dim")),
        "attentionHeads": as_int(config.get("num_attention_heads") or config.get("n_head")),
        "keyValueHeads": as_int(config.get("num_key_value_heads") or config.get("n_kv_heads")),
        "quantizationBits": as_int(quant.get("bits") or quant.get("bits_per_weight")),
        "quantizationGroupSize": as_int(quant.get("group_size") or quant.get("q_group_size")),
        "hasJangConfig": jang is not None,
        "hasSafetensorsIndex": os.path.isfile(os.path.join(path, "model.safetensors.index.json")),
        "weightFormat": (jang or {}).get("weight_format") or (jang or {}).get("format") or config.get("weight_format"),
        "hasStateSpaceHints": state_hints(config, model_type, arch),
    }
    cache = cache_class(meta)
    full = hashlib.sha256("\n".join([f["relativePath"] + " " + f["sha256"] for f in files]).encode("utf-8")).hexdigest()
    return {
        "path": path,
        "displayName": os.path.basename(path),
        "fullBundleHash": full,
        "bundleHash": full[:16],
        "identityMode": "identity_files_only",
        "files": files,
        "metadata": meta,
        "cacheClass": cache,
        "compatibleModes": [] if cache == "unknown" else ["replica"],
        "compatibilityWarnings": warnings(cache),
    }

roots = sys.argv[1:] or [os.path.expanduser("~/models")]
model_paths = set()
errors = []
for root in roots:
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        errors.append({"path": root, "message": "root does not exist"})
        continue
    if os.path.isfile(os.path.join(root, "config.json")):
        model_paths.add(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in [".git", ".build"]]
        if "config.json" in filenames:
            model_paths.add(dirpath)

models = []
for path in sorted(model_paths):
    try:
        models.append(build(path))
    except Exception as exc:
        errors.append({"path": path, "message": str(exc)})
print(json.dumps({"models": models, "errors": errors}, sort_keys=True))
"""#
}

private enum InventoryError: Error, CustomStringConvertible {
    case remoteInventoryFailed(String)

    var description: String {
        switch self {
        case .remoteInventoryFailed(let message):
            return "remote inventory failed: \(message)"
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

    return ProcessResult(
        exitCode: process.terminationStatus,
        stdout: String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "",
        stderr: String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "")
}

private func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private func tail(_ text: String, maxCharacters: Int = 800) -> String {
    if text.count <= maxCharacters { return text }
    return String(text.suffix(maxCharacters))
}
