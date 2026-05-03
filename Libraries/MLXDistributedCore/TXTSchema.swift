import Foundation

/// DNS-SD TXT-record codec for the `dist.*` schema (engine spec §10).
///
/// Encodes to `[String: String]` rather than the raw `[String: Data]` that
/// Foundation's `NetService` consumes — call sites do the final UTF-8 →
/// `Data` step at the Foundation boundary so this codec stays pure and
/// trivially testable.
public enum TXTSchema {
    public static let schemaVersion: UInt8 = 1

    public static func encode(_ peer: Peer) throws -> [String: String] {
        var out: [String: String] = [:]

        out["dist.v"] = String(schemaVersion)
        out["dist.peer.id"] = peer.id.uuidString.lowercased()

        let modeTokens = peer.capabilities.modes.compactMap { $0.rawCSV }.sorted()
        out["dist.modes"] = modeTokens.joined(separator: ",")

        for ep in peer.endpoints {
            switch ep {
            case .tls(_, let port, let fp):
                // host travels in the Bonjour A record; only port + fp go in TXT.
                out["dist.tls.port"] = String(port)
                out["dist.tls.fp"] = fp.lowercased()
            case .rdma(let gid, let devs):
                out["dist.rdma.gid"] = gid.lowercased()
                out["dist.rdma.devs"] = devs.joined(separator: ",")
            }
        }

        switch peer.modelHashes {
        case .explicit(let hashes):
            out["dist.models"] = hashes.joined(separator: ",")
        case .overflow:
            out["dist.models"] = "*"
        }

        if let mem = peer.memFreeMiB {
            out["dist.mem.free"] = String(mem)
        }

        out["dist.coord"] = peer.willingToBeCoordinator ? "1" : "0"

        return out
    }

    public static func decode(
        _ txt: [String: String],
        hostname: String
    ) throws -> Peer {
        guard let v = txt["dist.v"], let parsed = UInt8(v), parsed == schemaVersion else {
            throw DistributionError.malformedTXT(
                "missing/unsupported dist.v (got \(txt["dist.v"] ?? "nil"))")
        }
        guard let idStr = txt["dist.peer.id"], let id = UUID(uuidString: idStr) else {
            throw DistributionError.malformedTXT("missing/invalid dist.peer.id")
        }
        guard let modesCSV = txt["dist.modes"] else {
            throw DistributionError.malformedTXT("missing dist.modes")
        }
        let modes = Set(modesCSV.split(separator: ",")
                                 .compactMap { Mode(rawCSV: String($0)) })
        guard !modes.isEmpty else {
            throw DistributionError.malformedTXT("dist.modes parsed empty")
        }
        guard let portStr = txt["dist.tls.port"], let port = UInt16(portStr) else {
            throw DistributionError.malformedTXT("missing/invalid dist.tls.port")
        }
        guard let fp = txt["dist.tls.fp"], fp.count == 64 else {
            throw DistributionError.malformedTXT("dist.tls.fp must be 64 hex chars")
        }

        var endpoints: [Endpoint] = [
            .tls(host: hostname, port: port, fingerprintSHA256: fp.lowercased())
        ]
        if let gid = txt["dist.rdma.gid"] {
            let devs = txt["dist.rdma.devs"]?
                .split(separator: ",").map(String.init) ?? []
            endpoints.append(.rdma(gid: gid.lowercased(), devices: devs))
        }

        let modelHashes: ModelHashSet
        switch txt["dist.models"] {
        case .none:
            throw DistributionError.malformedTXT("missing dist.models")
        case .some("*"):
            modelHashes = .overflow
        case .some(let csv):
            let list = csv.split(separator: ",").map(String.init)
            modelHashes = .explicit(list)
        }

        let memFree = txt["dist.mem.free"].flatMap(UInt64.init)
        let coord = txt["dist.coord"] == "1"

        return Peer(
            id: id,
            hostname: hostname,
            capabilities: PeerCapabilities(modes: modes),
            endpoints: endpoints,
            modelHashes: modelHashes,
            memFreeMiB: memFree,
            willingToBeCoordinator: coord
        )
    }
}
