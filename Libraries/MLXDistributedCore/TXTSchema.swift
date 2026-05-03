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
}
