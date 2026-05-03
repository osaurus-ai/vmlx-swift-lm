import Foundation

/// Pluggable discovery surface. The default implementation is
/// `BonjourDiscoveryProvider`; osaurus supplies its own bridge that reuses
/// its existing `_osaurus._tcp.` advertiser/browser via this protocol.
public protocol DiscoveryProvider: Sendable {
    /// Stream of "peer set updated" events. Each emission is the **full**
    /// current peer set; consumers diff against their previous view.
    /// Stable peer identity is the `Peer.id` UUID.
    func peerStream() -> AsyncStream<[Peer]>

    /// Begin advertising this peer on the network. Idempotent — calling
    /// again replaces the advertised TXT record.
    func advertise(_ peer: Peer) async throws

    /// Stop advertising; safe to call when not advertising.
    func stopAdvertising() async
}
