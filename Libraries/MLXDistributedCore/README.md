# MLXDistributedCore

Phase-1A skeleton for the multi-host distributed inference system specified in
[`docs/superpowers/specs/2026-05-02-distributed-inference-engine-design.md`](../../docs/superpowers/specs/2026-05-02-distributed-inference-engine-design.md).

This target intentionally has **no MLX/MLXLLM dependencies** — it defines the
public types, the `DiscoveryProvider` protocol, the canonical TXT-record codec,
a default Bonjour discovery implementation, and `ClusterSession` (single-node
path only in Phase 1A). The actual generation execution is injected via
`LocalGenerator`, so the same target can be linked into any host that brings
its own runtime: `osaurus`, `RunBench`, or third-party SDK consumers.

## What's here (Phase 1A)

| File | Responsibility |
|------|----------------|
| `Peer.swift` | `Peer`, `Endpoint`, `PeerCapabilities`, `ModelHashSet` — wire-side identity types |
| `Mode.swift` | `Mode`, `TrustPolicy`, `DistributionError` — caller-facing knobs + errors |
| `DiscoveryProvider.swift` | Pluggable discovery protocol |
| `Generation.swift` | `ModelHandle`, `GenerateRequest`, `Token`, `LocalGenerator`, `ParallelPlan` |
| `TXTSchema.swift` | Canonical codec for the `dist.*` TXT keys (engine spec §10) |
| `BonjourDiscoveryProvider.swift` | Default `DiscoveryProvider` over Foundation `NetService` |
| `ClusterSession.swift` | User-facing entry point |

## What's NOT here (later phases)

- **Replica routing** — happens inside osaurus's `ChatEngine` via the
  `OsaurusBonjourBridge: DiscoveryProvider`. See Plan 1B.
- **TLS-NIO transport for PP** — Phase 2.
- **Per-family sharding plans** — Phase 3.
- **JACCL / RDMA bindings** — Phases 4–6.
- **Failure / replan / heartbeats** — Phase 7.

## Usage (current shape)

```swift
import MLXDistributedCore

// Inject your own LocalGenerator (e.g. an MLXLLM-backed one in osaurus)
struct MyGenerator: LocalGenerator {
    func generate(_ request: GenerateRequest) -> AsyncStream<Token> { ... }
}

let session = try await ClusterSession(
    discovery: BonjourDiscoveryProvider(),
    localGenerator: MyGenerator(),
    mode: .auto
)

let plan = try await session.plan(model: handle)
for await token in session.generate(request, plan: plan) {
    // ...
}
```

Standalone consumers (RunBench, SDK) use the bundled `BonjourDiscoveryProvider`
on `_vmlx._tcp.`. Osaurus injects its own bridge to its existing
`BonjourBrowser`/`BonjourAdvertiser` on `_osaurus._tcp.` (Plan 1B) — same
TXT-record schema, different service type.

## Tests

```sh
swift test --filter "PeerTests|ModeTests|TXTSchema|ClusterSessionTests|BonjourAdvertiseTests"
```

The opt-in real-Bonjour round-trip test is gated on
`VMLX_RUN_BONJOUR_TESTS=1` to keep CI deterministic; run it locally to
validate mDNS works on your machine:

```sh
VMLX_RUN_BONJOUR_TESTS=1 swift test --filter BonjourRoundTripTests
```
