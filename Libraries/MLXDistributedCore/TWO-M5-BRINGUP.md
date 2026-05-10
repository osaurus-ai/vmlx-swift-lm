# Distributed Inference: Two M5 Max Bring-Up

Status date: 2026-05-08.

This is the active bring-up plan for two symmetric M5 Max 128 GB laptops.
The goal is to make distributed inference usable by Osaurus without guessing
about transport, cache, or model-family behavior.

JACCL/RDMA backend enablement and robust discovery edge cases are tracked in
[`JACCL-RDMA-DISCOVERY-BRINGUP.md`](JACCL-RDMA-DISCOVERY-BRINGUP.md).

## Current Baseline

The reference package contains the distributed foundations:

- `MLXDistributedCore`: peer identity, `dist.*` TXT schema, discovery hooks,
  placement planning, and local fallback.
- `MLXDistributedTransport`: TLS stage client/server, activation frame codec,
  trust helpers, and a two-rank prompt/token path.
- `MLXDistributedTP`: TP groups, single-rank-safe collectives, sharding plans,
  and `TPRankWorker`.
- `MLXDistributedJACCL`: local backend probes for the MLX distributed backend.
- `DistributedProbe`: local readiness probe for the two-laptop setup.
- `DistributedModelInventory`: local/remote model discovery and compatibility
  reporting. It emits schema-versioned JSON and distinguishes candidate
  replica matches from mismatches and scan errors.
- `DistributedPeerSmoke`: TLS/fingerprint-pinned frame smoke for loopback and
  two-host Thunderbolt Bridge validation.
- `DistributedReplicaSmoke`: request-level replica smoke using
  `ClusterSession(mode: .replica)`, matching model manifests, and a real remote
  `RunBench` generation.

The current local probe result is intentionally conservative:

- `librdma.dylib` is loadable on the host.
- `JACCL.isAvailable()` is currently false in this package build.
- Thunderbolt Bridge is visible after the TB5 cable is plugged in.
- Thunderbolt Bridge IPv6 link-local connectivity works in both directions.
- TCP reachability over Thunderbolt Bridge works in both directions.
- `MLX_IBV_DEVICES` is not configured, so RDMA is not configured yet.

That means replica and TLS pipeline work can move forward, but wired TP/RDMA
must remain disabled until the backend and two-rank collective smoke are proven.

## Two-Laptop Live Baseline

The current two-M5 setup proves the cable/link layer, not RDMA collectives:

- Both laptops expose `Thunderbolt Bridge` as `bridge0`.
- The local probe now reports Thunderbolt candidates on `bridge0`.
- IPv6 link-local ping succeeds over `bridge0` in both directions.
- SSH/TCP reachability succeeds over `bridge0` in both directions.
- `DistributedPeerSmoke` succeeds from this Mac to the other Mac over
  `fe80::3472:12ff:feed:e480%bridge0` with pinned TLS fingerprint
  `eb17ab0fb24af2c55e5fedf0beb07779fb7bce531029be70f859f5e4d4886a84`.
- `DistributedReplicaSmoke` succeeds with `Laguna-XS.2-JANGTQ` after matching
  local/remote manifests. The peer generated `distributed-ok` and the local
  session reported `placement: replicaOnPeer(...)`, `ok: true`, and
  `endReason: completed`. The current identity hash is
  `c853d16e89e9cd37`.
- `DistributedModelInventory` scanned `~/models` on both machines:
  21 local bundles, 20 remote bundles, 13 matching replica candidates, one
  same-name mismatch (`Kimi-K2.6-Small-JANGTQ`), and no scan errors.
- Both laptops can load `librdma.dylib`.
- The local package build still compiles the MLX distributed stubs
  (`no_ring.cpp`, `no_jaccl.cpp`), and `JACCL.isAvailable()` remains false.

For app code, treat the Thunderbolt Bridge as a preferred network interface
first. Do not report it as RDMA-ready until the MLX/JACCL backend reports
availability and a two-rank collective smoke passes.

## Modes

| Mode | Purpose | First usable target | Current gate |
| --- | --- | --- | --- |
| Replica fan-out | Route whole requests to the other laptop when it already has the same model. | Osaurus peer routing over existing chat endpoint. | Peer discovery, TLS trust, exact model hash/template match. |
| TLS pipeline parallel | Split layers/stages and move typed activations over TLS/NIO. | Dense Llama/Gemma-style stage split first. | Activation envelope, layer-range contract, cache identity. |
| Wired TP+PP | Tensor-parallel collectives plus pipeline stages over wired backend. | Two-rank TP smoke over TB5/RDMA/JACCL. | JACCL available, RDMA devices configured, collective smoke succeeds. |
| Tailnet / relay replica | Use a trusted remote Mac with spare memory from outside the LAN. | Remote Osaurus serving the same model over TLS. | Auth, trust pin, model hash match, latency budget, no TP advertisement. |

Replica comes first because it proves discovery, trust, routing, cancellation,
and Osaurus UI without touching MLX collectives. Pipeline parallel comes next
because it uses explicit activation frames instead of all-reduce collectives.
Wired TP comes after backend readiness is real.

## Bring-Up Commands

Build the probe:

```sh
swift build -c debug --product DistributedProbe
```

Build the TLS peer smoke:

```sh
swift build -c debug --product DistributedPeerSmoke
```

Build the replica smoke:

```sh
swift build -c release --product DistributedReplicaSmoke
```

Build the model inventory CLI:

```sh
swift build -c release --product DistributedModelInventory
```

Run a local readiness snapshot:

```sh
.build/debug/DistributedProbe
.build/debug/DistributedProbe --json
```

After plugging in the TB5 cable, re-run the probe and check for a Thunderbolt
candidate such as `bridge0` or a hardware port named `Thunderbolt Bridge`.
This only proves the network interface exists; it does not prove RDMA or TP.

Check bidirectional Thunderbolt Bridge reachability before debugging Osaurus:

```sh
ping6 -c 2 -I bridge0 <remote-link-local-ipv6>%bridge0
ssh <remote-link-local-ipv6>%bridge0 hostname
```

Use link-local IPv6 with an interface scope for the first bring-up because
macOS may not assign symmetric IPv4 addresses to both sides automatically.
Static IPv4 can be added later for convenience, but it is not required for
the initial TLS/replica control plane.

Run a local loopback transport smoke:

```sh
.build/debug/DistributedPeerSmoke --self-test
```

Run a two-host Thunderbolt Bridge smoke without modifying the remote repo:

```sh
scp .build/debug/DistributedPeerSmoke \
  '[<remote-link-local-ipv6>%bridge0]:/tmp/DistributedPeerSmoke'

ssh '<remote-link-local-ipv6>%bridge0' \
  '/tmp/DistributedPeerSmoke --listen --host "::" --port 7901 \
   --advertise-host "<remote-link-local-ipv6>%bridge0" \
   --interface bridge0 --duration 45'

.build/debug/DistributedPeerSmoke \
  --connect '<remote-link-local-ipv6>%bridge0' \
  --port 7901 \
  --fingerprint <listener-fingerprint-sha256> \
  --interface bridge0
```

Passing output must show `encryptedTransport: true`,
`peerIdentityPinned: true`, `linkClass: thunderboltNetwork`, and
`rdmaReady: false` until JACCL and collectives are actually available.

Run a request-level replica smoke with a model present on both machines:

```sh
swift build -c release --product RunBench
scp .build/release/RunBench <ssh-host>:/tmp/vmlx-runbench-smoke
scp .build/arm64-apple-macosx/release/default.metallib \
    .build/arm64-apple-macosx/release/mlx.metallib \
    <ssh-host>:/tmp/

.build/release/DistributedReplicaSmoke \
  --host <ssh-host> \
  --local-model ~/models/JANGQ/Laguna-XS.2-JANGTQ \
  --remote-model ~/models/JANGQ/Laguna-XS.2-JANGTQ \
  --remote-runbench /tmp/vmlx-runbench-smoke \
  --prompt 'Answer with exactly this text: distributed-ok' \
  --max-tokens 8
```

The 2026-05-08 proof used the matching manifest hash
`c853d16e89e9cd37` for `Laguna-XS.2-JANGTQ`; local and remote loader,
tokenizer, template/media sidecar, JANG config, generation config, and
safetensors-index identity files matched. This is request-level replica
inference over SSH/RunBench for bring-up. The final Osaurus path should
replace that injected transport with the app's trusted streaming chat endpoint.

Run the inventory over both laptops:

```sh
.build/release/DistributedModelInventory \
  --root ~/models \
  --remote-host <ssh-host> \
  --remote-root ~/models
```

The inventory hash mode is `identity_files_only`. It is a fast routing
candidate gate, not a full weight-shard integrity proof and not a load proof.

When Osaurus has a real TLS listener and certificate fingerprint, preview the
Bonjour TXT payload without advertising a false peer:

```sh
.build/debug/DistributedProbe \
  --modes replica,pp \
  --tls-port 7901 \
  --tls-fingerprint <64-hex-sha256> \
  --models <model-bundle-hash>
```

Do not include `tp` until all wired gates are true:

```sh
.build/debug/DistributedProbe \
  --modes tp \
  --tls-port 7901 \
  --tls-fingerprint <64-hex-sha256> \
  --rdma-gid <gid> \
  --rdma-devices <device[,device]> \
  --models <model-bundle-hash>
```

## False-Positive Rules

- `.auto` must remain local until peer health, failure, and replan behavior are
  wired.
- A peer must not advertise replica or pipeline capability without a TLS
  endpoint and a valid 64-hex SHA-256 certificate fingerprint.
- A peer must not advertise wired TP merely because a TB5 cable is connected.
  Wired TP requires JACCL availability, RDMA device configuration, and a
  two-rank collective smoke.
- A peer reached through a tailnet or internet relay must never advertise
  wired TP. Those paths are for TLS replica fan-out and, later, high-latency
  explicit pipeline modes only.
- Cross-peer cache reuse is blocked unless model hash, tokenizer/template hash,
  tool/reasoning mode, media salt, and family-specific cache state match.
- Ling/Bailing recurrent state, ZAYA CCA state, DSV4 CSA/HSA/SWA state, and
  VL media state are not plain KV and must keep their own cache contracts.

## Progress

| Slice | Status | Notes |
| --- | --- | --- |
| Base package products | Started | `DistributedProbe` product added. `MLXDistributedJACCL` now directly links the distributed C shim. |
| TXT schema hardening | Started | Encode/decode reject invalid TLS fingerprints, missing TLS endpoints, empty model lists, and non-hex model hashes. |
| Local readiness probe | Started | Reports JACCL, `librdma`, RDMA env, interfaces, Thunderbolt candidates, and TXT preview. |
| Model inventory CLI | Started | Local/remote SSH inventory passes on both M5 laptops; reports 13 matching replica candidates and one same-name mismatch. |
| TB5 network reachability | Passing locally | `bridge0` is active; bidirectional IPv6 link-local ping and TCP reachability are proven. |
| TLS peer smoke | Passing locally and two-host | Loopback and Thunderbolt Bridge handshakes pass with pinned TLS identity; client mode no longer advertises fake TXT/listener state. |
| Focused core tests | Passing locally | `ClusterSessionTests` and `TXTSchema*` pass under the full Xcode toolchain. |
| Focused transport tests | Passing locally | `PipelineRoundTripTests`, `CertificateAuthorityTests`, and `TrustVerifierTests` pass, including wrong-fingerprint rejection. |
| Typed wire payloads | Started | Prefill, token stream, completion, cache identity, layer ranges, tensor descriptors, and activation-forward payloads round-trip under focused tests. `WireStageHandler` now adapts typed frames to an injected stage runtime. Current live path still sends prompt/token work. |
| JACCL availability tests | Passing with backend skip | `librdma` loads, but this package build still reports no JACCL backend, so multi-rank JACCL readiness remains blocked. |
| Osaurus control plane | Not started here | Add `dist.*` TXT fields, peer trust/status, and peer list UI in Osaurus. |
| Replica routing | First proof passing | `DistributedReplicaSmoke` proves `ClusterSession(mode: .replica)` can select a matching peer and stream real remote Laguna output back through an injected transport. Osaurus still needs its production chat-endpoint `ReplicaTransport`. |
| Tailnet / relay routing | Design update needed | Same trust/model gates as replica; add explicit latency and privacy UI. |
| TLS activation PP | Design present | Replace prompt/token echo with typed activation envelopes. |
| Wired TP/RDMA | Blocked | Current build reports JACCL unavailable; RDMA devices are unset. |

## Next Implementation Steps

1. Wire Osaurus to depend on `MLXDistributedCore` and `MLXDistributedTransport`.
2. Extend Osaurus Bonjour records with `dist.*` fields using `TXTSchema`.
3. Add an Osaurus peer list that shows local-only, replica-ready, PP-ready, and
   wired-blocked states separately.
4. Add interface classification in the Osaurus peer model: Wi-Fi/LAN,
   Thunderbolt Bridge, tailnet, manual host, and relay.
5. Implement `ReplicaTransport` over Osaurus' existing OpenAI-compatible
   streaming endpoint.
6. Implement the first MLX-backed `WireStageRuntime`, then replace the current prompt/token TLS path with real activation
   production/consumption using the typed wire payload contract.
7. Revisit JACCL build wiring in `mlx-swift` before any TP advertisement.
