# Osaurus Distributed Integration Spec

Status date: 2026-05-08.

This is the tracked host-app spec for wiring the distributed inference
foundations into Osaurus. The engine owns placement and data-plane safety;
Osaurus owns discovery, trust, peer UI, routing policy, and user-visible
failure states.

## Goals

- One peer model for LAN, Thunderbolt Bridge, tailnet/manual hosts, and future
  authenticated relay.
- Start with replica fan-out because it proves discovery, trust, model hash
  matching, routing, cancellation, and UI without tensor collectives.
- Add TLS pipeline parallel after typed activation envelopes and stage/cache
  contracts are proven.
- Keep wired TP/RDMA behind hard runtime gates. A cable is not enough.
- Allow a trusted remote machine with spare memory to serve replica requests
  over a tailnet or relay without pretending that path supports collectives.

## Topology Classes

| Topology | Discovery path | First mode | TP/RDMA allowed? | UI label |
| --- | --- | --- | --- | --- |
| Local | Existing local runtime | Local | No | Local |
| LAN / Wi-Fi | Bonjour `_osaurus._tcp.` TXT | Replica | No | LAN |
| Thunderbolt Bridge | Bonjour/manual endpoint on `bridge0` | Replica, then TLS-PP | Only after wired gates | Wired |
| Tailnet | Tailscale/manual endpoint | Replica | No | Tailnet |
| Internet relay | Authenticated rendezvous/relay | Replica | No | Relay |
| Wired RDMA | Explicit JACCL/RDMA rank launch | TP+PP | Yes, after gates | RDMA |

Thunderbolt Bridge reachability proves a low-latency network interface. It
does not prove RDMA or tensor collectives.

## Peer Capability States

Osaurus should not use one "distributed ready" boolean. Track these states
separately:

| State | Meaning | UI behavior |
| --- | --- | --- |
| Discovered | Peer TXT/manual endpoint parsed, but not trusted. | Show as untrusted. |
| Trusted | TLS fingerprint is pinned and health verifies it. | Eligible for checks. |
| Replica-ready | Same model hash/template exists and chat endpoint streams. | Can route whole requests. |
| PP-candidate | Combined memory can fit the model and the engine has a stage plan. | Offer distributed mode. |
| Wired-blocked | Thunderbolt reachable, but JACCL/RDMA/collective gate is missing. | Show exact blocker. |
| Relay-ready | Authenticated outside-LAN peer with same model hash. | Can route with latency warning. |

## TXT Keys

Extend the existing `_osaurus._tcp.` record. `TXTSchema` in
`MLXDistributedCore` is the source of truth for validation.

| Key | Encoding | Required for |
| --- | --- | --- |
| `dist.v` | decimal uint8 | All distributed records |
| `dist.peer.id` | UUID | All distributed records |
| `dist.modes` | csv of `replica`, `pp`, `tp` | All distributed records |
| `dist.tls.port` | decimal uint16 | Replica and PP |
| `dist.tls.fp` | 64-char lowercase SHA-256 hex | Replica and PP |
| `dist.models` | csv of 16-64 char hex hashes or `*` | Model admission |
| `dist.mem.free` | MiB | Placement scoring |
| `dist.coord` | `0` or `1` | Coordinator hints |
| `dist.rdma.gid` | hex GID | Wired TP only |
| `dist.rdma.devs` | csv device names | Wired TP only |

False-positive rule: never advertise `tp` because `bridge0` is reachable.
Advertise `tp` only after the wired gates pass.

## Wired Gates

All of these must be true before Osaurus shows RDMA-ready or passes
`Mode.wired` to the engine:

1. `JACCL.isAvailable()` returns true in the running package build.
2. RDMA devices are configured for the rank processes.
3. A two-rank collective smoke succeeds.
4. The selected model family has an explicit TP-safe sharding plan.
5. The cache planner can preserve the model family's state contract.

Current two-M5 bring-up status: Thunderbolt Bridge network reachability is
proven, a fingerprint-pinned TLS peer smoke passes over `bridge0`, and a
request-level replica smoke has generated on the peer with
`Laguna-XS.2-JANGTQ` after matching local/remote model manifests. Both systems
can load `librdma.dylib`, but the active package build still reports no JACCL
backend. Wired TP/RDMA remains blocked.

## Inventory CLI Contract

`DistributedModelInventory` is the current engine-side source for model-list
and local/remote comparison data. It is intentionally independent from the
Osaurus app and can scan a peer over SSH during bring-up:

```sh
.build/release/DistributedModelInventory \
  --root ~/models \
  --remote-host <ssh-host> \
  --remote-root ~/models
```

The CLI emits snake-case JSON with:

| Field | Meaning |
| --- | --- |
| `schema_version` | JSON contract version. Current value: `1`. |
| `local.scan.models[]` | Local `DistributedModelManifest` rows. |
| `remote.scan.models[]` | Optional peer rows gathered over SSH. |
| `comparison.replica_matches[]` | Same identity hash on both machines. Candidate only. |
| `comparison.name_hash_mismatches[]` | Same display name but different identity hash. Do not route. |
| `comparison.local_only[]` / `remote_only[]` | Bundles present on only one side. |
| `scan.errors[]` | Per-root or per-model scan failures. A bad model must not hide good models. |

Manifest fields to preserve in Osaurus:

| Field | Meaning |
| --- | --- |
| `full_bundle_hash` | 64-hex SHA-256 over identity-file hash lines. |
| `bundle_hash` | 16-hex advertised prefix used by `dist.models`. |
| `identity_mode` | Currently `identity_files_only`; this is not a full weight-shard proof. |
| `files[]` | Loader/tokenizer/template/media/JANG/generation/safetensors-index identity files and SHA-256 hashes. |
| `cache_class` | `standard_kv`, `hybrid_state`, `multimodal`, or `unknown`. |
| `compatible_modes` | Candidate wire modes. Unknown configs emit no modes. |
| `compatibility_warnings[]` | Reasons Osaurus should block pipeline, TP, or cache sharing. |

Replica routing still needs these gates after an inventory match:

1. Peer trust pin is valid.
2. Health endpoint or smoke path proves reachability.
3. Selected model loads or has a current load/generation proof on the peer.
4. The requested route is request-level replica, not cache sharing.

Cross-peer cache reuse remains blocked unless model, tokenizer, template,
tool/reasoning mode, media salt, and family-specific state all match. An
`identity_files_only` inventory match is necessary but not sufficient for cache
admission.

## Remote Relay Rules

Relay is replica-first:

1. Remote Osaurus authenticates with the relay.
2. Local Osaurus pins/verifies the peer TLS identity.
3. Requests route only when model bundle hash and tokenizer/template hash
   match exactly.
4. Responses stream back as normal OpenAI-compatible SSE.
5. Prefix/cache reuse remains blocked unless the engine proves exact cache
   identity, including family-specific recurrent/media/hybrid state.

The engine wire payload can carry cache identity now, but tokenizer/template
hashes are optional in the current prompt/token TLS path. Osaurus must treat a
missing tokenizer/template/family/media identity as "not cache-share eligible".
It is not a wildcard and must not admit cross-peer prefix reuse.

Relay and tailnet peers must never advertise wired TP/RDMA. Later relay PP is
allowed only for explicit activation envelopes after latency, cancellation,
and failure recovery are measured.

## Osaurus UI Requirements

The Distributed Compute panel should include:

- Master toggle, off by default.
- Transport selector: Auto, Local only, Replica only, Network PP, Wired only.
- Peer table with name, link type, trust state, modes, models, free memory,
  latency, and last health error.
- Clear badges for LAN, Wired, Tailnet, Relay, RDMA-ready, and Wired-blocked.
- Trust manager for TOFU pins and revocation.
- Manual endpoint entry for tailnet/headless/relay cases.
- Per-request "served by" and transport events in chat/debug UI.

## Test Matrix

| Test | Required signal |
| --- | --- |
| Local fallback | Distributed off still uses the existing local engine path. |
| TXT validation | Invalid TLS fingerprint or missing TLS endpoint is rejected. |
| Trust pin | Changed TLS fingerprint blocks routing. |
| TLS peer smoke | Loopback and two-host `DistributedPeerSmoke` return encrypted transport and pinned identity. |
| Typed wire payload | Prefill/token/completion and activation-forward payloads round-trip with request ID, layer range, dtype, shape, and cache identity. |
| Typed stage adapter | Malformed frames become `.error`; valid prefill/activation frames call an injected stage runtime. |
| Replica smoke | Same-model peer receives one request through `ClusterSession(mode: .replica)` and returns real remote model text. |
| LAN replica | Osaurus chat endpoint receives one streamed request and preserves SSE framing. |
| Thunderbolt network | `bridge0` peer is reachable and preferred over Wi-Fi. |
| Wired false-positive guard | Thunderbolt reachable but JACCL unavailable shows wired-blocked. |
| Tailnet/manual replica | Manual endpoint can be trusted and used for replica. |
| Relay replica | Auth, trust, same model hash, and latency state are all visible. |
| Cache admission | Cross-peer prefix hit is blocked on model/template/media/state mismatch. |
| Mid-stream failure | Fallback happens only when the same model is also local. |

## Implementation Order

1. Add Osaurus peer manager and peer panel using `MLXDistributedCore`.
2. Extend Bonjour TXT with validated `dist.*` fields.
3. Add trusted health/model-list endpoints using the `DistributedModelInventory`
   JSON shape as the initial schema.
4. Route replica requests through the existing chat completion endpoint.
5. Add network/TB preference logic for replica transport selection.
6. Add the first MLX-backed stage runtime behind `WireStageRuntime`.
7. Wire JACCL/RDMA only after the engine-side wired gates pass.
