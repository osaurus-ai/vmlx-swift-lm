# Exo Architecture Review For vMLX Distributed Inference

Status date: 2026-05-08.

Source reviewed: `exo-explore/exo` at commit
`414132ae9cdf0f5a9a63b4ee95d0c3d465b2c448`.

This note captures what is worth borrowing from Exo and what should be changed
for the Swift/vMLX/Osaurus stack. It is an architecture review only; it does not
make runtime readiness claims beyond the local bring-up docs.

## Useful Exo Patterns

### Discovery And Control Plane

Exo uses a Rust libp2p layer with mDNS, ping, signed gossipsub topics, bootstrap
peers, and a private namespace derived from `EXO_LIBP2P_NAMESPACE` or a network
version. The useful ideas are:

- cluster namespace isolation so unrelated local clusters do not join each
  other.
- bootstrap/manual peers for networks where mDNS is unavailable.
- recurring reconnect attempts for known peers.
- typed control-plane messages rather than ad-hoc string payloads.

For vMLX, keep the control plane native Swift first. Bonjour, manual endpoints,
SSH bootstrap for developer bring-up, tailnet endpoints, peer-card import, and a
future relay should all feed one peer state machine. Avoid making libp2p a hard
dependency for the Osaurus app path unless a later transport comparison proves it
is worth the added surface area.

### Topology Graph

Exo models topology as a directed graph with explicit `SocketConnection` and
`RDMAConnection` edges. It gathers:

- network interfaces and interface class.
- static node identity: hardware model, chip, OS version, OS build.
- Thunderbolt domain UUIDs from `system_profiler SPThunderboltDataType -json`.
- `rdma_ctl status`.
- Thunderbolt Bridge enabled/disabled state.

This is the right shape. vMLX should add stronger evidence fields:

- discovery source and timestamp.
- latency and bandwidth probe results.
- TLS fingerprint and trust state.
- exact model manifest hash and cache class.
- backend build identity: real Ring/JACCL sources vs stubs.
- first blocker for each mode.

Do not infer RDMA readiness from `bridge0`. Exo's newer RDMA edge creation is
properly gated on both endpoints reporting `rdma_ctl` enabled, but vMLX must also
require a real backend build, valid `MLX_IBV_DEVICES`, strict two-rank init, and a
collective smoke before advertising TP.

### Placement

Exo placement filters candidate cycles by memory, model sharding support, tensor
divisibility constraints, RDMA edge availability, and model download progress. It
then generates concrete MLX launch material:

- Ring: per-node `MLX_HOSTFILE`, `MLX_RANK`, and `MLX_RING_VERBOSE`.
- JACCL: square `MLX_IBV_DEVICES` matrix, per-node
  `MLX_JACCL_COORDINATOR`, and `MLX_RANK`.

The launch-plan idea should be copied. In Swift, make it a typed value:

- `DistributedRingLaunchPlan`
- `DistributedJACCLLaunchPlan`
- `RankEnvironment`
- `RankFileMaterialization`

The planner should validate and serialize those plans before a runner process is
spawned. Do not let Osaurus or a UI panel assemble distributed env vars directly.

### Runner Lifecycle

Exo runs model workers as supervised subprocesses and sequences multi-rank
startup through explicit states: create runner, download model, initialize
distributed backend, load model, warm up, then accept tasks.

vMLX should use the same isolation principle. Strict distributed backend init can
throw through C++/C ABI boundaries, so the host app should ask a child process to
prove backend readiness and report structured output. A failed strict init must
not crash Osaurus.

### MLX Sharding Concepts

Exo has family-specific pipeline and tensor sharding logic, including per-family
attention head, KV head, MLP, MoE, and cache handling. The useful lesson is not
the Python monkey-patching; it is the existence of model-family gates.

Swift should represent this as explicit sharding plans:

- dense Llama/Gemma-style plan first.
- Qwen/Gemma MoE after dense proof.
- JANG/JANGTQ only after packed-weight sharding and sidecar metadata contracts
  are explicit.
- hybrid SSM/CCA/DSV4/VL families only after their cache-state contracts are
  encoded in admission.

### Remote Prefill

Exo's remote prefill path is a separate distributed mode from tensor
parallelism: send prompt tokens to a remote prefill worker, receive cache state,
then decode elsewhere.

This is worth keeping as a future vMLX mode because it can use a high-memory
remote Mac without requiring RDMA. It must carry strict cache identity:

- model manifest hash.
- tokenizer and chat-template hash.
- reasoning/tool mode.
- media salt and modality metadata.
- family-specific recurrent or hybrid cache state identity.

## Exo Choices To Avoid Or Improve

- Do not rely on mDNS alone. Keep Bonjour plus manual endpoint, peer-card import,
  SSH bootstrap, tailnet, relay, and optional subnet probe.
- Do not disable IPv6 as a blanket assumption. Two-M5 bring-up already proved
  Thunderbolt Bridge over IPv6 link-local with `%bridge0` scope.
- Do not make address changes fatal. Peers on macOS can move between Wi-Fi,
  bridge, tailnet, and relay candidates.
- Do not use a fresh ephemeral node ID every launch. Persist local node identity
  and handle duplicate detection explicitly.
- Do not use gossipsub for activation tensors, model data, or large cache
  payloads. Keep control-plane messages small and move data over dedicated TLS,
  ring, JACCL, or future relay streams.
- Do not score placement only by available memory and download progress. Include
  measured link quality, backend readiness, current load, model cache class, and
  family capability.
- Do not hard-code family bans as final policy. Record the blocker and the proof
  needed to lift it.
- Do not let a size-1 fallback group count as distributed readiness.

## vMLX Implementation Direction

### 1. Discovery Multiplexer

Add a single `DistributedPeerRegistry` fed by:

- Bonjour `_vmlx._tcp.` and Osaurus `_osaurus._tcp.` `dist.*` TXT records.
- interface-scoped Bonjour for `bridge0` preference.
- manual endpoint plus TLS fingerprint.
- SSH bootstrap inventory for developer bring-up only.
- tailnet endpoint records.
- peer-card import.
- authenticated relay rendezvous later.
- optional local subnet probe as candidate-only evidence.

Every source should produce `PeerEvidence`, not immediate capability. The registry
merges evidence by stable peer identity and surfaces the highest proven state.

### 2. Topology And Edge Evidence

Add typed edges:

- `SocketEdge`: host, port, interface, link class, latency, bandwidth, TLS state.
- `RdmaEdge`: source device, sink device, `rdma_ctl` state, OS build evidence,
  backend build identity, validation timestamp.
- `RelayEdge`: relay identity, region, auth state, latency, bandwidth.

The graph should expose both "best replica path" and "wired blocker" without
collapsing them into one boolean.

### 3. Launch Plans

Introduce typed plan objects before more runtime wiring:

- `DistributedRingLaunchPlan` validates `MLX_HOSTFILE`.
- `DistributedJACCLLaunchPlan` validates `MLX_IBV_DEVICES`,
  `MLX_JACCL_COORDINATOR`, rank count, and selected rank.
- `DistributedRunnerPlan` combines model manifest, rank env, binary path, and
  expected backend build identity.

The CLI should be able to emit these plans as JSON for Osaurus and for two-host
manual testing.

### 4. Proof Sequence

Keep the existing order, with Exo-derived additions:

1. Extend probe output with OS build, Thunderbolt UUID/interface evidence,
   `rdma_ctl` status, backend build identity, and mode blockers.
2. Add RDMA device enumeration and `MLX_IBV_DEVICES` validation.
3. Build the real MLX Ring/JACCL variant.
4. Spawn two rank child processes and prove strict Ring init.
5. Spawn two rank child processes and prove strict JACCL init.
6. Prove a tiny collective over each backend.
7. Prove dense-model TP bit identity or tolerance.
8. Add JANG/JANGTQ sharding only after packed-weight shard contracts exist.

## Test Matrix Additions

Add tests for:

- discovery merge and de-duplication across Bonjour, manual, tailnet, and SSH
  bootstrap evidence.
- persistent node identity and duplicate-node handling.
- stale peer expiry after failed health checks.
- address change from Wi-Fi to Thunderbolt Bridge without crashing the registry.
- IPv6 link-local endpoint parsing with interface scope.
- TXT records that advertise `tp` without backend proof are rejected or
  downgraded to blocked.
- RDMA status changes remove or block RDMA edges.
- Thunderbolt Bridge present but JACCL unavailable yields `wired-blocked`, not
  `tpReady`.
- generated Ring hostfiles and JACCL device matrices validate shape and rank
  count.
- JACCL coordinator selection does not use Thunderbolt Bridge as a side-channel
  address when RDMA devices need those ports.
- cross-peer cache admission fails on tokenizer/template/media/family-state
  mismatch.
- model same-name/different-manifest mismatch blocks replica routing.

## Immediate Next Work

1. Done: add `DistributedPeerRegistry` and `PeerEvidence` types in
   `MLXDistributedCore`.
2. Started: add topology edge evidence for socket, RDMA/JACCL, and relay paths.
3. Extend `DistributedProbe` JSON with Exo-style system evidence:
   OS version/build, Thunderbolt identifiers, `rdma_ctl`, and Thunderbolt Bridge
   service state.
4. Done: add launch-plan data types and validators for Ring and JACCL.
5. Add a plan-emitting CLI mode before enabling any new runtime path.
6. Keep the current request-level replica smoke as the first usable path while
   Ring/JACCL backend enablement is worked separately.
