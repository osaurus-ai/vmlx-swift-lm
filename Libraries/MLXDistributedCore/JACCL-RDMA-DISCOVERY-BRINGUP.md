# JACCL, RDMA, And Discovery Bring-Up

Status date: 2026-05-08.

This page tracks the next engine-side work needed before wired tensor
parallelism can be exposed to Osaurus. It intentionally separates ordinary
network discovery from JACCL/RDMA readiness so a Thunderbolt cable cannot
produce a false "TP-ready" signal.

## Current Verified State

- Thunderbolt Bridge is visible as `bridge0` and has both IPv4 link-local and
  IPv6 link-local addresses.
- `DistributedProbe --json` reports `librdmaLoadable: true`.
- `DistributedProbe --json` reports `jacclAvailable: false` and
  `anyDistributedBackendAvailable: false`.
- `MLX_IBV_DEVICES` is not configured, so RDMA rank launch is not configured.
- The active `mlx-swift` SwiftPM checkout still excludes the real MLX
  distributed sources:
  - `mlx/mlx/distributed/ring/ring.cpp`
  - `mlx/mlx/distributed/jaccl/jaccl.cpp`
  - `mlx/mlx/distributed/jaccl/mesh.cpp`
  - `mlx/mlx/distributed/jaccl/ring.cpp`
  - `mlx/mlx/distributed/jaccl/utils.cpp`
- Because the real sources are excluded, the package compiles the MLX
  `no_ring.cpp` / `no_jaccl.cpp` stubs. In that state,
  `JACCL.isAvailable() == false` is expected even when `librdma.dylib` can be
  loaded by the OS.

## Hard Blockers

### 1. Real Distributed MLX Build Variant

Create an explicit build path that includes real MLX Ring/JACCL sources and
excludes their `no_*` stubs. This should be opt-in until it builds cleanly on
both M5 laptops.

Required properties:

- Use the same source selection as upstream MLX CMake:
  - real Ring when CPU build is enabled and the platform is not Windows.
  - real JACCL only on Darwin with macOS SDK `>= 26.2`.
- Keep MPI/NCCL excluded for the Apple Silicon path unless a separate Linux
  backend is intentionally added later.
- Expose enough C symbols through a stable Swift target so `Group`,
  collectives, and diagnostics do not depend on private `@_silgen_name`
  declarations long-term.
- Build and run both ranks with the same package/backend variant. Mixed
  real/stub builds must be treated as a failure.

Do not make this the default package build until CI and normal local builds are
unaffected.

### 2. Backend Diagnostics That Cannot Be Fooled By Size-1 Fallback

`mlx_distributed_init(strict: false, backend: ...)` can fall back to a trivial
size-1 group. That is useful for no-op single-rank tests, but unsafe as a
readiness signal.

Add a diagnostic surface that reports:

- whether the package was built with real or stub distributed sources.
- `librdma.dylib` load status.
- `JACCL.isAvailable()`.
- `JACCL.anyBackendAvailable()`.
- macOS version, SDK version, Swift version, and developer directory.
- presence of `MLX_RANK`, `MLX_IBV_DEVICES`, `MLX_JACCL_COORDINATOR`, and
  `MLX_JACCL_RING`.
- whether `MLX_IBV_DEVICES` exists, parses as JSON, has a square rank matrix,
  and is valid as mesh and/or ring.
- whether the selected rank is inside the device-file rank count.
- the final group rank/size from a child-process strict init smoke.

The child process matters because C++ exceptions from strict backend init must
not crash the host app.

### 3. RDMA Device Discovery And Device-File Generation

Real MLX JACCL expects:

- `MLX_RANK`
- `MLX_IBV_DEVICES`
- `MLX_JACCL_COORDINATOR`
- optional `MLX_JACCL_RING`

`MLX_IBV_DEVICES` is a JSON rank-connectivity matrix. For two ranks, the first
usable generator should produce one of these shapes after device names are
discovered and confirmed on both hosts:

```json
[
  [null, "device-for-rank0-to-rank1"],
  ["device-for-rank1-to-rank0", null]
]
```

Next code work:

- add an RDMA device-list probe, preferably through MLX/JACCL's IBV wrapper or
  a small C shim over `ibv_get_device_list`.
- add a validator for user-provided and generated device JSON.
- add a two-host helper that writes the same validated file to both ranks.
- include the chosen device names and rank count in `DistributedProbe --json`.

Never infer device names from `bridge0`. Thunderbolt Bridge is a network
interface; JACCL uses RDMA device names from the verbs stack.

## Discovery Methods To Support

Osaurus should merge several discovery sources into one peer state machine.
Each source can find a peer, but none by itself proves every mode.

| Source | Covers | First allowed mode | Notes |
| --- | --- | --- | --- |
| Bonjour / mDNS | same LAN, Wi-Fi, Thunderbolt Bridge | replica | Use `_osaurus._tcp.` with `dist.*` TXT. Include model hashes and TLS fingerprint. |
| Interface-scoped Bonjour | multi-interface LAN/TB edge cases | replica | Prefer `bridge0` when the same peer is reachable through Wi-Fi and TB. |
| Manual endpoint | tailnet, static IP, headless peer | replica | User enters host/port/fingerprint or imports a peer card. |
| SSH bootstrap | developer bring-up without daemon | inventory, smoke only | Good for `~/models` scans and staging binaries; not production discovery. |
| Tailnet route | Tailscale/ZeroTier/WireGuard | replica | Treat as trusted/manual network. Never advertise TP/RDMA. |
| Internet relay | off-LAN spare Mac | replica | Authenticated rendezvous plus TLS identity. No collectives. |
| Optional local subnet scan | mDNS-broken LANs | candidate only | User-enabled fallback; verify with TLS health before showing as usable. |
| QR / peer-card import | NAT, DNS, or mDNS failures | replica | Contains host candidates and TLS fingerprint, not secrets. |

## Peer State Machine

Avoid a single `distributedReady` boolean. Track blockers and evidence per
state:

1. `discovered`: TXT/manual/relay record parsed.
2. `reachable`: TCP/TLS health endpoint responds.
3. `trusted`: fingerprint or account identity accepted.
4. `inventoried`: model inventory loaded with no fatal scan error.
5. `replicaReady`: requested model hash and template identity match and remote
   generation has a current load proof.
6. `pipelineCandidate`: typed activation/runtime stage contract exists for the
   model family.
7. `wiredCandidate`: same peer is reachable on a wired interface.
8. `rdmaConfigured`: JACCL env and device file validate for both ranks.
9. `rdmaReady`: two-rank strict group init and collective smoke pass.
10. `tpReady`: selected model family has a TP-safe sharding plan and cache
    contract.

Osaurus should show the highest achieved state plus the first blocker. Example:
`wired-blocked: JACCL backend is stubbed in this build`.

## False-Positive Guards

- Do not advertise `dist.modes=tp` because `bridge0` exists.
- Do not advertise `dist.rdma.*` when JACCL is unavailable.
- Do not treat `librdmaLoadable` as JACCL readiness.
- Do not treat a size-1 `Group` as successful distributed init.
- Do not route a same-name model mismatch. Use manifest hashes.
- Do not share cross-peer cache unless model, tokenizer, chat template,
  reasoning/tool mode, media salt, and family-specific state match exactly.
- Do not allow relay or tailnet peers to advertise wired TP.
- Do not keep stale mDNS peers alive after failed health checks or fingerprint
  changes.

## Proofs Required Before Enabling Wired TP

1. Build proof: real Ring/JACCL sources are compiled on both laptops, and the
   probe reports a non-stub backend.
2. Local diagnostic proof: `DistributedProbe --json` reports JACCL available,
   valid RDMA env, valid device JSON, and a Thunderbolt candidate.
3. Two-rank init proof: strict rank 0/rank 1 JACCL init returns `size == 2`.
4. Collective proof: all-sum/all-gather/sum-scatter produce expected values on
   both ranks and do not hang on cleanup.
5. Failure proof: wrong rank, missing device file, bad coordinator, and bad
   device name fail with structured errors and timeouts.
6. Tiny TP proof: synthetic sharded linear matches a single-rank baseline.
7. Real dense-model TP proof: start with a small dense Llama/Gemma-style text
   model before MoE, JANGTQ, VL, or hybrid cache families.
8. Replan proof: if one rank dies mid-request, Osaurus reports the failure and
   falls back only when the model is available locally or as a replica route.

Until all eight proofs exist, wired TP should remain a disabled/blocked state,
not an automatic option.

## Implementation Order

1. Add `DistributedJACCLDiagnostics` data types and JSON output in
   `DistributedProbe`.
2. Add RDMA device enumeration and `MLX_IBV_DEVICES` validation.
3. Add an opt-in real distributed `mlx-swift` build variant.
4. Add a child-process strict init smoke that cannot crash the parent.
5. Add a two-host collective smoke using the generated device file.
6. Teach `TXTSchema` and Osaurus peer UI to carry blocker/state evidence, not
   just mode booleans.
7. Promote `dist.modes=tp` only after the proof matrix is green.
