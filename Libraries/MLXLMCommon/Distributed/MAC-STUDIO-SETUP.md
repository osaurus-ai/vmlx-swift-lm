# Distributed pipeline inference on two Macs — setup guide

**Target:** pair a Mac Studio (worker) with any other Mac (driver) on
the local network, measure real transport throughput, and — once the
Qwen 3 adapter lands in the next commit — drive a real forward pass
across both.

No config files. No SSH keys. Just run the binary on each Mac.

## Prerequisites

- Both Macs on the same subnet (same Wi-Fi / same Ethernet / same
  Thunderbolt Bridge). Bonjour / mDNS multicast runs at link-local
  scope, so a VPN hop or a router-blocked-multicast configuration
  won't work.
- Swift 6.1 toolchain on both Macs (either install Xcode or use the
  standalone Swift toolchain from swift.org).
- `feature/distributed-inference` branch of this repo checked out on
  both. (Phase 1 builds require the patched
  `osaurus-0.31.3-distributed-phase0` mlx-swift branch — resolved
  automatically on first build.)

## One-time build

On each Mac:

```sh
cd /path/to/vmlx-swift-lm
swift build --product PipelineRun -c release
```

First build resolves + compiles mlx-swift (several minutes). Subsequent
builds are fast.

## Mac Studio side (worker)

```sh
./.build/release/PipelineRun server
```

The worker advertises itself on the local network under the Bonjour
service type `_vmlx-pipeline._tcp` and blocks for a driver to connect.
Default port is **7437** — change with `--port` if already in use.

Expected output:

```
[vmlx] starting pipeline worker
[vmlx]   service: _vmlx-pipeline._tcp
[vmlx]   name: <system hostname>
[vmlx]   port: 7437
[vmlx] advertising — waiting for driver
```

Leave it running. Ctrl-C stops.

## Driver Mac side (everything else)

### Quick discover check

Before wiring up a pipeline, confirm the driver Mac can see the worker:

```sh
./.build/release/PipelineRun discover
```

You should see your Mac Studio show up by hostname + IP + port. If
not, the two machines aren't on the same multicast-reachable subnet
— see "Troubleshooting" below.

### Measure transport throughput

```sh
./.build/release/PipelineRun driver
```

This:
1. Browses Bonjour for workers (default 5s).
2. Picks the first worker found.
3. Sets `MLX_HOSTS=localhost,<worker-host>` and initializes the mlx
   ring backend.
4. Runs the transport probe across four payload sizes (4 KB, 64 KB,
   1 MB, 16 MB), 50 round-trips each.
5. Prints median + p99 one-way latency and median bandwidth.

Example output on a TB4 bridge between two M-series Macs:

```
[vmlx] --- transport probe results ---
[vmlx] payload=    4096 B  iters=  50  latency p50=  180.2 us p99=  320.1 us  bw=   22.7 MB/s ( 0.19 Gbps)
[vmlx] payload=   65536 B  iters=  50  latency p50=  410.5 us p99=  580.3 us  bw=  159.7 MB/s ( 1.34 Gbps)
[vmlx] payload= 1048576 B  iters=  50  latency p50= 2450.1 us p99= 3100.4 us  bw=  427.9 MB/s ( 3.58 Gbps)
[vmlx] payload=16777216 B  iters=  50  latency p50=34200.8 us p99=41200.5 us  bw=  490.7 MB/s ( 4.11 Gbps)
```

Larger payloads hit higher absolute bandwidth because setup cost
amortizes; smaller payloads expose raw round-trip latency. Pipeline
parallelism for LLM decode typically ships `[1, 1, H]` tensors — ~12
KB for an 8K-hidden bf16 model — so the ~500 μs round-trip budget at
the 4-64 KB size is what matters.

### Use an explicit peer (skip Bonjour)

```sh
./.build/release/PipelineRun driver --peer 192.168.1.42:7437
```

Useful when:
- You're connected over TB Bridge where Bonjour isn't advertised
  consistently (macOS's TB Bridge is a second interface the mDNS
  responder doesn't always hit).
- You want to force a specific interface (e.g. 10 GbE vs Wi-Fi).
- You're debugging and need a fixed target.

### Connect over Thunderbolt Bridge

For the fastest path between Macs, use TB4 / TB5 with the Thunderbolt
Bridge interface (`bridge0`). Apple's docs:
<https://support.apple.com/guide/mac-help/set-up-a-thunderbolt-bridge-mchle083ce4a/mac>

1. Cable both Macs TB-to-TB.
2. On each Mac: **System Settings → Network → Thunderbolt Bridge →
   Manually**. Assign static IPs on the same subnet (e.g. `10.0.0.1`
   and `10.0.0.2`, netmask `255.255.255.0`).
3. On the driver Mac: `PipelineRun driver --peer 10.0.0.2:7437` (or
   whatever IP you gave the Mac Studio). Skip Bonjour — it's more
   reliable to be explicit for the TB Bridge.

## What this actually does today

Phase 1 scaffold. As of this commit:

- ✅ Bonjour discovery end-to-end (tested on localhost loopback — the
  driver finds the worker and resolves its IP + port).
- ✅ mlx-core ring backend initialization on both Macs once discovery
  completes.
- ✅ Transport probe (4 payload sizes × 50 iterations) reports real
  bandwidth.
- ⏳ Actual Qwen 3 forward pass across the two Macs — lands in the
  next commit. Today the driver stops after the probe with a note.

This is by design — we're stacking the bring-up in small, testable
pieces. Each step is a regression-proof commit. Ferebee's cache bugs
got fixed by not monkey-patching; distributed pipeline inference gets
built the same way.

## Troubleshooting

### Discover finds no workers

- Check both Macs report the same network in **System Settings →
  Network**. Bonjour / mDNS is link-local only; VPN / inter-subnet
  routing breaks it.
- Firewall: System Settings → Network → Firewall. Allow incoming
  connections for `PipelineRun` on the worker.
- Try the explicit `--peer HOST:PORT` form to sidestep discovery.

### Driver shows "mlx distributed failed to go multi-rank"

The ring backend initialized but the peer didn't join. Likely causes:
- Port blocked by firewall on the worker side.
- Port already in use by another process on the worker (try a
  different `--port`).
- Two workers running — the driver connects to the first it discovers
  but might be expecting a different one. Use `PipelineRun discover`
  first to see what's advertised.

### `MLX_HOSTS` env var already set

If you export `MLX_HOSTS` manually, both driver and worker use that
list verbatim and skip Bonjour. Unset it if you want the discovery
flow back.

## What's coming next

- Qwen 3 `PipelineStageModel` adapter — makes the existing
  `Libraries/MLXLLM/Models/Qwen3.swift` run one stage's worth of
  layers. Follows in the next commit.
- `PipelineRun driver` gains `--model PATH` / `--prompt TEXT` flags
  that load a model on both Macs, build a 2-stage partition, and
  drive the forward pass.
- Byte-identical regression test that runs on a single-host 2-process
  loopback so `swift test` catches any divergence before hardware.
- Jaccl backend enablement on the mlx-swift fork — Apple-native
  collective lib, potentially lower-latency than ring on TB5 pairs.
