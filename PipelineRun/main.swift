// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `PipelineRun` — launcher for vmlx-swift-lm distributed pipeline
// inference. This is the binary you run on each Mac in a pipeline
// cluster. One Mac is the worker (advertises on Bonjour, accepts the
// ring handshake, runs the downstream stages), another Mac is the
// driver (discovers the worker, drives the prompt + first stage,
// collects generated tokens).
//
// Phase 1 today — concrete bring-up. Phase 1 next — adds Qwen 3
// adapter and real model inference.
//
// ## Modes
//
//   PipelineRun server [--name NAME] [--port PORT]
//       Run as a worker. Advertise `_vmlx-pipeline._tcp` on the local
//       network, listen on PORT (default 7437), block until the
//       driver connects and drives the pipeline forward pass.
//
//   PipelineRun driver [--timeout SEC] [--peer HOST:PORT]
//       Run as the driver. Either discover a worker via Bonjour
//       (default) or connect to an explicit HOST:PORT. Runs the
//       transport measurement probe against the worker and prints
//       bandwidth / latency numbers. When the model adapter lands,
//       this also runs the actual pipeline forward pass.
//
//   PipelineRun probe [--peer HOST:PORT]
//       Just run the transport measurement probe. Useful for
//       isolating "is the network working?" from "is the pipeline
//       working?".
//
//   PipelineRun discover [--timeout SEC]
//       Browse the local network for pipeline workers and print what
//       comes back. No forward pass, no probe — just `ls` for
//       advertised peers.
//
// ## Environment variables (override CLI where conflicting)
//
//   MLX_HOSTFILE=/path       Path to JSON hostfile for the ring backend.
//                            Format: [["ip:port", "ip:port"], ...] —
//                            one outer entry per rank. When pre-set,
//                            the launcher honors it verbatim.
//   MLX_RANK=N               This process's rank. Auto-set by the
//                            launcher based on server vs driver role.
//   VMLX_PIPELINE_BACKEND    mlx distributed backend to init with.
//                            Defaults to "ring". Valid: any, ring,
//                            jaccl (jaccl requires build-time enable
//                            in mlx-swift — not yet on osaurus-
//                            0.31.3-distributed-phase0).

import Foundation
import MLXLMCommon

// MARK: - CLI parsing

enum Mode {
    case server(name: String?, port: Int)
    case driver(timeout: TimeInterval, explicitPeer: String?)
    case probe(explicitPeer: String?)
    case discover(timeout: TimeInterval)
}

func printUsage() {
    print("""
    Usage: PipelineRun <mode> [options]

    Modes:
      server [--name NAME] [--port PORT]
          Advertise on Bonjour and accept pipeline drivers.
          Default PORT: 7437. Default NAME: system hostname.

      driver [--timeout SEC] [--peer HOST:PORT]
          Discover or connect to a worker, run the pipeline.
          Default timeout: 5.0s.

      probe [--peer HOST:PORT]
          Run the transport measurement probe against a worker.
          Peer either discovered via Bonjour or given explicitly.

      discover [--timeout SEC]
          List Bonjour workers on the local network and exit.

    Environment variables:
      MLX_HOSTS=ip1,ip2          Explicit ring-backend host list.
      VMLX_PIPELINE_BACKEND=ring mlx distributed backend (default ring).
    """)
}

func parseMode(_ args: [String]) -> Mode? {
    guard args.count >= 1 else { return nil }

    switch args[0] {
    case "server":
        var name: String? = nil
        var port = 7437
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--name":
                if i + 1 < args.count { name = args[i + 1]; i += 2 }
                else { return nil }
            case "--port":
                if i + 1 < args.count, let p = Int(args[i + 1]) {
                    port = p; i += 2
                } else { return nil }
            default: return nil
            }
        }
        return .server(name: name, port: port)

    case "driver":
        var timeout: TimeInterval = 5.0
        var peer: String? = nil
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--timeout":
                if i + 1 < args.count, let t = Double(args[i + 1]) {
                    timeout = t; i += 2
                } else { return nil }
            case "--peer":
                if i + 1 < args.count { peer = args[i + 1]; i += 2 }
                else { return nil }
            default: return nil
            }
        }
        return .driver(timeout: timeout, explicitPeer: peer)

    case "probe":
        var peer: String? = nil
        var i = 1
        while i < args.count {
            if args[i] == "--peer", i + 1 < args.count {
                peer = args[i + 1]; i += 2
            } else { return nil }
        }
        return .probe(explicitPeer: peer)

    case "discover":
        var timeout: TimeInterval = 5.0
        var i = 1
        while i < args.count {
            if args[i] == "--timeout", i + 1 < args.count,
               let t = Double(args[i + 1])
            {
                timeout = t; i += 2
            } else { return nil }
        }
        return .discover(timeout: timeout)

    default:
        return nil
    }
}

// MARK: - Mode handlers

func runServer(name: String?, port: Int) async throws {
    print("[vmlx] starting pipeline worker")
    let env = ProcessInfo.processInfo.environment
    let backend = env["VMLX_PIPELINE_BACKEND"] ?? "ring"
    let hasHostfile = env["MLX_HOSTFILE"] != nil && env["MLX_RANK"] != nil

    // Bonjour advertising is for peer DISCOVERY. When MLX_HOSTFILE is
    // pre-set the operator already knows the topology — skip Bonjour
    // so the same port can be used by the ring backend's own listener
    // (Bonjour would otherwise hold the port and the ring init would
    // fail to bind, leaving both ranks stuck "accepting"). This is
    // the typical path for explicit-peer operations (Tailscale, VPN,
    // cross-subnet cases where mDNS doesn't work anyway).
    var advertiser: BonjourAdvertiser?
    if !hasHostfile {
        print("[vmlx]   service: \(BonjourAdvertiser.serviceType)")
        print("[vmlx]   name: \(name ?? "<system hostname>")")
        print("[vmlx]   port: \(port)")
        let adv = BonjourAdvertiser(serviceName: name, port: port)
        try await adv.start()
        advertiser = adv
        print("[vmlx] advertising — waiting for driver")
    } else {
        print("[vmlx]   (skipping Bonjour advertise — MLX_HOSTFILE pre-set)")
    }
    print("[vmlx] (press ctrl-c to stop)")

    if hasHostfile {
        print("[vmlx] MLX_HOSTFILE=\(env["MLX_HOSTFILE"] ?? "")")
        print("[vmlx] MLX_RANK=\(env["MLX_RANK"] ?? "")")
        _ = MLXDistributed.initialize(strict: false, backend: backend)
        if let world = MLXDistributed.worldGroup, world.isMultiRank {
            let peerRank = world.rank == 0 ? 1 : 0
            print("[vmlx] mlx distributed up: rank=\(world.rank), size=\(world.size)")
            print("[vmlx] running transport probe responder (peer rank \(peerRank))")
            TransportProbe.runResponder(peerRank: peerRank)
            print("[vmlx] responder exited")
        } else {
            print("[vmlx] warning: MLX_HOSTFILE set but initialize gave single-rank world")
            print("[vmlx]   check that the file is reachable and lists this host at rank \(env["MLX_RANK"] ?? "?")")
        }
    } else {
        print("[vmlx] MLX_HOSTFILE not set — blocking on Bonjour advertise only")
        print("[vmlx] launch this binary with:")
        print("[vmlx]   MLX_HOSTFILE=/path/to/hosts.json MLX_RANK=1 PipelineRun server")
        print("[vmlx] after the driver has written the hostfile and SSH'd it to you.")
        while !Task.isCancelled {
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    if let advertiser { await advertiser.stop() }
}

func runDriver(timeout: TimeInterval, explicitPeer: String?) async throws {
    print("[vmlx] starting pipeline driver")
    let env = ProcessInfo.processInfo.environment

    let peer: DiscoveredPeer
    if let explicit = explicitPeer {
        guard let parsed = parseHostPort(explicit) else {
            print("[vmlx] error: invalid --peer \(explicit). Expected HOST:PORT.")
            exit(2)
        }
        peer = parsed
        print("[vmlx] using explicit peer: \(peer)")
    } else {
        print("[vmlx] browsing for workers (timeout \(timeout)s)…")
        let browser = BonjourBrowser()
        let peers = try await browser.discover(timeout: timeout)
        guard let first = peers.first else {
            print("[vmlx] no workers found on local network.")
            print("[vmlx] hint: run `PipelineRun server` on the Mac you want")
            print("[vmlx]       to use as a worker, then re-run `driver`.")
            exit(3)
        }
        if peers.count > 1 {
            print("[vmlx] \(peers.count) workers found; using the first")
            for p in peers { print("[vmlx]   \(p)") }
        }
        peer = first
    }

    print("[vmlx] peer selected: \(peer)")

    // mlx-core's ring backend reads a JSON HOSTFILE + MLX_RANK.
    // Format: [["ip:port"], ["ip:port"]] — one outer entry per rank.
    //
    // If the operator has already set MLX_HOSTFILE + MLX_RANK we
    // honor those — typical when the same hostfile has been copied
    // to both machines and the driver is using it verbatim.
    // Otherwise we write a default hostfile to /tmp and set the envs.
    let hadPresetHostfile = env["MLX_HOSTFILE"] != nil
        && env["MLX_RANK"] != nil
    if hadPresetHostfile {
        print("[vmlx] honoring pre-set MLX_HOSTFILE=\(env["MLX_HOSTFILE"] ?? "")")
        print("[vmlx] honoring pre-set MLX_RANK=\(env["MLX_RANK"] ?? "")")
        if let contents = try? String(
            contentsOfFile: env["MLX_HOSTFILE"] ?? "", encoding: .utf8)
        {
            print("[vmlx] hostfile contents:")
            for line in contents.split(separator: "\n") {
                print("[vmlx]   \(line)")
            }
        }
    } else {
        let driverHost = env["VMLX_DRIVER_HOST"] ?? "localhost"
        let driverPort = env["VMLX_DRIVER_PORT"] ?? "7436"
        let hostfileJSON = """
        [
          ["\(driverHost):\(driverPort)"],
          ["\(peer.host):\(peer.port)"]
        ]
        """
        let hostfilePath = "/tmp/vmlx-pipeline-hosts-\(getpid()).json"
        try hostfileJSON.write(
            toFile: hostfilePath, atomically: true, encoding: .utf8)
        setenv("MLX_HOSTFILE", hostfilePath, 1)
        setenv("MLX_RANK", "0", 1)

        print("[vmlx] wrote hostfile: \(hostfilePath)")
        print("[vmlx] MLX_RANK=0 (driver)")
        print("[vmlx] hostfile contents:")
        for line in hostfileJSON.split(separator: "\n") {
            print("[vmlx]   \(line)")
        }
        print("[vmlx] worker must be running with:")
        print("[vmlx]   scp \(hostfilePath) \(peer.host):\(hostfilePath)")
        print("[vmlx]   ssh \(peer.host) MLX_HOSTFILE=\(hostfilePath) MLX_RANK=1 PipelineRun server")
        print("")
    }

    let backend = env["VMLX_PIPELINE_BACKEND"] ?? "ring"
    _ = MLXDistributed.initialize(strict: false, backend: backend)
    guard let world = MLXDistributed.worldGroup, world.isMultiRank else {
        print("[vmlx] mlx distributed failed to go multi-rank — check that")
        print("[vmlx] the worker is running with the same hostfile at")
        print("[vmlx] MLX_RANK=1, and that \(peer.host):\(peer.port) is reachable.")
        exit(4)
    }

    print("[vmlx] mlx distributed up: rank=\(world.rank), size=\(world.size)")
    print("[vmlx] running transport probe initiator")
    let results = TransportProbe.runInitiator(
        peerRank: 1,
        payloads: [4096, 65_536, 1_048_576, 16_777_216],
        iterations: 50
    )
    print("")
    print("[vmlx] --- transport probe results ---")
    for r in results {
        print("[vmlx] \(r.summary())")
    }
    print("")

    // Phase 1 close will: load Qwen 3-0.6B here, build the PipelineEngine,
    // call `engine.runPrompt(...)`. For now we've proven the network
    // pair works end-to-end.
    print("[vmlx] model-inference step — coming in the next commit.")
    print("[vmlx] this commit proves the transport pair works.")
}

func runProbe(explicitPeer: String?) async throws {
    // Probe mode = driver without the "run a real forward" hint.
    // Simpler alias so operators can test transport in isolation.
    try await runDriver(timeout: 5.0, explicitPeer: explicitPeer)
}

func runDiscover(timeout: TimeInterval) async throws {
    print("[vmlx] browsing for \(BonjourAdvertiser.serviceType) workers")
    print("[vmlx]   timeout: \(timeout)s")
    let browser = BonjourBrowser()
    let peers = try await browser.discover(timeout: timeout)

    if peers.isEmpty {
        print("[vmlx] no workers found.")
        exit(0)
    }
    print("[vmlx] found \(peers.count) worker(s):")
    for peer in peers {
        print("[vmlx]   \(peer)")
    }
}

// MARK: - Helpers

func parseHostPort(_ s: String) -> DiscoveredPeer? {
    let parts = s.split(separator: ":", maxSplits: 1).map(String.init)
    guard parts.count == 2, let port = Int(parts[1]) else { return nil }
    return DiscoveredPeer(name: parts[0], host: parts[0], port: port)
}

// MARK: - Entry

let args = Array(CommandLine.arguments.dropFirst())

guard let mode = parseMode(args) else {
    printUsage()
    exit(1)
}

do {
    switch mode {
    case .server(let name, let port):
        try await runServer(name: name, port: port)
    case .driver(let timeout, let peer):
        try await runDriver(timeout: timeout, explicitPeer: peer)
    case .probe(let peer):
        try await runProbe(explicitPeer: peer)
    case .discover(let timeout):
        try await runDiscover(timeout: timeout)
    }
} catch {
    print("[vmlx] error: \(error)")
    exit(10)
}
