# MLXDistributedTransport

Phase 2 of the multi-host distributed inference rollout. Adds the
TLS-backed pipeline-parallel transport that turns `Mode.pipelined` from a
`notImplementedYet` stub in `MLXDistributedCore` into a working two-rank
PP path over the network.

This target depends on `MLXDistributedCore` plus the SwiftNIO stack
(`NIOCore`, `NIOPosix`, `NIOSSL`, `NIOHTTP2`) and `swift-certificates` /
`swift-crypto` for self-signed cert generation. **No MLX deps** — the
runtime that consumes activations / produces tokens is injected by the
caller (osaurus or RunBench), same pattern as Phase 1A.

## What's here so far (Phase 2 in progress)

| File | Purpose |
|------|---------|
| `ActivationFrame.swift` | 24-byte big-endian envelope for PP wire frames |
| `CertificateBundle.swift` | Self-signed cert generator (P256/ECDSA, swift-certificates) |
| `TrustVerifier.swift` | Consults `TrustPolicy` (TOFU / allowlist / denyAll) |

## What's coming (Phase 2 remaining)

- `PipelineStageServer` — TLS listener that accepts inbound stage handoffs.
- `PipelineStageClient` — TLS client that opens / multiplexes outbound streams.
- `PipelineRuntime` — wires `ClusterSession.Mode.pipelined` to the above.
- `StageHandler` — protocol for the consumer-supplied stage execution unit.
- Integration test: two `ClusterSession`s in the same process round-trip a
  token stream over loopback TLS (gated on `VMLX_RUN_NIO_TESTS=1`).

## Wire-frame format

```
offset  size  field
     0     4  magic           ("VMLX" = 0x564D4C58)
     4     4  schemaVersion   (uint32, == 1)
     8     4  frameType       (uint32: 1=prefillRequest, 2=decodeRequest,
                                       3=activationsForward, 4=tokenStream,
                                       5=error)
    12     4  reserved        (zero in v1)
    16     8  payloadLen      (uint64 big-endian)
    24    *   payload         (raw bytes)
```

Decoder is restartable: insufficient bytes / bad magic / wrong version /
unknown type / truncated payload all throw without consuming bytes, so
callers can wait on more data without losing state.
