# MLXDistributedTransport

TLS-backed pipeline transport for the multi-host distributed inference
rollout. `Mode.pipelined` is explicit opt-in from `MLXDistributedCore` and
currently supports a two-rank prompt/token path over the network.

Replica fan-out is not implemented in this target. The core exposes
`ReplicaTransport`; osaurus should back it with the existing chat-completion
HTTP/SSE endpoint.

This target depends on `MLXDistributedCore` plus the SwiftNIO stack
(`NIOCore`, `NIOPosix`, `NIOSSL`, `NIOHTTP2`) and `swift-certificates` /
`swift-crypto` for self-signed cert generation. **No MLX deps** — the
runtime that consumes activations / produces tokens is injected by the
caller (osaurus or RunBench), same pattern as Phase 1A.

## What's here

| File | Purpose |
|------|---------|
| `ActivationFrame.swift` | 24-byte big-endian envelope for PP wire frames |
| `CertificateBundle.swift` | Self-signed cert generator (P256/ECDSA, swift-certificates) |
| `TrustVerifier.swift` | Consults `TrustPolicy` (TOFU / allowlist / denyAll) |
| `PipelineStageServer.swift` | TLS listener that accepts inbound stage handoffs |
| `PipelineStageClient.swift` | TLS client for outbound stage calls |
| `StageHandler.swift` | Consumer-supplied stage execution protocol |
| `TLSPipelinedTransport.swift` | `PipelinedTransport` implementation used by `ClusterSession` |

## What's coming

- Activation frames with shape, dtype, layer-range, and cache identity.
- Multi-stage runtime beyond the current first remote stage.
- Backpressure, heartbeat, timeout, and retry policy.
- Cache-aware admission from the distributed planner.

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
