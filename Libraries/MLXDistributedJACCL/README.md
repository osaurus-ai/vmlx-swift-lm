# MLXDistributedJACCL

Swift binding to MLX's JACCL distributed backend (RDMA over Thunderbolt 5
on Apple Silicon, macOS 26.3+).

## Phase 4 status (2026-05-02)

**`JACCL.isAvailable()` returns true on macOS 26.3.2 / M5 Max.** Smoke
tests pass. The full upstream MLX C++ JACCL backend compiles cleanly
against `MacOSX26.4.sdk` and links against the system-cached
`librdma.dylib`.

This target ships only the capability probe in Phase 4. The real
collective bindings (`Group`, `allSum`, `allGather`, `send`, `recv`,
`sumScatter`) land in **Phase 5**, on top of this foundation.

## Required upstream change in `osaurus-ai/mlx-swift`

`MLXDistributedJACCL` works in this repo today because we patch the
mlx-swift checkout's `Package.swift` locally. To make it durable across
`swift package resolve` cycles, the following diff needs to land in
the `osaurus-ai/mlx-swift` fork (the package vmlx-swift-lm depends on):

```diff
        // example code + mlx-c distributed
        "mlx-c/examples",
-       "mlx-c/mlx/c/distributed.cpp",
-       "mlx-c/mlx/c/distributed_group.cpp",

-       // do not build distributed support (yet)
-       "mlx/mlx/distributed/mpi/mpi.cpp",
-       "mlx/mlx/distributed/ring/ring.cpp",
-       "mlx/mlx/distributed/nccl/nccl.cpp",
-       "mlx/mlx/distributed/nccl/nccl_stub",
-       "mlx/mlx/distributed/jaccl/jaccl.cpp",
-       "mlx/mlx/distributed/jaccl/mesh.cpp",
-       "mlx/mlx/distributed/jaccl/ring.cpp",
-       "mlx/mlx/distributed/jaccl/utils.cpp",
+       // distributed: keep mpi + nccl excluded (no Apple Silicon use
+       // case); build jaccl + ring against macOS librdma. The "no_*"
+       // stubs define the same symbols as the real cpps so we exclude
+       // them when the real implementations are in the build.
+       "mlx/mlx/distributed/mpi/mpi.cpp",
+       "mlx/mlx/distributed/nccl/nccl.cpp",
+       "mlx/mlx/distributed/nccl/nccl_stub",
+       "mlx/mlx/distributed/jaccl/no_jaccl.cpp",
+       "mlx/mlx/distributed/ring/no_ring.cpp",
```

Also: `Cmlx` should be added to the `products:` array as a public
library so consumers can `import Cmlx` instead of using `@_silgen_name`
shims:

```diff
    products: [
        .library(name: "MLX", targets: ["MLX"]),
+       .library(name: "Cmlx", targets: ["Cmlx"]),
        ...
    ],
```

After both changes land, this Swift binding can `import Cmlx` directly
and drop the `@_silgen_name` declarations in `JACCL.swift`.

## Usage

```swift
import MLXDistributedJACCL

if JACCL.isAvailable() {
    // JACCL backend is ready; can call mlx_distributed_init with
    // MLX_RANK / MLX_IBV_DEVICES / MLX_JACCL_COORDINATOR set.
}
```

## Tests

```sh
swift test --filter JACCLAvailabilityTests
```

Tests skip automatically on hosts where `dlopen("librdma.dylib")`
fails (older macOS, Linux without RDMA stack, sandboxed CI).

## Hardware checked

- M5 Max MacBook Pro, macOS 26.3.2 (Darwin 25.3.0): ✅
- M4 Max MacBook Pro (`oldmbp`), macOS 26.3: not yet smoke-tested.
- Linux/InfiniBand: out of scope (Apple Silicon target).
