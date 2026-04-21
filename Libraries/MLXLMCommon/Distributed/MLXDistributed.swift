// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Cmlx
import Foundation
import MLX
import os

/// Swift bindings for mlx-core's `distributed` module.
///
/// This is Phase 0 of vmlx-swift-lm's distributed inference effort
/// (see `Libraries/MLXLMCommon/Distributed/DISTRIBUTED-DESIGN.md`).
///
/// mlx-core (C++) ships a working distributed layer with four backends:
///
/// - `jaccl` — Apple-native collective communication, used by exo and the
///   preferred path on Apple Silicon.
/// - `ring` — generic TCP backend, works over any IP (including macOS's
///   built-in Thunderbolt Bridge `bridge0`), our fallback for Phase 0/1.
/// - `mpi` — classic HPC backend, needs an external MPI installation.
/// - `nccl` — CUDA only, not relevant on Apple Silicon.
///
/// `mlx-swift` (upstream and our osaurus-0.31.3 fork at
/// `.build/checkouts/mlx-swift`) does not expose any of these. This file
/// adds the Swift surface directly via the `Cmlx` C bridge.
///
/// ## Lifetime note
///
/// `mlx_distributed_group` owns a heap-allocated `mlx::core::distributed::Group`
/// instance via its `void* ctx` field. The upstream C header does not expose a
/// public `mlx_distributed_group_free` function — only a private inline helper
/// in `mlx-c/mlx/c/private/distributed_group.h` — so we cannot deterministically
/// destroy the C-side group from Swift.
///
/// In practice this is fine: `init` is called once per process and the world
/// group lives for the lifetime of the process. `split` is rarer; each split
/// group also lives to the end. The Swift ``MLXDistributedGroup`` class holds
/// a reference-counted copy of the handle struct but does NOT call the private
/// free on deinit — the C-side allocation stays until process exit.
public enum MLXDistributed {

    private static let logger = Logger(subsystem: "vmlx", category: "MLXDistributed")

    // MARK: - Availability + init

    /// Check whether distributed is available at all, or for a specific backend.
    ///
    /// - Parameter backend: `"jaccl"`, `"ring"`, `"mpi"`, `"nccl"`, or `nil` /
    ///   `"any"` to check whether ANY backend is available.
    /// - Returns: `true` if the named backend (or any backend) can be used.
    ///
    /// This does not initialize distributed. Call ``initialize(strict:backend:)``
    /// for that.
    public static func isAvailable(backend: String? = nil) -> Bool {
        if let backend {
            return backend.withCString { mlx_distributed_is_available($0) }
        }
        return mlx_distributed_is_available(nil)
    }

    /// Initialize the distributed world group for this process.
    ///
    /// Must be called exactly once per process before any collective op runs.
    /// The returned group represents the process's membership in the cluster.
    ///
    /// ## Side effect: error-handler installation
    ///
    /// mlx-c's default C error handler calls `exit(-1)` on any caught C++
    /// exception (see `mlx-c/mlx/c/error.cpp:14`). That's fatal for any
    /// recoverable condition surfaced through the distributed surface — for
    /// example calling `split` on a size-1 group raises "Cannot split the
    /// distributed group further". To let callers handle such conditions
    /// from Swift, ``initialize(strict:backend:)`` replaces the default
    /// handler with one that logs via the `vmlx / MLXDistributed` subsystem
    /// and returns normally. The binding's Swift methods already expose
    /// error conditions through nil-returning or fallible APIs (e.g.
    /// ``MLXDistributedGroup/split(color:key:)`` returns `Optional`) so
    /// callers never see an unhandled termination.
    ///
    /// The error handler is process-wide, so this affects every mlx-c
    /// operation from this point on — not just distributed ops. That is
    /// the correct Swift-application behavior: a library should never
    /// `exit()` on a host app's behalf.
    ///
    /// - Parameters:
    ///   - strict: when `true`, fail hard if the requested backend is not
    ///     available. When `false`, fall back to a single-rank stand-in so
    ///     callers can unconditionally write distributed-aware code that
    ///     runs correctly on a single machine with no external launcher.
    ///   - backend: `"jaccl"`, `"ring"`, `"mpi"`, `"nccl"`, or `"any"` (the
    ///     default) to let mlx-core pick the first available backend.
    /// - Returns: the world group for this process.
    @discardableResult
    public static func initialize(
        strict: Bool = false,
        backend: String = "any"
    ) -> MLXDistributedGroup {
        installErrorHandlerIfNeeded()
        let raw = backend.withCString { cStr in
            mlx_distributed_init(strict, cStr)
        }
        let group = MLXDistributedGroup(raw: raw)
        Self.logger.info(
            "distributed initialized: backend=\(backend, privacy: .public), rank=\(group.rank), size=\(group.size)"
        )
        Self.storedWorldGroup = group
        return group
    }

    /// Install a non-exiting error handler on the process-wide mlx-c slot.
    ///
    /// Idempotent — the first call replaces the default handler and
    /// subsequent calls are a no-op.
    nonisolated(unsafe) private static var errorHandlerInstalled = false

    private static func installErrorHandlerIfNeeded() {
        guard !errorHandlerInstalled else { return }
        errorHandlerInstalled = true
        mlx_set_error_handler(
            { msg, _ in
                // Just log; do not exit(-1) like the default handler.
                if let msg = msg {
                    let text = String(cString: msg)
                    MLXDistributed.logger.error(
                        "mlx C error: \(text, privacy: .public)"
                    )
                }
            },
            nil,
            nil
        )
    }

    /// Process-wide shared world group handle, populated by ``initialize(strict:backend:)``.
    ///
    /// Accessors on collective ops default to this group when no explicit group
    /// is supplied. Reading before ``initialize(strict:backend:)`` has been
    /// called returns `nil`.
    public static var worldGroup: MLXDistributedGroup? {
        storedWorldGroup
    }

    nonisolated(unsafe) private static var storedWorldGroup: MLXDistributedGroup?

    // MARK: - Collective operations

    /// All-reduce (sum). Every rank contributes `x`; every rank receives the
    /// element-wise sum across all ranks.
    ///
    /// All ranks must call this with arrays of the same shape and dtype.
    ///
    /// - Parameters:
    ///   - x: local contribution.
    ///   - group: which group to reduce within. Defaults to the world group.
    ///   - stream: stream or device to evaluate on.
    /// - Returns: the sum, replicated on every rank.
    public static func allSum(
        _ x: MLXArray,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_all_sum(&result, x.ctx, g, stream.ctx)
        return MLXArray(result)
    }

    /// All-gather: every rank contributes `x`; every rank receives a single
    /// array that is the concatenation along the outermost axis of every
    /// rank's `x` in rank order.
    ///
    /// All ranks must contribute arrays with the same trailing shape and
    /// dtype; only the outer dimension can differ per rank (and in the
    /// common case is equal).
    public static func allGather(
        _ x: MLXArray,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_all_gather(&result, x.ctx, g, stream.ctx)
        return MLXArray(result)
    }

    /// All-reduce (max). Element-wise maximum across all ranks.
    public static func allMax(
        _ x: MLXArray,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_all_max(&result, x.ctx, g, stream.ctx)
        return MLXArray(result)
    }

    /// All-reduce (min). Element-wise minimum across all ranks.
    public static func allMin(
        _ x: MLXArray,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_all_min(&result, x.ctx, g, stream.ctx)
        return MLXArray(result)
    }

    /// Sum-scatter: every rank contributes `x`; rank `i` receives the `i`-th
    /// shard of the summed result along the outermost axis. Inverse of
    /// `allGather` composed with a local reduce.
    ///
    /// `x.dim(0)` must be divisible by the group size.
    public static func sumScatter(
        _ x: MLXArray,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_sum_scatter(&result, x.ctx, g, stream.ctx)
        return MLXArray(result)
    }

    /// Point-to-point send. Sends `x` from the caller's rank to `dst`.
    ///
    /// Pipeline parallel uses this to hand an activation tensor from stage N
    /// to stage N+1. Must be paired with a matching ``recv(shape:dtype:src:group:stream:)``
    /// (or ``recvLike(_:src:group:stream:)``) on the destination rank.
    ///
    /// - Returns: the same `x` (mlx returns a placeholder to preserve the
    ///   eval-graph dependency on the send).
    @discardableResult
    public static func send(
        _ x: MLXArray,
        dst: Int,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_send(&result, x.ctx, Int32(dst), g, stream.ctx)
        return MLXArray(result)
    }

    /// Point-to-point receive. Blocking on the source rank's matching
    /// ``send(_:dst:group:stream:)``.
    ///
    /// - Parameters:
    ///   - shape: expected shape of the incoming array.
    ///   - dtype: expected dtype.
    ///   - src: source rank.
    ///   - group: group to receive within.
    ///   - stream: stream or device to evaluate on.
    public static func recv(
        shape: [Int],
        dtype: DType,
        src: Int,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        let shape32 = shape.map { Int32($0) }
        shape32.withUnsafeBufferPointer { ptr in
            _ = mlx_distributed_recv(
                &result,
                ptr.baseAddress,
                shape32.count,
                dtype.cmlxDtype,
                Int32(src),
                g,
                stream.ctx
            )
        }
        return MLXArray(result)
    }

    /// Point-to-point receive using another array's shape and dtype as
    /// templates. Convenient when sender and receiver agree on a schema.
    public static func recvLike(
        _ template: MLXArray,
        src: Int,
        group: MLXDistributedGroup? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        let g = (group ?? storedWorldGroup)?.raw ?? emptyGroup
        var result = mlx_array_new()
        _ = mlx_distributed_recv_like(&result, template.ctx, Int32(src), g, stream.ctx)
        return MLXArray(result)
    }

    // MARK: - Internals

    /// Return an "empty" group literal — ctx is nullptr. mlx-core treats this
    /// as "use the process default group" for all collectives. We fall back
    /// to this when neither an explicit group nor the stored world group is
    /// available (i.e. distributed was never initialized).
    ///
    /// Computed per call because `mlx_distributed_group` is a C struct that
    /// is not `Sendable`; storing it in a static `let` would trip strict
    /// concurrency. The construction is effectively free.
    private static var emptyGroup: mlx_distributed_group {
        mlx_distributed_group(ctx: nil)
    }
}

/// A Swift handle onto an `mlx_distributed_group` from the C layer.
///
/// Instances are created by ``MLXDistributed/initialize(strict:backend:)`` and
/// ``MLXDistributedGroup/split(color:key:)``. The underlying C handle is
/// leaked at process exit (see the lifetime note on ``MLXDistributed``).
public final class MLXDistributedGroup: @unchecked Sendable {

    /// Raw C handle — package-private. Use the public `rank`/`size`/`split`
    /// API from Swift.
    let raw: mlx_distributed_group

    /// This rank's index within the group. Always in `0 ..< size`.
    public let rank: Int

    /// Number of ranks in the group.
    public let size: Int

    /// Whether this group represents multiple distinct ranks.
    ///
    /// A single-rank group behaves correctly for every collective op (they
    /// reduce to the identity on size 1) but callers sometimes want to
    /// short-circuit around a collective entirely for clarity.
    public var isMultiRank: Bool { size > 1 }

    init(raw: mlx_distributed_group) {
        self.raw = raw
        // If `raw.ctx` is nullptr we're wrapping an error-case group that
        // `mlx_distributed_group_{rank,size}` cannot dereference without
        // null-derefing. The mlx-c bridge does catch C++ exceptions from
        // rank/size and return 0, but the accessor does still hit
        // `*static_cast<Group*>(d.ctx)` before the catch, so defend here.
        if raw.ctx != nil {
            self.rank = Int(mlx_distributed_group_rank(raw))
            self.size = Int(mlx_distributed_group_size(raw))
        } else {
            self.rank = 0
            self.size = 0
        }
    }

    /// Whether this group is a valid (non-empty) handle. An invalid handle is
    /// returned from the mlx-c bridge on error paths (see ``split(color:key:)``).
    public var isValid: Bool { raw.ctx != nil }

    /// Split this group into sub-groups by `color`. Ranks that pass the same
    /// `color` value land in the same sub-group; `key` breaks ties for the
    /// new sub-group's rank ordering.
    ///
    /// Mirrors MPI's `MPI_Comm_split`. Use this to carve out per-layer
    /// tensor-parallel sub-groups inside a larger pipeline-parallel world.
    ///
    /// Returns `nil` when the underlying call failed — for example, calling
    /// `split` on a size-1 group ("Cannot split the distributed group
    /// further"). The mlx-c bridge logs the specific reason via
    /// `mlx_error(...)` on the way out.
    public func split(color: Int, key: Int = -1) -> MLXDistributedGroup? {
        let sub = mlx_distributed_group_split(raw, Int32(color), Int32(key))
        // On error the mlx-c bridge returns a zero-initialized group
        // (ctx = nullptr). Reading rank/size from that would null-deref.
        guard sub.ctx != nil else { return nil }
        return MLXDistributedGroup(raw: sub)
    }
}

// MARK: - DType bridging

extension DType {
    /// Bridge a Swift `DType` to the `mlx_dtype` C enum.
    ///
    /// Mirrors the one-to-one mapping in `mlx-swift`'s own `DType.swift`.
    /// Kept internal to the Distributed module because Cmlx bridging is the
    /// only consumer outside `MLX` itself.
    fileprivate var cmlxDtype: mlx_dtype {
        switch self {
        case .bool: return MLX_BOOL
        case .uint8: return MLX_UINT8
        case .uint16: return MLX_UINT16
        case .uint32: return MLX_UINT32
        case .uint64: return MLX_UINT64
        case .int8: return MLX_INT8
        case .int16: return MLX_INT16
        case .int32: return MLX_INT32
        case .int64: return MLX_INT64
        case .float16: return MLX_FLOAT16
        case .bfloat16: return MLX_BFLOAT16
        case .float32: return MLX_FLOAT32
        case .float64: return MLX_FLOAT64
        case .complex64: return MLX_COMPLEX64
        }
    }
}
