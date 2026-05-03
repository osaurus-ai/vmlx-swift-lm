import Foundation
import MLX
import CmlxDistributedShim

/// Tensor-level collectives. Each is a no-op on a size-1 group (matches
/// Python's `mx.distributed.all_sum` etc. semantics) so call sites don't
/// need to gate on `group.isMultiRank`.
public enum Collectives {

    /// All-reduce sum: every rank ends up with the elementwise sum of
    /// `x` across the group.
    public static func allSum(_ x: MLXArray, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invoke(x, group: group) { res, src, grp, str in
            vmlx_all_sum(res, src, grp, str)
        }
    }

    /// All-gather: returns concatenation of every rank's `x` along the
    /// 0th axis (Python ref: `mx.distributed.all_gather`).
    public static func allGather(_ x: MLXArray, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invoke(x, group: group) { res, src, grp, str in
            vmlx_all_gather(res, src, grp, str)
        }
    }

    /// Sum-scatter: every rank gets a slice of the elementwise sum
    /// (Python ref: `mx.distributed.sum_scatter`).
    public static func sumScatter(_ x: MLXArray, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invoke(x, group: group) { res, src, grp, str in
            vmlx_sum_scatter(res, src, grp, str)
        }
    }

    /// Send `x` to `dst`; returns the input array (used for graph
    /// dependency in MLX's lazy-eval model).
    public static func send(_ x: MLXArray, to dst: Int, group: Group) -> MLXArray {
        if !group.isMultiRank { return x }
        return invokeWithInt(x, intArg: dst, group: group) { res, src, n, grp, str in
            vmlx_send(res, src, Int32(n), grp, str)
        }
    }

    /// Receive an array shaped like `like` from `src` rank.
    public static func recvLike(_ like: MLXArray, from src: Int, group: Group) -> MLXArray {
        if !group.isMultiRank { return like }
        return invokeWithInt(like, intArg: src, group: group) { res, lik, n, grp, str in
            vmlx_recv_like(res, lik, Int32(n), grp, str)
        }
    }

    // MARK: - Internal bridging

    /// Calls a 4-arg collective (res, x, group, stream).
    private static func invoke(
        _ x: MLXArray,
        group: Group,
        _ call: (
            UnsafeMutablePointer<UnsafeMutableRawPointer?>,
            UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?
        ) -> Int32
    ) -> MLXArray {
        var resPtr: UnsafeMutableRawPointer? = nil
        let srcPtr = unsafeBitCast(x.ctx, to: UnsafeMutableRawPointer?.self)
        let grpPtr = group.handle.raw.ctx
        let stmPtr: UnsafeMutableRawPointer? = nil  // default stream
        let rc = call(&resPtr, srcPtr, grpPtr, stmPtr)
        precondition(rc == 0, "mlx-c collective failed with rc=\(rc)")
        return mlxArrayFromCtx(resPtr)
    }

    /// Calls a 5-arg collective with an int between x and group.
    private static func invokeWithInt(
        _ x: MLXArray,
        intArg: Int,
        group: Group,
        _ call: (
            UnsafeMutablePointer<UnsafeMutableRawPointer?>,
            UnsafeMutableRawPointer?,
            Int,
            UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?
        ) -> Int32
    ) -> MLXArray {
        var resPtr: UnsafeMutableRawPointer? = nil
        let srcPtr = unsafeBitCast(x.ctx, to: UnsafeMutableRawPointer?.self)
        let grpPtr = group.handle.raw.ctx
        let stmPtr: UnsafeMutableRawPointer? = nil
        let rc = call(&resPtr, srcPtr, intArg, grpPtr, stmPtr)
        precondition(rc == 0, "mlx-c send/recv failed with rc=\(rc)")
        return mlxArrayFromCtx(resPtr)
    }
}

/// Reconstruct an MLXArray from the void* ctx of a freshly synthesized
/// mlx_array. We can't directly call MLXArray.init(ctx:) without
/// importing Cmlx; we use unsafeBitCast on a Swift struct that is
/// layout-compatible with mlx_array (both are single-pointer aggregates).
internal func mlxArrayFromCtx(_ ptr: UnsafeMutableRawPointer?) -> MLXArray {
    // mlx_array is { void* ctx } — same shape as a single pointer.
    // MLXArray's stored ctx field accepts an mlx_array value; we
    // construct one by reinterpreting the pointer.
    let raw = _MLXArrayCtxBox(ctx: ptr)
    return unsafeBitCast(raw, to: MLXArray.self)
}

/// Layout-compatible mirror of mlx_array. Used only for the
/// `unsafeBitCast` round-trip in `mlxArrayFromCtx`.
internal struct _MLXArrayCtxBox {
    var ctx: UnsafeMutableRawPointer?
}
