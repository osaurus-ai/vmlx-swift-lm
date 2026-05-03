import Foundation
import MLX
import MLXNN

/// What kind of TP sharding to apply.
///
/// Mirrors the string tags in Python's `mlx.nn.layers.distributed`:
/// - `.allToSharded` ("all-to-sharded") for column-output projections
///   (qkv, gate, up_proj) — each rank holds a row-shard of the weight,
///   the result is sharded across the group.
/// - `.shardedToAll` ("sharded-to-all") for row-output projections
///   (o_proj, down_proj) — each rank holds a column-shard, partial
///   outputs are all-reduced into the full output.
public enum Sharding: String, Sendable {
    case allToSharded = "all-to-sharded"
    case shardedToAll = "sharded-to-all"
}

/// Convert a dense `Linear` into a tensor-parallel layer of the
/// requested kind. Returns `UnaryLayer` so callers can substitute it
/// transparently for the original module.
///
/// Mirror of Python's `mx.nn.utils.shard_linear`.
public func shardLinear(
    _ linear: Linear,
    sharding: Sharding,
    segments: Int = 1,
    group: Group? = nil
) -> any UnaryLayer {
    switch sharding {
    case .allToSharded:
        return AllToShardedLinear.from(linear, group: group, segments: segments)
    case .shardedToAll:
        return ShardedToAllLinear.from(linear, group: group, segments: segments)
    }
}
