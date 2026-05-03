import Foundation
import MLX
import MLXNN
import MLXRandom

/// Tensor-parallel linear layer: each rank holds a row-shard of the
/// weight, applies the local matmul, and the result is sharded across
/// the group along the output axis. The aggregate equivalent of the
/// original (output_dims) is the concatenation of every rank's local
/// output.
///
/// Mirror of Python's `mlx.nn.layers.distributed.AllToShardedLinear`.
/// In the inference path (no gradients), forward is just a standard
/// linear with the locally-shaped weight. The Python reference wraps
/// the input in `sum_gradients(group)` for backward-pass aggregation
/// — that's a no-op on the forward path so we omit it.
open class AllToShardedLinear: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let group: Group

    /// Initialise from scratch with random weights, sharded across `group`.
    public init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        bias: Bool = true,
        group: Group? = nil
    ) {
        let g = group ?? Group(strict: false)
        precondition(
            outputDimensions % g.size == 0,
            "AllToShardedLinear: output_dims (\(outputDimensions)) must be divisible by group size (\(g.size))")

        let scale = sqrt(1.0 / Float(inputDimensions))
        let perRankOut = outputDimensions / g.size

        self.weight = MLXRandom.uniform(-scale ..< scale, [perRankOut, inputDimensions])
        if bias {
            self.bias = MLXRandom.uniform(-scale ..< scale, [perRankOut])
        } else {
            self.bias = nil
        }
        self.group = g
        super.init()
    }

    /// Initialise from an existing dense Linear layer by sharding its
    /// weights along the output axis. Equivalent to Python's
    /// `from_linear` classmethod. `segments` allows splitting fused
    /// weights (e.g. fused QKV with 3 segments) so each segment is
    /// sharded independently rather than splitting through a logical
    /// boundary.
    public static func from(
        _ linear: Linear,
        group: Group? = nil,
        segments: Int = 1
    ) -> AllToShardedLinear {
        let g = group ?? Group(strict: false)
        let (out, _) = linear.shape
        precondition(
            out % g.size == 0,
            "AllToShardedLinear.from: output_dims (\(out)) must be divisible by group size (\(g.size))")
        precondition(segments >= 1, "segments must be >= 1")
        precondition(out % segments == 0,
                     "output_dims (\(out)) must be divisible by segments (\(segments))")

        // Per-rank slice plan: split the weight into `segments` chunks
        // along axis 0, take rank-r's slice from each chunk, concat.
        let perSegmentOut = out / segments
        let perSegmentPerRank = perSegmentOut / g.size
        precondition(perSegmentOut % g.size == 0,
                     "(output_dims/segments) must be divisible by group size")

        var rows: [MLXArray] = []
        for seg in 0 ..< segments {
            let segStart = seg * perSegmentOut
            let rankStart = segStart + g.rank * perSegmentPerRank
            let rankEnd = rankStart + perSegmentPerRank
            rows.append(linear.weight[rankStart ..< rankEnd, 0...])
        }
        let shardedWeight = concatenated(rows, axis: 0)

        var shardedBias: MLXArray? = nil
        if let b = linear.bias {
            var biasRows: [MLXArray] = []
            for seg in 0 ..< segments {
                let segStart = seg * perSegmentOut
                let rankStart = segStart + g.rank * perSegmentPerRank
                let rankEnd = rankStart + perSegmentPerRank
                biasRows.append(b[rankStart ..< rankEnd])
            }
            shardedBias = concatenated(biasRows, axis: 0)
        }

        return AllToShardedLinear(
            preShardedWeight: shardedWeight,
            preShardedBias: shardedBias,
            group: g)
    }

    /// Subclass-friendly init that takes already-sharded weights.
    /// Used by `from(_:group:segments:)`.
    public init(preShardedWeight: MLXArray, preShardedBias: MLXArray?, group: Group) {
        self.weight = preShardedWeight
        self.bias = preShardedBias
        self.group = group
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let bias {
            return addMM(bias, x, weight.T)
        }
        return matmul(x, weight.T)
    }
}

/// Tensor-parallel linear layer: each rank holds a column-shard of
/// the weight, applies the local matmul on its slice of the input,
/// and the **partial** outputs are summed via all-reduce across the
/// group to produce the full output.
///
/// Mirror of Python's `mlx.nn.layers.distributed.ShardedToAllLinear`.
open class ShardedToAllLinear: Module, UnaryLayer {

    public let weight: MLXArray
    public let bias: MLXArray?
    public let group: Group

    public init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        bias: Bool = true,
        group: Group? = nil
    ) {
        let g = group ?? Group(strict: false)
        precondition(
            inputDimensions % g.size == 0,
            "ShardedToAllLinear: input_dims (\(inputDimensions)) must be divisible by group size (\(g.size))")

        let scale = sqrt(1.0 / Float(inputDimensions))
        let perRankIn = inputDimensions / g.size

        self.weight = MLXRandom.uniform(-scale ..< scale, [outputDimensions, perRankIn])
        if bias {
            // Bias is full-width and added once after the all-reduce.
            self.bias = MLXRandom.uniform(-scale ..< scale, [outputDimensions])
        } else {
            self.bias = nil
        }
        self.group = g
        super.init()
    }

    /// Build from a dense Linear by column-sharding its weight.
    /// `segments` mirrors the AllToSharded variant — useful for fused
    /// projections that need to be split before column-shard.
    public static func from(
        _ linear: Linear,
        group: Group? = nil,
        segments: Int = 1
    ) -> ShardedToAllLinear {
        let g = group ?? Group(strict: false)
        let (_, inDim) = linear.shape
        precondition(inDim % g.size == 0,
                     "ShardedToAllLinear.from: input_dims (\(inDim)) must be divisible by group size (\(g.size))")
        precondition(segments >= 1, "segments must be >= 1")

        let perSegmentIn = inDim / segments
        let perSegmentPerRank = perSegmentIn / g.size
        precondition(perSegmentIn % g.size == 0,
                     "(input_dims/segments) must be divisible by group size")

        var cols: [MLXArray] = []
        for seg in 0 ..< segments {
            let segStart = seg * perSegmentIn
            let rankStart = segStart + g.rank * perSegmentPerRank
            let rankEnd = rankStart + perSegmentPerRank
            cols.append(linear.weight[0..., rankStart ..< rankEnd])
        }
        let shardedWeight = concatenated(cols, axis: 1)

        return ShardedToAllLinear(
            preShardedWeight: shardedWeight,
            bias: linear.bias,
            group: g)
    }

    /// Subclass-friendly init for callers that already sharded weights.
    public init(preShardedWeight: MLXArray, bias: MLXArray?, group: Group) {
        self.weight = preShardedWeight
        self.bias = bias
        self.group = group
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let partial = matmul(x, weight.T)
        let summed = Collectives.allSum(partial, group: group)
        if let bias {
            return summed + bias
        }
        return summed
    }
}
