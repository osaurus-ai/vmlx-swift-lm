import XCTest
import MLX
import MLXNN
@testable import MLXDistributedTP

final class ShardingHelperTests: XCTestCase {

    private func makeLinear(in inDim: Int, out outDim: Int) -> Linear {
        let w = MLXArray(
            (0 ..< outDim * inDim).map { Float($0) * 0.001 - 0.05 },
            [outDim, inDim])
        let b = MLXArray((0 ..< outDim).map { Float($0) * 0.01 })
        return Linear(weight: w, bias: b)
    }

    func testShardLinearAllToShardedDispatches() {
        let dense = makeLinear(in: 4, out: 8)
        let sharded = shardLinear(dense, sharding: .allToSharded)
        XCTAssertTrue(sharded is AllToShardedLinear,
                      "sharding=.allToSharded should produce AllToShardedLinear")
    }

    func testShardLinearShardedToAllDispatches() {
        let dense = makeLinear(in: 4, out: 8)
        let sharded = shardLinear(dense, sharding: .shardedToAll)
        XCTAssertTrue(sharded is ShardedToAllLinear,
                      "sharding=.shardedToAll should produce ShardedToAllLinear")
    }

    func testShardLinearAllToShardedNumericMatchOnSize1() {
        let dense = makeLinear(in: 4, out: 8)
        let sharded = shardLinear(dense, sharding: .allToSharded)
        let x = MLXArray((0 ..< 12).map { Float($0) * 0.1 }, [3, 4])
        let yDense = dense(x)
        let ySharded = sharded(x)
        for (lhs, rhs) in zip(yDense.asArray(Float.self), ySharded.asArray(Float.self)) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-5)
        }
    }

    func testSharingEnumRawStringsMatchPython() {
        XCTAssertEqual(Sharding.allToSharded.rawValue, "all-to-sharded")
        XCTAssertEqual(Sharding.shardedToAll.rawValue, "sharded-to-all")
    }
}
