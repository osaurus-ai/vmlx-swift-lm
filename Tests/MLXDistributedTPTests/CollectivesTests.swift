import XCTest
import MLX
@testable import MLXDistributedTP

final class CollectivesSingleRankTests: XCTestCase {

    /// On a size-1 group every collective is a no-op — the input array
    /// passes through unchanged. Same semantics as the Python reference.

    func testAllSumIsIdentityOnSize1Group() {
        let g = Group(strict: false)
        let x = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
        let y = Collectives.allSum(x, group: g)
        XCTAssertEqual(y.asArray(Float.self), [1.0, 2.0, 3.0, 4.0])
    }

    func testAllGatherIsIdentityOnSize1Group() {
        let g = Group(strict: false)
        let x = MLXArray([7.0, 8.0] as [Float])
        let y = Collectives.allGather(x, group: g)
        XCTAssertEqual(y.asArray(Float.self), [7.0, 8.0])
    }

    func testSumScatterIsIdentityOnSize1Group() {
        let g = Group(strict: false)
        let x = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float])
        let y = Collectives.sumScatter(x, group: g)
        XCTAssertEqual(y.asArray(Float.self), [1.0, 2.0, 3.0, 4.0])
    }

    func testSendIsIdentityOnSize1Group() {
        let g = Group(strict: false)
        let x = MLXArray([42.0] as [Float])
        let y = Collectives.send(x, to: 0, group: g)
        XCTAssertEqual(y.asArray(Float.self), [42.0])
    }

    func testRecvLikeIsIdentityOnSize1Group() {
        let g = Group(strict: false)
        let like = MLXArray([0.0, 0.0, 0.0] as [Float])
        let y = Collectives.recvLike(like, from: 0, group: g)
        XCTAssertEqual(y.asArray(Float.self), [0.0, 0.0, 0.0])
    }
}
