import XCTest
@testable import MLXDistributedTP

final class GroupTests: XCTestCase {
    /// In a single-process build with no MLX_RANK/MLX_IBV_DEVICES env,
    /// `init(strict: false)` returns a degenerate size-1 group — same
    /// semantics as `mx.distributed.init()` in Python.
    func testNonStrictInitReturnsSize1Group() {
        let g = Group(strict: false)
        XCTAssertEqual(g.size, 1)
        XCTAssertEqual(g.rank, 0)
        XCTAssertFalse(g.isMultiRank)
    }

    func testSplitOnSize1GroupReturnsSelf() {
        // Splitting a size-1 group is a no-op in our wrapper: mlx-c
        // refuses ("cannot split further") and returns an empty handle,
        // which the wrapper re-maps to the original group rather than
        // exposing the malformed empty handle.
        let g = Group(strict: false)
        let sub = g.split(color: 0, key: 0)
        XCTAssertEqual(sub.size, 1)
        XCTAssertEqual(sub.rank, 0)
    }

    // strict=true behavior is intentionally NOT tested here: mlx-c
    // catches std::exception inside init and prints to mlx_error
    // (stderr) before returning a size-1 fallback group. The print
    // output disrupts XCTest's reporter pipeline. Exercising strict
    // init belongs in a multi-rank end-to-end test (Phase 6) where
    // env vars are actually set and the success path runs.
}
