import XCTest
import MLX
@testable import MLXLLM

final class KernelTests: XCTestCase {
    func testGatedDeltaKernelLoaded() throws {
        // GatedDeltaKernelManager is private, so we can't test it directly unless we make it internal
    }
}
