import XCTest
import Darwin  // dlopen
@testable import MLXDistributedJACCL

final class JACCLAvailabilityTests: XCTestCase {

    /// Probe whether `librdma.dylib` is loadable on this host. macOS 11+
    /// keeps system libraries in the dyld shared cache, so a file-system
    /// existence check returns false even when dlopen succeeds. The
    /// JACCL C++ uses dlopen internally; mirror that.
    private func librdmaAvailable() -> Bool {
        if let h = dlopen("librdma.dylib", RTLD_LAZY) {
            dlclose(h)
            return true
        }
        return false
    }

    /// Smoke test for Phase 4 build infra: the JACCL backend must report
    /// available on macOS 26.3+ where Apple ships `librdma.dylib` in the
    /// dyld shared cache. If this fails on a machine where dlopen of the
    /// library succeeds, the upstream `osaurus-ai/mlx-swift` build flag
    /// is regressed — the four jaccl/ring `.cpp` files must NOT be in the
    /// Cmlx target's exclude list, AND the corresponding `no_*.cpp`
    /// stubs MUST be excluded.
    func testJACCLReportsAvailableOnMacOS263OrLater() throws {
        guard librdmaAvailable() else {
            throw XCTSkip("librdma.dylib not loadable on this host")
        }
        XCTAssertTrue(JACCL.isAvailable(),
                      "librdma loadable but JACCL.isAvailable() returned false")
    }

    func testAnyBackendIsAvailableWhenJACCLIs() throws {
        guard librdmaAvailable() else {
            throw XCTSkip("librdma.dylib not loadable")
        }
        XCTAssertTrue(JACCL.anyBackendAvailable(),
                      "any-backend probe should agree with the JACCL probe")
    }
}
