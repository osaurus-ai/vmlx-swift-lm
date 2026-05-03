import XCTest
@testable import MLXDistributedCore

final class ModeTests: XCTestCase {
    func testModeRawCSV() {
        XCTAssertEqual(Mode.replica.rawCSV, "replica")
        XCTAssertEqual(Mode.pipelined.rawCSV, "pp")
        XCTAssertEqual(Mode.wired.rawCSV, "tp")
        XCTAssertNil(Mode.auto.rawCSV)
        XCTAssertNil(Mode.localOnly.rawCSV)
    }

    func testModeFromCSV() {
        XCTAssertEqual(Mode(rawCSV: "replica"), .replica)
        XCTAssertEqual(Mode(rawCSV: "pp"), .pipelined)
        XCTAssertEqual(Mode(rawCSV: "tp"), .wired)
        XCTAssertNil(Mode(rawCSV: "auto"),
                     "auto/localOnly are caller-side, never advertised")
        XCTAssertNil(Mode(rawCSV: "garbage"))
    }

    func testNotImplementedYetIsAnError() {
        let err: Error = DistributionError.notImplementedYet(.replica)
        XCTAssertNotNil(err as? DistributionError)
        guard case let DistributionError.notImplementedYet(mode)? = err as? DistributionError else {
            XCTFail("expected notImplementedYet")
            return
        }
        XCTAssertEqual(mode, .replica)
    }
}
