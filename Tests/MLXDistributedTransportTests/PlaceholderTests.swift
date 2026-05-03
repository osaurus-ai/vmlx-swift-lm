import XCTest
@testable import MLXDistributedTransport

final class TransportPlaceholderTests: XCTestCase {
    func testTransportTargetCompilesAndLinksNIO() {
        XCTAssertTrue(MLXDistributedTransportPlaceholder.sanityCheckNIO())
    }
}
