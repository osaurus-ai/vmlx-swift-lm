import XCTest
import NIOCore
@testable import MLXDistributedTransport

final class ActivationPayloadTests: XCTestCase {

    func testPrefillPayloadRoundTripPreservesStageAndCacheMetadata() throws {
        let cache = try WireCacheIdentity(
            modelHash: "0123456789abcdef",
            tokenizerHash: "aaaaaaaaaaaaaaaa",
            chatTemplateHash: "bbbbbbbbbbbbbbbb",
            reasoningMode: "off",
            toolMode: "none",
            mediaSalt: nil,
            familyStateHash: "cccccccccccccccc"
        )
        let stage = try WireStageDescriptor(
            rank: 1,
            worldSize: 2,
            layers: WireLayerRange(start: 24, endExclusive: 48),
            input: WireTensorDescriptor(name: "hidden_states", dtype: .float16, shape: [1, 128, 4096]),
            output: WireTensorDescriptor(name: "hidden_states", dtype: .float16, shape: [1, 128, 4096])
        )
        let payload = try WirePrefillRequestPayload(
            requestID: UUID(uuidString: "11111111-1111-1111-1111-111111111111")!,
            prompt: "hello distributed world",
            maxTokens: 32,
            cache: cache,
            stage: stage
        )

        let buffer = try WirePayloadCodec.encode(payload)
        let decoded = try WirePayloadCodec.decode(WirePrefillRequestPayload.self, from: buffer)

        XCTAssertEqual(decoded, payload)
        XCTAssertEqual(decoded.stage?.layers.count, 24)
    }

    func testRejectsInvalidLayerRange() {
        XCTAssertThrowsError(try WireLayerRange(start: 7, endExclusive: 7)) { error in
            XCTAssertEqual(error as? WirePayloadValidationError, .invalidLayerRange(start: 7, endExclusive: 7))
        }
    }

    func testRejectsInvalidTensorShape() {
        XCTAssertThrowsError(try WireTensorDescriptor(
            name: "bad",
            dtype: .float16,
            shape: [1, 0, 4096]
        )) { error in
            XCTAssertEqual(error as? WirePayloadValidationError, .invalidTensorShape([1, 0, 4096]))
        }
    }

    func testRejectsInvalidCacheHash() {
        XCTAssertThrowsError(try WireCacheIdentity(
            modelHash: "not-a-hash",
            tokenizerHash: "aaaaaaaaaaaaaaaa",
            chatTemplateHash: "bbbbbbbbbbbbbbbb",
            reasoningMode: "off",
            toolMode: "none"
        )) { error in
            XCTAssertEqual(error as? WirePayloadValidationError, .invalidHexHash(field: "modelHash", value: "not-a-hash"))
        }
    }

    func testActivationPayloadRoundTripCarriesBinaryBytes() throws {
        let cache = try WireCacheIdentity(
            modelHash: "0123456789abcdef",
            tokenizerHash: "aaaaaaaaaaaaaaaa",
            chatTemplateHash: "bbbbbbbbbbbbbbbb",
            reasoningMode: "off",
            toolMode: "none"
        )
        let payload = try WireActivationForwardPayload(
            requestID: UUID(uuidString: "22222222-2222-2222-2222-222222222222")!,
            fromRank: 0,
            toRank: 1,
            layers: WireLayerRange(start: 0, endExclusive: 24),
            tensor: WireTensorDescriptor(
                name: "hidden_states",
                dtype: .bfloat16,
                shape: [1, 64, 4096],
                byteCount: 524_288
            ),
            cache: cache,
            bytes: Data([0xde, 0xad, 0xbe, 0xef])
        )

        let buffer = try WirePayloadCodec.encode(payload)
        let decoded = try WirePayloadCodec.decode(WireActivationForwardPayload.self, from: buffer)

        XCTAssertEqual(decoded, payload)
        XCTAssertEqual(decoded.bytes, Data([0xde, 0xad, 0xbe, 0xef]))
    }
}
