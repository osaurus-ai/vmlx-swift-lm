import XCTest
import NIOCore
@testable import MLXDistributedTransport

final class WireStageHandlerTests: XCTestCase {
    func testPrefillFrameCallsRuntimeAndEncodesResponses() async throws {
        let requestID = UUID(uuidString: "33333333-3333-3333-3333-333333333333")!
        let runtime = RecordingWireStageRuntime()
        runtime.prefillResponses = [
            .tokenStream(WireTokenStreamPayload(requestID: requestID, text: "hi", tokenIDs: [1])),
            .tokensComplete(WireTokensCompletePayload(requestID: requestID, reason: "stop"))
        ]
        let handler = WireStageHandler(runtime: runtime)
        let payload = try WirePrefillRequestPayload(
            requestID: requestID,
            prompt: "hello",
            maxTokens: 8,
            cache: WireCacheIdentity(
                modelHash: "0123456789abcdef",
                reasoningMode: "off",
                toolMode: "none"
            ),
            stage: nil
        )

        let replies = await collect(handler.handle(ActivationFrame(
            frameType: .prefillRequest,
            payload: try WirePayloadCodec.encode(payload)
        )))

        XCTAssertEqual(runtime.prefillRequests, [payload])
        XCTAssertEqual(replies.map(\.frameType), [.tokenStream, .tokensComplete])
        XCTAssertEqual(
            try WirePayloadCodec.decode(WireTokenStreamPayload.self, from: replies[0].payload),
            WireTokenStreamPayload(requestID: requestID, text: "hi", tokenIDs: [1])
        )
        XCTAssertEqual(
            try WirePayloadCodec.decode(WireTokensCompletePayload.self, from: replies[1].payload),
            WireTokensCompletePayload(requestID: requestID, reason: "stop")
        )
    }

    func testActivationFrameCallsRuntimeAndCanForwardActivation() async throws {
        let requestID = UUID(uuidString: "44444444-4444-4444-4444-444444444444")!
        let runtime = RecordingWireStageRuntime()
        let activation = try makeActivation(requestID: requestID, fromRank: 0, toRank: 1)
        let forwarded = try makeActivation(requestID: requestID, fromRank: 1, toRank: 0)
        runtime.activationResponses = [.activationForward(forwarded)]
        let handler = WireStageHandler(runtime: runtime)

        let replies = await collect(handler.handle(ActivationFrame(
            frameType: .activationsForward,
            payload: try WirePayloadCodec.encode(activation)
        )))

        XCTAssertEqual(runtime.activationRequests, [activation])
        XCTAssertEqual(replies.map(\.frameType), [.activationsForward])
        XCTAssertEqual(
            try WirePayloadCodec.decode(WireActivationForwardPayload.self, from: replies[0].payload),
            forwarded
        )
    }

    func testInvalidTypedPayloadReturnsErrorFrame() async throws {
        let runtime = RecordingWireStageRuntime()
        let handler = WireStageHandler(runtime: runtime)
        var bad = ByteBufferAllocator().buffer(capacity: 3)
        bad.writeString("{")

        let replies = await collect(handler.handle(ActivationFrame(
            frameType: .prefillRequest,
            payload: bad
        )))

        XCTAssertTrue(runtime.prefillRequests.isEmpty)
        XCTAssertEqual(replies.map(\.frameType), [.error])
        var payload = replies[0].payload
        XCTAssertTrue(
            (payload.readString(length: payload.readableBytes) ?? "").contains("failed to decode prefillRequest")
        )
    }

    func testUnsupportedFrameTypeReturnsErrorFrame() async throws {
        let runtime = RecordingWireStageRuntime()
        let handler = WireStageHandler(runtime: runtime)

        let replies = await collect(handler.handle(ActivationFrame(
            frameType: .decodeRequest,
            payload: ByteBuffer()
        )))

        XCTAssertEqual(replies.map(\.frameType), [.error])
        var payload = replies[0].payload
        XCTAssertTrue(
            (payload.readString(length: payload.readableBytes) ?? "").contains("unsupported stage frame type")
        )
    }
}

private final class RecordingWireStageRuntime: WireStageRuntime, @unchecked Sendable {
    var prefillRequests: [WirePrefillRequestPayload] = []
    var activationRequests: [WireActivationForwardPayload] = []
    var prefillResponses: [WireStageResponse] = []
    var activationResponses: [WireStageResponse] = []

    func handlePrefill(_ request: WirePrefillRequestPayload) -> AsyncStream<WireStageResponse> {
        prefillRequests.append(request)
        return stream(prefillResponses)
    }

    func handleActivation(_ activation: WireActivationForwardPayload) -> AsyncStream<WireStageResponse> {
        activationRequests.append(activation)
        return stream(activationResponses)
    }
}

private func makeActivation(
    requestID: UUID,
    fromRank: Int,
    toRank: Int
) throws -> WireActivationForwardPayload {
    try WireActivationForwardPayload(
        requestID: requestID,
        fromRank: fromRank,
        toRank: toRank,
        layers: WireLayerRange(start: 0, endExclusive: 4),
        tensor: WireTensorDescriptor(
            name: "hidden_states",
            dtype: .float16,
            shape: [1, 2, 8],
            byteCount: 32
        ),
        cache: WireCacheIdentity(
            modelHash: "0123456789abcdef",
            reasoningMode: "off",
            toolMode: "none"
        ),
        bytes: Data([1, 2, 3, 4])
    )
}

private func stream(_ responses: [WireStageResponse]) -> AsyncStream<WireStageResponse> {
    AsyncStream { continuation in
        for response in responses {
            continuation.yield(response)
        }
        continuation.finish()
    }
}

private func collect(_ stream: AsyncStream<ActivationFrame>) async -> [ActivationFrame] {
    var frames: [ActivationFrame] = []
    for await frame in stream {
        frames.append(frame)
    }
    return frames
}
