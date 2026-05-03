import XCTest
import NIOCore
@testable import MLXDistributedTransport

final class ActivationFrameCodecTests: XCTestCase {

    private func bb(_ bytes: [UInt8]) -> ByteBuffer {
        var b = ByteBufferAllocator().buffer(capacity: bytes.count)
        b.writeBytes(bytes)
        return b
    }

    func testRoundTripEmptyPayload() throws {
        let frame = ActivationFrame(
            frameType: .decodeRequest,
            payload: ByteBufferAllocator().buffer(capacity: 0))
        var encoded = ActivationFrameCodec.encode(frame)
        XCTAssertEqual(encoded.readableBytes, ActivationFrame.headerSize)

        let decoded = try ActivationFrameCodec.decode(&encoded)
        XCTAssertEqual(decoded.frameType, .decodeRequest)
        XCTAssertEqual(decoded.payload.readableBytes, 0)
        XCTAssertEqual(encoded.readableBytes, 0, "all bytes consumed")
    }

    func testRoundTripWithPayload() throws {
        let payload = bb([0xDE, 0xAD, 0xBE, 0xEF, 0x42])
        let frame = ActivationFrame(frameType: .activationsForward, payload: payload)
        var encoded = ActivationFrameCodec.encode(frame)

        XCTAssertEqual(encoded.readableBytes, ActivationFrame.headerSize + 5)
        let decoded = try ActivationFrameCodec.decode(&encoded)
        XCTAssertEqual(decoded.frameType, .activationsForward)
        XCTAssertEqual(Array(decoded.payload.readableBytesView), [0xDE, 0xAD, 0xBE, 0xEF, 0x42])
    }

    func testTwoFramesBackToBack() throws {
        let p1 = bb([0xAA, 0xBB])
        let p2 = bb([0xCC])
        var combined = ActivationFrameCodec.encode(
            ActivationFrame(frameType: .prefillRequest, payload: p1))
        var second = ActivationFrameCodec.encode(
            ActivationFrame(frameType: .tokenStream, payload: p2))
        combined.writeBuffer(&second)

        let f1 = try ActivationFrameCodec.decode(&combined)
        let f2 = try ActivationFrameCodec.decode(&combined)
        XCTAssertEqual(f1.frameType, .prefillRequest)
        XCTAssertEqual(Array(f1.payload.readableBytesView), [0xAA, 0xBB])
        XCTAssertEqual(f2.frameType, .tokenStream)
        XCTAssertEqual(Array(f2.payload.readableBytesView), [0xCC])
        XCTAssertEqual(combined.readableBytes, 0)
    }

    func testTooShortHeaderThrowsAndPreservesBuffer() {
        var buf = bb([0x56, 0x4D, 0x4C, 0x58])  // 4 bytes, well below 24
        let savedReader = buf.readerIndex
        XCTAssertThrowsError(try ActivationFrameCodec.decode(&buf)) { err in
            guard case ActivationFrameCodecError.bufferTooShort = err else {
                return XCTFail("expected bufferTooShort, got \(err)")
            }
        }
        XCTAssertEqual(buf.readerIndex, savedReader,
                       "decoder must not consume bytes on insufficient input")
    }

    func testWrongMagicRejectedAndCursorRestored() {
        var buf = ByteBufferAllocator().buffer(capacity: ActivationFrame.headerSize)
        buf.writeInteger(UInt32(0xBAD0_F00D), endianness: .big, as: UInt32.self)
        buf.writeInteger(ActivationFrame.schemaVersion, endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(1), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(0), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt64(0), endianness: .big, as: UInt64.self)

        let savedReader = buf.readerIndex
        XCTAssertThrowsError(try ActivationFrameCodec.decode(&buf)) { err in
            guard case ActivationFrameCodecError.wrongMagic = err else {
                return XCTFail("expected wrongMagic")
            }
        }
        XCTAssertEqual(buf.readerIndex, savedReader)
    }

    func testUnsupportedVersionRejected() {
        var buf = ByteBufferAllocator().buffer(capacity: ActivationFrame.headerSize)
        buf.writeInteger(ActivationFrame.magic, endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(99), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(1), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(0), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt64(0), endianness: .big, as: UInt64.self)
        XCTAssertThrowsError(try ActivationFrameCodec.decode(&buf)) { err in
            guard case ActivationFrameCodecError.unsupportedSchemaVersion = err else {
                return XCTFail("expected unsupportedSchemaVersion")
            }
        }
    }

    func testUnknownFrameTypeRejected() {
        var buf = ByteBufferAllocator().buffer(capacity: ActivationFrame.headerSize)
        buf.writeInteger(ActivationFrame.magic, endianness: .big, as: UInt32.self)
        buf.writeInteger(ActivationFrame.schemaVersion, endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(999), endianness: .big, as: UInt32.self)  // not a FrameType
        buf.writeInteger(UInt32(0), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt64(0), endianness: .big, as: UInt64.self)
        XCTAssertThrowsError(try ActivationFrameCodec.decode(&buf)) { err in
            guard case ActivationFrameCodecError.unknownFrameType = err else {
                return XCTFail("expected unknownFrameType")
            }
        }
    }

    func testPayloadTruncatedRejectedAndCursorRestored() {
        var buf = ByteBufferAllocator().buffer(capacity: ActivationFrame.headerSize + 2)
        buf.writeInteger(ActivationFrame.magic, endianness: .big, as: UInt32.self)
        buf.writeInteger(ActivationFrame.schemaVersion, endianness: .big, as: UInt32.self)
        buf.writeInteger(ActivationFrame.FrameType.prefillRequest.rawValue,
                         endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt32(0), endianness: .big, as: UInt32.self)
        buf.writeInteger(UInt64(10), endianness: .big, as: UInt64.self)  // claim 10 bytes
        buf.writeBytes([0x01, 0x02])  // only 2 actually present
        let savedReader = buf.readerIndex
        XCTAssertThrowsError(try ActivationFrameCodec.decode(&buf)) { err in
            guard case ActivationFrameCodecError.payloadTruncated = err else {
                return XCTFail("expected payloadTruncated, got \(err)")
            }
        }
        XCTAssertEqual(buf.readerIndex, savedReader,
                       "decoder must not consume bytes on truncated payload")
    }
}
