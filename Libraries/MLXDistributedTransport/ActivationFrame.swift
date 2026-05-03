import Foundation
import NIOCore

/// One framed message on the pipeline-parallel wire. Activations,
/// requests, tokens, and errors all share this envelope so the codec
/// stays trivial.
///
/// Wire format (network byte order, 24-byte header + payload):
/// ```
///   offset  size  field
///        0     4  magic           ("VMLX" = 0x564D4C58)
///        4     4  schemaVersion   (uint32)
///        8     4  frameType       (uint32, see FrameType raw values)
///       12     4  reserved        (zero in v1)
///       16     8  payloadLen      (uint64)
///       24    *   payload         (raw bytes)
/// ```
public struct ActivationFrame: Sendable, Equatable {
    public static let magic: UInt32 = 0x564D_4C58  // "VMLX"
    public static let schemaVersion: UInt32 = 1
    public static let headerSize: Int = 24

    public let frameType: FrameType
    public let payload: ByteBuffer

    public init(frameType: FrameType, payload: ByteBuffer) {
        self.frameType = frameType
        self.payload = payload
    }

    public enum FrameType: UInt32, Sendable, Equatable {
        /// Driver → first stage: prompt + sampling params.
        case prefillRequest = 1
        /// Driver → first stage: continue decoding.
        case decodeRequest = 2
        /// Stage N → Stage N+1: hidden states for the next layer block.
        case activationsForward = 3
        /// Last stage → driver: one or more output tokens.
        case tokenStream = 4
        /// Any rank → driver: structured failure (frame text payload).
        case error = 5
        /// Last stage → driver: end-of-stream for the current request.
        /// Sent after the final `tokenStream`, signals the caller can
        /// stop reading. Payload is empty.
        case tokensComplete = 6
    }
}

public enum ActivationFrameCodecError: Error, Equatable {
    case bufferTooShort(expected: Int, available: Int)
    case wrongMagic(found: UInt32)
    case unsupportedSchemaVersion(found: UInt32)
    case unknownFrameType(raw: UInt32)
    case payloadTruncated(declared: UInt64, available: Int)
}

public enum ActivationFrameCodec {

    /// Encode a frame into a fresh ByteBuffer.
    public static func encode(
        _ frame: ActivationFrame,
        allocator: ByteBufferAllocator = ByteBufferAllocator()
    ) -> ByteBuffer {
        var out = allocator.buffer(
            capacity: ActivationFrame.headerSize + frame.payload.readableBytes)
        out.writeInteger(ActivationFrame.magic, endianness: .big, as: UInt32.self)
        out.writeInteger(ActivationFrame.schemaVersion, endianness: .big, as: UInt32.self)
        out.writeInteger(frame.frameType.rawValue, endianness: .big, as: UInt32.self)
        out.writeInteger(UInt32(0), endianness: .big, as: UInt32.self)  // reserved
        out.writeInteger(UInt64(frame.payload.readableBytes), endianness: .big, as: UInt64.self)
        var payload = frame.payload
        out.writeBuffer(&payload)
        return out
    }

    /// Decode a frame from the front of `buffer`. On success the read
    /// cursor advances past the consumed bytes. On insufficient bytes,
    /// the buffer is left untouched and `bufferTooShort` is thrown so
    /// the caller can wait for more data.
    public static func decode(_ buffer: inout ByteBuffer) throws -> ActivationFrame {
        guard buffer.readableBytes >= ActivationFrame.headerSize else {
            throw ActivationFrameCodecError.bufferTooShort(
                expected: ActivationFrame.headerSize,
                available: buffer.readableBytes)
        }

        let savedReader = buffer.readerIndex

        let magic: UInt32 = buffer.readInteger(endianness: .big)!
        guard magic == ActivationFrame.magic else {
            buffer.moveReaderIndex(to: savedReader)
            throw ActivationFrameCodecError.wrongMagic(found: magic)
        }

        let version: UInt32 = buffer.readInteger(endianness: .big)!
        guard version == ActivationFrame.schemaVersion else {
            buffer.moveReaderIndex(to: savedReader)
            throw ActivationFrameCodecError.unsupportedSchemaVersion(found: version)
        }

        let typeRaw: UInt32 = buffer.readInteger(endianness: .big)!
        guard let frameType = ActivationFrame.FrameType(rawValue: typeRaw) else {
            buffer.moveReaderIndex(to: savedReader)
            throw ActivationFrameCodecError.unknownFrameType(raw: typeRaw)
        }

        let _: UInt32 = buffer.readInteger(endianness: .big)!  // reserved
        let payloadLen: UInt64 = buffer.readInteger(endianness: .big)!

        guard payloadLen <= UInt64(Int.max),
              buffer.readableBytes >= Int(payloadLen) else {
            buffer.moveReaderIndex(to: savedReader)
            throw ActivationFrameCodecError.payloadTruncated(
                declared: payloadLen,
                available: buffer.readableBytes)
        }

        let payload = buffer.readSlice(length: Int(payloadLen))!
        return ActivationFrame(frameType: frameType, payload: payload)
    }
}
