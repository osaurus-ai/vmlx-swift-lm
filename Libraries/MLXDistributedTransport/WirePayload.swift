import Foundation
import NIOCore

public enum WirePayloadValidationError: Error, Equatable, Sendable {
    case invalidLayerRange(start: Int, endExclusive: Int)
    case invalidRank(rank: Int, worldSize: Int)
    case invalidTensorShape([Int])
    case invalidByteCount(Int)
    case invalidMaxTokens(Int)
    case invalidHexHash(field: String, value: String)
}

public enum WirePayloadCodecError: Error, Equatable, Sendable {
    case encodingFailed(String)
    case decodingFailed(String)
}

public enum WirePayloadCodec {
    public static func encode<T: Encodable>(
        _ payload: T,
        allocator: ByteBufferAllocator = ByteBufferAllocator()
    ) throws -> ByteBuffer {
        do {
            let data = try JSONEncoder.wire.encode(payload)
            var buffer = allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            return buffer
        } catch {
            throw WirePayloadCodecError.encodingFailed(String(describing: error))
        }
    }

    public static func decode<T: Decodable>(
        _ type: T.Type,
        from payload: ByteBuffer
    ) throws -> T {
        do {
            let data = Data(payload.readableBytesView)
            return try JSONDecoder.wire.decode(type, from: data)
        } catch {
            throw WirePayloadCodecError.decodingFailed(String(describing: error))
        }
    }
}

public struct WireLayerRange: Codable, Equatable, Sendable {
    public let start: Int
    public let endExclusive: Int

    public init(start: Int, endExclusive: Int) throws {
        guard start >= 0, endExclusive > start else {
            throw WirePayloadValidationError.invalidLayerRange(
                start: start,
                endExclusive: endExclusive
            )
        }
        self.start = start
        self.endExclusive = endExclusive
    }

    public var count: Int { endExclusive - start }
}

public enum WireDType: String, Codable, Equatable, Sendable {
    case float16
    case bfloat16
    case float32
    case int32
    case uint32
    case int64
    case uint8
}

public struct WireTensorDescriptor: Codable, Equatable, Sendable {
    public let name: String?
    public let dtype: WireDType
    public let shape: [Int]
    public let byteCount: Int?

    public init(
        name: String?,
        dtype: WireDType,
        shape: [Int],
        byteCount: Int? = nil
    ) throws {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw WirePayloadValidationError.invalidTensorShape(shape)
        }
        if let byteCount, byteCount < 0 {
            throw WirePayloadValidationError.invalidByteCount(byteCount)
        }
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.byteCount = byteCount
    }
}

public struct WireStageDescriptor: Codable, Equatable, Sendable {
    public let rank: Int
    public let worldSize: Int
    public let layers: WireLayerRange
    public let input: WireTensorDescriptor
    public let output: WireTensorDescriptor

    public init(
        rank: Int,
        worldSize: Int,
        layers: WireLayerRange,
        input: WireTensorDescriptor,
        output: WireTensorDescriptor
    ) throws {
        guard worldSize > 0, rank >= 0, rank < worldSize else {
            throw WirePayloadValidationError.invalidRank(rank: rank, worldSize: worldSize)
        }
        self.rank = rank
        self.worldSize = worldSize
        self.layers = layers
        self.input = input
        self.output = output
    }
}

public struct WireCacheIdentity: Codable, Equatable, Sendable {
    public let modelHash: String
    public let tokenizerHash: String?
    public let chatTemplateHash: String?
    public let reasoningMode: String
    public let toolMode: String
    public let mediaSalt: String?
    public let familyStateHash: String?

    public init(
        modelHash: String,
        tokenizerHash: String? = nil,
        chatTemplateHash: String? = nil,
        reasoningMode: String,
        toolMode: String,
        mediaSalt: String? = nil,
        familyStateHash: String? = nil
    ) throws {
        try Self.validateHash(field: "modelHash", value: modelHash)
        if let tokenizerHash {
            try Self.validateHash(field: "tokenizerHash", value: tokenizerHash)
        }
        if let chatTemplateHash {
            try Self.validateHash(field: "chatTemplateHash", value: chatTemplateHash)
        }
        if let familyStateHash {
            try Self.validateHash(field: "familyStateHash", value: familyStateHash)
        }
        self.modelHash = modelHash.lowercased()
        self.tokenizerHash = tokenizerHash?.lowercased()
        self.chatTemplateHash = chatTemplateHash?.lowercased()
        self.reasoningMode = reasoningMode
        self.toolMode = toolMode
        self.mediaSalt = mediaSalt
        self.familyStateHash = familyStateHash?.lowercased()
    }

    private static func validateHash(field: String, value: String) throws {
        guard (16...64).contains(value.count),
              value.allSatisfy({ $0.isHexDigit }) else {
            throw WirePayloadValidationError.invalidHexHash(field: field, value: value)
        }
    }
}

public struct WirePrefillRequestPayload: Codable, Equatable, Sendable {
    public let requestID: UUID
    public let prompt: String
    public let maxTokens: Int
    public let cache: WireCacheIdentity
    public let stage: WireStageDescriptor?

    public init(
        requestID: UUID,
        prompt: String,
        maxTokens: Int,
        cache: WireCacheIdentity,
        stage: WireStageDescriptor?
    ) throws {
        guard maxTokens > 0 else {
            throw WirePayloadValidationError.invalidMaxTokens(maxTokens)
        }
        self.requestID = requestID
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.cache = cache
        self.stage = stage
    }
}

public struct WireActivationForwardPayload: Codable, Equatable, Sendable {
    public let requestID: UUID
    public let fromRank: Int
    public let toRank: Int
    public let layers: WireLayerRange
    public let tensor: WireTensorDescriptor
    public let cache: WireCacheIdentity
    public let bytes: Data

    public init(
        requestID: UUID,
        fromRank: Int,
        toRank: Int,
        layers: WireLayerRange,
        tensor: WireTensorDescriptor,
        cache: WireCacheIdentity,
        bytes: Data
    ) throws {
        guard fromRank >= 0, toRank >= 0, fromRank != toRank else {
            throw WirePayloadValidationError.invalidRank(rank: toRank, worldSize: max(fromRank, toRank))
        }
        self.requestID = requestID
        self.fromRank = fromRank
        self.toRank = toRank
        self.layers = layers
        self.tensor = tensor
        self.cache = cache
        self.bytes = bytes
    }
}

public struct WireTokenStreamPayload: Codable, Equatable, Sendable {
    public let requestID: UUID
    public let text: String
    public let tokenIDs: [Int]

    public init(requestID: UUID, text: String, tokenIDs: [Int] = []) {
        self.requestID = requestID
        self.text = text
        self.tokenIDs = tokenIDs
    }
}

public struct WireTokensCompletePayload: Codable, Equatable, Sendable {
    public let requestID: UUID
    public let reason: String

    public init(requestID: UUID, reason: String) {
        self.requestID = requestID
        self.reason = reason
    }
}

private extension JSONEncoder {
    static var wire: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static var wire: JSONDecoder {
        JSONDecoder()
    }
}
