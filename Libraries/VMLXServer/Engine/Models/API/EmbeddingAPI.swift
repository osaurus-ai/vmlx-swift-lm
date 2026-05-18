//
//  EmbeddingAPI.swift
//  osaurus
//
//  Request/response models for embedding endpoints.
//  Supports both OpenAI (/v1/embeddings) and Ollama (/api/embed) formats.
//

import Foundation

// MARK: - Shared Input Type

/// Decodes both `"single string"` and `["array", "of", "strings"]` from JSON.
public enum EmbeddingInput: Codable, Sendable {
    case single(String)
    case multiple([String])

    public var texts: [String] {
        switch self {
        case .single(let s): [s]
        case .multiple(let a): a
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .single(str)
        } else if let arr = try? container.decode([String].self) {
            self = .multiple(arr)
        } else {
            throw DecodingError.typeMismatch(
                EmbeddingInput.self,
                .init(codingPath: decoder.codingPath, debugDescription: "Expected a string or array of strings")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let s): try container.encode(s)
        case .multiple(let a): try container.encode(a)
        }
    }
}

// MARK: - Shared Request

/// Both OpenAI and Ollama endpoints accept the same request shape.
public struct EmbeddingRequest: Codable, Sendable {
    public let model: String
    public let input: EmbeddingInput
    public let encoding_format: String?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(String.self, forKey: .model)
        input = try c.decode(EmbeddingInput.self, forKey: .input)
        encoding_format = try c.decodeIfPresent(String.self, forKey: .encoding_format)
    }
}

// MARK: - OpenAI Response (/v1/embeddings)

public struct OpenAIEmbeddingResponse: Codable, Sendable {
    public let object: String
    public let data: [OpenAIEmbeddingObject]
    public let model: String
    public let usage: OpenAIEmbeddingUsage

    public init(data: [OpenAIEmbeddingObject], model: String, usage: OpenAIEmbeddingUsage) {
        self.object = "list"
        self.data = data
        self.model = model
        self.usage = usage
    }
}

public struct OpenAIEmbeddingObject: Codable, Sendable {
    public let object: String
    public let embedding: [Float]
    public let index: Int

    public init(embedding: [Float], index: Int) {
        self.object = "embedding"
        self.embedding = embedding
        self.index = index
    }
}

public struct OpenAIEmbeddingUsage: Codable, Sendable {
    public let prompt_tokens: Int
    public let total_tokens: Int
}

// MARK: - Ollama Response (/api/embed)

public struct OllamaEmbedResponse: Codable, Sendable {
    public let model: String
    public let embeddings: [[Float]]
}
