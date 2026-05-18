//
//  ChatEngineProtocol.swift
//  osaurus
//

import Foundation

public protocol ChatEngineProtocol: Sendable {
    func streamChat(request: ChatCompletionRequest) async throws -> AsyncThrowingStream<String, Error>
    func completeChat(request: ChatCompletionRequest) async throws -> ChatCompletionResponse
}

/// Classified error thrown by chat-engine implementations so the HTTP
/// layer can emit 4xx/5xx instead of a generic 500. Implementations
/// throw cases here; engine HTTPHandler catches by this type.
public struct ChatEngineError: Error, LocalizedError {
    public enum Kind: Sendable {
        case modelNotFound(requested: String)
        case noServiceAvailable(requested: String)
    }

    public let kind: Kind

    public init(kind: Kind) {
        self.kind = kind
    }

    public var errorDescription: String? {
        switch kind {
        case .modelNotFound(let requested):
            return "Model '\(requested)' is not installed or registered with any provider."
        case .noServiceAvailable(let requested):
            return "No service is currently available to handle model '\(requested)'."
        }
    }

    public var httpStatus: Int {
        switch kind {
        case .modelNotFound: return 404
        case .noServiceAvailable: return 503
        }
    }
}
