//
//  InternalMessage.swift
//  osaurus
//
//  Extracted from MLXService for reuse across services.
//

import Foundation

/// Message role for chat interactions
public enum MessageRole: String, Codable, Sendable {
    case system
    case user
    case assistant
    case tool
}

/// Chat message structure
public struct Message: Codable, Sendable {
    public let role: MessageRole
    public let content: String

    public init(role: MessageRole, content: String) {
        self.role = role
        self.content = content
    }
}
