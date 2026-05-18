//
//  AnthropicAPI.swift
//  osaurus
//
//  Anthropic Messages API compatible request/response models for SDK-compatible clients.
//

import Foundation

// MARK: - Request Models

/// Anthropic Messages API request
public struct AnthropicMessagesRequest: Codable, Sendable {
    public let model: String
    public let max_tokens: Int
    public let system: AnthropicSystemContent?
    public let messages: [AnthropicMessage]
    public let stream: Bool?
    public let temperature: Double?
    public let top_p: Double?
    public let top_k: Int?
    public let stop_sequences: [String]?
    public let tools: [AnthropicTool]?
    public let tool_choice: AnthropicToolChoice?
    public let metadata: AnthropicMetadata?

    public init(
        model: String,
        max_tokens: Int,
        system: AnthropicSystemContent? = nil,
        messages: [AnthropicMessage],
        stream: Bool? = nil,
        temperature: Double? = nil,
        top_p: Double? = nil,
        top_k: Int? = nil,
        stop_sequences: [String]? = nil,
        tools: [AnthropicTool]? = nil,
        tool_choice: AnthropicToolChoice? = nil,
        metadata: AnthropicMetadata? = nil
    ) {
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.messages = messages
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self.tools = tools
        self.tool_choice = tool_choice
        self.metadata = metadata
    }
}

/// System content can be a string or array of content blocks
public enum AnthropicSystemContent: Codable, Sendable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicSystemContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or [AnthropicContentBlock]"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text):
            try container.encode(text)
        case .blocks(let blocks):
            try container.encode(blocks)
        }
    }

    /// Extract plain text from system content
    public var plainText: String {
        switch self {
        case .text(let text):
            return text
        case .blocks(let blocks):
            return blocks.compactMap { block in
                if case .text(let textBlock) = block {
                    return textBlock.text
                }
                return nil
            }.joined(separator: "\n")
        }
    }
}

/// Anthropic message in conversation
public struct AnthropicMessage: Codable, Sendable {
    public let role: String  // "user" or "assistant"
    public let content: AnthropicMessageContent

    public init(role: String, content: AnthropicMessageContent) {
        self.role = role
        self.content = content
    }
}

/// Message content can be a string or array of content blocks
public enum AnthropicMessageContent: Codable, Sendable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicMessageContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or [AnthropicContentBlock]"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text):
            try container.encode(text)
        case .blocks(let blocks):
            try container.encode(blocks)
        }
    }

    /// Extract plain text from message content
    public var plainText: String {
        switch self {
        case .text(let text):
            return text
        case .blocks(let blocks):
            return blocks.compactMap { block in
                if case .text(let textBlock) = block {
                    return textBlock.text
                }
                return nil
            }.joined(separator: "\n")
        }
    }

    /// Get all content blocks
    public var blocks: [AnthropicContentBlock] {
        switch self {
        case .text(let text):
            return [.text(AnthropicTextBlock(text: text))]
        case .blocks(let blocks):
            return blocks
        }
    }
}

/// Content block types
public enum AnthropicContentBlock: Codable, Sendable {
    case text(AnthropicTextBlock)
    case image(AnthropicImageBlock)
    case toolUse(AnthropicToolUseBlock)
    case toolResult(AnthropicToolResultBlock)

    private enum CodingKeys: String, CodingKey {
        case type
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            self = .text(try AnthropicTextBlock(from: decoder))
        case "image":
            self = .image(try AnthropicImageBlock(from: decoder))
        case "tool_use":
            self = .toolUse(try AnthropicToolUseBlock(from: decoder))
        case "tool_result":
            self = .toolResult(try AnthropicToolResultBlock(from: decoder))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown content block type: \(type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .text(let block):
            try block.encode(to: encoder)
        case .image(let block):
            try block.encode(to: encoder)
        case .toolUse(let block):
            try block.encode(to: encoder)
        case .toolResult(let block):
            try block.encode(to: encoder)
        }
    }
}

/// Text content block
public struct AnthropicTextBlock: Codable, Sendable {
    public let type: String
    public let text: String

    public init(text: String) {
        self.type = "text"
        self.text = text
    }

    private enum CodingKeys: String, CodingKey {
        case type, text
    }
}

/// Image content block
public struct AnthropicImageBlock: Codable, Sendable {
    public let type: String
    public let source: AnthropicImageSource

    public struct AnthropicImageSource: Codable, Sendable {
        public let type: String  // "base64" or "url"
        public let media_type: String?  // e.g., "image/png"
        public let data: String?  // base64 data
        public let url: String?  // URL if type is "url"
    }
}

/// Tool use content block (assistant requesting tool invocation)
public struct AnthropicToolUseBlock: Codable, Sendable {
    public let type: String
    public let id: String
    public let name: String
    public let input: [String: AnyCodableValue]

    public init(id: String, name: String, input: [String: AnyCodableValue]) {
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input
    }

    private enum CodingKeys: String, CodingKey {
        case type, id, name, input
    }
}

/// Tool result content block (user providing tool output)
public struct AnthropicToolResultBlock: Codable, Sendable {
    public let type: String
    public let tool_use_id: String
    public let content: AnthropicToolResultContent?
    public let is_error: Bool?

    public init(
        type: String,
        tool_use_id: String,
        content: AnthropicToolResultContent? = nil,
        is_error: Bool? = nil
    ) {
        self.type = type
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error
    }
}

/// Tool result content can be a string or array of content blocks
public enum AnthropicToolResultContent: Codable, Sendable {
    case text(String)
    case blocks([AnthropicContentBlock])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
        } else {
            throw DecodingError.typeMismatch(
                AnthropicToolResultContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected String or [AnthropicContentBlock]"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text):
            try container.encode(text)
        case .blocks(let blocks):
            try container.encode(blocks)
        }
    }

    public var plainText: String {
        switch self {
        case .text(let text):
            return text
        case .blocks(let blocks):
            return blocks.compactMap { block in
                if case .text(let textBlock) = block {
                    return textBlock.text
                }
                return nil
            }.joined(separator: "\n")
        }
    }
}

/// Tool definition
public struct AnthropicTool: Codable, Sendable {
    public let name: String
    public let description: String?
    public let input_schema: JSONValue?

    public init(name: String, description: String?, input_schema: JSONValue?) {
        self.name = name
        self.description = description
        self.input_schema = input_schema
    }
}

/// Tool choice specification
public enum AnthropicToolChoice: Codable, Sendable {
    case auto
    case any
    case none
    case tool(name: String)

    private enum CodingKeys: String, CodingKey {
        case type, name
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "auto":
            self = .auto
        case "any":
            self = .any
        case "none":
            self = .none
        case "tool":
            let name = try container.decode(String.self, forKey: .name)
            self = .tool(name: name)
        default:
            self = .auto
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .auto:
            try container.encode("auto", forKey: .type)
        case .any:
            try container.encode("any", forKey: .type)
        case .none:
            try container.encode("none", forKey: .type)
        case .tool(let name):
            try container.encode("tool", forKey: .type)
            try container.encode(name, forKey: .name)
        }
    }
}

/// Request metadata
public struct AnthropicMetadata: Codable, Sendable {
    public let user_id: String?
}

// MARK: - Response Models

/// Anthropic Messages API response (non-streaming)
public struct AnthropicMessagesResponse: Codable, Sendable {
    public let id: String
    public let type: String
    public let role: String
    public let content: [AnthropicResponseContentBlock]
    public let model: String
    public let stop_reason: String?
    public let stop_sequence: String?
    public let usage: AnthropicUsage

    public init(
        id: String,
        model: String,
        content: [AnthropicResponseContentBlock],
        stopReason: String?,
        usage: AnthropicUsage
    ) {
        self.id = id
        self.type = "message"
        self.role = "assistant"
        self.content = content
        self.model = model
        self.stop_reason = stopReason
        self.stop_sequence = nil
        self.usage = usage
    }
}

/// Response content block (simplified for encoding)
public enum AnthropicResponseContentBlock: Codable, Sendable {
    case text(type: String, text: String)
    case toolUse(type: String, id: String, name: String, input: [String: AnyCodableValue])

    private enum CodingKeys: String, CodingKey {
        case type, text, id, name, input
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(type: type, text: text)
        case "tool_use":
            let id = try container.decode(String.self, forKey: .id)
            let name = try container.decode(String.self, forKey: .name)
            let input = try container.decode([String: AnyCodableValue].self, forKey: .input)
            self = .toolUse(type: type, id: id, name: name, input: input)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown response content block type: \(type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let type, let text):
            try container.encode(type, forKey: .type)
            try container.encode(text, forKey: .text)
        case .toolUse(let type, let id, let name, let input):
            try container.encode(type, forKey: .type)
            try container.encode(id, forKey: .id)
            try container.encode(name, forKey: .name)
            try container.encode(input, forKey: .input)
        }
    }

    public static func textBlock(_ text: String) -> AnthropicResponseContentBlock {
        .text(type: "text", text: text)
    }

    public static func toolUseBlock(id: String, name: String, input: [String: AnyCodableValue])
        -> AnthropicResponseContentBlock
    {
        .toolUse(type: "tool_use", id: id, name: name, input: input)
    }
}

/// Token usage information
public struct AnthropicUsage: Codable, Sendable {
    public let input_tokens: Int
    public let output_tokens: Int

    public init(inputTokens: Int, outputTokens: Int) {
        self.input_tokens = inputTokens
        self.output_tokens = outputTokens
    }
}

/// Anthropic error response
public struct AnthropicError: Codable, Sendable {
    public let type: String
    public let error: AnthropicErrorDetail

    public init(type: String = "error", message: String, errorType: String = "invalid_request_error") {
        self.type = type
        self.error = AnthropicErrorDetail(type: errorType, message: message)
    }

    public struct AnthropicErrorDetail: Codable, Sendable {
        public let type: String
        public let message: String
    }
}

// MARK: - Streaming Event Models

/// Base streaming event
public protocol AnthropicStreamEvent: Codable, Sendable {
    var type: String { get }
}

/// message_start event
public struct MessageStartEvent: Codable, Sendable {
    public let type: String
    public let message: MessageStartPayload

    public struct MessageStartPayload: Codable, Sendable {
        public let id: String
        public let type: String
        public let role: String
        public let content: [AnthropicResponseContentBlock]
        public let model: String
        public let stop_reason: String?
        public let stop_sequence: String?
        public let usage: AnthropicUsage
    }

    public init(id: String, model: String, inputTokens: Int) {
        self.type = "message_start"
        self.message = MessageStartPayload(
            id: id,
            type: "message",
            role: "assistant",
            content: [],
            model: model,
            stop_reason: nil,
            stop_sequence: nil,
            usage: AnthropicUsage(inputTokens: inputTokens, outputTokens: 0)
        )
    }
}

/// content_block_start event
public struct ContentBlockStartEvent: Codable, Sendable {
    public let type: String
    public let index: Int
    public let content_block: ContentBlockStart

    public enum ContentBlockStart: Codable, Sendable {
        case text(TextBlockStart)
        case toolUse(ToolUseBlockStart)
        case thinking(ThinkingBlockStart)

        public struct TextBlockStart: Codable, Sendable {
            public let type: String
            public let text: String

            public init() {
                self.type = "text"
                self.text = ""
            }
        }

        public struct ToolUseBlockStart: Codable, Sendable {
            public let type: String
            public let id: String
            public let name: String
            public let input: [String: AnyCodableValue]

            public init(id: String, name: String) {
                self.type = "tool_use"
                self.id = id
                self.name = name
                self.input = [:]
            }
        }

        /// Anthropic extended-thinking content block start. The block opens
        /// empty and is filled by subsequent `thinking_delta` events.
        public struct ThinkingBlockStart: Codable, Sendable {
            public let type: String
            public let thinking: String

            public init() {
                self.type = "thinking"
                self.thinking = ""
            }
        }

        private enum CodingKeys: String, CodingKey {
            case type, text, id, name, input, thinking
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "text":
                self = .text(TextBlockStart())
            case "tool_use":
                let id = try container.decode(String.self, forKey: .id)
                let name = try container.decode(String.self, forKey: .name)
                self = .toolUse(ToolUseBlockStart(id: id, name: name))
            case "thinking":
                self = .thinking(ThinkingBlockStart())
            default:
                self = .text(TextBlockStart())
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .text(let block):
                try container.encode(block.type, forKey: .type)
                try container.encode(block.text, forKey: .text)
            case .toolUse(let block):
                try container.encode(block.type, forKey: .type)
                try container.encode(block.id, forKey: .id)
                try container.encode(block.name, forKey: .name)
                try container.encode(block.input, forKey: .input)
            case .thinking(let block):
                try container.encode(block.type, forKey: .type)
                try container.encode(block.thinking, forKey: .thinking)
            }
        }
    }

    public init(index: Int, textBlock: Bool = true) {
        self.type = "content_block_start"
        self.index = index
        self.content_block = .text(ContentBlockStart.TextBlockStart())
    }

    public init(index: Int, toolId: String, toolName: String) {
        self.type = "content_block_start"
        self.index = index
        self.content_block = .toolUse(ContentBlockStart.ToolUseBlockStart(id: toolId, name: toolName))
    }

    /// Initializer for Anthropic extended-thinking content blocks.
    public init(thinkingBlockAt index: Int) {
        self.type = "content_block_start"
        self.index = index
        self.content_block = .thinking(ContentBlockStart.ThinkingBlockStart())
    }
}

/// content_block_delta event
public struct ContentBlockDeltaEvent: Codable, Sendable {
    public let type: String
    public let index: Int
    public let delta: ContentBlockDelta

    public enum ContentBlockDelta: Codable, Sendable {
        case textDelta(TextDelta)
        case inputJsonDelta(InputJsonDelta)
        case thinkingDelta(ThinkingDelta)

        public struct TextDelta: Codable, Sendable {
            public let type: String
            public let text: String

            public init(text: String) {
                self.type = "text_delta"
                self.text = text
            }
        }

        public struct InputJsonDelta: Codable, Sendable {
            public let type: String
            public let partial_json: String

            public init(partialJson: String) {
                self.type = "input_json_delta"
                self.partial_json = partialJson
            }
        }

        /// Anthropic extended-thinking incremental delta — appended to the
        /// currently-open `thinking` content block.
        public struct ThinkingDelta: Codable, Sendable {
            public let type: String
            public let thinking: String

            public init(thinking: String) {
                self.type = "thinking_delta"
                self.thinking = thinking
            }
        }

        private enum CodingKeys: String, CodingKey {
            case type, text, partial_json, thinking
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "text_delta":
                let text = try container.decode(String.self, forKey: .text)
                self = .textDelta(TextDelta(text: text))
            case "input_json_delta":
                let json = try container.decode(String.self, forKey: .partial_json)
                self = .inputJsonDelta(InputJsonDelta(partialJson: json))
            case "thinking_delta":
                let thinking = try container.decode(String.self, forKey: .thinking)
                self = .thinkingDelta(ThinkingDelta(thinking: thinking))
            default:
                self = .textDelta(TextDelta(text: ""))
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .textDelta(let delta):
                try container.encode(delta.type, forKey: .type)
                try container.encode(delta.text, forKey: .text)
            case .inputJsonDelta(let delta):
                try container.encode(delta.type, forKey: .type)
                try container.encode(delta.partial_json, forKey: .partial_json)
            case .thinkingDelta(let delta):
                try container.encode(delta.type, forKey: .type)
                try container.encode(delta.thinking, forKey: .thinking)
            }
        }
    }

    public init(index: Int, text: String) {
        self.type = "content_block_delta"
        self.index = index
        self.delta = .textDelta(ContentBlockDelta.TextDelta(text: text))
    }

    public init(index: Int, partialJson: String) {
        self.type = "content_block_delta"
        self.index = index
        self.delta = .inputJsonDelta(ContentBlockDelta.InputJsonDelta(partialJson: partialJson))
    }

    /// Initializer for Anthropic extended-thinking deltas. Pairs with
    /// `ContentBlockStartEvent(thinkingBlockAt:)`.
    public init(thinkingAt index: Int, text: String) {
        self.type = "content_block_delta"
        self.index = index
        self.delta = .thinkingDelta(ContentBlockDelta.ThinkingDelta(thinking: text))
    }
}

/// content_block_stop event
public struct ContentBlockStopEvent: Codable, Sendable {
    public let type: String
    public let index: Int

    public init(index: Int) {
        self.type = "content_block_stop"
        self.index = index
    }
}

/// message_delta event
public struct MessageDeltaEvent: Codable, Sendable {
    public let type: String
    public let delta: MessageDelta
    public let usage: MessageDeltaUsage

    public struct MessageDelta: Codable, Sendable {
        public let stop_reason: String?
        public let stop_sequence: String?
    }

    public struct MessageDeltaUsage: Codable, Sendable {
        public let output_tokens: Int
    }

    public init(stopReason: String?, outputTokens: Int) {
        self.type = "message_delta"
        self.delta = MessageDelta(stop_reason: stopReason, stop_sequence: nil)
        self.usage = MessageDeltaUsage(output_tokens: outputTokens)
    }
}

/// message_stop event
public struct MessageStopEvent: Codable, Sendable {
    public let type: String

    public init() {
        self.type = "message_stop"
    }
}

/// ping event (for keep-alive)
public struct PingEvent: Codable, Sendable {
    public let type: String

    public init() {
        self.type = "ping"
    }
}

// MARK: - Helper: Generic Codable Value

/// A type-erased codable value for handling arbitrary JSON
public struct AnyCodableValue: Codable, @unchecked Sendable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            value = NSNull()
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let array = try? container.decode([AnyCodableValue].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodableValue].self) {
            value = dict.mapValues { $0.value }
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unable to decode value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case is NSNull:
            try container.encodeNil()
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodableValue($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodableValue($0) })
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - Conversion Helpers

extension AnthropicMessagesRequest {
    /// Convert Anthropic request to OpenAI-compatible ChatCompletionRequest
    public func toChatCompletionRequest() -> ChatCompletionRequest {
        var openAIMessages: [ChatMessage] = []

        // Add system message if present
        if let system = system {
            openAIMessages.append(ChatMessage(role: "system", content: system.plainText))
        }

        // Convert messages
        for msg in messages {
            switch msg.role {
            case "user":
                // Check for tool_result blocks
                let blocks = msg.content.blocks
                var hasToolResult = false
                for block in blocks {
                    if case .toolResult(let result) = block {
                        hasToolResult = true
                        let content = result.content?.plainText ?? ""
                        openAIMessages.append(
                            ChatMessage(
                                role: "tool",
                                content: content,
                                tool_calls: nil,
                                tool_call_id: result.tool_use_id
                            )
                        )
                    }
                }
                if !hasToolResult {
                    openAIMessages.append(ChatMessage(role: "user", content: msg.content.plainText))
                }
            case "assistant":
                // Check for tool_use blocks
                let blocks = msg.content.blocks
                var toolCalls: [ToolCall] = []
                var textContent = ""

                for block in blocks {
                    switch block {
                    case .text(let textBlock):
                        textContent += textBlock.text
                    case .toolUse(let toolUse):
                        let argsData =
                            try? JSONSerialization.data(
                                withJSONObject: toolUse.input.mapValues { $0.value }
                            )
                        let argsString = argsData.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                        toolCalls.append(
                            ToolCall(
                                id: toolUse.id,
                                type: "function",
                                function: ToolCallFunction(name: toolUse.name, arguments: argsString)
                            )
                        )
                    default:
                        break
                    }
                }

                if !toolCalls.isEmpty {
                    openAIMessages.append(
                        ChatMessage(
                            role: "assistant",
                            content: textContent.isEmpty ? nil : textContent,
                            tool_calls: toolCalls,
                            tool_call_id: nil
                        )
                    )
                } else {
                    openAIMessages.append(ChatMessage(role: "assistant", content: textContent))
                }
            default:
                openAIMessages.append(ChatMessage(role: msg.role, content: msg.content.plainText))
            }
        }

        // Convert tools
        var openAITools: [Tool]? = nil
        if let tools = tools {
            openAITools = tools.map { tool in
                Tool(
                    type: "function",
                    function: ToolFunction(
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.input_schema
                    )
                )
            }
        }

        // Convert tool_choice
        var openAIToolChoice: ToolChoiceOption? = nil
        if let choice = tool_choice {
            switch choice {
            case .auto:
                openAIToolChoice = .auto
            case .none:
                openAIToolChoice = ToolChoiceOption.none
            case .any:
                // "any" means the model must call a tool - map to auto as closest equivalent
                openAIToolChoice = .auto
            case .tool(let name):
                openAIToolChoice = .function(
                    ToolChoiceOption.FunctionName(
                        type: "function",
                        function: ToolChoiceOption.Name(name: name)
                    )
                )
            }
        }

        return ChatCompletionRequest(
            model: model,
            messages: openAIMessages,
            temperature: temperature.map { Float($0) },
            max_tokens: max_tokens,
            stream: stream,
            top_p: top_p.map { Float($0) },
            frequency_penalty: nil,
            presence_penalty: nil,
            stop: stop_sequences,
            n: nil,
            tools: openAITools,
            tool_choice: openAIToolChoice,
            session_id: nil
        )
    }
}

// MARK: - Models API Response

/// Response from the Anthropic `/v1/models` endpoint
public struct AnthropicModelsResponse: Codable, Sendable {
    public let data: [AnthropicModelInfo]
    public let has_more: Bool
    public let first_id: String?
    public let last_id: String?
}

/// A single model entry from the Anthropic Models API
public struct AnthropicModelInfo: Codable, Sendable {
    public let id: String
    public let display_name: String
    public let created_at: String
    public let type: String
}
