//
//  RequestLog.swift
//  osaurus
//
//  Model for in-memory request/response logging used by InsightsService.
//

import Foundation

/// Represents a logged tool call within an inference
public struct ToolCallLog: Identifiable, Sendable {
    public let id: UUID
    public let name: String
    public let arguments: String
    public let result: String?
    public let durationMs: Double?
    public let isError: Bool

    public init(
        id: UUID = UUID(),
        name: String,
        arguments: String,
        result: String? = nil,
        durationMs: Double? = nil,
        isError: Bool = false
    ) {
        self.id = id
        self.name = name
        self.arguments = arguments
        self.result = result
        self.durationMs = durationMs
        self.isError = isError
    }
}

/// Source of the request
public enum RequestSource: String, Sendable, CaseIterable {
    case chatUI = "Chat UI"
    case httpAPI = "HTTP API"
    case plugin = "Plugin"

    public var displayName: String {
        switch self {
        case .chatUI: return L("Chat UI")
        case .httpAPI: return L("HTTP API")
        case .plugin: return L("Plugin")
        }
    }
}

/// Represents a single request log entry with optional inference data
public struct RequestLog: Identifiable, Sendable {
    public let id: UUID
    public let timestamp: Date
    public let source: RequestSource

    // HTTP request/response fields
    public let method: String
    public let path: String
    public let statusCode: Int
    public let durationMs: Double
    public let requestBody: String?
    public let responseBody: String?
    public let userAgent: String?

    // Plugin attribution (nil for non-plugin requests)
    public let pluginId: String?

    // Optional inference fields (only for chat endpoints)
    public let model: String?
    public let inputTokens: Int?
    public let outputTokens: Int?
    public let tokensPerSecond: Double?
    public let temperature: Float?
    public let maxTokens: Int?
    public let toolCalls: [ToolCallLog]?
    public let finishReason: FinishReason?
    public let errorMessage: String?

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        source: RequestSource,
        method: String,
        path: String,
        statusCode: Int,
        durationMs: Double,
        requestBody: String? = nil,
        responseBody: String? = nil,
        userAgent: String? = nil,
        pluginId: String? = nil,
        model: String? = nil,
        inputTokens: Int? = nil,
        outputTokens: Int? = nil,
        temperature: Float? = nil,
        maxTokens: Int? = nil,
        toolCalls: [ToolCallLog]? = nil,
        finishReason: FinishReason? = nil,
        errorMessage: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.source = source
        self.method = method
        self.path = path
        self.statusCode = statusCode
        self.durationMs = durationMs
        self.requestBody = requestBody
        self.responseBody = responseBody
        self.userAgent = userAgent
        self.pluginId = pluginId
        self.model = model
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.errorMessage = errorMessage

        // Calculate tokens per second if we have inference data
        if let outputTokens = outputTokens, durationMs > 0 {
            self.tokensPerSecond = Double(outputTokens) / (durationMs / 1000.0)
        } else {
            self.tokensPerSecond = nil
        }
    }

    public enum FinishReason: String, Sendable {
        case stop = "stop"
        case length = "length"
        case toolCalls = "tool_calls"
        case error = "error"
        case cancelled = "cancelled"
    }

    // MARK: - Computed Properties

    /// Whether this is a plugin console log entry (not an API call)
    public var isPluginLog: Bool {
        method == "LOG"
    }

    /// Whether this is an inference request (chat endpoint)
    public var isInference: Bool {
        path.contains("chat")
    }

    /// Whether the request was successful (2xx status)
    public var isSuccess: Bool {
        statusCode >= 200 && statusCode < 300
    }

    /// Is this an error state?
    public var isError: Bool {
        !isSuccess || finishReason == .error || errorMessage != nil
    }

    /// Formatted timestamp for display
    public var formattedTimestamp: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: timestamp)
    }

    /// Formatted duration for display
    public var formattedDuration: String {
        if durationMs < 1000 {
            return String(format: "%.0fms", durationMs)
        } else {
            return String(format: "%.1fs", durationMs / 1000)
        }
    }

    /// Formatted tokens per second
    public var formattedSpeed: String {
        if let speed = tokensPerSecond, speed > 0 {
            return String(format: "%.1f tok/s", speed)
        }
        return "-"
    }

    /// Short model name for display
    public var shortModelName: String {
        guard let model = model else { return "-" }
        if model.lowercased() == "foundation" { return "Foundation" }
        if let lastPart = model.split(separator: "/").last {
            return String(lastPart)
        }
        return model
    }

    /// Truncated request body for display (max 500 chars)
    public var truncatedRequestBody: String? {
        guard let body = requestBody else { return nil }
        if body.count > 500 {
            return String(body.prefix(500)) + "..."
        }
        return body
    }

    /// Truncated response body for display (max 1000 chars)
    public var truncatedResponseBody: String? {
        guard let body = responseBody else { return nil }
        if body.count > 1000 {
            return String(body.prefix(1000)) + "..."
        }
        return body
    }

    /// Pretty-printed request body if JSON
    public var formattedRequestBody: String? {
        guard let body = requestBody, let data = body.data(using: .utf8) else { return requestBody }
        if let json = try? JSONSerialization.jsonObject(with: data, options: []),
            let prettyData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
            let prettyString = String(data: prettyData, encoding: .utf8)
        {
            return prettyString
        }
        return body
    }

    /// Pretty-printed response body if JSON
    public var formattedResponseBody: String? {
        guard let body = responseBody, let data = body.data(using: .utf8) else { return responseBody }
        if let json = try? JSONSerialization.jsonObject(with: data, options: []),
            let prettyData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
            let prettyString = String(data: prettyData, encoding: .utf8)
        {
            return prettyString
        }
        return body
    }

    /// Number of tool definitions sent with the request, parsed on demand
    /// from `requestBody`. Returns nil for non-chat or non-JSON bodies, or
    /// when the request did not include a `tools` array. Computed lazily so
    /// the parse cost is only paid for visible rows.
    public var toolDefinitionCount: Int? {
        guard isInference,
            let body = requestBody,
            let data = body.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let tools = obj["tools"] as? [Any]
        else { return nil }
        return tools.isEmpty ? nil : tools.count
    }
}

/// Pending inference metadata captured at start
public struct PendingInference: Sendable {
    public let id: UUID
    public let startTime: Date
    public let source: RequestSource
    public let model: String
    public let inputTokens: Int
    public let temperature: Float
    public let maxTokens: Int

    public init(
        id: UUID = UUID(),
        startTime: Date = Date(),
        source: RequestSource,
        model: String,
        inputTokens: Int,
        temperature: Float,
        maxTokens: Int
    ) {
        self.id = id
        self.startTime = startTime
        self.source = source
        self.model = model
        self.inputTokens = inputTokens
        self.temperature = temperature
        self.maxTokens = maxTokens
    }
}

// MARK: - Legacy type alias for backward compatibility

public typealias InferenceLog = RequestLog
public typealias InferenceSource = RequestSource
