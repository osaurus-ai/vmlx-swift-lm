//
//  MLXService.swift
//  osaurus
//
//  Migrated to Swift 6 actors; delegates runtime state to ModelManager/ModelRuntime.
//

import Foundation

/// Lightweight reference to a local MLX model (name + repo id)
private struct LocalModelRef {
    public let name: String
    public let modelId: String
}

public actor MLXService: ToolCapableService {

    public init() {}

    /// Shared instance for convenience (actor is stateless, delegates to ModelRuntime.shared)
    public static let shared = MLXService()

    public nonisolated var id: String { "mlx" }

    // MARK: - Availability / Routing

    public nonisolated func isAvailable() -> Bool {
        return !Self.getAvailableModels().isEmpty
    }

    public nonisolated func handles(requestedModel: String?) -> Bool {
        let trimmed = (requestedModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        return Self.findModel(named: trimmed) != nil
    }

    // MARK: - Static discovery wrappers (delegate to injected locator)

    public nonisolated static func getAvailableModels() -> [String] {
        return InferenceServices.modelLocator.installedModelNames()
    }

    fileprivate nonisolated static func findModel(named name: String) -> LocalModelRef? {
        if let found = InferenceServices.modelLocator.findInstalledModel(named: name) {
            return LocalModelRef(name: found.name, modelId: found.id)
        }
        return nil
    }

    // MARK: - ModelService

    public func streamDeltas(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error> {
        let model = try selectModel(requestedName: requestedModel)
        return try await ModelRuntime.shared.streamWithTools(
            messages: messages,
            parameters: parameters,
            stopSequences: stopSequences,
            tools: [],
            toolChoice: nil,
            modelId: model.modelId,
            modelName: model.name
        )
    }

    public func generateOneShot(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?
    ) async throws -> String {
        let stream = try await streamDeltas(
            messages: messages,
            parameters: parameters,
            requestedModel: requestedModel,
            stopSequences: []
        )
        var out = ""
        for try await s in stream {
            // `streamDeltas` wraps `ModelRuntime.streamWithTools`, which
            // encodes non-token events (reasoning, stats, tool calls) as
            // in-band `\u{FFFE}…` sentinel strings so the SSE/NDJSON writer
            // can peel them off and route to their own response channels.
            // For non-streaming `chat/completions` the caller wants a plain
            // text answer; concatenating sentinels verbatim made them leak
            // into `content` — e.g. a reasoning model's thought content
            // arrived as
            // `"\u{FFFE}reasoning:thought…\u{FFFE}stats:80;8.83"` embedded
            // in the response. Skip every delta that starts with the
            // sentinel marker; `StreamingToolHint.isSentinel` covers
            // tool/args/done, reasoning, stats, and any future sentinel
            // that adheres to the `\u{FFFE}` prefix contract.
            if StreamingToolHint.isSentinel(s) { continue }
            out += s
        }
        return out
    }

    // MARK: - Message-based Tool-capable bridge

    public func respondWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> String {
        let model = try selectModel(requestedName: requestedModel)
        return try await ModelRuntime.shared.respondWithTools(
            messages: messages,
            parameters: parameters,
            stopSequences: stopSequences,
            tools: tools,
            toolChoice: toolChoice,
            modelId: model.modelId,
            modelName: model.name
        )
    }

    public func streamWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error> {
        let model = try selectModel(requestedName: requestedModel)
        return try await ModelRuntime.shared.streamWithTools(
            messages: messages,
            parameters: parameters,
            stopSequences: stopSequences,
            tools: tools,
            toolChoice: toolChoice,
            modelId: model.modelId,
            modelName: model.name
        )
    }

    // MARK: - Runtime cache management

    public func cachedRuntimeSummaries() async -> [ModelRuntime.ModelCacheSummary] {
        await ModelRuntime.shared.cachedModelSummaries()
    }

    public func unloadRuntimeModel(named name: String) async {
        await ModelRuntime.shared.unload(name: name)
    }

    public func clearRuntimeCache() async {
        await ModelRuntime.shared.clearAll()
    }

    // MARK: - Helpers

    private func selectModel(requestedName: String?) throws -> LocalModelRef {
        let trimmed = (requestedName ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw NSError(
                domain: "MLXService",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Requested model is required"]
            )
        }
        if let m = Self.findModel(named: trimmed) {
            return m
        }
        throw NSError(
            domain: "MLXService",
            code: 4,
            userInfo: [NSLocalizedDescriptionKey: "Requested model not found: \(trimmed)"]
        )
    }
}
