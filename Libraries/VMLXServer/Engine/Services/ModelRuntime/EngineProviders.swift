import Foundation

public protocol ServerConfigurationProvider: Sendable {
    func load() async -> ServerConfiguration?
}

public protocol ModelDirectoryProvider: Sendable {
    func effectiveModelsDirectory() -> URL
}

public protocol ModelLocator: Sendable {
    func installedModelNames() -> [String]
    func findInstalledModel(named name: String) -> (name: String, id: String)?
}

public protocol ModelListProvider: Sendable {
    func isFoundationModelAvailable() -> Bool
    func availableRemoteModels() async -> [OpenAIModel]
}

public protocol Telemetry: Sendable {
    func logRequest(
        method: String,
        path: String,
        userAgent: String?,
        requestBody: String?,
        responseBody: String?,
        responseStatus: Int,
        durationMs: Double,
        model: String?,
        tokensInput: Int?,
        tokensOutput: Int?,
        temperature: Float?,
        maxTokens: Int?,
        toolCalls: [ToolCallLog]?,
        finishReason: RequestLog.FinishReason?,
        errorMessage: String?
    )
}

/// Engine-side surface for agent-scoped chat completions and
/// `/agent/{id}/...` routing. The wider agent concept (pairing,
/// invites, listing, per-agent crypto keys) is app-specific and
/// stays on `AgentManager` directly in HTTPHandler endpoints that
/// will relocate out of the engine in a later phase.
public protocol AgentProvider: Sendable {
    func effectiveModel(for agentId: UUID) async -> String?
    func autonomousExecEnabled(for agentId: UUID) async -> Bool
    func resolveAgentId(_ identifier: String) async -> UUID?
}

public struct ToolListEntry: Sendable {
    public let name: String
    public let description: String
    public let parameters: JSONValue?

    public init(name: String, description: String, parameters: JSONValue?) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

/// Host-supplied tool registry used by HTTPHandler. The "always-loaded
/// specs" call resolves capability/folder filters internally — the
/// engine just passes the `autonomousEnabled` agent flag through.
public protocol ToolExecutor: Sendable {
    func alwaysLoadedSpecs(autonomousEnabled: Bool) async -> [Tool]
    func listEnabledTools() async -> [ToolListEntry]
    func parameters(forTool name: String) async -> JSONValue?
    func execute(name: String, argumentsJSON: String) async throws -> String
}

/// Narrow surface that HTTPHandler's memory-ingest + agents-list
/// endpoints need. Standalone CLI uses the no-op default; this entire
/// endpoint group is app-specific and will likely move out of
/// HTTPHandler in a later refactor.
public protocol MemoryProvider: Sendable {
    var isOpen: Bool { get }
    func deleteTranscriptForConversation(_ conversationId: String) throws
    func insertTranscriptTurn(
        agentId: String, conversationId: String, chunkIndex: Int,
        role: String, content: String, tokenCount: Int, createdAt: String?
    ) throws
    func agentIdsWithPinnedFacts() throws -> [(agentId: String, count: Int)]
}

/// Enriches a chat completion request with agent-scoped context
/// (system prompt + memory section) before it hits the engine. Host
/// resolves agent id → agent record → prompt composer; CLI uses the
/// no-op default and passes the request through unchanged.
public protocol AgentEnricher: Sendable {
    func enrich(_ request: ChatCompletionRequest, agentId: String) async -> ChatCompletionRequest
}

public protocol TunnelResolver: Sendable {
    func tunnelBaseURL(for agentId: UUID) async -> String?
}

public protocol DownloadVerifier: Sendable {
    func ensureComplete(modelId: String, name: String, directory: URL) async
    func resolveURL(repoId: String, path: String) -> URL?
}

/// Persists a completed chat-completion round. Engine HTTPHandler invokes
/// this after streaming finishes; Mac app writes to ChatHistoryDatabase,
/// CLI uses the no-op default.
public protocol ChatHistoryPersister: Sendable {
    func persist(
        sourceTag: String,
        sourcePluginId: String?,
        agentId: UUID?,
        externalKey: String?,
        finalMessages: [ChatMessage],
        model: String
    ) async
}

/// Computes vector embeddings for /v1/embeddings.
public protocol EmbeddingProvider: Sendable {
    var modelName: String { get }
    func embed(texts: [String]) async throws -> [[Float]]
}

/// Handles /v1/audio/transcriptions. Returns text and optional duration.
public protocol SpeechProvider: Sendable {
    func transcribe(audioURL: URL) async throws -> (text: String, durationSeconds: TimeInterval?)
}

/// Engine-side outcome for `APIKeyValidating.validate`. Mirrors the
/// Mac app's `AccessKeyValidationResult` cases (minus the issuer
/// payload) so the engine HTTP layer can react uniformly across hosts.
public enum APIKeyValidationOutcome: Sendable {
    case valid
    case expired
    case revoked
    case invalid(reason: String)
}

/// Validates incoming API keys. Mac app's struct conforms via a
/// thin shim; CLI uses `NoOpAPIKeyValidator` and accepts any token.
public protocol APIKeyValidating: Sendable {
    var hasKeys: Bool { get }
    /// Renamed to avoid clashing with app-side `APIKeyValidator.validate`
    /// which returns the richer `AccessKeyValidationResult`. Engine callers
    /// use this one; app callers keep using the original.
    func validateKey(rawKey: String) -> APIKeyValidationOutcome
}

/// Provides the chat-completion brain (model routing + service dispatch).
/// Mac app registers the full `ChatEngine` with `RemoteProviderManager`
/// + `InsightsService`; CLI registers a slimmer engine that only knows
/// about MLX. Engine HTTPHandler reads from the seam when no explicit
/// engine was passed to its init.
public protocol ChatEngineProvider: Sendable {
    func makeChatEngine() -> any ChatEngineProtocol
}

/// Backs `/agents/{id}/dispatch`, `GET /tasks/{id}`, `DELETE /tasks/{id}`.
/// Mac app implements via `TaskDispatcher`/`BackgroundTaskManager`; CLI
/// uses the no-op default and these endpoints return errors.
public protocol BackgroundTaskService: Sendable {
    /// Returns resolved task id (may differ from requestId when the
    /// dispatcher reattaches to an existing session), or nil if the
    /// task limit was reached.
    func dispatchHTTPTask(
        requestId: UUID,
        prompt: String,
        agentId: UUID,
        title: String?,
        externalSessionKey: String?
    ) async -> UUID?

    /// Serialized task state JSON, or nil if the task is not found.
    func taskStateJSON(id: UUID) async -> String?

    /// Fire-and-forget cancel. Matches the host's existing semantics.
    func cancel(id: UUID) async
}

public struct DefaultServerConfigurationProvider: ServerConfigurationProvider {
    public init() {}
    public func load() async -> ServerConfiguration? { nil }
}

/// Defaults to `~/.osaurus/models`. CLI overrides via `--model-dir`.
public struct DefaultModelDirectoryProvider: ModelDirectoryProvider {
    public init() {}
    public func effectiveModelsDirectory() -> URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".osaurus/models", isDirectory: true)
    }
}

public struct NoOpModelLocator: ModelLocator {
    public init() {}
    public func installedModelNames() -> [String] { [] }
    public func findInstalledModel(named name: String) -> (name: String, id: String)? { nil }
}

public struct NoOpModelListProvider: ModelListProvider {
    public init() {}
    public func isFoundationModelAvailable() -> Bool { false }
    public func availableRemoteModels() async -> [OpenAIModel] { [] }
}

public struct NoOpTelemetry: Telemetry {
    public init() {}
    public func logRequest(
        method: String, path: String, userAgent: String?,
        requestBody: String?, responseBody: String?,
        responseStatus: Int, durationMs: Double, model: String?,
        tokensInput: Int?, tokensOutput: Int?, temperature: Float?,
        maxTokens: Int?, toolCalls: [ToolCallLog]?,
        finishReason: RequestLog.FinishReason?, errorMessage: String?
    ) {}
}

public struct NoOpAgentProvider: AgentProvider {
    public init() {}
    public func effectiveModel(for agentId: UUID) async -> String? { nil }
    public func autonomousExecEnabled(for agentId: UUID) async -> Bool { false }
    public func resolveAgentId(_ identifier: String) async -> UUID? { nil }
}

public struct NoOpToolExecutor: ToolExecutor {
    public init() {}
    public func alwaysLoadedSpecs(autonomousEnabled: Bool) async -> [Tool] { [] }
    public func listEnabledTools() async -> [ToolListEntry] { [] }
    public func parameters(forTool name: String) async -> JSONValue? { nil }
    public func execute(name: String, argumentsJSON: String) async throws -> String {
        throw NSError(
            domain: "ToolExecutor", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "No tool executor registered: \(name)"]
        )
    }
}

public struct NoOpMemoryProvider: MemoryProvider {
    public init() {}
    public var isOpen: Bool { false }
    public func deleteTranscriptForConversation(_ conversationId: String) throws {}
    public func insertTranscriptTurn(
        agentId: String, conversationId: String, chunkIndex: Int,
        role: String, content: String, tokenCount: Int, createdAt: String?
    ) throws {}
    public func agentIdsWithPinnedFacts() throws -> [(agentId: String, count: Int)] { [] }
}

public struct NoOpAgentEnricher: AgentEnricher {
    public init() {}
    public func enrich(_ request: ChatCompletionRequest, agentId: String) async -> ChatCompletionRequest {
        request
    }
}

public struct NoOpTunnelResolver: TunnelResolver {
    public init() {}
    public func tunnelBaseURL(for agentId: UUID) async -> String? { nil }
}

public struct NoOpDownloadVerifier: DownloadVerifier {
    public init() {}
    public func ensureComplete(modelId: String, name: String, directory: URL) async {}
    public func resolveURL(repoId: String, path: String) -> URL? { nil }
}

public struct NoOpChatHistoryPersister: ChatHistoryPersister {
    public init() {}
    public func persist(
        sourceTag: String,
        sourcePluginId: String?,
        agentId: UUID?,
        externalKey: String?,
        finalMessages: [ChatMessage],
        model: String
    ) async {}
}

public struct NoOpEmbeddingProvider: EmbeddingProvider {
    public init() {}
    public var modelName: String { "" }
    public func embed(texts: [String]) async throws -> [[Float]] {
        throw NSError(
            domain: "EmbeddingProvider", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "No embedding provider registered"]
        )
    }
}

public struct NoOpSpeechProvider: SpeechProvider {
    public init() {}
    public func transcribe(audioURL: URL) async throws -> (text: String, durationSeconds: TimeInterval?) {
        throw NSError(
            domain: "SpeechProvider", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "No speech provider registered"]
        )
    }
}

public struct NoOpAPIKeyValidator: APIKeyValidating {
    public init() {}
    public let hasKeys = false
    public func validateKey(rawKey: String) -> APIKeyValidationOutcome { .valid }
}

public struct NoOpChatEngine: ChatEngineProtocol {
    public init() {}
    public func streamChat(request: ChatCompletionRequest) async throws -> AsyncThrowingStream<String, Error> {
        throw ChatEngineError(kind: .noServiceAvailable(requested: request.model))
    }
    public func completeChat(request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        throw ChatEngineError(kind: .noServiceAvailable(requested: request.model))
    }
}

public struct NoOpChatEngineProvider: ChatEngineProvider {
    public init() {}
    public func makeChatEngine() -> any ChatEngineProtocol { NoOpChatEngine() }
}

public struct NoOpBackgroundTaskService: BackgroundTaskService {
    public init() {}
    public func dispatchHTTPTask(
        requestId: UUID,
        prompt: String,
        agentId: UUID,
        title: String?,
        externalSessionKey: String?
    ) async -> UUID? { nil }
    public func taskStateJSON(id: UUID) async -> String? { nil }
    public func cancel(id: UUID) async {}
}
