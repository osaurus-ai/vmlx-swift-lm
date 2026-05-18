import Foundation

/// Process-wide registry of host-supplied engine seams. CLI keeps the
/// defaults; Mac app registers real adapters at startup.
public enum InferenceServices {
    private static let lock = NSLock()

    nonisolated(unsafe) private static var _progressReporter: any ProgressReporter
        = NoOpProgressReporter()
    nonisolated(unsafe) private static var _serverConfig: any ServerConfigurationProvider
        = DefaultServerConfigurationProvider()
    nonisolated(unsafe) private static var _modelDirectory: any ModelDirectoryProvider
        = DefaultModelDirectoryProvider()
    nonisolated(unsafe) private static var _downloadVerifier: any DownloadVerifier
        = NoOpDownloadVerifier()
    nonisolated(unsafe) private static var _modelLocator: any ModelLocator
        = NoOpModelLocator()
    nonisolated(unsafe) private static var _modelList: any ModelListProvider
        = NoOpModelListProvider()
    nonisolated(unsafe) private static var _telemetry: any Telemetry
        = NoOpTelemetry()
    nonisolated(unsafe) private static var _tunnelResolver: any TunnelResolver
        = NoOpTunnelResolver()
    nonisolated(unsafe) private static var _memory: any MemoryProvider
        = NoOpMemoryProvider()
    nonisolated(unsafe) private static var _tools: any ToolExecutor
        = NoOpToolExecutor()
    nonisolated(unsafe) private static var _agents: any AgentProvider
        = NoOpAgentProvider()
    nonisolated(unsafe) private static var _agentEnricher: any AgentEnricher
        = NoOpAgentEnricher()
    nonisolated(unsafe) private static var _chatHistory: any ChatHistoryPersister
        = NoOpChatHistoryPersister()
    nonisolated(unsafe) private static var _embedding: any EmbeddingProvider
        = NoOpEmbeddingProvider()
    nonisolated(unsafe) private static var _speech: any SpeechProvider
        = NoOpSpeechProvider()
    nonisolated(unsafe) private static var _backgroundTasks: any BackgroundTaskService
        = NoOpBackgroundTaskService()
    nonisolated(unsafe) private static var _chatEngine: any ChatEngineProvider
        = NoOpChatEngineProvider()

    public static var progressReporter: any ProgressReporter {
        lock.withLock { _progressReporter }
    }
    public static var serverConfig: any ServerConfigurationProvider {
        lock.withLock { _serverConfig }
    }
    public static var modelDirectory: any ModelDirectoryProvider {
        lock.withLock { _modelDirectory }
    }
    public static var downloadVerifier: any DownloadVerifier {
        lock.withLock { _downloadVerifier }
    }
    public static var modelLocator: any ModelLocator {
        lock.withLock { _modelLocator }
    }
    public static var modelList: any ModelListProvider {
        lock.withLock { _modelList }
    }
    public static var telemetry: any Telemetry {
        lock.withLock { _telemetry }
    }
    public static var tunnelResolver: any TunnelResolver {
        lock.withLock { _tunnelResolver }
    }
    public static var memory: any MemoryProvider {
        lock.withLock { _memory }
    }
    public static var tools: any ToolExecutor {
        lock.withLock { _tools }
    }
    public static var agents: any AgentProvider {
        lock.withLock { _agents }
    }
    public static var agentEnricher: any AgentEnricher {
        lock.withLock { _agentEnricher }
    }
    public static var chatHistory: any ChatHistoryPersister {
        lock.withLock { _chatHistory }
    }
    public static var embedding: any EmbeddingProvider {
        lock.withLock { _embedding }
    }
    public static var speech: any SpeechProvider {
        lock.withLock { _speech }
    }
    public static var backgroundTasks: any BackgroundTaskService {
        lock.withLock { _backgroundTasks }
    }
    public static var chatEngine: any ChatEngineProvider {
        lock.withLock { _chatEngine }
    }

    public static func register(progressReporter: any ProgressReporter) {
        lock.withLock { _progressReporter = progressReporter }
    }
    public static func register(serverConfig: any ServerConfigurationProvider) {
        lock.withLock { _serverConfig = serverConfig }
    }
    public static func register(modelDirectory: any ModelDirectoryProvider) {
        lock.withLock { _modelDirectory = modelDirectory }
    }
    public static func register(downloadVerifier: any DownloadVerifier) {
        lock.withLock { _downloadVerifier = downloadVerifier }
    }
    public static func register(modelLocator: any ModelLocator) {
        lock.withLock { _modelLocator = modelLocator }
    }
    public static func register(modelList: any ModelListProvider) {
        lock.withLock { _modelList = modelList }
    }
    public static func register(telemetry: any Telemetry) {
        lock.withLock { _telemetry = telemetry }
    }
    public static func register(tunnelResolver: any TunnelResolver) {
        lock.withLock { _tunnelResolver = tunnelResolver }
    }
    public static func register(memory: any MemoryProvider) {
        lock.withLock { _memory = memory }
    }
    public static func register(tools: any ToolExecutor) {
        lock.withLock { _tools = tools }
    }
    public static func register(agents: any AgentProvider) {
        lock.withLock { _agents = agents }
    }
    public static func register(agentEnricher: any AgentEnricher) {
        lock.withLock { _agentEnricher = agentEnricher }
    }
    public static func register(chatHistory: any ChatHistoryPersister) {
        lock.withLock { _chatHistory = chatHistory }
    }
    public static func register(embedding: any EmbeddingProvider) {
        lock.withLock { _embedding = embedding }
    }
    public static func register(speech: any SpeechProvider) {
        lock.withLock { _speech = speech }
    }
    public static func register(backgroundTasks: any BackgroundTaskService) {
        lock.withLock { _backgroundTasks = backgroundTasks }
    }
    public static func register(chatEngine: any ChatEngineProvider) {
        lock.withLock { _chatEngine = chatEngine }
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
