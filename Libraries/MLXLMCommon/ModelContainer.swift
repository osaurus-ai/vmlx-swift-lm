// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN
import os

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``ModelContext``):
///
/// ```swift
/// let messages = [["role": "user", "content": prompt]]
/// let promptTokens = try await modelContainer.perform { context in
///     try context.tokenizer.applyChatTemplate(messages: messages)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let userInput: UserInput
/// let result = await modelContainer.perform { context in
///     let input = try await context.processor.prepare(input: userInput)
///     return generate(
///         input: input, parameters: generateParameters, context: context
///     ) { tokens in
///     ...
///     }
/// }
/// ```
public final class ModelContainer: Sendable {
    private let context: SerialAccessContainer<ModelContext>

    // MARK: - Multi-tier KV Cache

    /// Locked storage for the optional cache coordinator.
    private let _cacheCoordinator = OSAllocatedUnfairLock<CacheCoordinator?>(initialState: nil)

    /// Optional cache coordinator for multi-tier KV caching.
    /// Enable via ``enableCaching(config:)`` after model loading.
    public var cacheCoordinator: CacheCoordinator? {
        _cacheCoordinator.withLock { $0 }
    }

    /// Enable multi-tier KV caching with the given configuration.
    /// Call after model loading. Safe to call multiple times (replaces previous coordinator).
    /// Auto-detects hybrid models and sets modelKey from configuration if not provided.
    public func enableCaching(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        var config = config
        // Auto-set modelKey from model configuration if not provided
        if config.modelKey == nil {
            // Will be set asynchronously after first access — for now use a placeholder
            // that prevents cross-model poisoning within the same process.
            config.modelKey = "\(ObjectIdentifier(self))"
        }
        let coordinator = CacheCoordinator(config: config)
        _cacheCoordinator.withLock { $0 = coordinator }
    }

    /// Enable caching with auto-detection of hybrid models.
    /// Call after model loading. Inspects the model's cache types to detect SSM layers.
    public func enableCachingAsync() async {
        var config = CacheCoordinatorConfig()
        let modelConfig = await context.read { $0.configuration }
        config.modelKey = modelConfig.name

        // Auto-detect hybrid: check if model creates MambaCache/ArraysCache layers
        let isHybrid = await context.read { ctx -> Bool in
            let testCache = ctx.model.newCache(parameters: nil)
            return testCache.contains { $0 is MambaCache || $0 is ArraysCache }
        }

        // 2026-05-05 (Ling-2.6-flash multi-turn fix): hybrid models with
        // ArraysCache (Linear-Attn / GLA recurrence) live or die by the
        // SSMStateCache fetch in CacheCoordinator.fetch — and that fetch
        // is gated INSIDE the paged-hit branch. With the default
        // `pagedBlockSize=64`, short chat prompts (≤ 64 tokens, common
        // for Bailing/Ling chat templates which render to ~30 tokens)
        // store ZERO paged blocks → coordinator misses → SSM state never
        // restored → the live cache passed across ChatSession turns goes
        // stale → incoherent Turn 2 output (or SIGKILL on full re-prefill
        // since recurrentGLA can't handle L>~30 reliably).
        //
        // Lower the paged block size for hybrid models so even short
        // chat turns store at least one block, enabling the SSM-state
        // restoration path to fire on Turn 2+. 16 tokens covers system-
        // only prefixes and short user messages while keeping hash-chain
        // cost negligible.
        if isHybrid {
            config.pagedBlockSize = 16
        }

        let coordinator = CacheCoordinator(config: config)
        coordinator.setHybrid(isHybrid)

        _cacheCoordinator.withLock { $0 = coordinator }
    }

    /// Disable caching and release all cached state.
    public func disableCaching() {
        _cacheCoordinator.withLock { coordinator in
            coordinator?.clear()
            coordinator = nil
        }
    }

    // MARK: - JangPress runtime

    /// Locked storage for the optional JangPress runtime. Settable
    /// from `loadModelContainer(from:using:loadConfiguration:)` so the
    /// runtime stays alive for the model's lifetime; callers (osaurus
    /// settings panel, JANG Studio inspector) poll status from
    /// anywhere via ``jangPressStatus()``.
    private let _jangPressRuntime = OSAllocatedUnfairLock<JangPressRuntime>(
        initialState: .none)

    /// Read the current JangPress runtime. `.none` when JangPress was
    /// not activated for this load (e.g. `LoadConfiguration.jangPress
    /// == .disabled` or auto-threshold not met).
    public var jangPressRuntime: JangPressRuntime {
        _jangPressRuntime.withLock { $0 }
    }

    /// Replace the JangPress runtime. Called once at load time by
    /// `loadModelContainer(from:using:loadConfiguration:)`. Safe to
    /// call multiple times — each call replaces; ARC drops the prior
    /// runtime and its tiers release.
    public func setJangPressRuntime(_ runtime: JangPressRuntime) {
        _jangPressRuntime.withLock { $0 = runtime }
    }

    /// Snapshot the current JangPress status. Returns
    /// `JangPressStatus.disabled` when the runtime is `.none`.
    /// Cheap enough to call on a polling timer (no heavy work; reads
    /// cached counters under a single lock).
    public func jangPressStatus() -> JangPressStatus {
        jangPressRuntime.status()
    }

    public var configuration: ModelConfiguration {
        get async {
            await context.read { $0.configuration }
        }
    }

    public var processor: UserInputProcessor {
        get async {
            await context.read { $0.processor }
        }
    }

    public var tokenizer: Tokenizer {
        get async {
            await context.read { $0.tokenizer }
        }
    }

    /// Whether this model supports vision/image input (is a VLM).
    public var isVLM: Bool {
        get async {
            await context.read { $0.isVLM }
        }
    }

    public init(context: consuming ModelContext) {
        self.context = .init(context)
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    public func perform<R: Sendable>(
        _ action: @Sendable (any LanguageModel, Tokenizer) throws -> sending R
    )
        async rethrows
        -> sending R
    {
        try await context.read {
            try action($0.model, $0.tokenizer)
        }
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try action($0.model, $0.tokenizer, values)
        }
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    ///
    /// - Note: The closure receives `ModelContext` which is not `Sendable`. This is intentional -
    ///   the closure runs within the actor's isolation, ensuring thread-safe access to the model.
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared) across
    ///   isolation boundaries, allowing non-Sendable types to be safely returned.
    public func perform<R: Sendable>(
        _ action: @Sendable (ModelContext) async throws -> sending R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V: Sendable, R: Sendable>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        try await context.read {
            try await action($0, values)
        }
    }

    /// Perform an action on the ``ModelContext`` with additional (non `Sendable`) context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    public func perform<V, R: Sendable>(
        nonSendable values: consuming V, _ action: @Sendable (ModelContext, V) async throws -> R
    ) async rethrows -> sending R {
        let values = SendableBox(values)
        return try await context.read {
            try await action($0, values.consume())
        }
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    public func update(_ action: @Sendable (inout ModelContext) -> Void) async {
        await context.update {
            action(&$0)
        }
    }

    // MARK: - Thread-safe convenience methods

    /// The resolved local model directory for the loaded container.
    public var modelDirectory: URL {
        get async throws {
            try (await configuration).modelDirectory
        }
    }

    /// The resolved local tokenizer directory for the loaded container.
    public var tokenizerDirectory: URL {
        get async throws {
            try (await configuration).tokenizerDirectory
        }
    }

    /// Prepare user input for generation.
    ///
    /// This method safely prepares input within the actor's isolation,
    /// avoiding the need for closure-based `perform` calls.
    ///
    /// - Parameter input: The user input to prepare
    /// - Returns: Prepared language model input (transferred via `sending`)
    /// - Note: The `sending` keyword indicates the return value is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func prepare(input: consuming sending UserInput) async throws -> sending LMInput {
        let processor = await self.processor
        return try await processor.prepare(input: input)
    }

    /// Generate tokens from prepared input, returning an AsyncStream.
    ///
    /// This method provides a thread-safe way to generate tokens without
    /// needing to use closure-based `perform` calls.
    ///
    /// Example:
    /// ```swift
    /// let input = try await modelContainer.prepare(input: userInput)
    /// let stream = try modelContainer.generate(input: input, parameters: parameters)
    /// for await generation in stream {
    ///     switch generation {
    ///     case .chunk(let text): print(text)
    ///     case .reasoning: break  // optional: route to think-pane
    ///     case .info(let info): print(info.tokensPerSecond)
    ///     case .toolCall(let call): handleToolCall(call)
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - input: Prepared language model input (transferred via `sending`)
    ///   - parameters: Generation parameters
    ///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination
    /// - Returns: An AsyncStream of generation events
    /// - Note: The `sending` parameter indicates the input is transferred (not shared),
    ///   allowing non-Sendable types like `LMInput` to safely cross isolation boundaries.
    public func generate(
        input: consuming sending LMInput,
        parameters: GenerateParameters,
        wiredMemoryTicket: WiredMemoryTicket? = nil
    ) async throws -> AsyncStream<Generation> {
        jangPressRuntime.recordPromptTokenActivity(
            input.text.tokens.reshaped(-1).asArray(Int.self))

        let input = SendableBox(input)
        let coordinator = self.cacheCoordinator

        // Note: this is only visiting the model exclusively
        // for the pre-fill time.  Beyond that there is no
        // shared mutable state.
        //
        // This means that there may be concurrent access to the
        // model weights themselves (but they are already evaluated).

        return try await context.read { context in
            try MLXLMCommon.generate(
                input: input.consume(),
                parameters: parameters,
                context: context,
                wiredMemoryTicket: wiredMemoryTicket,
                cacheCoordinator: coordinator
            )
        }
    }

    /// Decode token IDs to a string.
    ///
    /// - Parameter tokenIds: Array of token IDs
    /// - Returns: Decoded string
    public func decode(tokenIds: [Int]) async -> String {
        let tokenizer = await self.tokenizer
        return tokenizer.decode(tokenIds: tokenIds)
    }

    @available(*, deprecated, renamed: "decode(tokenIds:)")
    public func decode(tokens: [Int]) async -> String {
        await decode(tokenIds: tokens)
    }

    /// Encode a string to token IDs.
    ///
    /// - Parameter text: Text to encode
    /// - Returns: Array of token IDs
    public func encode(_ text: String) async -> [Int] {
        let tokenizer = await self.tokenizer
        return tokenizer.encode(text: text)
    }

    /// Apply chat template to messages and return token IDs.
    ///
    /// - Parameter messages: Array of message dictionaries with "role" and "content" keys
    /// - Returns: Array of token IDs
    @available(*, deprecated, message: "Use applyChatTemplate directly on tokenizer")
    public func applyChatTemplate(messages: [[String: String]]) async throws -> [Int] {
        let tokenizer = await self.tokenizer
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}
