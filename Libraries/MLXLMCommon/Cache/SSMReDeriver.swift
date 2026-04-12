// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Manages async idle-time re-derivation of SSM state for thinking models.
///
/// **Problem:** Thinking models (Qwen3.5, MiniMax) append thinking tokens after the
/// user prompt during generation. These tokens contaminate the cumulative SSM state --
/// a cached snapshot taken at the end of generation includes contributions from thinking
/// tokens that will not be present in the next request with the same prompt prefix.
///
/// **Solution:** After generation completes, enqueue a re-derive job. During idle time
/// (no active generation requests), the caller re-runs prefill with *only* the original
/// prompt tokens (no thinking tokens) to produce a clean SSM state, then stores it via
/// ``completeJob(promptTokens:boundary:cleanSSMStates:)``.
///
/// The actor manages the job queue and bookkeeping. The actual model prefill is performed
/// by the caller (e.g., BatchEngine or scheduler) because the model reference cannot be
/// safely passed into the actor.
public actor SSMReDeriver {

    /// A single re-derive job describing the prompt that needs a clean SSM state snapshot.
    public struct ReDeriverJob: Sendable {
        /// The original prompt tokens (excluding any thinking/generation tokens).
        public let promptTokens: [Int]
        /// The boundary (number of tokens from the start) used as the cache key.
        public let boundary: Int
        /// An opaque identifier for the originating request (for logging/dedup).
        public let requestId: String
    }

    // MARK: - Properties

    private var queue: [ReDeriverJob] = []
    private var isProcessing: Bool = false
    private let ssmStateCache: SSMStateCache

    // MARK: - Initialization

    /// Creates a new re-deriver backed by the given SSM state cache.
    /// - Parameter ssmStateCache: The cache where clean SSM states will be stored.
    public init(ssmStateCache: SSMStateCache) {
        self.ssmStateCache = ssmStateCache
    }

    // MARK: - Queue Management

    /// Queue a re-derive job. Called after generation completes for thinking models.
    ///
    /// - Parameters:
    ///   - promptTokens: The original prompt tokens (no thinking/generation tokens).
    ///   - boundary: The token boundary for the cache key.
    ///   - requestId: An opaque identifier for the originating request.
    public func enqueue(promptTokens: [Int], boundary: Int, requestId: String) {
        queue.append(ReDeriverJob(promptTokens: promptTokens, boundary: boundary, requestId: requestId))
    }

    /// Dequeue the next job for processing. Called during idle time by the caller.
    ///
    /// Returns `nil` if a job is already being processed or the queue is empty.
    /// The caller is responsible for running prefill with the returned job's
    /// ``ReDeriverJob/promptTokens`` and then calling
    /// ``completeJob(promptTokens:boundary:cleanSSMStates:)`` with the results.
    public func processNext() -> ReDeriverJob? {
        guard !isProcessing, !queue.isEmpty else { return nil }
        isProcessing = true
        return queue.removeFirst()
    }

    /// Called after the caller has run prefill and extracted clean SSM states.
    ///
    /// Stores the clean states in the SSM state cache and marks processing as complete,
    /// allowing the next queued job to be dequeued.
    ///
    /// - Parameters:
    ///   - promptTokens: The prompt tokens that were prefilled.
    ///   - boundary: The token boundary for the cache key.
    ///   - cleanSSMStates: The SSM state arrays produced by a clean prefill (no thinking tokens).
    public func completeJob(promptTokens: [Int], boundary: Int, cleanSSMStates: [MLXArray]) {
        ssmStateCache.store(ssmStates: cleanSSMStates, tokens: promptTokens, boundary: boundary)
        isProcessing = false
    }

    /// Cancel the current in-progress job (e.g., a new request arrived during re-derive).
    ///
    /// This does *not* remove queued jobs -- only resets the processing flag so that
    /// ``processNext()`` can return the next job when idle time resumes.
    public func cancelProcessing() {
        isProcessing = false
    }

    // MARK: - Status

    /// The number of jobs waiting in the queue (excludes any in-progress job).
    public var pendingCount: Int { queue.count }

    /// Whether there are any jobs waiting in the queue.
    public var hasPending: Bool { !queue.isEmpty }

    /// Remove all queued jobs and reset the processing flag.
    public func clear() {
        queue.removeAll()
        isProcessing = false
    }
}
