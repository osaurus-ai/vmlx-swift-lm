import Foundation

/// Opaque handle to a model bundle on disk. Identified by content hash so
/// peers can compare what's locally available without sharing weights.
/// Phase 1 carries only what the engine spec requires; per-family fields
/// (sliding-window size, MoE expert count, etc.) join in Phase 3 when the
/// sharding planner needs them.
public struct ModelHandle: Sendable, Hashable {
    /// 16-hex-char trunc(SHA-256), per engine spec §10.
    public let bundleHash: String
    public let displayName: String

    public init(bundleHash: String, displayName: String) {
        self.bundleHash = bundleHash
        self.displayName = displayName
    }
}

/// Caller-supplied generation request. Phase 1 is a thin wrapper — real
/// fields (messages, sampling, tools) get added when osaurus's ChatEngine
/// starts producing these instead of its existing types.
public struct GenerateRequest: Sendable {
    public let model: ModelHandle
    public let prompt: String
    public let maxTokens: Int

    public init(model: ModelHandle, prompt: String, maxTokens: Int) {
        self.model = model
        self.prompt = prompt
        self.maxTokens = maxTokens
    }
}

/// One streamed output token (or end-of-stream marker).
public enum Token: Sendable, Equatable {
    case text(String)
    case end(reason: EndReason)

    public enum EndReason: Sendable, Equatable {
        case completed
        case maxTokens
        case error(String)
    }
}

/// Callback the engine uses for the local-execution path. Implemented by
/// the consumer (osaurus, RunBench, future TLS/JACCL backends). Keeps
/// `MLXDistributedCore` free of MLX/MLXLLM dependencies so it can be
/// linked into hosts that bring their own runtime.
public protocol LocalGenerator: Sendable {
    func generate(_ request: GenerateRequest) -> AsyncStream<Token>
}

/// Output of `ClusterSession.plan(model:)`. Phase 1 only encodes
/// "where does this run". TP/PP details land in Phase 5/6.
public struct ParallelPlan: Sendable, Equatable {
    public enum Placement: Sendable, Equatable {
        case local
        case replicaOnPeer(UUID)        // Phase 1B (osaurus router)
        case pipelinedOver([UUID])      // Phase 2+
        case wiredOver([UUID])          // Phase 6+
    }

    public let placement: Placement
    public let model: ModelHandle

    public init(placement: Placement, model: ModelHandle) {
        self.placement = placement
        self.model = model
    }
}
