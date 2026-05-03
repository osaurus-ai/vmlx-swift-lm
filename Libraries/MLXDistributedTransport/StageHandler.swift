import Foundation
import NIOCore

/// What a server-side rank does with a received frame. Implementations
/// might run a slice of model layers, fetch a KV cache, or echo back for
/// tests. The handler streams zero or more reply frames; the server
/// flushes them in order on the same TLS stream.
///
/// Sendable because handler instances are shared across NIO event-loop
/// threads. Implementations should avoid storing mutable state outside
/// of explicit actor isolation.
public protocol StageHandler: Sendable {
    func handle(_ frame: ActivationFrame) -> AsyncStream<ActivationFrame>
}

/// Convenience handler that calls a closure per inbound frame. Mostly
/// useful in tests; production stages likely conform to `StageHandler`
/// directly to hold model state cleanly.
public struct ClosureStageHandler: StageHandler {
    public typealias Handler = @Sendable (ActivationFrame) -> AsyncStream<ActivationFrame>

    private let handler: Handler

    public init(_ handler: @escaping Handler) {
        self.handler = handler
    }

    public func handle(_ frame: ActivationFrame) -> AsyncStream<ActivationFrame> {
        handler(frame)
    }
}
