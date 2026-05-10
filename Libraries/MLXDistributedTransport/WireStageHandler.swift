import Foundation
import NIOCore

/// Runtime hook for a typed distributed stage.
///
/// Implementations are expected to own model/stage state. This protocol is
/// still tensor-agnostic: real MLX activation execution should decode the
/// `WireActivationForwardPayload.bytes` field into the runtime's native tensor
/// representation at the edge of the stage implementation.
public protocol WireStageRuntime: Sendable {
    func handlePrefill(_ request: WirePrefillRequestPayload) -> AsyncStream<WireStageResponse>
    func handleActivation(_ activation: WireActivationForwardPayload) -> AsyncStream<WireStageResponse>
}

public enum WireStageResponse: Sendable {
    case tokenStream(WireTokenStreamPayload)
    case tokensComplete(WireTokensCompletePayload)
    case activationForward(WireActivationForwardPayload)
    case error(String)

    func frame(allocator: ByteBufferAllocator = ByteBufferAllocator()) -> ActivationFrame {
        switch self {
        case .tokenStream(let payload):
            return ActivationFrame(
                frameType: .tokenStream,
                payload: (try? WirePayloadCodec.encode(payload, allocator: allocator))
                    ?? Self.errorPayload("failed to encode tokenStream", allocator: allocator)
            )
        case .tokensComplete(let payload):
            return ActivationFrame(
                frameType: .tokensComplete,
                payload: (try? WirePayloadCodec.encode(payload, allocator: allocator))
                    ?? Self.errorPayload("failed to encode tokensComplete", allocator: allocator)
            )
        case .activationForward(let payload):
            return ActivationFrame(
                frameType: .activationsForward,
                payload: (try? WirePayloadCodec.encode(payload, allocator: allocator))
                    ?? Self.errorPayload("failed to encode activationsForward", allocator: allocator)
            )
        case .error(let message):
            return ActivationFrame(frameType: .error, payload: Self.errorPayload(message, allocator: allocator))
        }
    }

    private static func errorPayload(
        _ message: String,
        allocator: ByteBufferAllocator
    ) -> ByteBuffer {
        var buffer = allocator.buffer(capacity: message.utf8.count)
        buffer.writeString(message)
        return buffer
    }
}

/// StageHandler adapter for the typed v1 wire payload contract.
///
/// `PipelineStageServer` remains frame-oriented. This adapter decodes supported
/// frame payloads into typed requests and rejects unknown or malformed frames
/// with a single `.error` reply.
public struct WireStageHandler: StageHandler {
    private let runtime: any WireStageRuntime
    private let allocator: ByteBufferAllocator

    public init(
        runtime: any WireStageRuntime,
        allocator: ByteBufferAllocator = ByteBufferAllocator()
    ) {
        self.runtime = runtime
        self.allocator = allocator
    }

    public func handle(_ frame: ActivationFrame) -> AsyncStream<ActivationFrame> {
        switch frame.frameType {
        case .prefillRequest:
            do {
                let request = try WirePayloadCodec.decode(
                    WirePrefillRequestPayload.self,
                    from: frame.payload
                )
                return map(runtime.handlePrefill(request))
            } catch {
                return oneError("failed to decode prefillRequest: \(String(describing: error))")
            }

        case .activationsForward:
            do {
                let activation = try WirePayloadCodec.decode(
                    WireActivationForwardPayload.self,
                    from: frame.payload
                )
                return map(runtime.handleActivation(activation))
            } catch {
                return oneError("failed to decode activationsForward: \(String(describing: error))")
            }

        default:
            return oneError("unsupported stage frame type: \(frame.frameType)")
        }
    }

    private func map(_ stream: AsyncStream<WireStageResponse>) -> AsyncStream<ActivationFrame> {
        AsyncStream { continuation in
            Task {
                for await response in stream {
                    continuation.yield(response.frame(allocator: allocator))
                }
                continuation.finish()
            }
        }
    }

    private func oneError(_ message: String) -> AsyncStream<ActivationFrame> {
        AsyncStream { continuation in
            continuation.yield(WireStageResponse.error(message).frame(allocator: allocator))
            continuation.finish()
        }
    }
}
