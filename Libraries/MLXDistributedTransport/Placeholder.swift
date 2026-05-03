// Placeholder source so MLXDistributedTransport has a compilable file
// before the real transport types land in Phase 2 Tasks 2-7. Removed
// once `ActivationFrame` and `PipelineStageServer` arrive.
import NIOCore

public enum MLXDistributedTransportPlaceholder {
    /// Sanity-check that NIO is linked correctly.
    public static func sanityCheckNIO() -> Bool {
        let buf = ByteBufferAllocator().buffer(capacity: 4)
        return buf.capacity == 4
    }
}
