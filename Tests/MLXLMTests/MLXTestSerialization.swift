import Foundation

private let mlxTestSerializationLock = NSRecursiveLock()

final class MLXTestSerializationToken {
    private var unlocked = false

    func unlock() {
        guard !unlocked else { return }
        unlocked = true
        mlxTestSerializationLock.unlock()
    }

    deinit {
        unlock()
    }
}

func lockSerializedMLXTest() -> MLXTestSerializationToken {
    mlxTestSerializationLock.lock()
    return MLXTestSerializationToken()
}
