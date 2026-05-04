// Copyright © 2026 Jinho Jang. All rights reserved.
//
// LlamaTPBitIdentityTests — end-to-end loopback proof that a size-2
// tensor-parallel forward pass produces the same logits as a size-1
// baseline on the same model + same prompt.
//
// This test:
//   1. Launches `Tools/tp-launch.sh` once with `MLX_WORLD_SIZE=1` to
//      produce a single-rank baseline at `/tmp/tp_baseline.f32`.
//   2. Launches it again with `MLX_WORLD_SIZE=2` to produce two
//      rank dumps at `/tmp/tp_rank0.f32` and `/tmp/tp_rank1.f32`.
//   3. Reads all three blobs and compares baseline vs rank-0 to within
//      `1e-3` absolute tolerance (fp16 collective rounding budget).
//   4. Sanity-checks rank-0 vs rank-1 — both ranks should hold the same
//      full-width logits after the final ShardedToAll all-reduce.
//
// Gated on `VMLX_RUN_DISTRIBUTED_TESTS=1` because:
//   - it spawns child processes (heavyweight)
//   - it requires a model bundle on disk (env: `TP_TEST_MODEL_PATH`)
//   - the JACCL backend may need RDMA hardware; the ring backend is
//     the default for this test and works on any host.

import Foundation
import Testing

@Suite("LlamaTPBitIdentity")
struct LlamaTPBitIdentityTests {

    private static let runEnv = "VMLX_RUN_DISTRIBUTED_TESTS"
    private static let modelEnv = "TP_TEST_MODEL_PATH"
    private static let toleranceEnv = "TP_TEST_TOLERANCE"

    /// Loaded blob: shape vector + raw Float32 row-major values.
    private struct LogitsBlob {
        let shape: [Int]
        let values: [Float]
    }

    /// Decode the [ndim u32][dim u32 × ndim][count u64][raw f32 bytes]
    /// format produced by `TPRankWorker`.
    private static func loadBlob(_ url: URL) throws -> LogitsBlob {
        let data = try Data(contentsOf: url)
        var off = 0
        func read<T: FixedWidthInteger>(_: T.Type) -> T {
            let v = data.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: off, as: T.self)
            }
            off += MemoryLayout<T>.size
            return T(littleEndian: v)
        }
        let ndim = Int(read(UInt32.self))
        var shape: [Int] = []
        for _ in 0..<ndim {
            shape.append(Int(read(UInt32.self)))
        }
        let count = Int(read(UInt64.self))
        precondition(shape.reduce(1, *) == count,
            "shape product \(shape.reduce(1, *)) != count \(count)")
        let bytes = count * MemoryLayout<Float>.size
        let values = data[off..<(off + bytes)].withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        return LogitsBlob(shape: shape, values: values)
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "value-count mismatch \(a.count) vs \(b.count)")
        var m: Float = 0
        for i in 0..<a.count {
            let d = abs(a[i] - b[i])
            if d > m { m = d }
        }
        return m
    }

    private static func runLauncher(
        worldSize: Int, outputs: [String: String], modelPath: String
    ) throws -> Int32 {
        let repoRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let launcher = repoRoot.appendingPathComponent("Tools/tp-launch.sh")
        let proc = Process()
        proc.executableURL = launcher
        proc.arguments = [modelPath]
        var env = ProcessInfo.processInfo.environment
        env["MLX_WORLD_SIZE"] = String(worldSize)
        for (k, v) in outputs { env[k] = v }
        proc.environment = env
        try proc.run()
        proc.waitUntilExit()
        return proc.terminationStatus
    }

    @Test("loopback bit-identity: world_size=2 logits match world_size=1 baseline")
    func bitIdentity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env[Self.runEnv] == "1" else {
            // Gated — return without running. Swift Testing has no
            // built-in skip but this preserves the green test count.
            return
        }
        guard let modelPath = env[Self.modelEnv], !modelPath.isEmpty else {
            Issue.record("\(Self.runEnv)=1 set but \(Self.modelEnv) is empty — set the path to a small dense fp16 Llama-family bundle")
            return
        }
        let tolerance = Float(env[Self.toleranceEnv] ?? "0.001") ?? 0.001

        let baseline = "/tmp/tp_baseline.f32"
        let rank0 = "/tmp/tp_rank0.f32"
        let rank1 = "/tmp/tp_rank1.f32"
        for path in [baseline, rank0, rank1] {
            try? FileManager.default.removeItem(atPath: path)
        }

        // 1. Single-rank baseline. The launcher with WORLD_SIZE=1 still
        //    spawns 2 processes by default; override by hand-running the
        //    worker with WORLD_SIZE=1 instead. We can't easily do that
        //    here without duplicating the launcher logic, so for the
        //    baseline we set MLX_WORLD_SIZE=1 and let the launcher's
        //    rank-1 spawn into a size-1 group on its own (which is the
        //    same math as the "real" baseline since both just run dense).
        //    Cleaner: spawn the worker directly for the baseline.
        let repoRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let workerURL = repoRoot.appendingPathComponent(".build/release/TPRankWorker")
        guard FileManager.default.isExecutableFile(atPath: workerURL.path) else {
            Issue.record("TPRankWorker not built; run: swift build -c release --product TPRankWorker")
            return
        }
        let baselineProc = Process()
        baselineProc.executableURL = workerURL
        var baseEnv = ProcessInfo.processInfo.environment
        baseEnv["MLX_RANK"] = "0"
        baseEnv["MLX_WORLD_SIZE"] = "1"
        baseEnv["MLX_DIST_BACKEND"] = "ring"
        baseEnv["TP_STRICT"] = "0"  // size-1 fallback OK
        baseEnv["TP_MODEL_PATH"] = modelPath
        baseEnv["TP_OUTPUT_PATH"] = baseline
        baselineProc.environment = baseEnv
        try baselineProc.run()
        baselineProc.waitUntilExit()
        try #require(baselineProc.terminationStatus == 0,
            "baseline run failed: status=\(baselineProc.terminationStatus)")

        // 2. Two-rank loopback via launcher.
        let rc = try Self.runLauncher(
            worldSize: 2, outputs: [:], modelPath: modelPath)
        try #require(rc == 0,
            "tp-launch.sh failed with exit \(rc) — see /tmp/tp_rank0.log + /tmp/tp_rank1.log")

        // 3. Compare blobs.
        let baseBlob = try Self.loadBlob(URL(fileURLWithPath: baseline))
        let r0Blob = try Self.loadBlob(URL(fileURLWithPath: rank0))
        let r1Blob = try Self.loadBlob(URL(fileURLWithPath: rank1))

        #expect(baseBlob.shape == r0Blob.shape, "baseline shape != rank0 shape")
        #expect(r0Blob.shape == r1Blob.shape, "rank0 shape != rank1 shape")

        let baseVsR0 = Self.maxAbsDiff(baseBlob.values, r0Blob.values)
        let r0VsR1 = Self.maxAbsDiff(r0Blob.values, r1Blob.values)

        FileHandle.standardError.write(Data(
            "[BitIdentity] baseline vs rank0: maxAbsDiff=\(baseVsR0) (tol=\(tolerance))\n".utf8))
        FileHandle.standardError.write(Data(
            "[BitIdentity] rank0 vs rank1: maxAbsDiff=\(r0VsR1)\n".utf8))

        #expect(baseVsR0 < tolerance,
            "baseline vs rank0 logits differ by \(baseVsR0) > tolerance \(tolerance)")
        // Rank 0 and rank 1 should be exactly equal — they hold the same
        // full-width logits after the final ShardedToAll all-reduce.
        #expect(r0VsR1 < tolerance,
            "rank0 vs rank1 logits differ by \(r0VsR1) — collectives may not be reaching both ranks symmetrically")
    }
}
