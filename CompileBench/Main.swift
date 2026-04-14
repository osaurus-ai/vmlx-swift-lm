import Foundation
import MLX
import MLXNN

@main
struct CompileBench {
    static func main() {
        // Override via BENCH_N / BENCH_DIM / BENCH_MODE env vars to sweep.
        let N = Int(ProcessInfo.processInfo.environment["BENCH_N"] ?? "10000") ?? 10000
        let DIM = Int(ProcessInfo.processInfo.environment["BENCH_DIM"] ?? "16") ?? 16
        let mode = ProcessInfo.processInfo.environment["BENCH_MODE"] ?? "compute_g"

        let aLog = MLXRandom.uniform(low: Float(0), high: Float(1), [DIM]).asType(.float32)
        let aArr = MLXRandom.uniform(low: Float(0), high: Float(1), [DIM])
        let dtBias = MLXRandom.uniform(low: Float(0), high: Float(1), [DIM])
        eval(aLog, aArr, dtBias)

        @Sendable func eager(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
            // Isolation modes — each does a minimal chain that tests one
            // suspect primitive. Compare against Python for per-op attribution.
            switch mode {
            case "astype_f32":     return aLog.asType(.float32)
            case "astype_bf16":    return aLog.asType(.bfloat16)
            case "astype_back":    return aLog.asType(.float32).asType(.bfloat16)
            case "softplus":       return softplus(a)
            case "unary_neg":      return -a
            case "mul":            return a * dtBias
            case "exp4":           return exp(exp(exp(exp(a))))
            case "neg_mul":        return -a * dtBias
            case "compute_g_no_cast":
                // compute_g minus the two type casts, to isolate whether the
                // cast path is the slow one.
                return exp(-exp(aLog) * softplus(a + dtBias))
            default:
                let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
                return decay.asType(a.dtype)
            }
        }

        let compiled: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
            compile(shapeless: true, eager)

        eval(eager(aLog, aArr, dtBias))
        eval(compiled(aLog, aArr, dtBias))

        let t0e = CFAbsoluteTimeGetCurrent()
        var resultsE: [MLXArray] = []
        resultsE.reserveCapacity(N)
        for _ in 0..<N {
            resultsE.append(eager(aLog, aArr, dtBias))
        }
        eval(resultsE)
        let eagerTotal = CFAbsoluteTimeGetCurrent() - t0e

        let t0c = CFAbsoluteTimeGetCurrent()
        var resultsC: [MLXArray] = []
        resultsC.reserveCapacity(N)
        for _ in 0..<N {
            resultsC.append(compiled(aLog, aArr, dtBias))
        }
        eval(resultsC)
        let compiledTotal = CFAbsoluteTimeGetCurrent() - t0c

        let eagerUs = eagerTotal * 1e6 / Double(N)
        let compiledUs = compiledTotal * 1e6 / Double(N)

        print("Swift mlx-swift  N=\(N)  DIM=\(DIM) (batched, single eval)")
        print(String(format: "  eager:    %8.2f us/call  total %.1f ms", eagerUs, eagerTotal * 1000))
        print(String(format: "  compiled: %8.2f us/call  total %.1f ms", compiledUs, compiledTotal * 1000))
        print(String(format: "  speedup:  %.2fx", eagerTotal / compiledTotal))
        print(String(format: "  savings:  %.2f us/call (%.0f%%)",
            eagerUs - compiledUs, (eagerUs - compiledUs) / eagerUs * 100))
    }
}
