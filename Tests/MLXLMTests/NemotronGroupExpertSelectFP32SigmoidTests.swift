// Pin the fp32 sigmoid precision floor for `groupExpertSelect` (the
// shared MoE router used by NemotronH AND Hy3).
//
// Python reference: `mlx_lm/models/nemotron_h.py:324` and
// `mlx_lm/models/dots1.py:116` both do:
//
//     orig_scores = scores = mx.sigmoid(gates.astype(mx.float32))
//
// — explicit fp32 cast BEFORE sigmoid. The cast is INSIDE the helper, so
// every caller (NemotronH, Hy3, future dots1-style routers) inherits the
// precision floor regardless of input dtype.
//
// Swift `Libraries/MLXLLM/Models/NemotronH.swift:groupExpertSelect`
// previously did `sigmoid(gates)` directly. NemotronH's caller passes
// bf16 gates (no pre-cast at `MoEGate.callAsFunction:459`), so production
// NemotronH MoE routing was running sigmoid on bf16 — the exact same
// precision drift that the Hy3 fp32 lm_head fix addresses, just at the
// MoE gate. (Hy3 pre-casts at `Hy3MoEGate.callAsFunction:307`, so Hy3
// was incidentally correct; the bug only fired for NemotronH.)
//
// Source-coverage style — the precision contract is structural and
// pinning it textually catches regressions without needing a real-model
// run. A future contributor that "simplifies" the cast away would be
// caught at test time.

import Foundation
import Testing

@Suite("NemotronH/Hy3 groupExpertSelect fp32-sigmoid precision floor")
struct NemotronGroupExpertSelectFP32SigmoidTests {

    private static func source(_ relativePath: String) throws -> String {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: repo.appendingPathComponent(relativePath), encoding: .utf8)
    }

    @Test("groupExpertSelect casts gates to fp32 before sigmoid (matches Python dots1/nemotron_h)")
    func groupExpertSelectCastsToFP32() throws {
        let src = try Self.source("Libraries/MLXLLM/Models/NemotronH.swift")

        // The fp32 cast must be on the line that initialises origScores.
        // Pin both the cast pattern AND the comment that explains why it's there
        // (so a future "cleanup" PR can't strip the comment without also
        // exposing the un-cast sigmoid line).
        #expect(
            src.contains("sigmoid(gates.asType(.float32))"),
            "groupExpertSelect must pre-cast gates to fp32 before sigmoid — see dots1.py:116 / nemotron_h.py:324.")

        // The previous bare `sigmoid(gates)` form must not be reintroduced.
        // We accept the line as long as it's `sigmoid(gates.asType(.float32))`.
        // A bare `sigmoid(gates)` (without `.asType(.float32)`) immediately
        // adjacent to the `origScores` assignment is the regression we're guarding.
        #expect(
            !src.contains("let origScores = sigmoid(gates)"),
            "groupExpertSelect must NOT use bare `sigmoid(gates)` — the bf16 sigmoid output drifts top-k expert picks (see Hy3 fp32 lm_head comment for the same class of drift).")
    }
}
