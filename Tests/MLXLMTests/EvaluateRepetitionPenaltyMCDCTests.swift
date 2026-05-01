// Copyright © 2026 osaurus.
//
// MC/DC (Modified Condition / Decision Coverage) tests for the
// `processor()` repetition-context guard added in Bug 3a fix
// (Evaluate.swift:289-292).
//
// Decision under test:
//
//   if let repetitionPenalty,           // A: P != nil
//      repetitionPenalty != 0,          // B: P ≠ 0
//      repetitionPenalty != 1.0,        // C: P ≠ 1.0   ← Bug 3a focus
//      repetitionContextSize > 0 {      // D: S > 0
//       repetitionContext = RepetitionContext(...)
//   }
//
// Decision = A ∧ B ∧ C ∧ D
//
// MC/DC for an N-AND chain requires pairs that flip exactly one input
// while all others remain TRUE, plus an all-true master case. Minimum
// 5 cases for 4 inputs:
//
// | Case | A | B | C | D | Decision | Independence demonstrated |
// |------|---|---|---|---|----------|---------------------------|
// | T1   | T | T | T | T |   TRUE   | master                    |
// | T2   | F | T | T | T |   FALSE  | A (P=nil)                 |
// | T3   | T | F | T | T |   FALSE  | B (P=0)                   |
// | T4   | T | T | F | T |   FALSE  | C (P=1.0) ← Bug 3a        |
// | T5   | T | T | T | F |   FALSE  | D (S=0)                   |
//
// Each F-row pairs with T1 to demonstrate that flipping that input
// (with all others held TRUE) flips the decision — the MC/DC criterion.

import MLX
import MLXLMCommon
import XCTest

public class EvaluateRepetitionPenaltyMCDCTests: XCTestCase {

    // MARK: - Master case (all conditions TRUE → decision TRUE)

    func test_T1_allConditionsTrue_buildsContext() {
        // A=T (1.1 != nil), B=T (1.1 != 0), C=T (1.1 != 1.0), D=T (20 > 0)
        let processor = GenerateParameters(
            repetitionPenalty: 1.1,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNotNil(processor, "all conditions TRUE must build RepetitionContext")
    }

    // MARK: - Independence pairs (flip exactly one input vs T1)

    func test_T2_penaltyNil_independence() {
        // A=F (nil), B=T (vacuous), C=T (vacuous), D=T (20 > 0)
        let processor = GenerateParameters(
            repetitionPenalty: nil,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNil(processor, "A=F (penalty nil) must produce no context")
    }

    func test_T3_penaltyZero_independence() {
        // A=T (0.0 != nil), B=F (0.0 == 0), C=T (0.0 != 1.0), D=T (20 > 0)
        let processor = GenerateParameters(
            repetitionPenalty: 0.0,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNil(processor, "B=F (penalty == 0) must produce no context")
    }

    func test_T4_penaltyOne_independence() {
        // A=T (1.0 != nil), B=T (1.0 != 0), C=F (1.0 == 1.0), D=T (20 > 0)
        // This is the Bug 3a regression — without the C guard, RepetitionContext
        // gets built for the math no-op and trips MLXArray subscript bounds check.
        let processor = GenerateParameters(
            repetitionPenalty: 1.0,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNil(processor, "C=F (penalty == 1.0) must produce no context — Bug 3a")
    }

    func test_T5_contextSizeZero_independence() {
        // A=T, B=T, C=T, D=F (S == 0)
        let processor = GenerateParameters(
            repetitionPenalty: 1.1,
            repetitionContextSize: 0
        ).processor()
        XCTAssertNil(processor, "D=F (context size 0) must produce no context")
    }

    // MARK: - Boundary + sanity rows (lock subtle cases the table doesn't cover)

    /// `Float(1.0).nextUp` is the smallest representable value strictly
    /// greater than 1.0. C must remain TRUE here (penalty ≠ 1.0 strictly),
    /// so the context IS built. Locks against a future regression where
    /// someone changes `!= 1.0` to a tolerance window.
    func test_boundary_penaltyJustAboveOne_buildsContext() {
        let processor = GenerateParameters(
            repetitionPenalty: Float(1.0).nextUp,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNotNil(processor, "penalty strictly > 1.0 must build context")
    }

    /// `Float(1.0).nextDown` — symmetric to nextUp, strictly below 1.0.
    func test_boundary_penaltyJustBelowOne_buildsContext() {
        let processor = GenerateParameters(
            repetitionPenalty: Float(1.0).nextDown,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNotNil(processor, "penalty strictly < 1.0 must build context")
    }

    /// Negative penalty is unusual but mathematically meaningful (boost
    /// repeated tokens). Not 0, not 1.0, not nil → context builds.
    func test_boundary_penaltyNegative_buildsContext() {
        let processor = GenerateParameters(
            repetitionPenalty: -1.0,
            repetitionContextSize: 20
        ).processor()
        XCTAssertNotNil(processor, "negative penalty must build context (not nil/0/1.0)")
    }

    /// Negative context size → D=FALSE → no context. The guard is
    /// `> 0` so 0 and any negative value both fail.
    func test_boundary_contextSizeNegative_independence() {
        let processor = GenerateParameters(
            repetitionPenalty: 1.1,
            repetitionContextSize: -1
        ).processor()
        XCTAssertNil(processor, "negative context size must produce no context")
    }

    /// Composition with other penalty processors — the C guard must
    /// only suppress repetition; presence/frequency penalties must still
    /// build their respective processors. Bug 3a fix must NOT regress
    /// other penalty paths.
    func test_composition_repetitionOneButOtherPenaltiesNonZero_stillBuildsProcessor() {
        let processor = GenerateParameters(
            repetitionPenalty: 1.0,             // C=F → no RepetitionContext
            repetitionContextSize: 20,
            presencePenalty: 0.5,               // → builds PresencePenaltyContext
            frequencyPenalty: 0.5               // → builds FrequencyPenaltyContext
        ).processor()
        XCTAssertNotNil(processor,
            "non-zero presence/frequency penalties must build processor even when repetition=1.0")
    }

    /// Symmetric compose: penalty != 1.0 with zero presence/frequency —
    /// repetition context alone drives the processor.
    func test_composition_onlyRepetition_buildsProcessor() {
        let processor = GenerateParameters(
            repetitionPenalty: 1.1,
            repetitionContextSize: 20,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0
        ).processor()
        XCTAssertNotNil(processor, "non-1.0 repetition alone must build processor")
    }
}
