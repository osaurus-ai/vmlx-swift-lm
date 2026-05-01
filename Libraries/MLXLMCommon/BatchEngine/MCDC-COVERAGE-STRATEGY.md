# MC/DC coverage strategy — vmlx-swift-lm engine + osaurus host

Pinned to vmlx revision `2a6c965`. This is the test-design contract
for every fix and feature shipped to `main`. New decisions added to
the engine MUST be accompanied by an MC/DC-shaped test suite before
the commit lands. New tests WITHOUT MC/DC structure are still
welcome but do not satisfy this contract.

---

## What MC/DC means here

Modified Condition / Decision Coverage requires that **every condition
within a decision independently affect the decision's outcome**. For
an N-condition decision (N-AND or N-OR chain) the minimum case count
is N+1: one master case where the decision is TRUE, and N pair-cases
where exactly one condition flips while all others remain at the
"determining" value (TRUE for AND, FALSE for OR).

**Why this discipline matters for an inference engine**:

- Engine guards (Bug-3a-style) are silent killers — a regression
  doesn't blow up at compile time, it crashes on the user's first
  decode against a specific weight bundle. Branch coverage isn't
  enough; we need pair-coverage that proves each branch independently
  matters.
- Multi-condition guards in cache topology / reasoning parsers /
  sampler dispatch evolve rapidly. Without MC/DC, a refactor that
  collapses two conditions into one (or splits one into two) can
  pass all existing tests while shifting behavior at boundaries.
- Truth-table-shaped tests double as documentation. Reading the
  table tells the next maintainer *exactly* which inputs drive
  which output. Comments rot; truth tables don't.

---

## What's covered today (engine + host fixes shipped on tip)

### vmlx-swift-lm

| File | Fix / feature | MC/DC test file | Cases | Coverage status |
|---|---|---|---|---|
| `Libraries/MLXLMCommon/Evaluate.swift:289` | Bug 3a — `repetitionPenalty != 1.0` no-op guard (4-AND chain) | `Tests/MLXLMTests/EvaluateRepetitionPenaltyMCDCTests.swift` | 11 | full table + boundary rows + composition |
| `Libraries/MLXLMCommon/ReasoningParser.swift:467` | `reasoningStampFromModelType` (D1 ∧ guard, D2 prefix branch, D3 disjunction over 9 prefixes) | `Tests/MLXLMTests/ReasoningStampMCDCTests.swift` | 17 | each prefix independently flips D3 + case-folding + bare-prefix boundary |
| `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift:181-310` | Capability-name → format precedence (multi-stage resolver: explicit > jangConfig > model_type heuristic) | (existing `ReasoningParserTests.swift` covers per-prefix; precedence chain not yet MC/DC-shaped) | — | **gap — see "deferred" below** |

### osaurus host

| File | Fix / feature | MC/DC test file | Cases | Coverage status |
|---|---|---|---|---|
| `Packages/OsaurusCore/Services/ModelRuntime.swift:462` | `isKnownHybridModel` (3 OR-blocks: nemotron / qwen+holo / minimax) | `Tests/Service/IsKnownHybridModelMCDCTests.swift` | 14 | each substring independently flips block + case-folding + master-FALSE |
| `Packages/OsaurusCore/Services/ModelRuntime.swift:1057` | `materializeMediaDataUrl` audit fix (D1 prefix guard, D2 comma, D3 base64, D6 audio-mime gate) | `Tests/Model/MaterializeMediaDataUrlMCDCTests.swift` | 11 | **D6 regression row** + each switch arm + invalid-payload guards |

---

## What's deferred (deliberate gaps with rationale)

Each of these is genuinely non-trivial to MC/DC-cover and the right
investment is more than a truth-table. Tracked as follow-up tasks
rather than blocking ship.

### 1. State-machine: `ReasoningParser` tag latching

The parser holds implicit state (`startInReasoning`, current channel,
unclosed-tag buffer). MC/DC over a single `if` doesn't catch wrong
state transitions across token streams. **Right tool**: property-based
tests over arbitrary token sequences with invariants (e.g. "no
`<think>` chars ever appear in `.chunk` events") — already exist in
`ReasoningParserTests.swift`'s 23 streaming tests, just not labeled
MC/DC.

### 2. Concurrency: `CacheCoordinator.setHybrid` race

`setHybrid(true)` flag and `admitPendingRequests` auto-flip can both
fire concurrently across decode threads. Single-thread MC/DC misses
ordering bugs. **Right tool**: `CacheCoordinatorConcurrencyTests`
(`testB8DiskConcurrencyStress`, `testConcurrentHybridFlagToggles`)
already do stress, but not MC/DC-shaped over the actual flag
transitions.

### 3. Floating-point edge cases for Bug 3a

`repetitionPenalty != 1.0` is true for `Float.nan`. NaN would build a
RepetitionContext. Whether that's intended or a latent bug is unclear.
**Right tool**: explicit row added once we decide the contract — the
test file leaves space for `test_boundary_penaltyNaN_…`.

### 4. `Mistral3Text.newCache` per-layer dispatch boundary

```swift
if layer.useSliding, let slidingWindow = args.slidingWindow {
    return RotatingKVCache(maxSize: slidingWindow)
} else {
    return KVCacheSimple()
}
```

**Real concern**: if a config has `useSliding=true` but
`sliding_window` field is missing/null, the optional binding fails and
the layer silently falls through to `KVCacheSimple` — wrong cache
type, no error. **Right tool**: refactor `newCache` to take pure
inputs (`(useSliding, slidingWindow?) → KVCache`) and unit-test the
function directly. Currently the function requires a fully-initialized
model. Tracked as engine refactor.

### 5. Multi-stage resolver precedence

`LLMModelFactory.swift:986-1010` — tool format resolution chain:
1. Explicit capability name (`jangConfig.tool_parser`)
2. `jangConfig.tool_parser` ≠ model_type heuristic disagreement → log + prefer explicit
3. Model_type heuristic fallback

MC/DC means: each stage's match must independently determine the
final format. Need a 3-stage truth table of (explicit?, jangConfig?,
model_type) input triples and the resolved output. Currently exercised
by per-family tests but no precedence-table.

### 6. Cross-repo integration: disableThinking → additionalContext → jinja → ReasoningParser

osaurus tests cover osaurus code; vmlx tests cover vmlx code. There's
no test that flips the `disableThinking` model option at the host
layer and asserts the vmlx-side reasoning parser actually deactivates.
**Right tool**: integration test in `osaurus-staging` that loads a
real reasoning-capable bundle, sets `disableThinking=true`, runs a
single decode, and asserts zero `.reasoning` events emitted. Heavy
but very high signal.

### 7. Bug 1 mlx-swift `Device::clear_library` env reverter

```cpp
if (getenv("MLX_CLEAR_LIBRARY_RELEASE") == "1") { /* old release path */ }
else { /* new leak-stale-pipelines path */ }
```

Single-condition decision, MC/DC trivial. The interesting test is
"old path under concurrent kernel dispatch crashes; new path doesn't"
— that's a stress test, not MC/DC. Existing `StabilityBench` S7
exercises the new path; the old path's crash is the bug being fixed.

---

## Adding MC/DC tests for new fixes — checklist

When you introduce a new decision in engine code, follow this
sequence before the commit lands:

1. **Identify the decision**: write the boolean expression in a
   comment (e.g. `D = A ∧ B ∧ C ∧ D`).
2. **Build the truth table**: for an N-AND, write N+1 rows; for an
   N-OR, write N+1 rows; mixed expressions need (Conditions+1) rows
   minimum.
3. **One test per row**: each test name should reference the
   independence claim (e.g. `test_T4_penaltyOne_independence`).
4. **Pair-completeness check**: each F-row must pair with the
   master-T row by flipping exactly one input.
5. **Add boundary rows**: floating-point edge cases (NaN, ±∞,
   nextUp/nextDown), integer edge cases (Int.max, -1, 0),
   string edge cases (empty, whitespace-only, locale-sensitive).
6. **Add composition rows**: how does the decision interact with
   adjacent decisions? (e.g. Bug 3a's repetition guard with
   non-zero presence/frequency penalties.)
7. **Add a master-FALSE row**: the "all-default" case where no
   branch fires. Catches drift where someone adds a new arm without
   updating the default-arm assumption.

---

## Coverage extraction

Run with branch coverage instrumentation enabled:

```bash
# vmlx-swift-lm
cd vmlx-swift-lm
swift test --enable-code-coverage \
    --filter "EvaluateRepetitionPenaltyMCDCTests|ReasoningStampMCDCTests"

# Find the .profdata
PROFDATA=$(find .build -name 'default.profdata' | head -1)
TEST_BIN=$(find .build -name 'vmlx-swift-lmPackageTests.xctest' | head -1)/Contents/MacOS/vmlx-swift-lmPackageTests

# Branch-level coverage report for the files under test
xcrun llvm-cov report \
    "$TEST_BIN" \
    -instr-profile="$PROFDATA" \
    Libraries/MLXLMCommon/Evaluate.swift \
    Libraries/MLXLMCommon/ReasoningParser.swift \
    --show-branches=count
```

Branch coverage % above 95% on the MC/DC-targeted files is the bar.
Lower than that means we missed an arm. Run on every PR that
touches engine code.

For osaurus host:

```bash
cd Packages/OsaurusCore
swift test --enable-code-coverage \
    --filter "IsKnownHybridModelMCDCTests|MaterializeMediaDataUrlMCDCTests"
PROFDATA=$(find .build -name 'default.profdata' | head -1)
TEST_BIN=$(find .build -name 'OsaurusCorePackageTests.xctest' | head -1)/Contents/MacOS/OsaurusCorePackageTests
xcrun llvm-cov report "$TEST_BIN" -instr-profile="$PROFDATA" \
    Sources/OsaurusCore/Services/ModelRuntime.swift \
    --show-branches=count
```

---

## Where this strategy lives long-term

This document is the source of truth for:

1. Which decisions are under MC/DC discipline
2. Which decisions are deferred and why
3. The checklist contributors follow when adding new decisions

When you add a new MC/DC test file, append a row to the "What's
covered today" tables. When you discover a new decision that warrants
deferral, document it in the "What's deferred" section with the
specific reason.

The bar to graduate a deferred item to "covered" is the same as
shipping any test: pair-completeness, boundary rows, composition,
master-FALSE. No exceptions for "the function is hard to test" — if
it's hard, refactor it to be testable.
