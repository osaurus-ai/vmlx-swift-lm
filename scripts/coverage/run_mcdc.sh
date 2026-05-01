#!/usr/bin/env bash
# Run the MC/DC test suites + extract branch coverage.
#
# Usage:
#   scripts/coverage/run_mcdc.sh [--osaurus PATH_TO_OSAURUS_CORE]
#
# By default runs only the vmlx-side MC/DC suites and prints branch
# coverage for the engine files under MC/DC discipline. Pass
# `--osaurus PATH` to additionally run the osaurus-side suites.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OSAURUS_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --osaurus) OSAURUS_PATH="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,11p' "$0" | sed 's/^# //'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "===================================================================="
echo "vmlx-swift-lm MC/DC coverage @ $(git rev-parse --short HEAD)"
echo "===================================================================="

# The two MC/DC-shaped suites currently in the engine. Add new suite
# names here when shipping new MC/DC tests for engine guards.
VMLX_FILTER="EvaluateRepetitionPenaltyMCDCTests|ReasoningStampMCDCTests|lagunaResolvesToThinkXml"

swift test \
  --enable-code-coverage \
  --filter "$VMLX_FILTER" 2>&1 | tail -10

# Locate profdata + test binary. The test binary path differs on
# different SwiftPM versions / Xcode toolchains, so glob.
PROFDATA=$(find .build -name 'default.profdata' -path '*/codecov/*' 2>/dev/null | head -1)
if [[ -z "$PROFDATA" ]]; then
  PROFDATA=$(find .build -name 'default.profdata' 2>/dev/null | head -1)
fi
TEST_BIN=$(find .build -name 'vmlx-swift-lmPackageTests.xctest' -type d 2>/dev/null | head -1)
if [[ -n "$TEST_BIN" ]]; then
  TEST_BIN="$TEST_BIN/Contents/MacOS/vmlx-swift-lmPackageTests"
fi

if [[ -z "$PROFDATA" || -z "$TEST_BIN" ]]; then
  echo "warn: could not locate profdata ($PROFDATA) or test binary ($TEST_BIN)"
  echo "      branch-coverage report skipped — tests still ran above"
  exit 0
fi

echo ""
echo "Branch coverage on MC/DC-targeted engine files:"
echo "----------------------------------------------------------------"
xcrun llvm-cov report \
  "$TEST_BIN" \
  -instr-profile="$PROFDATA" \
  Libraries/MLXLMCommon/Evaluate.swift \
  Libraries/MLXLMCommon/ReasoningParser.swift \
  --show-branches=count 2>/dev/null || echo "(report unavailable — files may not be instrumented if not exercised)"

if [[ -n "$OSAURUS_PATH" ]]; then
  echo ""
  echo "===================================================================="
  echo "OsaurusCore MC/DC coverage @ $OSAURUS_PATH"
  echo "===================================================================="
  pushd "$OSAURUS_PATH" >/dev/null
  swift test \
    --enable-code-coverage \
    --filter "IsKnownHybridModelMCDCTests|MaterializeMediaDataUrlMCDCTests" 2>&1 | tail -10

  PROFDATA=$(find .build -name 'default.profdata' 2>/dev/null | head -1)
  TEST_BIN=$(find .build -name 'OsaurusCorePackageTests.xctest' -type d 2>/dev/null | head -1)
  if [[ -n "$TEST_BIN" ]]; then
    TEST_BIN="$TEST_BIN/Contents/MacOS/OsaurusCorePackageTests"
  fi
  if [[ -n "$PROFDATA" && -n "$TEST_BIN" ]]; then
    echo ""
    echo "Branch coverage on MC/DC-targeted host files:"
    xcrun llvm-cov report "$TEST_BIN" -instr-profile="$PROFDATA" \
      Sources/OsaurusCore/Services/ModelRuntime.swift \
      --show-branches=count 2>/dev/null || true
  fi
  popd >/dev/null
fi

echo ""
echo "Done. See Libraries/MLXLMCommon/BatchEngine/MCDC-COVERAGE-STRATEGY.md"
echo "for the test-design contract and the deferred-coverage list."
