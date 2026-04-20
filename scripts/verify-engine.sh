#!/bin/bash
# BatchEngine verification runner.
#
# Exercises every engine bench scenario against locally-cached models and
# the unit-test suite. Prints a single-line verdict per scenario so
# operator output stays shallow. Any non-zero exit code surfaces as
# "FAIL" and the run continues to the next scenario — it's a cumulative
# smoke, not a fail-fast.
#
# Usage:
#   ./scripts/verify-engine.sh                # full sweep
#   ./scripts/verify-engine.sh --quick        # skip the slow 35B runs
#   ./scripts/verify-engine.sh --tests-only   # just the unit suite
#
# Environment overrides:
#   Q06B, Q36, VL4B, G4E2B: override default model paths.
#   BENCH_MAX_TOKENS: decoder limit per run (default 30).

set -u
QUICK=0
TESTS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --quick)       QUICK=1 ;;
    --tests-only)  TESTS_ONLY=1 ;;
    -h|--help)
      /usr/bin/sed -n '2,20p' "$0"
      exit 0 ;;
  esac
done

# Resolve script-relative repo root so the template path works regardless
# of where the script is invoked from.
SCRIPT_DIR="$(cd "$(/usr/bin/dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default model paths — override by exporting before invocation.
# Q06B is content-addressed in the HF hub cache; auto-resolve via snapshot
# walk if the default isn't set.
if [ -z "${Q06B:-}" ]; then
  _hub="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots"
  if [ -d "$_hub" ]; then
    Q06B="$(/usr/bin/find "$_hub" -maxdepth 1 -mindepth 1 -type d | /usr/bin/head -1)"
  else
    Q06B=""
  fi
fi
: "${Q36:=$HOME/.mlxstudio/models/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2}"
: "${VL4B:=$HOME/.mlxstudio/models/MLXModels/dealignai/Qwen3.5-VL-4B-JANG_4S-CRACK}"
: "${G4E2B:=$HOME/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit}"
: "${BENCH_MAX_TOKENS:=30}"
: "${G4_TEMPLATE:=$REPO_ROOT/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja}"

cd "$REPO_ROOT"

PASS=0
FAIL=0
SKIP=0
FAILURES=()

pass() { PASS=$((PASS + 1)); printf '  [ok]    %s\n' "$1"; }
fail() { FAIL=$((FAIL + 1)); FAILURES+=("$1"); printf '  [FAIL]  %s\n' "$1"; }
skip() { SKIP=$((SKIP + 1)); printf '  [skip]  %s\n' "$1"; }

run_scenario() {
  local label="$1"
  shift
  # Serialize — the scheduler + Metal can't handle overlapping bench runs.
  /usr/bin/pkill -9 RunBench 2>/dev/null || true
  /bin/sleep 1
  if "$@" > "/tmp/verify-engine-$$-$RANDOM.log" 2>&1; then
    pass "$label"
  else
    fail "$label (exit $?)"
  fi
}

check_model() {
  [ -e "$1" ]
}

# ---------------------------------------------------------------------------
echo "=== Building ==="
if ! swift build 2>&1 | /usr/bin/tail -1; then
  echo "Build failed — aborting."
  exit 1
fi
echo ""

# ---------------------------------------------------------------------------
echo "=== Unit tests (121 XCTest + Swift Testing suites) ==="
/usr/bin/pkill -9 xctest 2>/dev/null || true
/bin/sleep 1
if swift test \
  --filter "BatchCompile|CompilableCacheList|CompilableMambaCache|CompilableRotatingKVCache|CompilableTurboQuantKVCache|RotatingKVCacheCompile|MambaCacheCompile|TurboQuantCompile|SSMStateCache|JangTokenizerFallback|Gemma4ChatTemplateProbe|CacheCoordinatorRotating|CacheCoordinatorMediaSalt|CacheCoordinatorConcurrency|UserInputInitSemantics|BucketHandle|BatchEngineCompileWiring|ChatTemplateOverrideIntegration|TokenizerClassSubstitution|JANGTQKernelsTests|DDTreeDesignTests" \
  2>&1 | /usr/bin/grep -E "Executed [0-9]+ tests.*in " | /usr/bin/tail -1; then
  pass "unit tests"
else
  fail "unit tests"
fi
echo ""

if [ "$TESTS_ONLY" = "1" ]; then
  printf '=== Summary: %d passed, %d failed, %d skipped ===\n' "$PASS" "$FAIL" "$SKIP"
  [ "$FAIL" = "0" ]
  exit $?
fi

# ---------------------------------------------------------------------------
echo "=== Qwen3-0.6B-8bit (dense) ==="
if check_model "$Q06B"; then
  run_scenario "Qwen3-0.6B BENCH_BATCH_CHAT"        env BENCH_MODEL="$Q06B" BENCH_BATCH_CHAT=1        BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_CROSS_VALIDATE"    env BENCH_MODEL="$Q06B" BENCH_CROSS_VALIDATE=1    BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_CONCURRENT"  env BENCH_MODEL="$Q06B" BENCH_BATCH_CONCURRENT=1  BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_CACHE_HIT"   env BENCH_MODEL="$Q06B" BENCH_BATCH_CACHE_HIT=1   BENCH_MAX_TOKENS=20 .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_DISK_RESTORE" env BENCH_MODEL="$Q06B" BENCH_BATCH_DISK_RESTORE=1 BENCH_MAX_TOKENS=20 .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_PERSLOT_SAMPLER" env BENCH_MODEL="$Q06B" BENCH_BATCH_PERSLOT_SAMPLER=1 BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_B4"          env BENCH_MODEL="$Q06B" BENCH_BATCH_B4=1          BENCH_MAX_TOKENS=20 .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_B8"          env BENCH_MODEL="$Q06B" BENCH_BATCH_B4=1 BENCH_B_SIZE=8 BENCH_MAX_TOKENS=20 .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_CANCEL"      env BENCH_MODEL="$Q06B" BENCH_BATCH_CANCEL=1      BENCH_MAX_TOKENS=60 .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_LONG_CONTEXT" env BENCH_MODEL="$Q06B" BENCH_BATCH_LONG_CONTEXT=1 BENCH_LONG_LEN=2048 BENCH_MAX_TOKENS=20 .build/debug/RunBench
  run_scenario "Qwen3-0.6B BENCH_BATCH_TQ_B2"       env BENCH_MODEL="$Q06B" BENCH_BATCH_TQ_B2=1       BENCH_MAX_TOKENS=25 .build/debug/RunBench
else
  skip "Qwen3-0.6B not cached at $Q06B"
fi
echo ""

# ---------------------------------------------------------------------------
if [ "$QUICK" = "0" ]; then
  echo "=== Qwen3.6-35B-A3B-JANGTQ2 (hybrid SSM) ==="
  if check_model "$Q36"; then
    run_scenario "Qwen3.6-35B BENCH_BATCH_CHAT"         env BENCH_MODEL="$Q36" BENCH_BATCH_CHAT=1         BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
    run_scenario "Qwen3.6-35B BENCH_CROSS_VALIDATE"     env BENCH_MODEL="$Q36" BENCH_CROSS_VALIDATE=1     BENCH_MAX_TOKENS=25 .build/debug/RunBench
    run_scenario "Qwen3.6-35B BENCH_BATCH_CONCURRENT"   env BENCH_MODEL="$Q36" BENCH_BATCH_CONCURRENT=1   BENCH_MAX_TOKENS=30 .build/debug/RunBench
    run_scenario "Qwen3.6-35B BENCH_BATCH_CACHE_HIT"    env BENCH_MODEL="$Q36" BENCH_BATCH_CACHE_HIT=1    BENCH_MAX_TOKENS=20 .build/debug/RunBench
    run_scenario "Qwen3.6-35B BENCH_BATCH_DISK_RESTORE" env BENCH_MODEL="$Q36" BENCH_BATCH_DISK_RESTORE=1 BENCH_MAX_TOKENS=20 .build/debug/RunBench
  else
    skip "Qwen3.6-35B-JANGTQ2 not cached at $Q36"
  fi
  echo ""
else
  skip "Qwen3.6-35B (--quick)"
fi

# ---------------------------------------------------------------------------
echo "=== Qwen3.5-VL-4B-JANG_4S-CRACK (VL) ==="
if check_model "$VL4B"; then
  run_scenario "VL-4B BENCH_VL_BATCH_CHAT"        env BENCH_MODEL="$VL4B" BENCH_VL_BATCH_CHAT=1        BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  run_scenario "VL-4B BENCH_VL_CROSS_VALIDATE"    env BENCH_MODEL="$VL4B" BENCH_VL_CROSS_VALIDATE=1    BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  run_scenario "VL-4B BENCH_VL_BATCH_MEDIASALT"   env BENCH_MODEL="$VL4B" BENCH_VL_BATCH_MEDIASALT=1   BENCH_MAX_TOKENS=20 .build/debug/RunBench
  run_scenario "VL-4B BENCH_VL_BATCH_CACHE_HIT"   env BENCH_MODEL="$VL4B" BENCH_VL_BATCH_CACHE_HIT=1   BENCH_MAX_TOKENS=25 .build/debug/RunBench
  run_scenario "VL-4B BENCH_VL_VIDEO"             env BENCH_MODEL="$VL4B" BENCH_VL_VIDEO=1             BENCH_MAX_TOKENS=40 .build/debug/RunBench
else
  skip "Qwen3.5-VL-4B-JANG not cached at $VL4B"
fi
echo ""

# ---------------------------------------------------------------------------
VL9B="${VL9B:-$HOME/.mlxstudio/models/MLXModels/mlx-community/Qwen3.5-VL-9B-8bit}"
echo "=== Qwen3.5-VL-9B mlx-community (validates TokenizersBackend substitution, iter 59) ==="
if check_model "$VL9B"; then
  run_scenario "VL-9B BENCH_VL_BATCH_CHAT" env BENCH_MODEL="$VL9B" BENCH_VL_BATCH_CHAT=1 BENCH_MAX_TOKENS=20 .build/debug/RunBench
else
  skip "Qwen3.5-VL-9B mlx-community not cached at $VL9B"
fi
echo ""

# ---------------------------------------------------------------------------
G4_TOOLS_TEMPLATE="${G4_TOOLS_TEMPLATE:-$REPO_ROOT/Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja}"
echo "=== Gemma-4-E2B-4bit (sliding-window, template override) ==="
if check_model "$G4E2B"; then
  if [ -e "$G4_TEMPLATE" ]; then
    run_scenario "Gemma-4-E2B BENCH_BATCH_CHAT (Gemma4Minimal)" \
      env BENCH_MODEL="$G4E2B" VMLX_CHAT_TEMPLATE_OVERRIDE="$G4_TEMPLATE" BENCH_BATCH_CHAT=1 BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  else
    skip "Gemma4Minimal.jinja missing at $G4_TEMPLATE"
  fi
  if [ -e "$G4_TOOLS_TEMPLATE" ]; then
    run_scenario "Gemma-4-E2B BENCH_BATCH_CHAT (Gemma4WithTools, iter 60)" \
      env BENCH_MODEL="$G4E2B" VMLX_CHAT_TEMPLATE_OVERRIDE="$G4_TOOLS_TEMPLATE" BENCH_BATCH_CHAT=1 BENCH_MAX_TOKENS=$BENCH_MAX_TOKENS .build/debug/RunBench
  else
    skip "Gemma4WithTools.jinja missing at $G4_TOOLS_TEMPLATE"
  fi
else
  skip "Gemma-4-E2B-4bit not cached at $G4E2B"
fi
echo ""

# ---------------------------------------------------------------------------
printf '=== Summary: %d passed, %d failed, %d skipped ===\n' "$PASS" "$FAIL" "$SKIP"
if [ "$FAIL" -gt 0 ]; then
  echo "Failures:"
  for f in "${FAILURES[@]}"; do
    printf '  - %s\n' "$f"
  done
fi

# Clean up run logs
/bin/rm -f /tmp/verify-engine-$$-*.log 2>/dev/null || true

[ "$FAIL" = "0" ]
