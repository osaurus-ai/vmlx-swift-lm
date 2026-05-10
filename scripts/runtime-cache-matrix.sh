#!/bin/bash
# Durable wrapper for cache/batching/runtime verification.
#
# This keeps run metadata, the summary stream, and per-scenario logs so results
# can be compared across model families without relying on terminal scrollback.
#
# Usage:
#   scripts/runtime-cache-matrix.sh --tests-only
#   scripts/runtime-cache-matrix.sh --quick
#   BENCH_MAX_TOKENS=64 scripts/runtime-cache-matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="${RUNTIME_MATRIX_LOG_ROOT:-$REPO_ROOT/build/runtime-cache-matrix}"
RUN_DIR="$LOG_ROOT/$RUN_ID"
SCENARIO_DIR="$RUN_DIR/scenarios"

/bin/mkdir -p "$SCENARIO_DIR"

cd "$REPO_ROOT"

if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'repo=%s\n' "$REPO_ROOT"
  printf 'date_utc=%s\n' "$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'date_local=%s\n' "$(/bin/date '+%Y-%m-%dT%H:%M:%S%z')"
  printf 'developer_dir=%s\n' "${DEVELOPER_DIR:-}"
  printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
  printf 'git_status_short<<EOF\n'
  git status --short 2>/dev/null || true
  printf 'EOF\n'
  printf 'swift_version<<EOF\n'
  swift --version 2>/dev/null || true
  printf 'EOF\n'
  printf 'runtime_env<<EOF\n'
  env | /usr/bin/sort | /usr/bin/grep -E '^(BENCH|Q06B|Q36|VL4B|VL9B|G4E2B|G4_|SPECDEC|VMLX|MLXPRESS|JANG|MLX|DEVELOPER_DIR|RUNTIME_MATRIX|VERIFY_ENGINE)=' || true
  printf 'EOF\n'
} > "$RUN_DIR/metadata.txt"

printf 'Runtime cache matrix run: %s\n' "$RUN_DIR"

set +e
VERIFY_ENGINE_LOG_DIR="$SCENARIO_DIR" "$SCRIPT_DIR/verify-engine.sh" "$@" 2>&1 | /usr/bin/tee "$RUN_DIR/summary.log"
status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
printf 'Summary: %s\n' "$RUN_DIR/summary.log"
printf 'Scenario logs: %s\n' "$SCENARIO_DIR"

exit "$status"
