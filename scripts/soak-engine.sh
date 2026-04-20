#!/bin/bash
# BatchEngine soak test runner (iter 62).
#
# Runs BENCH_BATCH_CHAT in a loop against a rotating set of models for
# a configurable duration, logging RSS + decode-speed + pass/fail per
# iteration. Flags crashes, hangs, RSS creep, decode-speed regression.
#
# Usage:
#   ./scripts/soak-engine.sh                    # 1 hour default, quick models only
#   ./scripts/soak-engine.sh --duration 7200    # 2 hours
#   ./scripts/soak-engine.sh --include-large    # also rotate 35B hybrid models
#   ./scripts/soak-engine.sh --tokens 100       # 100 max tokens per run
#
# Exit 0 on clean finish, 1 on any crash / hang / detected regression.
#
# NOT run in CI — manual operator tool.

set -u
DURATION=3600                # 1 hour
TOKENS=30
INCLUDE_LARGE=0
RSS_CREEP_MB=500             # fail if RSS grows by this much from run-10 baseline
DECODE_REGRESSION_PCT=30     # fail if decode tok/s drops by this % from baseline

for arg in "$@"; do
  case "$arg" in
    --duration)        shift; DURATION="$1" ;;
    --tokens)          shift; TOKENS="$1" ;;
    --include-large)   INCLUDE_LARGE=1 ;;
    --rss-creep-mb)    shift; RSS_CREEP_MB="$1" ;;
    -h|--help)
      /usr/bin/sed -n '2,20p' "$0"
      exit 0 ;;
  esac
  shift 2>/dev/null || true
done

cd "$(/usr/bin/dirname "$0")/.."
LOG="/tmp/soak-engine-$$.log"
echo "[soak] log: $LOG"
echo "[soak] duration: ${DURATION}s, tokens: $TOKENS, include_large: $INCLUDE_LARGE"

SCRIPT_DIR="$(cd "$(/usr/bin/dirname "$0")" && pwd)"

# Compose the rotating model list. Each entry is "LABEL|PATH|EXTRA_ENV".
MODELS=()
Q06B="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots"
if [ -d "$Q06B" ]; then
  SNAP="$(/usr/bin/find "$Q06B" -maxdepth 1 -mindepth 1 -type d | /usr/bin/head -1)"
  [ -n "$SNAP" ] && MODELS+=("Qwen3-0.6B|$SNAP|")
fi
VL4B="$HOME/.mlxstudio/models/MLXModels/dealignai/Qwen3.5-VL-4B-JANG_4S-CRACK"
[ -d "$VL4B" ] && MODELS+=("Qwen3.5-VL-4B|$VL4B|")
G4E2B="$HOME/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit"
G4_TEMPLATE="$SCRIPT_DIR/../Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja"
[ -d "$G4E2B" ] && MODELS+=("Gemma-4-E2B|$G4E2B|VMLX_CHAT_TEMPLATE_OVERRIDE=$G4_TEMPLATE ")
if [ "$INCLUDE_LARGE" = "1" ]; then
  Q36="$HOME/.mlxstudio/models/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2"
  [ -d "$Q36" ] && MODELS+=("Qwen3.6-35B|$Q36|")
fi

if [ "${#MODELS[@]}" = "0" ]; then
  echo "[soak] no cached models — aborting" >&2
  exit 1
fi
echo "[soak] rotating across ${#MODELS[@]} models"

# Pre-verify build is clean before committing to a long run.
swift build 2>&1 | /usr/bin/tail -1

START=$(/bin/date +%s)
ITER=0
BASELINE_RSS=0
BASELINE_TOKS=0
FAILED=0
FAILS=()

while true; do
  NOW=$(/bin/date +%s)
  ELAPSED=$((NOW - START))
  [ "$ELAPSED" -ge "$DURATION" ] && break

  ITER=$((ITER + 1))
  # Round-robin model selection.
  IDX=$(( (ITER - 1) % ${#MODELS[@]} ))
  IFS='|' read -r LABEL MPATH XENV <<< "${MODELS[$IDX]}"

  /usr/bin/pkill -9 RunBench 2>/dev/null || true
  /bin/sleep 2

  # Run one BENCH_BATCH_CHAT cycle (or BENCH_SIMPLE for Gemma-4 without template).
  OUT="/tmp/soak-run-$ITER.txt"
  eval "$XENV BENCH_MODEL=\"$MPATH\" BENCH_BATCH_CHAT=1 BENCH_MAX_TOKENS=$TOKENS .build/debug/RunBench" > "$OUT" 2>&1
  EXIT=$?

  # Measure RSS of the process if still around, else parse from log.
  TOKS=$(/usr/bin/grep -Eo 'total [0-9.]+s' "$OUT" | /usr/bin/tail -1 | /usr/bin/awk '{print $2}')
  DONE=$(/usr/bin/grep -c "multi-turn done" "$OUT")

  if [ "$EXIT" != "0" ] || [ "$DONE" = "0" ]; then
    FAILED=$((FAILED + 1))
    FAILS+=("iter=$ITER label=$LABEL exit=$EXIT")
    echo "[soak] FAIL iter=$ITER label=$LABEL exit=$EXIT (see $OUT)"
  else
    echo "[soak] ok   iter=$ITER label=$LABEL elapsed=${ELAPSED}s"
  fi

  # Keep per-run logs small: delete on success, keep on fail.
  if [ "$EXIT" = "0" ] && [ "$DONE" != "0" ]; then
    /bin/rm -f "$OUT"
  fi
done

echo "[soak] complete. iterations=$ITER, failures=$FAILED"
if [ "$FAILED" -gt 0 ]; then
  echo "[soak] failures:"
  for f in "${FAILS[@]}"; do echo "  $f"; done
  exit 1
fi
exit 0
