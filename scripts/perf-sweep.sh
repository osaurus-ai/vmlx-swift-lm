#!/usr/bin/env bash
# Deterministic decode-tok/s sweep across MLX models + configs.
#
# Usage: ./scripts/perf-sweep.sh [output.csv]
#   Writes CSV rows: commit,model,variant,genTokens,genSec,tokps_median,tokps_best
#
# Environment overrides:
#   BENCH_MAX_TOKENS   — decode budget per run (default 128)
#   BENCH_PERF_WARMUP  — warmup turns (default 1)
#   BENCH_PERF_RUNS    — measurement turns (default 3)
#
# Pre-flight: kills xctest / RunBench / ollama / lms between runs so
# GPU VRAM is never fragmented across models.

set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$REPO/perf-sweep-$(date +%Y%m%d-%H%M%S).csv"}

: "${BENCH_MAX_TOKENS:=128}"
: "${BENCH_PERF_WARMUP:=1}"
: "${BENCH_PERF_RUNS:=3}"

MODELS_ROOT=${MODELS_ROOT:-"$HOME/.mlxstudio/models/MLXModels"}

# model path :: variant label
CONFIGS=(
  "$MODELS_ROOT/JANGQ-AI/Qwen3.5-35B-A3B-4bit::qwen3.5-35b-a3b-4bit"
  "$MODELS_ROOT/OsaurusAI/Qwen3.6-35B-A3B-MXFP4::qwen3.6-35b-a3b-mxfp4"
  "$MODELS_ROOT/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2::qwen3.6-35b-a3b-jangtq2"
  "$MODELS_ROOT/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ4::qwen3.6-35b-a3b-jangtq4"
  "$MODELS_ROOT/OsaurusAI/gemma-4-e2b-it-4bit::gemma-4-e2b-4bit"
  "$MODELS_ROOT/OsaurusAI/gemma-4-e4b-it-4bit::gemma-4-e4b-4bit"
)

preflight() {
  pkill -f xctest 2>/dev/null
  pkill -f RunBench 2>/dev/null
  pkill -f ollama 2>/dev/null
  pkill -f lms 2>/dev/null
  sleep 2
}

echo 'commit,model,variant,genTokens,genSec,tokps_median,tokps_best' > "$OUT"

echo "Building release..." >&2
swift build -c release 2>&1 | tail -3

for cfg in "${CONFIGS[@]}"; do
  model_path="${cfg%%::*}"
  variant="${cfg##*::}"
  if [ ! -d "$model_path" ]; then
    echo "SKIP missing: $model_path" >&2
    continue
  fi
  preflight
  extra_env=""
  if [[ "$variant" == gemma-* ]]; then
    extra_env="VMLX_CHAT_TEMPLATE_OVERRIDE=$REPO/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja"
  fi
  # Run
  line=$(env BENCH_PERF=1 \
    BENCH_PERF_VARIANT="$variant" \
    BENCH_MAX_TOKENS="$BENCH_MAX_TOKENS" \
    BENCH_PERF_WARMUP="$BENCH_PERF_WARMUP" \
    BENCH_PERF_RUNS="$BENCH_PERF_RUNS" \
    BENCH_MODEL="$model_path" \
    $extra_env \
    "$REPO/.build/release/RunBench" 2>&1 | grep '^PERF ')
  if [ -z "$line" ]; then
    echo "FAIL: no PERF line from $variant" >&2
    continue
  fi
  echo "$line" >&2
  # Parse: PERF model=M variant=V commit=C genTokens=N genSec=F tokps_median=F tokps_best=F runs=...
  commit=$(echo "$line" | sed -nE 's/.* commit=([^ ]+) .*/\1/p')
  model=$(echo "$line" | sed -nE 's/.* model=([^ ]+) .*/\1/p')
  gt=$(echo "$line" | sed -nE 's/.* genTokens=([0-9]+) .*/\1/p')
  gs=$(echo "$line" | sed -nE 's/.* genSec=([0-9.]+) .*/\1/p')
  median=$(echo "$line" | sed -nE 's/.* tokps_median=([0-9.]+) .*/\1/p')
  best=$(echo "$line" | sed -nE 's/.* tokps_best=([0-9.]+) .*/\1/p')
  echo "$commit,$model,$variant,$gt,$gs,$median,$best" >> "$OUT"
done

echo
echo "=== Results written to $OUT ==="
cat "$OUT"
