#!/bin/bash
# tp-launch-2host.sh — distributed TP launcher across two hosts.
#
# Rank 0: this host (M5 Max, Tailscale 100.107.102.99 by default)
# Rank 1: SSH target `oldmbp` (M4 Max, Tailscale 100.72.8.105)
#
# Both hosts must already have:
#   - a built TPRankWorker binary at the matching path on each
#   - the same model bundle copied locally on each
#
# Outputs (on this host):
#   /tmp/tp_rank0.f32  — rank-0 logits
#   /tmp/tp_rank1.f32  — rank-1 logits (scp'd back from oldmbp at end)
#
# Usage:
#   ./Tools/tp-launch-2host.sh <model-dir-on-this-host>
#
# Env knobs:
#   TP_RANK0_HOST       = 100.107.102.99  (this M5 Tailscale IP)
#   TP_RANK1_HOST       = 100.72.8.105    (oldmbp Tailscale IP)
#   TP_RANK1_SSH        = oldmbp
#   TP_RANK1_MODEL      = ~/tp_test_model
#   TP_RANK1_WORKER     = ~/vmlx-swift-lm-tp/.build/release/TPRankWorker
#   TP_PROMPT_TOKEN_IDS = 1,2,3,4,5,6,7,8

set -u
MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
  echo "usage: $0 <model-dir-on-this-host>" >&2
  exit 64
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "model dir not found: $MODEL_PATH" >&2
  exit 65
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$REPO_ROOT/.build/release/TPRankWorker"
if [ ! -x "$WORKER" ]; then
  echo "TPRankWorker not built locally; run: swift build -c release --product TPRankWorker" >&2
  exit 66
fi

R0_HOST="${TP_RANK0_HOST:-100.107.102.99}"
R1_HOST="${TP_RANK1_HOST:-100.72.8.105}"
R1_SSH="${TP_RANK1_SSH:-oldmbp}"
R1_MODEL="${TP_RANK1_MODEL:-tp_test_model}"
R1_WORKER="${TP_RANK1_WORKER:-vmlx-swift-lm-tp/.build/release/TPRankWorker}"
TOKENS="${TP_PROMPT_TOKEN_IDS:-1,2,3,4,5,6,7,8}"

# Ring backend hostfile — JSON, two ports per rank for the full ring mesh.
RING_HOSTFILE="/tmp/tp_ring_hosts_2host.json"
BASE="${TP_RING_BASE_PORT:-29600}"
cat > "$RING_HOSTFILE" <<EOF
[
  ["${R0_HOST}:$((BASE+0))", "${R0_HOST}:$((BASE+1))"],
  ["${R1_HOST}:$((BASE+2))", "${R1_HOST}:$((BASE+3))"]
]
EOF
echo "[tp-2host] wrote $RING_HOSTFILE"

# Push the same hostfile to oldmbp.
scp -q "$RING_HOSTFILE" "$R1_SSH:/tmp/tp_ring_hosts_2host.json"

echo "[tp-2host] r0=$R0_HOST r1=$R1_HOST tokens=$TOKENS"

# Rank 0 — local M5.
mkdir -p /tmp
MLX_RANK=0 \
MLX_WORLD_SIZE=2 \
MLX_DIST_BACKEND=ring \
MLX_HOSTFILE="$RING_HOSTFILE" \
MLX_RING_VERBOSE=1 \
TP_STRICT=1 \
TP_MODEL_PATH="$MODEL_PATH" \
TP_OUTPUT_PATH=/tmp/tp_rank0.f32 \
TP_PROMPT_TOKEN_IDS="$TOKENS" \
  "$WORKER" > /tmp/tp_rank0.log 2>&1 &
RANK0_PID=$!
echo "[tp-2host] rank 0 (local) pid=$RANK0_PID"

# Stagger so rank 0 binds its ports first.
sleep 2

# Rank 1 — oldmbp via SSH.
ssh "$R1_SSH" "MLX_RANK=1 MLX_WORLD_SIZE=2 MLX_DIST_BACKEND=ring \
  MLX_HOSTFILE=/tmp/tp_ring_hosts_2host.json \
  MLX_RING_VERBOSE=1 \
  TP_STRICT=1 \
  TP_MODEL_PATH='$R1_MODEL' \
  TP_OUTPUT_PATH=/tmp/tp_rank1.f32 \
  TP_PROMPT_TOKEN_IDS='$TOKENS' \
  '$R1_WORKER' > /tmp/tp_rank1.log 2>&1" &
RANK1_PID=$!
echo "[tp-2host] rank 1 (oldmbp) pid=$RANK1_PID (ssh)"

RC0=0
RC1=0
wait $RANK0_PID || RC0=$?
wait $RANK1_PID || RC1=$?

echo "[tp-2host] rank 0 exited $RC0"
echo "[tp-2host] rank 1 (oldmbp) exited $RC1"

# Bring rank 1's logit blob + log back for inspection / verification.
scp -q "$R1_SSH:/tmp/tp_rank1.f32" /tmp/tp_rank1.f32 2>/dev/null || true
scp -q "$R1_SSH:/tmp/tp_rank1.log" /tmp/tp_rank1.oldmbp.log 2>/dev/null || true

if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  echo "[tp-2host] FAILED" >&2
  echo "--- rank 0 (local) tail ---"
  tail -25 /tmp/tp_rank0.log
  echo "--- rank 1 (oldmbp) tail ---"
  tail -25 /tmp/tp_rank1.oldmbp.log 2>/dev/null
  exit $(( RC0 | RC1 ))
fi

echo "[tp-2host] OK — outputs at /tmp/tp_rank{0,1}.f32"
exit 0
