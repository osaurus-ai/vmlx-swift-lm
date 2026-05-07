#!/bin/bash
# tp-launch-2host.sh — distributed TP launcher across two configured hosts.
#
# Both hosts must already have:
#   - a built TPRankWorker binary at the configured path
#   - the same model bundle copied locally
#   - network reachability on the advertised ring host/ports
#
# Usage:
#   TP_RANK0_HOST=<rank0-ip-or-host> \
#   TP_RANK1_HOST=<rank1-ip-or-host> \
#   TP_RANK1_SSH=<ssh-target> \
#   TP_RANK1_MODEL=<model-dir-on-rank1> \
#   TP_RANK1_WORKER=<rank1-TPRankWorker-path> \
#     ./Tools/tp-launch-2host.sh <model-dir-on-this-host>
#
# Optional:
#   TP_PROMPT_TOKEN_IDS   token CSV, default 1,2,3,4,5,6,7,8
#   TP_RING_BASE_PORT     first ring port, default 29600
#   TP_REMOTE_HOSTFILE    remote hostfile path, default /tmp/tp_ring_hosts_2host.json

set -eu

MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
  echo "usage: $0 <model-dir-on-this-host>" >&2
  exit 64
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "model dir not found: $MODEL_PATH" >&2
  exit 65
fi

: "${TP_RANK0_HOST:?set TP_RANK0_HOST to rank 0's reachable host/IP}"
: "${TP_RANK1_HOST:?set TP_RANK1_HOST to rank 1's reachable host/IP}"
: "${TP_RANK1_SSH:?set TP_RANK1_SSH to rank 1's ssh target}"
: "${TP_RANK1_MODEL:?set TP_RANK1_MODEL to rank 1's local model path}"
: "${TP_RANK1_WORKER:?set TP_RANK1_WORKER to rank 1's TPRankWorker path}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$REPO_ROOT/.build/release/TPRankWorker"
if [ ! -x "$WORKER" ]; then
  echo "TPRankWorker not built locally; run: swift build -c release --product TPRankWorker" >&2
  exit 66
fi

R0_HOST="$TP_RANK0_HOST"
R1_HOST="$TP_RANK1_HOST"
R1_SSH="$TP_RANK1_SSH"
R1_MODEL="$TP_RANK1_MODEL"
R1_WORKER="$TP_RANK1_WORKER"
TOKENS="${TP_PROMPT_TOKEN_IDS:-1,2,3,4,5,6,7,8}"
REMOTE_HOSTFILE="${TP_REMOTE_HOSTFILE:-/tmp/tp_ring_hosts_2host.json}"

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

scp -q "$RING_HOSTFILE" "$R1_SSH:$REMOTE_HOSTFILE"

echo "[tp-2host] r0=$R0_HOST r1=$R1_HOST tokens=$TOKENS"

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
echo "[tp-2host] rank 0 pid=$RANK0_PID"

# Let rank 0 bind its ring ports before the remote worker joins.
sleep 2

ssh "$R1_SSH" "MLX_RANK=1 MLX_WORLD_SIZE=2 MLX_DIST_BACKEND=ring \
  MLX_HOSTFILE='$REMOTE_HOSTFILE' \
  MLX_RING_VERBOSE=1 \
  TP_STRICT=1 \
  TP_MODEL_PATH='$R1_MODEL' \
  TP_OUTPUT_PATH=/tmp/tp_rank1.f32 \
  TP_PROMPT_TOKEN_IDS='$TOKENS' \
  '$R1_WORKER' > /tmp/tp_rank1.log 2>&1" &
RANK1_PID=$!
echo "[tp-2host] rank 1 pid=$RANK1_PID (ssh $R1_SSH)"

RC0=0
RC1=0
wait $RANK0_PID || RC0=$?
wait $RANK1_PID || RC1=$?

echo "[tp-2host] rank 0 exited $RC0"
echo "[tp-2host] rank 1 exited $RC1"

scp -q "$R1_SSH:/tmp/tp_rank1.f32" /tmp/tp_rank1.f32 2>/dev/null || true
scp -q "$R1_SSH:/tmp/tp_rank1.log" /tmp/tp_rank1.remote.log 2>/dev/null || true

if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  echo "[tp-2host] FAILED" >&2
  echo "--- rank 0 tail ---"
  tail -25 /tmp/tp_rank0.log
  echo "--- rank 1 tail ---"
  tail -25 /tmp/tp_rank1.remote.log 2>/dev/null
  exit $(( RC0 | RC1 ))
fi

echo "[tp-2host] OK — outputs at /tmp/tp_rank{0,1}.f32"
exit 0
