#!/bin/bash
# tp-launch.sh — single-host loopback launcher for the size-2 TP proof.
#
# Spawns rank-0 + rank-1 as two child processes of TPRankWorker, with
# the env vars `mlx_distributed_init` reads at process start. Default
# backend is "ring" (TCP loopback) for first-pass validation; flip to
# "jaccl" via MLX_DIST_BACKEND=jaccl to exercise RDMA loopback.
#
# Usage:
#   ./Tools/tp-launch.sh <model-dir>
#   MLX_DIST_BACKEND=jaccl ./Tools/tp-launch.sh <model-dir>
#
# Outputs:
#   /tmp/tp_rank0.f32  — rank-0 logits dump (header + Float32 LE)
#   /tmp/tp_rank1.f32  — rank-1 logits dump
#
# Exit code:
#   0 — both ranks exited 0
#   N — first non-zero rank exit code

set -u
MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
  echo "usage: $0 <model-dir>" >&2
  exit 64
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "model dir not found: $MODEL_PATH" >&2
  exit 65
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$REPO_ROOT/.build/release/TPRankWorker"
if [ ! -x "$WORKER" ]; then
  echo "TPRankWorker not built; run: swift build -c release --product TPRankWorker" >&2
  exit 66
fi

BACKEND="${MLX_DIST_BACKEND:-ring}"
COORD_HOST="${MLX_JACCL_COORDINATOR_HOST:-127.0.0.1}"
COORD_PORT="${MLX_JACCL_COORDINATOR_PORT:-29500}"
COORD="${COORD_HOST}:${COORD_PORT}"
TOKENS="${TP_PROMPT_TOKEN_IDS:-1,2,3,4,5,6,7,8}"

# Backend-specific config files.
#
# Ring (TCP loopback): MLX_HOSTFILE points to a JSON list of per-rank
# `host:port` arrays. We use two ports per rank for two TCP links so
# the ring backend can run its full mesh.
#
# JACCL (RDMA): MLX_IBV_DEVICES JSON describes per-(src,dst) IBV device
# names. For loopback both ranks talk to themselves through the same
# device. The "" entry is self-to-self per the JACCL convention.
RING_HOSTFILE="/tmp/tp_ring_hosts.json"
RING_BASE_PORT="${TP_RING_BASE_PORT:-29600}"
if [ "$BACKEND" = "ring" ]; then
  R0_P0=$(( RING_BASE_PORT + 0 ))
  R0_P1=$(( RING_BASE_PORT + 1 ))
  R1_P0=$(( RING_BASE_PORT + 2 ))
  R1_P1=$(( RING_BASE_PORT + 3 ))
  cat > "$RING_HOSTFILE" <<EOF
[
  ["127.0.0.1:${R0_P0}", "127.0.0.1:${R0_P1}"],
  ["127.0.0.1:${R1_P0}", "127.0.0.1:${R1_P1}"]
]
EOF
  echo "[tp-launch] wrote $RING_HOSTFILE (ports ${R0_P0}-${R1_P1})"
fi

IBV_JSON="/tmp/tp_ibv_devices.json"
IBV_DEV="${MLX_IBV_DEVICE_NAME:-mlx0}"
if [ "$BACKEND" = "jaccl" ]; then
  cat > "$IBV_JSON" <<EOF
[
  ["", "${IBV_DEV}"],
  ["${IBV_DEV}", ""]
]
EOF
  echo "[tp-launch] wrote $IBV_JSON (device=${IBV_DEV})"
fi

# Common env exported to every rank.
export TP_MODEL_PATH="$MODEL_PATH"
export TP_PROMPT_TOKEN_IDS="$TOKENS"
export MLX_WORLD_SIZE=2
export MLX_DIST_BACKEND="$BACKEND"
[ "$BACKEND" = "ring" ] && export MLX_HOSTFILE="$RING_HOSTFILE"
[ "$BACKEND" = "jaccl" ] && export MLX_IBV_DEVICES="$IBV_JSON" && export MLX_JACCL_COORDINATOR="$COORD"

mkdir -p /tmp

echo "[tp-launch] backend=$BACKEND coord=$COORD"
echo "[tp-launch] model=$MODEL_PATH"
echo "[tp-launch] tokens=$TOKENS"

# Rank 0 — coordinator side.
MLX_RANK=0 TP_OUTPUT_PATH=/tmp/tp_rank0.f32 \
  "$WORKER" > /tmp/tp_rank0.log 2>&1 &
RANK0_PID=$!
echo "[tp-launch] rank 0 pid=$RANK0_PID"

# Small stagger so rank 0 binds the coordinator port first.
sleep 1

# Rank 1 — joins coordinator.
MLX_RANK=1 TP_OUTPUT_PATH=/tmp/tp_rank1.f32 \
  "$WORKER" > /tmp/tp_rank1.log 2>&1 &
RANK1_PID=$!
echo "[tp-launch] rank 1 pid=$RANK1_PID"

# Wait for both, capture exit codes individually.
RC0=0
RC1=0
wait $RANK0_PID || RC0=$?
wait $RANK1_PID || RC1=$?

echo "[tp-launch] rank 0 exited $RC0"
echo "[tp-launch] rank 1 exited $RC1"

if [ $RC0 -ne 0 ] || [ $RC1 -ne 0 ]; then
  echo "[tp-launch] FAILED — see /tmp/tp_rank0.log + /tmp/tp_rank1.log" >&2
  echo "--- rank 0 tail ---"
  tail -20 /tmp/tp_rank0.log
  echo "--- rank 1 tail ---"
  tail -20 /tmp/tp_rank1.log
  exit $(( RC0 | RC1 ))
fi

echo "[tp-launch] OK — outputs at /tmp/tp_rank{0,1}.f32"
exit 0
