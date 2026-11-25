#!/usr/bin/env bash
# Lock the GPU's graphics clock so boost behaviour doesn't add run-to-run variance.
#
# Usage: sudo scripts/lock_clocks.sh <gpu_index> <sm_clock_mhz>
#
# Pick a clock your card can SUSTAIN under load without hitting power or thermal limits:
#   1. run `kb4090 baseline --tasks L1-001-matmul` once and watch `nvidia-smi -q -d CLOCK,PERFORMANCE`
#   2. choose a value at or a little below the sustained SM clock you saw
#   3. list valid values with: nvidia-smi -i <gpu> -q -d SUPPORTED_CLOCKS
# Record the clock you chose in your write-up. Undo with scripts/unlock_clocks.sh.
set -euo pipefail
GPU="${1:?usage: lock_clocks.sh <gpu_index> <sm_clock_mhz>}"
CLK="${2:?usage: lock_clocks.sh <gpu_index> <sm_clock_mhz>}"
LOCK_DIR="${KB4090_LOCK_DIR:-/var/tmp/kb4090-locks}"

nvidia-smi -i "$GPU" -pm 1
nvidia-smi -i "$GPU" --lock-gpu-clocks="$CLK,$CLK"

mkdir -p "$LOCK_DIR"
chmod 1777 "$LOCK_DIR" 2>/dev/null || true
cat > "$LOCK_DIR/clocks-gpu${GPU}.json" <<EOF
{"gpu": $GPU, "sm_clock_mhz": $CLK, "locked_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
chmod 644 "$LOCK_DIR/clocks-gpu${GPU}.json"
echo "Locked GPU $GPU graphics clock at $CLK MHz (marker: $LOCK_DIR/clocks-gpu${GPU}.json)"
nvidia-smi -i "$GPU" --query-gpu=clocks.sm,clocks.mem,power.limit,persistence_mode --format=csv
