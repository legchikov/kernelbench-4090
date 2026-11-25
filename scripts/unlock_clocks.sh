#!/usr/bin/env bash
# Undo scripts/lock_clocks.sh.  Usage: sudo scripts/unlock_clocks.sh <gpu_index>
set -euo pipefail
GPU="${1:?usage: unlock_clocks.sh <gpu_index>}"
LOCK_DIR="${KB4090_LOCK_DIR:-/var/tmp/kb4090-locks}"
nvidia-smi -i "$GPU" --reset-gpu-clocks
rm -f "$LOCK_DIR/clocks-gpu${GPU}.json"
echo "GPU $GPU clocks reset"
