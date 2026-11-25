#!/usr/bin/env bash
# Minimal bubblewrap sandbox for evaluation workers and agents.
#
#   scripts/sandbox.sh [--no-net] [--hide-home] [--rw DIR]... [--ro DIR]... -- command args...
#
# The whole filesystem is mounted read-only, /dev is passed through (the GPU needs /dev/nvidia*),
# /tmp is private, and only the --rw directories are writable. --no-net removes network access
# (use it for evaluation workers; agents need the network for their model API). --hide-home
# replaces $HOME with an empty tmpfs; add back what is needed with --ro/--rw.
#
# This is defence in depth, not a security boundary you should bet secrets on: keep API keys and
# SSH keys out of the machine (or VM) where agents run. Test it before relying on it:
#   scripts/sandbox.sh --no-net --rw /tmp/x -- bash -c 'touch ~/nope; curl -sI https://example.com'
set -euo pipefail
command -v bwrap >/dev/null || { echo "bubblewrap (bwrap) is not installed: sudo apt install bubblewrap" >&2; exit 2; }

NET=1; HIDE_HOME=0; RW=(); RO=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-net) NET=0; shift ;;
    --hide-home) HIDE_HOME=1; shift ;;
    --rw) RW+=("$2"); shift 2 ;;
    --ro) RO+=("$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "no command given" >&2; exit 2; }

LOCK_DIR="${KB4090_LOCK_DIR:-/var/tmp/kb4090-locks}"
mkdir -p "$LOCK_DIR"

ARGS=(--ro-bind / / --dev-bind /dev /dev --proc /proc --tmpfs /tmp --die-with-parent --new-session
      --bind "$LOCK_DIR" "$LOCK_DIR")
if [ "$HIDE_HOME" = 1 ]; then ARGS+=(--tmpfs "$HOME"); fi
for d in "${RO[@]}"; do [ -e "$d" ] && ARGS+=(--ro-bind "$d" "$d"); done
for d in "${RW[@]}"; do [ -e "$d" ] && ARGS+=(--bind "$d" "$d"); done  # paths must already exist
if [ "$NET" = 0 ]; then ARGS+=(--unshare-net); fi

exec bwrap "${ARGS[@]}" -- "$@"
