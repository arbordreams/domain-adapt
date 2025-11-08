#!/usr/bin/env bash
# MedTokAlign shared shell library - tmux orchestration
set -euo pipefail
IFS=$'\n\t'

TMUX_SESSION="${TMUX_SESSION:-medtokalign}"

tmux_available() { command -v tmux >/dev/null 2>&1; }

tmux_ensure_session() {
  if ! tmux_available; then
    return 1
  fi
  if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${TMUX_SESSION}" -n "prep"
    tmux set-option -t "${TMUX_SESSION}" -g remain-on-exit on
  fi
  return 0
}

tmux_run() {
  # window_name, command
  local win="$1"; shift
  local cmd="$*"
  if ! tmux_ensure_session; then
    echo "tmux not available" >&2
    return 1
  fi
  if tmux list-windows -t "${TMUX_SESSION}" | awk '{print $2}' | sed 's/:$//' | grep -qx "${win}"; then
    tmux kill-window -t "${TMUX_SESSION}:${win}" || true
  fi
  tmux new-window -t "${TMUX_SESSION}" -n "${win}"
  tmux send-keys -t "${TMUX_SESSION}:${win}" "${cmd}" C-m
}

tmux_attach() {
  if tmux_ensure_session; then
    tmux attach-session -t "${TMUX_SESSION}"
  fi
}

tmux_detach() {
  if [ -n "${TMUX:-}" ]; then
    tmux detach-client || true
  fi
}

# End of tmux.sh

