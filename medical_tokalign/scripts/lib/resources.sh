#!/usr/bin/env bash
# MedTokAlign shared shell library - resource sizing
set -euo pipefail
IFS=$'\n\t'

# Expect common.sh sourced first for logging helpers if available
_rs_log() { printf "[%s] [RES] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" "${*}"; }

clamp_int() {
  local val="$1"; local min="$2"; local max="$3"
  if [ "${val}" -lt "${min}" ]; then echo "${min}"; return 0; fi
  if [ "${val}" -gt "${max}" ]; then echo "${max}"; return 0; fi
  echo "${val}"
}

detect_vcpus() {
  # cgroup v2: /sys/fs/cgroup/cpu.max (quota period)
  if [ -f /sys/fs/cgroup/cpu.max ]; then
    local max_line
    max_line="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || true)"
    # format: "<quota> <period>" or "max <period>"
    local quota period
    quota="$(echo "${max_line}" | awk '{print $1}')"
    period="$(echo "${max_line}" | awk '{print $2}')"
    if [ "${quota}" != "max" ] && [ -n "${quota}" ] && [ -n "${period}" ] && [ "${period}" -gt 0 ]; then
      local cpus=$(( (quota + period - 1) / period ))
      cpus="$(clamp_int "${cpus}" 1 64)"
      echo "${cpus}"
      return 0
    fi
  fi
  # cgroup v1
  if [ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -f /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
    local quota period
    quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1)"
    period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)"
    if [ "${quota}" -gt 0 ] && [ "${period}" -gt 0 ]; then
      local cpus=$(( (quota + period - 1) / period ))
      cpus="$(clamp_int "${cpus}" 1 64)"
      echo "${cpus}"
      return 0
    fi
  fi
  # fallback: nproc
  if command -v nproc >/dev/null 2>&1; then
    local np
    np="$(nproc || echo 1)"
    np="$(clamp_int "${np}" 1 64)"
    echo "${np}"
    return 0
  fi
  echo 4
}

detect_ram_mb() {
  if [ -r /proc/meminfo ]; then
    local kb
    kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    if [ -n "${kb}" ] && [ "${kb}" -gt 0 ]; then
      echo $(( kb / 1024 ))
      return 0
    fi
  fi
  # macOS fallback
  if command -v sysctl >/dev/null 2>&1; then
    local bytes
    bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    if [ -n "${bytes}" ] && [ "${bytes}" -gt 0 ]; then
      echo $(( bytes / 1024 / 1024 ))
      return 0
    fi
  fi
  echo 16384
}

setup_threads() {
  local vcpus="$1"
  local half=$(( vcpus / 2 ))
  [ "${half}" -lt 1 ] && half=1
  local thr="$(clamp_int "${half}" 4 16)"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${thr}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${thr}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${thr}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${thr}}"
}

setup_glove_params() {
  local vcpus="$1"
  local ram_mb="$2"
  if [ "${EMBEDDING_BACKEND:-fasttext}" = "glove" ]; then
    local mem_est=$(( ram_mb * 30 / 100 ))
    mem_est="$(clamp_int "${mem_est}" 4096 65536)"
    local shuf_mem="${mem_est}"
    if [ "${shuf_mem}" -gt 16384 ]; then shuf_mem=16384; fi
    local threads=$(( vcpus - 1 ))
    if [ "${threads}" -lt 1 ]; then threads=1; fi
    threads="$(clamp_int "${threads}" 1 64)"
    export GLOVE_MEMORY_MB="${GLOVE_MEMORY_MB:-${mem_est}}"
    export GLOVE_SHUFFLE_MEMORY_MB="${GLOVE_SHUFFLE_MEMORY_MB:-${shuf_mem}}"
    export GLOVE_THREADS="${GLOVE_THREADS:-${threads}}"
  fi
}

setup_resources() {
  local vcpus ram_mb
  vcpus="$(detect_vcpus)"
  ram_mb="$(detect_ram_mb)"
  setup_threads "${vcpus}"
  setup_glove_params "${vcpus}" "${ram_mb}"
  _rs_log "vCPUs=${vcpus} RAM_MB=${ram_mb} OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS} EMBEDDING_BACKEND=${EMBEDDING_BACKEND:-fasttext} GLOVE_MEMORY_MB=${GLOVE_MEMORY_MB:-} GLOVE_SHUFFLE_MEMORY_MB=${GLOVE_SHUFFLE_MEMORY_MB:-} GLOVE_THREADS=${GLOVE_THREADS:-}"
}

# Runtime environment preflight: log core library versions and HF settings
preflight_runtime() {
  if ! command -v python >/dev/null 2>&1; then
    _rs_log "python=missing (cannot preflight)"
    return 0
  fi
  python - <<'PY' 2>/dev/null || true
import os
def ver(m):
  try:
    mod=__import__(m); return getattr(mod, "__version__", "unknown")
  except Exception as e:
    return f"missing({type(e).__name__})"
mods=["torch","torchvision","torchaudio","transformers","datasets","huggingface_hub","hf_transfer","gensim","numpy"]
print("[VERS]", " ".join(f"{m}={ver(m)}" for m in mods))
print("[HF]", f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER','')}",
             f"TRANSFORMERS_NO_TORCHVISION={os.environ.get('TRANSFORMERS_NO_TORCHVISION','')}")
PY
}

# End of resources.sh

