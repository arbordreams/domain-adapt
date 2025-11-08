#!/usr/bin/env bash
# MedTokAlign shared shell library - corpus building
set -euo pipefail
IFS=$'\n\t'

# Requires common.sh for REPO_ROOT/PROJECT_DIR and logging

_corpus_root_from_cfg() {
  # crude parse: if output_dir present, prefer it; else default to medical_tokalign/data/biomed_corpus
  local cfg="$1"
  local def="${PROJECT_DIR}/data/biomed_corpus"
  if command -v python >/dev/null 2>&1; then
    python - "$cfg" <<PY 2>/dev/null || echo "${def}"
import sys, os, yaml
cfg = yaml.safe_load(open(sys.argv[1], 'r', encoding='utf-8'))
out = cfg.get('output_dir') or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'biomed_corpus')
print(out)
PY
  else
    echo "${def}"
  fi
}

_summary_complete() {
  local sjson="$1"
  if [ ! -f "${sjson}" ]; then return 1; fi
  if command -v python >/dev/null 2>&1; then
    python - "$sjson" <<PY 2>/dev/null
import sys, json
try:
  s=json.load(open(sys.argv[1],'r',encoding='utf-8'))
  tb=int(s.get('total_bytes',0)); tt=int(s.get('target_total_bytes',0))
  print('OK' if (tt>0 and tb>=tt) or bool(s.get('complete', False)) else 'NO')
except Exception:
  print('NO')
PY
  else
    echo "NO"
  fi
}

_make_quick_overlay() {
  local in_cfg="$1"; local out_cfg="$2"
  # Reduce target_total_bytes and per-source target_bytes; disable problematic sources
  if ! command -v python >/dev/null 2>&1; then
    cp "${in_cfg}" "${out_cfg}"
    return 0
  fi
  python - "${in_cfg}" "${out_cfg}" <<PY
import sys, yaml
inp, outp = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(inp, 'r', encoding='utf-8'))
cfg['defaults'] = cfg.get('defaults', {})
cfg['defaults']['min_chars'] = int(cfg['defaults'].get('min_chars', 200))
cfg['defaults']['max_chars'] = int(cfg['defaults'].get('max_chars', 20000))
# cap totals to ~20MB overall; per-source to 5–20MB range
cfg['target_total_bytes'] = int(min(20_000_000, int(cfg.get('target_total_bytes', 20_000_000))))
srcs = cfg.get('sources', {}) or {}
# Disable sources known to require scripts (datasets 3.x incompatible)
problematic = ['meddialog', 'ccdv_pubmed_summ', 'pubmed_abstracts']
for k, sc in srcs.items():
    if not isinstance(sc, dict): 
        continue
    # Disable problematic sources in quick mode
    if k in problematic:
        sc['enabled'] = False
        continue
    tb = int(sc.get('target_bytes', 0))
    if tb <= 0:
        sc['target_bytes'] = 5_000_000
    else:
        sc['target_bytes'] = min(tb, 10_000_000)
yaml.safe_dump(cfg, open(outp, 'w', encoding='utf-8'), sort_keys=False)
PY
}

# build_corpus CONFIG [quick=0]
build_corpus() {
  local cfg="$1"; local quick="${2:-0}"
  local use_cfg="${cfg}"
  if [ "${quick}" = "1" ]; then
    local tmpdir="${RUNS_DIR}/tmp"
    mkdir -p "${tmpdir}"
    use_cfg="${tmpdir}/$(basename "${cfg%.yaml}").quick.yaml"
    _make_quick_overlay "${cfg}" "${use_cfg}"
    info "Quick mode: using overlay config ${use_cfg}"
  fi
  local out_root; out_root="$(_corpus_root_from_cfg "${use_cfg}")"
  local sjson="${out_root}/summary.json"
  if [ "${RESUME_FLAG:-0}" -eq 1 ] && [ -f "${sjson}" ]; then
    local st="$(_summary_complete "${sjson}")"
    if [ "${st}" = "OK" ]; then
      info "Corpus summary indicates completed target; skipping build-corpus"
      return 0
    fi
  fi
  # Preflight (best-effort in quick mode)
  if [ "${quick}" != "1" ]; then
    ( cd "${REPO_ROOT}" && HF_HUB_ENABLE_HF_TRANSFER=1 python -m medical_tokalign.src.cli build-corpus --config "${use_cfg}" --preflight_only ) || {
      error "Preflight failed for ${use_cfg}"
      return 2
    }
  fi
  # Run build (non-strict in quick mode to tolerate partial availability)
  if [ "${quick}" = "1" ]; then
    ( cd "${REPO_ROOT}" && HF_HUB_ENABLE_HF_TRANSFER=1 STRICT_SOURCES=0 python -m medical_tokalign.src.cli build-corpus --config "${use_cfg}" ) || {
      warn "Build-corpus had errors in quick mode; continuing anyway"
      return 0
    }
  else
    ( cd "${REPO_ROOT}" && HF_HUB_ENABLE_HF_TRANSFER=1 python -m medical_tokalign.src.cli build-corpus --config "${use_cfg}" --strict_sources )
  fi
}

# End of corpus.sh

