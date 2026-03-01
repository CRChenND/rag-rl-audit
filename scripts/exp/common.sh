#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXP_RESUME=0
EXP_RETRIES=0
EXP_RETRY_DELAY=5

log() {
  printf '[exp] %s\n' "$*"
}

die() {
  printf '[exp][error] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

join_by_comma() {
  local out=""
  for x in "$@"; do
    if [[ -z "${out}" ]]; then
      out="${x}"
    else
      out="${out},${x}"
    fi
  done
  printf '%s' "${out}"
}

parse_seeds() {
  local raw="${1:-1,2,3}"
  local arr=()
  IFS=',' read -r -a arr <<<"${raw}"
  printf '%s\n' "${arr[@]}"
}

run_train_with_overrides() {
  # Args:
  # 1 base_config
  # 2 out_config
  # 3 out_dir
  # 4 seed
  # 5 train_path (optional)
  # 6 eval_path (optional)
  # 7 documents_path (optional)
  # 8 rm_base_model (optional)
  # 9 rm_adapter_path (optional)
  local base_cfg="${1}"
  local out_cfg="${2}"
  local out_dir="${3}"
  local seed="${4}"
  local train_path="${5:-}"
  local eval_path="${6:-}"
  local documents_path="${7:-}"
  local rm_base_model="${8:-}"
  local rm_adapter_path="${9:-}"

  if [[ "${EXP_RESUME}" == "1" ]] && model_output_ready "${out_dir}"; then
    log "Resume skip: ${out_dir} already looks complete."
    return 0
  fi

  cp "${base_cfg}" "${out_cfg}"
  cat >>"${out_cfg}" <<EOF

training:
  output_dir: ${out_dir}
  seed: ${seed}
EOF

  if [[ -n "${train_path}" || -n "${eval_path}" || -n "${documents_path}" ]]; then
    cat >>"${out_cfg}" <<EOF

data:
EOF
    [[ -n "${train_path}" ]] && echo "  train_path: ${train_path}" >>"${out_cfg}"
    [[ -n "${eval_path}" ]] && echo "  eval_path: ${eval_path}" >>"${out_cfg}"
    [[ -n "${documents_path}" ]] && echo "  documents_path: ${documents_path}" >>"${out_cfg}"
  fi

  if [[ -n "${rm_base_model}" || -n "${rm_adapter_path}" ]]; then
    cat >>"${out_cfg}" <<EOF

reward_model:
  base_model_name: ${rm_base_model}
  adapter_path: ${rm_adapter_path}
  adapter_trainable: false
  freeze: true
  use_lora: false
EOF
  fi

  local max_attempts=$((EXP_RETRIES + 1))
  local attempt=1
  while [[ "${attempt}" -le "${max_attempts}" ]]; do
    log "Train attempt ${attempt}/${max_attempts}: out_dir=${out_dir} seed=${seed}"
    if (cd "${REPO_ROOT}" && bash scripts/train.sh --config "${out_cfg}"); then
      return 0
    fi

    if [[ "${attempt}" -lt "${max_attempts}" ]]; then
      log "Attempt ${attempt} failed; retry in ${EXP_RETRY_DELAY}s"
      sleep "${EXP_RETRY_DELAY}"
    fi
    attempt=$((attempt + 1))
  done
  log "All attempts failed for out_dir=${out_dir} seed=${seed}"
  return 1
}

ensure_reports_dir() {
  mkdir -p "${REPO_ROOT}/reports"
}

parse_run_control_args() {
  # Supported args:
  # --resume
  # --retries <int>
  # --retry-delay <seconds>
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --resume)
        EXP_RESUME=1
        shift
        ;;
      --retries)
        [[ $# -ge 2 ]] || die "--retries requires a value"
        EXP_RETRIES="$2"
        shift 2
        ;;
      --retry-delay)
        [[ $# -ge 2 ]] || die "--retry-delay requires a value"
        EXP_RETRY_DELAY="$2"
        shift 2
        ;;
      *)
        die "Unsupported argument: $1"
        ;;
    esac
  done
}

model_output_ready() {
  local out_dir_rel="$1"
  local out_dir_abs="${REPO_ROOT}/${out_dir_rel}"
  [[ -d "${out_dir_abs}" ]] || return 1
  [[ -f "${out_dir_abs}/adapter_config.json" ]] && return 0
  [[ -f "${out_dir_abs}/config.json" ]] && return 0
  [[ -f "${out_dir_abs}/model.safetensors" ]] && return 0
  [[ -f "${out_dir_abs}/pytorch_model.bin" ]] && return 0
  [[ -f "${out_dir_abs}/tokenizer_config.json" ]] && return 0
  return 1
}
