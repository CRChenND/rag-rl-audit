#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ARGS=("$@")
PREPARE_DATA="${PREPARE_DATA:-1}"
RUN_E1="${RUN_E1:-1}"
RUN_E2="${RUN_E2:-1}"
RUN_E3="${RUN_E3:-1}"
RUN_E4="${RUN_E4:-1}"
RUN_E5="${RUN_E5:-1}"

if [[ "${PREPARE_DATA}" == "1" ]]; then
  bash "${SCRIPT_DIR}/prepare_fixed_data.sh"
fi

# Experiments.
if [[ "${RUN_E1}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_e1.sh" "${RUN_ARGS[@]}"
fi
if [[ "${RUN_E2}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_e2.sh" "${RUN_ARGS[@]}"
fi
if [[ "${RUN_E3}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_e3.sh" "${RUN_ARGS[@]}"
fi
if [[ "${RUN_E4}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_e4.sh" "${RUN_ARGS[@]}"
fi
if [[ "${RUN_E5}" == "1" ]]; then
  bash "${SCRIPT_DIR}/run_e5.sh" "${RUN_ARGS[@]}"
fi

printf '[exp] run_all.sh completed.\n'
