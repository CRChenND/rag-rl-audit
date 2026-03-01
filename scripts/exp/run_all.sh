#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ARGS=("$@")

# Prepare data once.
bash "${SCRIPT_DIR}/prepare_fixed_data.sh"

# Experiments.
bash "${SCRIPT_DIR}/run_e1.sh" "${RUN_ARGS[@]}"
bash "${SCRIPT_DIR}/run_e2.sh" "${RUN_ARGS[@]}"
bash "${SCRIPT_DIR}/run_e3.sh" "${RUN_ARGS[@]}"
bash "${SCRIPT_DIR}/run_e4.sh" "${RUN_ARGS[@]}"
bash "${SCRIPT_DIR}/run_e5.sh" "${RUN_ARGS[@]}"

printf '[exp] run_all.sh completed.\n'
