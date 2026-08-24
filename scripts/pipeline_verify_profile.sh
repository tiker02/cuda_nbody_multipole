#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CPU_BIN="${ROOT_DIR}/build/taichi_cpu"
GPU_BIN="${ROOT_DIR}/build/taichi_gpu"

VAL_INPUT="${ROOT_DIR}/data/plummer/plummer_1000.dat"
PROF_INPUT="${ROOT_DIR}/data/plummer/plummer_100000.dat"

VAL_DIR="${ROOT_DIR}/build/val_runs"
REPORT_DIR="${ROOT_DIR}/ncu-reports"

mkdir -p "${VAL_DIR}/cpu" "${VAL_DIR}/gpu" "${REPORT_DIR}"

MODE="${1:---verify}"

# Numerical Verification Run
echo "=== [1/2] Running Numerical Check ==="
rm -f "${VAL_DIR}"/cpu/snapshot_*.hdf5 "${VAL_DIR}"/gpu/snapshot_*.hdf5

(cd "${VAL_DIR}/cpu" && ln -sf "${VAL_INPUT}" input.dat && "${CPU_BIN}" 0 input.dat 1 10 0.25)
(cd "${VAL_DIR}/gpu" && ln -sf "${VAL_INPUT}" input.dat && "${GPU_BIN}" 0 input.dat 1 10 0.25)

python3 "${ROOT_DIR}/scripts/compare_num_results.py" \
  "${VAL_DIR}/cpu/snapshot_0.hdf5" \
  "${VAL_DIR}/gpu/snapshot_0.hdf5" \
  1e-3

# NCU Profiling
if [ "${MODE}" == "--profile" ]; then
    echo -e "\n=== [2/2] Running Nsight Compute on 100k Particles ==="
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    REPORT_PATH="${REPORT_DIR}/cup2p_${TIMESTAMP}"

    ncu \
      --set full \
      --kernel-name regex:cuP2P \
      --export "${REPORT_PATH}" \
      --force-overwrite \
      "${GPU_BIN}" 0 "${PROF_INPUT}" 1 1 0.25

    echo -e "\n[SUCCESS] NCU CSV Report: ${REPORT_PATH}.csv"
fi