#!/usr/bin/env bash
set -euo pipefail

# Binary paths and parameters
CPU_BIN="Taichi"
GPU_BIN="cuEFMM/build/Taichi"
INPUT_FILE="plummer_data_gen/plummer_10000.dat"
ITERATIONS=2
DT=0.25
SEED=0

# Directories for outputs
CPU_DIR="./run_cpu"
GPU_DIR="./run_gpu"

mkdir -p "${CPU_DIR}" "${GPU_DIR}"

# Cleanup old snapshots
rm -f "${CPU_DIR}"/snapshot_*.hdf5 "${GPU_DIR}"/snapshot_*.hdf5

echo "=== Running CPU Implementation (${ITERATIONS} steps) ==="
(
  cd "${CPU_DIR}"
  ls
  "../${CPU_BIN}" "${SEED}" "../${INPUT_FILE}" "${ITERATIONS}" 1 "${DT}"
)

echo -e "\n=== Running GPU Implementation (${ITERATIONS} steps) ==="
(
  cd "${GPU_DIR}"
  "../${GPU_BIN}" "${SEED}" "../${INPUT_FILE}" "${ITERATIONS}" 1 "${DT}"
)

# Comparison
FIRST_SNAPSHOT=0
LAST_SNAPSHOT=$((ITERATIONS - 1))

echo -e "\n=============================================="
echo "Comparing Initial Snapshot (Step ${FIRST_SNAPSHOT})"
echo "=============================================="
python3 compare_num_results.py \
  "${CPU_DIR}/snapshot_${FIRST_SNAPSHOT}.hdf5" \
  "${GPU_DIR}/snapshot_${FIRST_SNAPSHOT}.hdf5"

if [ "$ITERATIONS" -gt 1 ]; then
    echo -e "\n=============================================="
    echo "Comparing Final Snapshot (Step ${LAST_SNAPSHOT})"
    echo "=============================================="
    python3 compare_num_results.py \
    "${CPU_DIR}/snapshot_${LAST_SNAPSHOT}.hdf5" \
    "${GPU_DIR}/snapshot_${LAST_SNAPSHOT}.hdf5"
fi