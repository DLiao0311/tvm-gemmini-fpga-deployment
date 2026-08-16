#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GEMMINI_SOFTWARE_HOME="${PROJECT_ROOT}/third_party/gemmini-rocc-tests"

# 可由外部指定，沒指定就用預設值
# ---------------------------------------------
DIM="${DIM:-16}"
SCRATCHPAD_KB="${SCRATCHPAD_KB:-512}"
ACCUMULATOR_KB="${ACCUMULATOR_KB:-256}"
BANK_NUM=4
# ---------------------------------------------

MLF_ROOT="${PROJECT_ROOT}/generated/mlf/runs"
if [ "$#" -ge 1 ]; then
  MLF_DIR="$1"
elif [ -d "${MLF_ROOT}" ]; then
  MLF_DIR="$(find "${MLF_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
else
  MLF_DIR=""
fi
if [ -z "${MLF_DIR}" ] || [ ! -d "${MLF_DIR}" ]; then
  echo "ERROR: no MLF run found. Run Step 2 without LUT first, or pass an MLF run directory as argument 1." >&2
  exit 1
fi
OUTPUT_ELF="${2:-${PROJECT_ROOT}/generated/elf/pc_gemmini_dim${DIM}_sp${SCRATCHPAD_KB}kb_acc${ACCUMULATOR_KB}kb_mlf.elf}"

SRC_DIR="${MLF_DIR}/codegen/host/src"
CODEGEN_INCLUDE_DIR="${MLF_DIR}/codegen/host/include"
RUNTIME_INCLUDE_DIR="${MLF_DIR}/runtime/include"

MAIN_C="${PROJECT_ROOT}/runtime/run.c"
LIB0_C="${SRC_DIR}/default_lib0.c"
LIB1_C="${SRC_DIR}/default_lib1.c"

echo "Using MLF_DIR=${MLF_DIR}"
echo "Using SRC_DIR=${SRC_DIR}"
echo "Output ELF=${OUTPUT_ELF}"
mkdir -p "$(dirname -- "${OUTPUT_ELF}")"


# -----------------------------
# 自動計算 Gemmini 參數
# -----------------------------

# scratchpad: int8 => 1 byte
# BANK_ROWS = scratchpad_bytes / BANK_NUM / DIM
SCRATCHPAD_BYTES=$((SCRATCHPAD_KB * 1024))
BANK_ROWS=$((SCRATCHPAD_BYTES / BANK_NUM / DIM))

# accumulator: int32 => 4 bytes
# ACC_ROWS = accumulator_bytes / 4 / DIM
ACCUMULATOR_BYTES=$((ACCUMULATOR_KB * 1024))
ACC_ROWS=$((ACCUMULATOR_BYTES / 4 / DIM))

# MAX_BLOCK_LEN = 64 / (DIM * 1)
MAX_BLOCK_LEN=$((64 / DIM))
if [ "${MAX_BLOCK_LEN}" -lt 1 ]; then
  MAX_BLOCK_LEN=1
fi

# MAX_BLOCK_LEN_ACC = 64 / (DIM * 4)
MAX_BLOCK_LEN_ACC=$((64 / (DIM * 4)))
if [ "${MAX_BLOCK_LEN_ACC}" -lt 1 ]; then
  MAX_BLOCK_LEN_ACC=1
fi

# 32x32 systolic array
# scratchpad = 256KB, 4 banks -> BANK_ROWS = 2048
# accumulator = 64KB, acc_t = int32, DIM = 32 -> ACC_ROWS = 512
# MAX_BLOCK_LEN = 64 / (32 * 1) = 2
# MAX_BLOCK_LEN_ACC = 1 because 64 / (32 * 4) < 1 and Gemmini header generator clamps it to 1
# -DPRINT_TILE=1 \  ----> debug 用，會在每次 tile 計算完後印出 tile 的內容，會讓執行時間變長很多

riscv64-linux-gnu-gcc \
  -static \
  -mcmodel=medany \
  -march=rv64gc \
  -mabi=lp64d \
  -DDIM="${DIM}" \
  -DBANK_NUM=4 \
  -DBANK_ROWS="${BANK_ROWS}" \
  -DACC_ROWS="${ACC_ROWS}" \
  -DMAX_BLOCK_LEN="${MAX_BLOCK_LEN}" \
  -DMAX_BLOCK_LEN_ACC="${MAX_BLOCK_LEN_ACC}" \
  -O2 \
  -I"${GEMMINI_SOFTWARE_HOME}/include" \
  -I"${GEMMINI_SOFTWARE_HOME}" \
  -I"${PROJECT_ROOT}/runtime" \
  -I"${CODEGEN_INCLUDE_DIR}" \
  -I"${RUNTIME_INCLUDE_DIR}" \
  "${MAIN_C}" \
  "${LIB0_C}" \
  "${LIB1_C}" \
  -Wl,--allow-multiple-definition \
  -lm \
  -o "${OUTPUT_ELF}"

  
if [ -f "${OUTPUT_ELF}" ]; then
  echo "OK: built ${OUTPUT_ELF}"
  echo "sp: ${BANK_ROWS} rows"
  echo "acc: ${ACC_ROWS} rows"
  echo "ELF size: $(du -h "${OUTPUT_ELF}" | cut -f1)"
  echo "ELF file: $(file "${OUTPUT_ELF}")"
else
  echo "ERROR: ELF was not generated: ${OUTPUT_ELF}" >&2
  exit 1
fi
