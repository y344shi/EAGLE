#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${SCRIPT_DIR}"
KERNEL_SRC=""
RUN_CHECK=1

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Run all CostDraftTree dump scripts in one shot and self-check outputs.

Options:
  --python <path>           Python executable (default: ${PYTHON_BIN})
  --output-dir <path>       Output directory for all case files (default: ${OUTPUT_DIR})
  --kernel-src <path>       Optional CUDA kernel source path for CUDA-backed dump scripts
  --skip-check              Skip final check_cost_draft_tree_goldens.sh validation
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --kernel-src) KERNEL_SRC="$2"; shift 2 ;;
    --skip-check) RUN_CHECK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[error] Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

EXPECTED_FILES=(
  "cost_draft_tree_score_case.txt"
  "cost_draft_tree_score_case_hot.txt"
  "cost_draft_tree_update_case.txt"
  "cost_draft_tree_controller_case.txt"
  "cost_draft_tree_fused_wiring_case.txt"
)

run_dump() {
  local step="$1"
  local total="$2"
  local desc="$3"
  shift 3
  echo "[dump ${step}/${total}] ${desc}"
  "$@"
}

SCORE_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_score_case.py"
UPDATE_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_update_case.py"
CONTROLLER_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_controller_case.py"
FUSED_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_fused_wiring_case.py"

for script in "${SCORE_SCRIPT}" "${UPDATE_SCRIPT}" "${CONTROLLER_SCRIPT}" "${FUSED_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "[error] Missing dump script: ${script}" >&2
    exit 1
  fi
done

score_cmd=("${PYTHON_BIN}" "${SCORE_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_score_case.txt")
score_hot_cmd=("${PYTHON_BIN}" "${SCORE_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_score_case_hot.txt" --use-hot-token-id)
if [[ -n "${KERNEL_SRC}" ]]; then
  score_cmd+=(--kernel-src "${KERNEL_SRC}")
  score_hot_cmd+=(--kernel-src "${KERNEL_SRC}")
fi

run_dump 1 5 "score case" "${score_cmd[@]}"
run_dump 2 5 "score hot-token case" "${score_hot_cmd[@]}"
update_cmd=("${PYTHON_BIN}" "${UPDATE_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_update_case.txt")
fused_cmd=("${PYTHON_BIN}" "${FUSED_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_fused_wiring_case.txt")
if [[ -n "${KERNEL_SRC}" ]]; then
  update_cmd+=(--kernel-src "${KERNEL_SRC}")
  fused_cmd+=(--kernel-src "${KERNEL_SRC}")
fi

run_dump 3 5 "update-state case" "${update_cmd[@]}"
run_dump 4 5 "controller case" "${PYTHON_BIN}" "${CONTROLLER_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_controller_case.txt"
run_dump 5 5 "fused wiring case" "${fused_cmd[@]}"

echo "[self-check] verifying expected filenames and non-empty files..."
missing=0
for f in "${EXPECTED_FILES[@]}"; do
  if [[ ! -s "${OUTPUT_DIR}/${f}" ]]; then
    echo "[missing] ${f}"
    missing=$((missing + 1))
  fi
done
if [[ ${missing} -ne 0 ]]; then
  echo "[error] Missing ${missing} expected dump files." >&2
  exit 1
fi

MANIFEST="${OUTPUT_DIR}/cost_draft_tree_case_manifest.txt"
: > "${MANIFEST}"
for f in "${EXPECTED_FILES[@]}"; do
  echo "${f}" >> "${MANIFEST}"
done

echo "[self-check] verifying manifest order..."
mapfile -t lines < "${MANIFEST}"
if [[ ${#lines[@]} -ne ${#EXPECTED_FILES[@]} ]]; then
  echo "[error] Manifest length mismatch." >&2
  exit 1
fi
for i in "${!EXPECTED_FILES[@]}"; do
  if [[ "${lines[$i]}" != "${EXPECTED_FILES[$i]}" ]]; then
    echo "[error] Manifest order mismatch at index ${i}: got='${lines[$i]}' expected='${EXPECTED_FILES[$i]}'" >&2
    exit 1
  fi
done

if [[ ${RUN_CHECK} -eq 1 ]]; then
  echo "[self-check] running full TB checks against dumped cases..."
  "${SCRIPT_DIR}/check_cost_draft_tree_goldens.sh" "${OUTPUT_DIR}"
fi

echo "[PASS] All CostDraftTree dump files generated with expected names/order."
