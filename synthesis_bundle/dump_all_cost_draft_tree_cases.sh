#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${SCRIPT_DIR}"
KERNEL_SRC=""
RUN_CHECK=1
E2E_CASE=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Run all CostDraftTree dump scripts in one shot and self-check outputs.

Options:
  --python <path>           Python executable (default: ${PYTHON_BIN})
  --output-dir <path>       Output directory for all case files (default: ${OUTPUT_DIR})
  --kernel-src <path>       Optional CUDA kernel source path for CUDA-backed dump scripts
  --e2e-case <path>         Optional draft E2E case used to seed orchestrator dump and sidecars
  --skip-check              Skip final check_cost_draft_tree_goldens.sh validation
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --kernel-src) KERNEL_SRC="$2"; shift 2 ;;
    --e2e-case) E2E_CASE="$2"; shift 2 ;;
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
  "cost_draft_tree_multilayer_orchestrator_case.txt"
)
E2E_FILES=(
  "cost_draft_tree_draft_e2e_case.txt"
  "cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin"
  "cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin"
)

run_dump() {
  local step="$1"
  local total="$2"
  local desc="$3"
  shift 3
  echo "[dump ${step}/${total}] ${desc}"
  "$@"
}

copy_e2e_artifacts() {
  local e2e_src="$1"
  local e2e_dst="${OUTPUT_DIR}/cost_draft_tree_draft_e2e_case.txt"
  local src_dir
  src_dir="$(cd "$(dirname "${e2e_src}")" && pwd)"
  cp -f "${e2e_src}" "${e2e_dst}"

  if ! "${PYTHON_BIN}" - "${e2e_src}" "${src_dir}" "${OUTPUT_DIR}" > /tmp/cdt_e2e_copy.log 2>&1 <<'PY'
import shutil
import sys
from pathlib import Path

case_path = Path(sys.argv[1]).resolve()
src_dir = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()

prefix_k = None
prefix_v = None
with case_path.open("r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        key = parts[0]
        count = int(parts[1])
        payload = parts[2:]
        if len(payload) != count:
            continue
        if key == "prefix_hbm_k_file" and count >= 1:
            prefix_k = payload[0]
        if key == "prefix_hbm_v_file" and count >= 1:
            prefix_v = payload[0]

def resolve(val, fallback_name):
    if val:
        p = Path(val)
        if not p.is_absolute():
            p = (src_dir / p).resolve()
        if p.exists():
            return p
    direct = (src_dir / fallback_name).resolve()
    if direct.exists():
        return direct
    raise FileNotFoundError(fallback_name)

k_src = resolve(prefix_k, "cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin")
v_src = resolve(prefix_v, "cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin")
shutil.copyfile(k_src, out_dir / "cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin")
shutil.copyfile(v_src, out_dir / "cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin")
print(f"[copied] {k_src}")
print(f"[copied] {v_src}")
PY
  then
    echo "[error] failed to resolve/copy E2E sidecars from ${e2e_src}" >&2
    cat /tmp/cdt_e2e_copy.log >&2
    exit 1
  fi
  sed -n '1,2p' /tmp/cdt_e2e_copy.log
}

SCORE_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_score_case.py"
UPDATE_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_update_case.py"
CONTROLLER_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_controller_case.py"
FUSED_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_fused_wiring_case.py"
ORCH_SCRIPT="${SCRIPT_DIR}/dump_cost_draft_tree_multilayer_orchestrator_case.py"

for script in "${SCORE_SCRIPT}" "${UPDATE_SCRIPT}" "${CONTROLLER_SCRIPT}" "${FUSED_SCRIPT}" "${ORCH_SCRIPT}"; do
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

run_dump 1 6 "score case" "${score_cmd[@]}"
run_dump 2 6 "score hot-token case" "${score_hot_cmd[@]}"
update_cmd=("${PYTHON_BIN}" "${UPDATE_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_update_case.txt")
fused_cmd=("${PYTHON_BIN}" "${FUSED_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_fused_wiring_case.txt")
orch_cmd=("${PYTHON_BIN}" "${ORCH_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_multilayer_orchestrator_case.txt")
if [[ -n "${KERNEL_SRC}" ]]; then
  update_cmd+=(--kernel-src "${KERNEL_SRC}")
  fused_cmd+=(--kernel-src "${KERNEL_SRC}")
fi

run_dump 3 6 "update-state case" "${update_cmd[@]}"
run_dump 4 6 "controller case" "${PYTHON_BIN}" "${CONTROLLER_SCRIPT}" --output "${OUTPUT_DIR}/cost_draft_tree_controller_case.txt"
run_dump 5 6 "fused wiring case" "${fused_cmd[@]}"
if [[ -n "${E2E_CASE}" ]]; then
  copy_e2e_artifacts "${E2E_CASE}"
  orch_cmd+=(--e2e-case "${OUTPUT_DIR}/cost_draft_tree_draft_e2e_case.txt")
fi
run_dump 6 6 "multilayer orchestrator case" "${orch_cmd[@]}"

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

if [[ -n "${E2E_CASE}" ]]; then
  e2e_missing=0
  for f in "${E2E_FILES[@]}"; do
    if [[ ! -s "${OUTPUT_DIR}/${f}" ]]; then
      echo "[missing] ${f}"
      e2e_missing=$((e2e_missing + 1))
    fi
  done
  if [[ ${e2e_missing} -ne 0 ]]; then
    echo "[error] Missing ${e2e_missing} expected E2E files." >&2
    exit 1
  fi
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
<<<<<<< Updated upstream
  bash "${SCRIPT_DIR}/check_cost_draft_tree_goldens.sh" "${OUTPUT_DIR}"
=======
  check_cmd=("${SCRIPT_DIR}/check_cost_draft_tree_goldens.sh")
  if [[ -n "${E2E_CASE}" ]]; then
    check_cmd+=(--require-e2e)
  fi
  check_cmd+=("${OUTPUT_DIR}")
  "${check_cmd[@]}"
>>>>>>> Stashed changes
fi

echo "[PASS] All CostDraftTree dump files generated with expected names/order."
