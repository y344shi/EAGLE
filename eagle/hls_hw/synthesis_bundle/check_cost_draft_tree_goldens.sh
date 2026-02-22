#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="${ROOT_DIR}"
DRY_RUN_ONLY=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] [case_dir]

Options:
  --dry-run   Parse-only checks (no numerical TB execution)
  --full      Run full TB comparisons (default)
  -h, --help  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN_ONLY=1; shift ;;
    --full) DRY_RUN_ONLY=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      CASE_DIR="$1"
      shift
      ;;
  esac
done

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -I.)

cd "${ROOT_DIR}"

echo "[info] synthesis bundle : ${ROOT_DIR}"
echo "[info] case directory   : ${CASE_DIR}"
if [[ ${DRY_RUN_ONLY} -eq 1 ]]; then
  echo "[info] mode           : dry-run parser checks"
else
  echo "[info] mode           : full TB comparisons"
fi

echo "[info] compiling dry-run checkers..."
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_score_tb.cpp -o /tmp/cdt_score_tb_check
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_update_tb.cpp -o /tmp/cdt_update_tb_check
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_controller_tb.cpp -o /tmp/cdt_controller_tb_check
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cdt_fused_wiring_tb_check

declare -a SPECS=(
  "cost_draft_tree_score_case.txt|/tmp/cdt_score_tb_check"
  "cost_draft_tree_score_case_hot.txt|/tmp/cdt_score_tb_check"
  "cost_draft_tree_update_case.txt|/tmp/cdt_update_tb_check --case-file"
  "cost_draft_tree_controller_case.txt|/tmp/cdt_controller_tb_check --case-file"
  "cost_draft_tree_fused_wiring_case.txt|/tmp/cdt_fused_wiring_tb_check --case-file"
)
declare -a EXPECTED_ORDER=(
  "cost_draft_tree_score_case.txt"
  "cost_draft_tree_score_case_hot.txt"
  "cost_draft_tree_update_case.txt"
  "cost_draft_tree_controller_case.txt"
  "cost_draft_tree_fused_wiring_case.txt"
)

missing=0
invalid=0
present=0

for spec in "${SPECS[@]}"; do
  IFS='|' read -r rel cmd_prefix <<< "${spec}"
  file="${CASE_DIR}/${rel}"
  if [[ ! -s "${file}" ]]; then
    echo "[missing] ${rel}"
    missing=$((missing + 1))
    continue
  fi

  echo "[check] ${rel}"
  read -r -a cmd_parts <<< "${cmd_prefix}"
  run_fail_log="/tmp/cdt_fullrun.log"
  if [[ ${DRY_RUN_ONLY} -eq 1 ]]; then
    run_fail_log="/tmp/cdt_dryrun.log"
    if "${cmd_parts[@]}" "${file}" --dry-run >"${run_fail_log}" 2>&1; then
      present=$((present + 1))
      sed -n '1,2p' "${run_fail_log}"
    else
      echo "[invalid] ${rel}"
      cat "${run_fail_log}"
      invalid=$((invalid + 1))
    fi
    continue
  fi

  if "${cmd_parts[@]}" "${file}" >"${run_fail_log}" 2>&1; then
    present=$((present + 1))
    sed -n '1,4p' "${run_fail_log}"
  else
    echo "[invalid] ${rel}"
    cat "${run_fail_log}"
    invalid=$((invalid + 1))
  fi
done

manifest="${CASE_DIR}/cost_draft_tree_case_manifest.txt"
if [[ -f "${manifest}" ]]; then
  echo "[check] cost_draft_tree_case_manifest.txt"
  mapfile -t manifest_lines < "${manifest}"
  if [[ ${#manifest_lines[@]} -ne ${#EXPECTED_ORDER[@]} ]]; then
    echo "[invalid] manifest length mismatch: got=${#manifest_lines[@]} expected=${#EXPECTED_ORDER[@]}"
    invalid=$((invalid + 1))
  else
    manifest_ok=1
    for i in "${!EXPECTED_ORDER[@]}"; do
      if [[ "${manifest_lines[$i]}" != "${EXPECTED_ORDER[$i]}" ]]; then
        echo "[invalid] manifest order mismatch at index ${i}: got='${manifest_lines[$i]}' expected='${EXPECTED_ORDER[$i]}'"
        manifest_ok=0
        break
      fi
    done
    if [[ ${manifest_ok} -eq 1 ]]; then
      echo "[order-ok] manifest names/order match expected."
    else
      invalid=$((invalid + 1))
    fi
  fi
else
  echo "[warn] manifest not found: ${manifest}"
fi

echo "[summary] present=${present} missing=${missing} invalid=${invalid}"
if [[ ${missing} -gt 0 || ${invalid} -gt 0 ]]; then
  exit 2
fi

echo "[PASS] all CostDraftTree golden case files are present and parse cleanly."
