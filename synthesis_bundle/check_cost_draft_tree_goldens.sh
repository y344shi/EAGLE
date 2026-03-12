#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="${ROOT_DIR}"
DRY_RUN_ONLY=0
STRICT_CLASSIC=0

# Orchestrator TB allocates large local buffers in non-synthesis C++ simulation.
# Ensure enough stack to avoid host-side segfaults in full checks.
ulimit -s unlimited || true

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] [case_dir]

Options:
  --dry-run   Parse-only checks (no numerical TB execution)
  --full      Run full TB comparisons (default)
  --strict-classic  Treat classic backend mismatches as fatal in orchestrator TB
  -h, --help  Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN_ONLY=1; shift ;;
    --full) DRY_RUN_ONLY=0; shift ;;
    --strict-classic) STRICT_CLASSIC=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      CASE_DIR="$1"
      shift
      ;;
  esac
done

if [[ ! -d "${CASE_DIR}" ]]; then
  echo "[error] case directory not found: ${CASE_DIR}" >&2
  exit 2
fi
CASE_DIR="$(cd "${CASE_DIR}" && pwd)"

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
if [[ ${STRICT_CLASSIC} -eq 1 ]]; then
  echo "[info] classic mode   : strict"
fi

echo "[info] compiling dry-run checkers..."
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_score_tb.cpp cost_draft_tree_score_hls.cpp -o /tmp/cdt_score_tb_check
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_update_tb.cpp cost_draft_tree_update_hls.cpp -o /tmp/cdt_update_tb_check
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_controller_tb.cpp cost_draft_tree_controller_hls.cpp -o /tmp/cdt_controller_tb_check
"${CXX}" "${CXXFLAGS[@]}" \
  cost_draft_tree_fused_wiring_tb.cpp \
  cost_draft_tree_fused_wiring_hls.cpp \
  cost_draft_tree_score_hls.cpp \
  cost_draft_tree_update_hls.cpp \
  cost_draft_tree_controller_hls.cpp \
  eagle4_lm_head_hls.cpp \
  eagle_tier1_lm_top.cpp \
  eagle_tier1_top.cpp \
  -o /tmp/cdt_fused_wiring_tb_check
"${CXX}" "${CXXFLAGS[@]}" \
  cost_draft_tree_multilayer_orchestrator_tb.cpp \
  cost_draft_tree_fused_wiring_hls.cpp \
  cost_draft_tree_score_hls.cpp \
  cost_draft_tree_update_hls.cpp \
  cost_draft_tree_controller_hls.cpp \
  eagle4_lm_head_hls.cpp \
  eagle_tier1_lm_top.cpp \
  eagle_tier1_top.cpp \
  -o /tmp/cdt_multilayer_orch_tb_check

declare -a SPECS=(
  "cost_draft_tree_score_case.txt|/tmp/cdt_score_tb_check"
  "cost_draft_tree_score_case_hot.txt|/tmp/cdt_score_tb_check"
  "cost_draft_tree_update_case.txt|/tmp/cdt_update_tb_check --case-file"
  "cost_draft_tree_controller_case.txt|/tmp/cdt_controller_tb_check --case-file"
  "cost_draft_tree_fused_wiring_case.txt|/tmp/cdt_fused_wiring_tb_check --case-file"
  "cost_draft_tree_multilayer_orchestrator_case.txt|/tmp/cdt_multilayer_orch_tb_check --case-file"
)
declare -a EXPECTED_ORDER=(
  "cost_draft_tree_score_case.txt"
  "cost_draft_tree_score_case_hot.txt"
  "cost_draft_tree_update_case.txt"
  "cost_draft_tree_controller_case.txt"
  "cost_draft_tree_fused_wiring_case.txt"
  "cost_draft_tree_multilayer_orchestrator_case.txt"
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
  run_cmd=("${cmd_parts[@]}" "${file}")
  if [[ ${STRICT_CLASSIC} -eq 1 && "${rel}" == "cost_draft_tree_multilayer_orchestrator_case.txt" ]]; then
    run_cmd+=(--strict-classic)
  fi
  run_fail_log="/tmp/cdt_fullrun.log"
  if [[ ${DRY_RUN_ONLY} -eq 0 && "${rel}" == "cost_draft_tree_multilayer_orchestrator_case.txt" ]]; then
    if [[ ${STRICT_CLASSIC} -eq 1 ]]; then
      capture_backend="$(awk '$1=="capture_backend"{for(i=3;i<=NF;++i) printf "%s%s",$i,(i<NF?" ":""); print ""}' "${file}" 2>/dev/null || true)"
      if [[ "${capture_backend}" != "eagle4_classic" && "${capture_backend}" != "classic_eagle" ]]; then
        echo "[invalid] ${rel}: full run requires capture_backend=eagle4_classic|classic_eagle (got='${capture_backend:-none}')"
        invalid=$((invalid + 1))
        continue
      fi
    fi
    required_keys=(
      "enable_prefill_stage"
      "prefill_input_hidden_states_3h"
      "prefill_input_embed_states"
      "enable_accepted_kv_compact"
      "accepted_draft_node_ids"
      "node_to_hbm_slot_init"
    )
    missing_fixture=0
    for key in "${required_keys[@]}"; do
      if ! awk -v k="${key}" '$1==k{found=1} END{exit(found?0:1)}' "${file}"; then
        echo "[invalid] ${rel}: missing required full-path fixture key '${key}'"
        missing_fixture=1
      fi
    done
    if [[ ${missing_fixture} -ne 0 ]]; then
      invalid=$((invalid + 1))
      continue
    fi
  fi
  if [[ ${DRY_RUN_ONLY} -eq 1 ]]; then
    run_fail_log="/tmp/cdt_dryrun.log"
    if "${run_cmd[@]}" --dry-run >"${run_fail_log}" 2>&1; then
      present=$((present + 1))
      sed -n '1,2p' "${run_fail_log}"
    else
      echo "[invalid] ${rel}"
      cat "${run_fail_log}"
      invalid=$((invalid + 1))
    fi
    continue
  fi

  if "${run_cmd[@]}" >"${run_fail_log}" 2>&1; then
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
