#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -I.)
HOST_LINK_INCLUDES=(
  -include eagle4_lm_head_hls.cpp
  -include eagle_tier1_lm_top.cpp
  -include eagle_tier1_top.cpp
  -include cost_draft_tree_fused_wiring_hls.cpp
)
RUN_CDT_E2E=0
CDT_E2E_ARGS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Run local C++ smoke tests. Optionally run CostDraftTree true E2E flow.

Options:
  --with-cdt-e2e                    Also run run_cost_draft_tree_e2e_flow.sh
  --cdt-case-dir <path>             Case dir for E2E flow
  --cdt-capture-case-dir <path>     Source capture dir to sync from
  --cdt-feature-profiles <csv>      Feature profiles (e.g. mid,low,high)
  --cdt-no-feature-sweep            Only run mid profile
  --cdt-no-require-e2e              Do not require E2E sidecars
  --cdt-strict                      Strict mode for E2E flow
  -h, --help                        Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-cdt-e2e) RUN_CDT_E2E=1; shift ;;
    --cdt-case-dir) CDT_E2E_ARGS+=(--case-dir "$2"); shift 2 ;;
    --cdt-capture-case-dir) CDT_E2E_ARGS+=(--capture-case-dir "$2"); shift 2 ;;
    --cdt-feature-profiles) CDT_E2E_ARGS+=(--feature-profiles "$2"); shift 2 ;;
    --cdt-no-feature-sweep) CDT_E2E_ARGS+=(--no-feature-sweep); shift ;;
    --cdt-no-require-e2e) CDT_E2E_ARGS+=(--no-require-e2e); shift ;;
    --cdt-strict) CDT_E2E_ARGS+=(--strict); shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[error] Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "[1/8] cost_draft_tree_controller_tb"
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_controller_tb.cpp -o /tmp/cost_draft_tree_controller_tb
/tmp/cost_draft_tree_controller_tb

echo "[2/8] cost_draft_tree_fused_wiring_tb"
"${CXX}" "${CXXFLAGS[@]}" "${HOST_LINK_INCLUDES[@]}" cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cost_draft_tree_fused_wiring_tb
/tmp/cost_draft_tree_fused_wiring_tb

echo "[3/8] cost_draft_tree_kv_cache_tb"
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_kv_cache_tb.cpp -o /tmp/cost_draft_tree_kv_cache_tb
/tmp/cost_draft_tree_kv_cache_tb

echo "[4/8] cost_draft_tree_update_tb"
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_update_tb.cpp -o /tmp/cost_draft_tree_update_tb
/tmp/cost_draft_tree_update_tb

echo "[5/8] fused_online_attention_pwl_tb"
"${CXX}" "${CXXFLAGS[@]}" fused_online_attention_pwl_tb.cpp -o /tmp/fused_online_attention_pwl_tb
/tmp/fused_online_attention_pwl_tb

echo "[6/8] deep_pipeline_lutmac_tb --smoke"
"${CXX}" "${CXXFLAGS[@]}" -DTMAC_ENABLE_LUTMAC_TB -Dmain2=main deep_pipeline_lutmac_tb.cpp -o /tmp/deep_pipeline_lutmac_tb
/tmp/deep_pipeline_lutmac_tb --smoke

echo "[7/8] test_eagle_top --smoke (default KV profile)"
"${CXX}" "${CXXFLAGS[@]}" test_eagle_top.cpp eagle_tier1_top.cpp -o /tmp/test_eagle_top
/tmp/test_eagle_top --smoke

echo "[8/8] test_eagle_top --smoke (KV8 profile compile check)"
"${CXX}" "${CXXFLAGS[@]}" -DTMAC_NUM_KV_HEADS=8 test_eagle_top.cpp eagle_tier1_top.cpp -o /tmp/test_eagle_top_kv8
/tmp/test_eagle_top_kv8 --smoke

if [[ "${RUN_CDT_E2E}" -eq 1 ]]; then
  echo "[9/9] run_cost_draft_tree_e2e_flow"
  bash "${ROOT_DIR}/run_cost_draft_tree_e2e_flow.sh" "${CDT_E2E_ARGS[@]}"
fi

echo "[PASS] Smoke suite completed."
