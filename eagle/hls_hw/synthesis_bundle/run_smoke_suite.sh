#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CXX="${CXX:-g++}"
CXXFLAGS=(-std=c++17 -I.)

echo "[1/8] cost_draft_tree_controller_tb"
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_controller_tb.cpp -o /tmp/cost_draft_tree_controller_tb
/tmp/cost_draft_tree_controller_tb

echo "[2/8] cost_draft_tree_fused_wiring_tb"
"${CXX}" "${CXXFLAGS[@]}" cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cost_draft_tree_fused_wiring_tb
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

echo "[PASS] Smoke suite completed."
