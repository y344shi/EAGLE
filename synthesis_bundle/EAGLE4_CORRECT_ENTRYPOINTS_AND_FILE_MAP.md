# EAGLE4 Correct Entry Points and File Map

This document is the canonical map for which top-level entry points to use and which files belong to each path.

## 0) Branch/Path Reality Note (2026-03-08)

- This file records the intended integration map from a fuller capture/validation branch.
- In this workspace, live HLS sources are under `hardware/EAGLE/synthesis_bundle` (not `hardware/EAGLE/eagle/hls_hw/synthesis_bundle`).
- Capture automation files (`capture_all_goldens.sh`, `export_hls_goldens.py`, capture-enabled `engine_test.py`) may be absent in some checkouts.
- For capture-flow recovery and exact port list, read:
  - `notes/GOLDEN_CAPTURE_WORKFLOW_HANDOFF.md`

## 0) End-to-End Candidate-Match Testbench Record

Primary end-to-end candidate-tree match bench:
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_fused_wiring_tb.cpp`

What it checks (file-driven case):
- top-k candidate selection outputs (`expected_cache_topk_indices`)
- fused hidden output (`expected_output_hidden_states`)
- frontier expansion (`expected_controller_frontier_out`)
- parent-visible KV metadata (`expected_controller_kv_indices`, `expected_controller_kv_mask`, `expected_controller_kv_lens`)
- exported frontier fields (`expected_controller_frontier_tokens`, `expected_controller_frontier_parent_ids`, `expected_controller_frontier_depths`, `expected_controller_frontier_cache_locs`)
- debug score/sort/parent traces and legacy update-state buffers

How this bench is run in the standard flow:
- `capture_all_goldens.sh` step `4/7` runs:
  - `dump_all_cost_draft_tree_cases.sh`
  - which runs `check_cost_draft_tree_goldens.sh`
  - which compiles and runs:
    - `cost_draft_tree_score_tb.cpp`
    - `cost_draft_tree_update_tb.cpp`
    - `cost_draft_tree_controller_tb.cpp`
    - `cost_draft_tree_fused_wiring_tb.cpp` (the end-to-end candidate match)

Direct one-off command for the end-to-end candidate bench:
- `g++ -std=c++17 -I. cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cdt_fused_wiring_tb && /tmp/cdt_fused_wiring_tb --case-file <path>/cost_draft_tree_fused_wiring_case.txt`

- `cd /home/y344shi/workspace/eagle4_adaptation/hardware/EAGLE/synthesis_bundle
g++ -std=c++17 -O2 -I. cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cdt_fused_wiring_tb
/tmp/cdt_fused_wiring_tb --multi-depth-steps 3`

Scope note:
- This bench now supports both:
  - single-step fused validation (`--multi-depth-steps` omitted or `1`)
  - multi-depth repeated fused-step validation (`--multi-depth-steps <N>`)
- Multi-depth mode chains frontier/state and validates candidate outputs each depth against a sequential reference pipeline.
- `cost_draft_tree_controller_tb.cpp` remains useful for focused parent-chain/controller-only topology checks.

Multi-depth command:
- `g++ -std=c++17 -I. cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cdt_fused_wiring_tb && /tmp/cdt_fused_wiring_tb --multi-depth-steps 3`



## 1) Correct Entry Points

### Runtime golden capture (Python/SGLang)
- Main orchestrator: `sglang-eagle4/eagle-project/eagle4/capture_all_goldens.sh`
- Runtime execution entry: `sglang-eagle4/eagle-project/eagle4/engine_test.py`
- Weight/golden export entry: `sglang-eagle4/eagle-project/eagle4/export_hls_goldens.py`

### HLS Tier1 compute entry (EAGLE4 layer-0 parity)
- Correct top: `eagle_tier1_top_eagle4_l0(...)`
- File: `hardware/EAGLE/synthesis_bundle/eagle_tier1_top.cpp`

### HLS Tier1 + LM head entry (EAGLE4 full path)
- Correct top: `eagle_tier1_lm_top_eagle4(...)`
- File: `hardware/EAGLE/synthesis_bundle/eagle_tier1_lm_top.cpp`

### Legacy wrapper (not preferred for new EAGLE4 integration)
- Legacy mixed wrapper: `eagle_tier1_lm_top(...)`
- File: `hardware/EAGLE/synthesis_bundle/eagle_tier1_lm_top.cpp`
- Notes:
  - Kept for compatibility/regression.
  - Carries legacy Eagle3-style LM ports.

### CostDraftTree fused entry
- Correct fused step API: `cost_draft_tree_fused_step_hls(...)`
- File: `hardware/EAGLE/synthesis_bundle/cost_draft_tree_fused_wiring_hls.hpp`

## 2) Required Files by Integration Path

### A) Tier1 + EAGLE4 LM head (recommended Vitis path)
- `hardware/EAGLE/synthesis_bundle/eagle_tier1_lm_top.cpp`
- `hardware/EAGLE/synthesis_bundle/eagle_tier1_lm_top.hpp`
- `hardware/EAGLE/synthesis_bundle/eagle_tier1_top.cpp`
- `hardware/EAGLE/synthesis_bundle/eagle_tier1_top.hpp`
- `hardware/EAGLE/synthesis_bundle/eagle4_lm_head_hls.hpp`
- `hardware/EAGLE/synthesis_bundle/tmac_utils.hpp`
- `hardware/EAGLE/synthesis_bundle/attention_solver.hpp`
- `hardware/EAGLE/synthesis_bundle/fused_online_attention_pwl.hpp`
- `hardware/EAGLE/synthesis_bundle/deep_pipeline_lutmac.hpp`
- `hardware/EAGLE/synthesis_bundle/kv_cache_manager.hpp`
- `hardware/EAGLE/synthesis_bundle/rms_norm_stream.hpp`
- `hardware/EAGLE/synthesis_bundle/rope_kernel.hpp`
- `hardware/EAGLE/synthesis_bundle/stream_utils.hpp`

### B) CostDraftTree fused wiring path
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_fused_wiring_hls.hpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_score_hls.hpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_update_hls.hpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_controller_hls.hpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_kv_cache_hls.hpp`

### C) CostDraftTree case-file IO (testbench-only)
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_tb_case_io.hpp`

## 3) Current Wiring Status (Important)

- `cost_draft_tree_fused_step_hls(...)` currently wires:
  - score
  - update
  - controller expand
  - parent-visible KV list/mask generation
  - frontier export
- `cost_draft_tree_fused_step_hls(...)` does **not** currently call the KV gather stage.
  - The KV gather helper exists as:
    - `cost_draft_tree_tree_kv_cache_gather_hls(...)` in `cost_draft_tree_fused_wiring_hls.hpp`
    - backed by `cdt_tree_kv_cache_gather_hls(...)` in `cost_draft_tree_kv_cache_hls.hpp`
  - This is available for integration but not yet consumed in the fused step entry.

## 4) Simulation / Validation Targets

### Tier1 + LM pipeline
- `hardware/EAGLE/synthesis_bundle/test_eagle_top_eagle4.cpp`
- `hardware/EAGLE/synthesis_bundle/test_eagle_top_eagle4_perop.cpp`
- `hardware/EAGLE/synthesis_bundle/test_eagle4_lm_head.cpp`
- `hardware/EAGLE/synthesis_bundle/test_eagle_tier1_lm_top_eagle4.cpp`

### CostDraftTree modules
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_score_tb.cpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_update_tb.cpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_controller_tb.cpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_fused_wiring_tb.cpp`
- `hardware/EAGLE/synthesis_bundle/cost_draft_tree_kv_cache_tb.cpp`

### Additional module TBs (not in default `capture_all_goldens.sh` flow)
- `hardware/EAGLE/synthesis_bundle/deep_pipeline_lutmac_tb.cpp`
- `hardware/EAGLE/synthesis_bundle/fused_online_attention_pwl_tb.cpp`
- `hardware/EAGLE/synthesis_bundle/fc1_compare_tb.cpp`

### CostDraftTree dump/check scripts
- Dump all cases: `hardware/EAGLE/synthesis_bundle/dump_all_cost_draft_tree_cases.sh`
- Validate dumped cases: `hardware/EAGLE/synthesis_bundle/check_cost_draft_tree_goldens.sh`

## 5) Vitis Top Recommendation

For EAGLE4 full chain integration, set top function to:
- `eagle_tier1_lm_top_eagle4`

Use `eagle_tier1_top_eagle4_l0` only when validating Tier1 block without LM head.

Avoid using `eagle_tier1_lm_top` as primary top for new EAGLE4 builds unless you explicitly need legacy wrapper compatibility.

## 6) Quick Port Checklist

1. Capture goldens with `capture_all_goldens.sh` using EAGLE4 config (`num_kv_heads=32`).
2. Build and pass Tier1/LM HLS testbench targets.
3. Build and pass CostDraftTree case-file checkers.
4. If integrating tree-KV streaming in fused path, wire `cost_draft_tree_tree_kv_cache_gather_hls(...)` into `cost_draft_tree_fused_step_hls(...)`.
5. Keep this file updated when any top function signature or wiring order changes.
