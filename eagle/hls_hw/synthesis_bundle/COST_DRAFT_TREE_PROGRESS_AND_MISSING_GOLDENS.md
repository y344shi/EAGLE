# CostDraftTree HLS Progress And Missing Goldens

Updated: 2026-02-20

## Scope of this tracker
Tracks implementation progress and missing CUDA/H100 golden artifacts for CostDraftTree HLS migration.

## Implemented (local smoke complete)
- [done] Score kernel mapping
  - `cost_draft_tree_score_hls.hpp`
  - `cost_draft_tree_score_tb.cpp`
  - supports token-path remap and top output token checks.
- [done] Update-state kernel mapping
  - `cost_draft_tree_update_hls.hpp`
  - `cost_draft_tree_update_tb.cpp`
- [done] Candidate-tree controller
  - `cost_draft_tree_controller_hls.hpp`
  - `cost_draft_tree_controller_tb.cpp`
  - strict parent-only KV visibility listing/mask.
- [done] Fused wiring step (`score -> update -> controller -> parent-visible KV`)
  - `cost_draft_tree_fused_wiring_hls.hpp`
  - `cost_draft_tree_fused_wiring_tb.cpp`
  - includes `#pragma HLS DATAFLOW` in top step.
- [done] Tree KV cache gather (controller KV list/mask -> per-candidate K/V streams)
  - `cost_draft_tree_kv_cache_hls.hpp`
  - `cost_draft_tree_kv_cache_tb.cpp`
  - validated for mask filtering, OOB filtering, per-query ordering, and zero-batch guard behavior.
- [done] File-driven dump scripts and one-shot runner
  - `dump_cost_draft_tree_score_case.py`
  - `dump_cost_draft_tree_update_case.py`
  - `dump_cost_draft_tree_controller_case.py`
  - `dump_cost_draft_tree_fused_wiring_case.py`
  - `dump_all_cost_draft_tree_cases.sh`
  - `check_cost_draft_tree_goldens.sh` now runs full TB comparisons by default and supports `--dry-run`.
- [done] CUDA-backed expected payload generation
  - update case now generated from CUDA score+update kernels.
  - fused wiring case now includes CUDA-backed score/update expected tensors and file-driven expected comparisons in fused TB.

## Golden case status (H100 trace-derived)
- [done] `cost_draft_tree_score_case.txt`
- [done] `cost_draft_tree_score_case_hot.txt`
- [done] `cost_draft_tree_update_case.txt`
- [done] `cost_draft_tree_controller_case.txt`
- [done] `cost_draft_tree_fused_wiring_case.txt`
- [done] `cost_draft_tree_case_manifest.txt`

## Runtime tensor gap status (from earlier capture contract)
- [done] Tier1 runtime tensors captured: `tensor_005/006/007/011/014`
- [done] packed projection weights/scales captured
- [done] CostDraftTree-specific runtime golden dumps (5/5 + manifest)

## Next integration work (after goldens)
- [next] Add single command regression script that runs all CostDraftTree TBs against golden files.

## One-shot command checklist for future H100 session
1. `python dump_cost_draft_tree_score_case.py --output cost_draft_tree_score_case_hot.txt --use-hot-token-id`
2. Run all case dumps and self checks:
   - `./dump_all_cost_draft_tree_cases.sh --output-dir <target_case_dir>`
3. Copy artifacts into `hardware/EAGLE/eagle/hls_hw/synthesis_bundle/goldens/`
4. Run smoke+golden regression locally.

## Local validation status (2026-02-20)
Run:
- `./check_cost_draft_tree_goldens.sh`

Result:
- present: `cost_draft_tree_score_case.txt`
- present: `cost_draft_tree_score_case_hot.txt`
- present: `cost_draft_tree_update_case.txt`
- present: `cost_draft_tree_controller_case.txt`
- present: `cost_draft_tree_fused_wiring_case.txt`
- invalid: none

Capture-folder parser-only run:
- `./check_cost_draft_tree_goldens.sh --dry-run /home/y344shi/workspace/eagle4_adaptation/capture`
- result: all expected case files parse cleanly.

One-shot local generation check:
- `./dump_all_cost_draft_tree_cases.sh --output-dir /tmp/cdt_cases_new --kernel-src <cost_draft_tree_kernel.cu>`
- result: 5/5 case files generated, manifest order check passed, full TB checks passed.
