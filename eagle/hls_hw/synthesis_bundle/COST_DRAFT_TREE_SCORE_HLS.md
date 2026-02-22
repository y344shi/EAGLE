# CostDraftTree Score HLS Check

This adds a testable HLS mapping for the score/index part of
`draft_tree_layer_gen_kernel` in:
`sglang-v0.5.6/python/sglang/srt/speculative/cost_draft_tree_kernel.cu`.

## Mapped logic
- `score = topk_probas_sampling[b, tid] * last_layer_scores[b, tid / node_top_k]`
- 64-lane bitonic sort (descending) on per-layer candidate scores
- output projection:
  - `sort_layer_scores`, `sort_layer_indices`
  - `cache_topk_indices = cumu_count + best_idx`
  - `parent_indices_in_layer = best_idx / node_top_k`
- hidden-state gather by parent index (same selection logic)
- token-path support for adaptation:
  - optional `topk_tokens_sampling` input
  - optional hot-token remap (`use_hot_token_id`, `hot_token_id`)
  - `expected_topk_tokens_sampling` (post-remap) check
  - `expected_output_tokens` check (top node_top_k tokens by sorted score index)

## Files
- `cost_draft_tree_score_hls.hpp`: synthesizable HLS-mapped kernel
- `cost_draft_tree_score_tb.cpp`: C-sim testbench with file-driven compare
- `dump_cost_draft_tree_score_case.py`: one-shot CUDA kernel runner + golden dump

## Run
```bash
cd hardware/EAGLE/eagle/hls_hw/synthesis_bundle

# 1) Run CUDA kernel once and dump golden tensors
python dump_cost_draft_tree_score_case.py --output cost_draft_tree_score_case.txt

# Optional: include hot-token remap validation path
python dump_cost_draft_tree_score_case.py --output cost_draft_tree_score_case_hot.txt --use-hot-token-id

# 2) Run HLS C-sim and compare against CUDA dump
g++ -std=c++17 -O2 -I. cost_draft_tree_score_tb.cpp -o /tmp/cdt_score_tb
/tmp/cdt_score_tb --case-file cost_draft_tree_score_case.txt
/tmp/cdt_score_tb --case-file cost_draft_tree_score_case.txt --dry-run
```

Expected result: all max diffs are `0` and mismatch counts are `0` (or `N/A` if the input case does not include token-path expected fields).
