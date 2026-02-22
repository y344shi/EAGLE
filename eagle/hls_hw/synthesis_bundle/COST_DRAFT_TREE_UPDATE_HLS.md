# CostDraftTree Update-State HLS Check

This adds a testable HLS mapping for `update_cumu_draft_state` from:
`sglang-v0.5.6/python/sglang/srt/speculative/cost_draft_tree_kernel.cu`.

## Mapped logic
- update `output_scores` from sorted scores (top `node_top_k`)
- update `output_tokens` using `(parent_idx, sorted_idx % node_top_k)` mapping
- copy selected parent tree-mask prefixes into `output_tree_mask`
- append current-layer nodes into cumulative buffers:
  - `cumu_tokens`, `cumu_scores`, `cumu_deltas`
  - `prev_indexs`, `next_indexs`, `side_indexs`
- update parent `next_indexs` pointers to first child of each parent branch
- update score maintenance buffers:
  - `work_scores` prefix update
  - `sort_scores` top-prefix merge from two descending score streams

## Files
- `cost_draft_tree_update_hls.hpp`: synthesizable HLS-mapped update kernel
- `cost_draft_tree_update_tb.cpp`: deterministic C-sim smoke TB

## Run smoke
```bash
cd hardware/EAGLE/eagle/hls_hw/synthesis_bundle
g++ -std=c++17 -O2 -I. cost_draft_tree_update_tb.cpp -o /tmp/cdt_update_tb
/tmp/cdt_update_tb
```

Expected result: both synthetic cases pass with zero diffs/mismatches.

## File-driven case mode
```bash
/tmp/cdt_update_tb --case-file cost_draft_tree_update_case.txt
/tmp/cdt_update_tb --case-file cost_draft_tree_update_case.txt --dry-run
```

Case file format uses `key count payload...` lines. Required keys:
- `meta` with 10 ints:
  - `batch_size node_top_k tree_width input_count cumu_count verify_num curr_depth max_input_size max_node_count max_verify_num`
- inputs:
  - `topk_probas`, `topk_tokens`, `sorted_scores`, `sorted_indexs`, `parent_indexs`, `topk_indexs`, `input_tree_mask`
- initial state:
  - `initial_cumu_tokens`, `initial_cumu_scores`, `initial_cumu_deltas`
  - `initial_prev_indexs`, `initial_next_indexs`, `initial_side_indexs`
  - `initial_output_scores`, `initial_output_tokens`
  - `initial_work_scores`, `initial_sort_scores`, `initial_output_tree_mask`
- optional expected state (if omitted, TB generates expected via reference C++):
  - `expected_*` mirrors all output buffers above.

## Current status
- This phase uses local synthetic smoke vectors only.
- CUDA/H100 golden capture for this kernel is intentionally deferred until the multi-kernel adaptation batch is complete.
