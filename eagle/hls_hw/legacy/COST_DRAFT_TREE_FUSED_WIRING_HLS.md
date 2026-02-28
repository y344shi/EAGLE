# CostDraftTree Fused Wiring (HLS)

This module wires the current HLS kernels into one layer-step pipeline:
- score/sort stage,
- cumulative state update stage,
- candidate-tree controller stage,
- parent-only KV list/mask stage.

## File
- `cost_draft_tree_fused_wiring_hls.hpp`
  - top API: `cost_draft_tree_fused_step_hls(...)`

## Dataflow usage
The top function applies `#pragma HLS DATAFLOW` and stages the following operations:
1. `cost_draft_tree_layer_score_hls_with_tokens`
2. `cost_draft_tree_update_state_hls`
3. `cdt_controller_expand_frontier`
4. `cdt_controller_build_parent_visible_kv`
5. `cdt_controller_export_frontier`

## Visibility contract
Per candidate query in the next frontier:
- visible KV = `prefix + ancestor chain(root->...->parent->self)`
- siblings/cousins are not visible.

## Smoke test
- `cost_draft_tree_fused_wiring_tb.cpp`
- compares fused pipeline output against sequential reference calls to stage kernels.

Run:
```bash
cd hardware/EAGLE/eagle/hls_hw/synthesis_bundle
g++ -std=c++17 -O2 -I. cost_draft_tree_fused_wiring_tb.cpp -o /tmp/cdt_fused_wiring_tb
/tmp/cdt_fused_wiring_tb
```

File-driven mode:
```bash
/tmp/cdt_fused_wiring_tb --case-file cost_draft_tree_fused_wiring_case.txt
/tmp/cdt_fused_wiring_tb --case-file cost_draft_tree_fused_wiring_case.txt --dry-run
```

Case file format uses `key count payload...` lines.
- `meta` has 17 ints:
  - `batch_size node_top_k tree_width hidden_size cumu_count input_count verify_num curr_depth max_input_size max_node_count max_verify_num max_tree_width max_prefix_len parent_width next_tree_width hot_vocab_size use_hot_token_id`
- required inputs:
  - `topk_probas_sampling`, `topk_tokens_sampling`, `last_layer_scores`, `input_hidden_states`
  - `topk_indexs_prev`, `input_tree_mask`
  - `prefix_kv_locs`, `prefix_lens`, `selected_cache_locs`
  - optional `hot_token_id` (identity mapping is used if absent)
- required initial legacy/controller state:
  - `legacy_*` buffers matching fused step input state
  - `controller_frontier_in`, `controller_node_count`, `controller_node_*`

The TB always compares fused output to its sequential stage reference, so explicit expected output payloads are not required for case files.
