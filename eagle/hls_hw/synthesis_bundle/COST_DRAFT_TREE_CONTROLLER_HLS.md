# CostDraftTree Candidate-Tree Controller (HLS)

This module adds a dedicated candidate-tree controller for HLS that tracks multi-layer draft candidates and enforces strict parent-only visibility.

## Why this exists
Current draft-state updates are kernel-local. This controller provides a persistent tree state abstraction for HLS integration:
- candidate listing/state bookkeeping across layers,
- parent/child/sibling linkage,
- deterministic frontier export,
- parent-only KV visibility mask/list generation.

## Parent-only visibility rule
For each frontier candidate query, visible keys are:
- prefix KV tokens, then
- ancestor chain from root to parent to self.

No sibling or cousin nodes are visible.

## Files
- `cost_draft_tree_controller_hls.hpp`
  - `cdt_controller_reset`
  - `cdt_controller_seed_frontier`
  - `cdt_controller_expand_frontier`
  - `cdt_controller_export_frontier`
  - `cdt_controller_build_parent_visible_kv`
- `cost_draft_tree_controller_tb.cpp`
  - deterministic smoke with two expansion layers and exact KV list checks.

## State model
Per batch, per node:
- token id
- parent id
- first child id
- last child id
- next sibling id
- depth
- cache location

Per batch, per frontier slot:
- frontier node id

## Mask/KV outputs
`cdt_controller_build_parent_visible_kv` emits:
- `kv_indices[batch, width, max_input_size]`
- `kv_mask[batch, width, max_input_size]`
- `kv_lens[batch, width]`

These are tree-shaped by construction and ready for downstream custom attention index/mask packing.

## Run smoke
```bash
cd hardware/EAGLE/eagle/hls_hw/synthesis_bundle
g++ -std=c++17 -O2 -I. cost_draft_tree_controller_tb.cpp -o /tmp/cdt_controller_tb
/tmp/cdt_controller_tb
```

Expected: `[PASS] cost_draft_tree controller HLS smoke passed.`

## File-driven case mode
```bash
/tmp/cdt_controller_tb --case-file cost_draft_tree_controller_case.txt
/tmp/cdt_controller_tb --case-file cost_draft_tree_controller_case.txt --dry-run
```

Case file format uses `key count payload...` lines. Required keys:
- scalar keys: `batch_size`, `width`, `max_tree_width`, `max_prefix_len`, `max_input_size`, `max_node_count`
- state/input keys:
  - `frontier_node_ids`
  - `prefix_kv_locs`, `prefix_lens`
  - `node_parent_ids`, `node_cache_locs`
- expected keys:
  - `expected_kv_indices`, `expected_kv_mask`, `expected_kv_lens`
  - optional: `expected_ancestor_node_ids`
