# EAGLE4 FPGA Migration Playbook (Kernel-by-Kernel)

This document is a practical execution plan for an expert implementing EAGLE4 CostDraftTree offload on FPGA/HLS.

## 1) Scope and assumptions
- Codebase: `sglang-eagle4/sglang-v0.5.6/python/sglang/srt/speculative/`
- Primary target file: `cost_draft_tree_kernel.cu`
- Keep Python orchestration unchanged first (`cost_draft_tree_main.py`), swap backend kernels progressively.
- Batch/system bounds (from current code):
  - `MAX_BATCH_SIZE = 128`
  - `verify_count <= 128`
  - `node_top_k` commonly `<= 8` (code comments mention up to 10)
  - hidden size commonly `4096`

## 2) What is already done
- `draft_tree_layer_score_index_gen_op` score path has an HLS-mapped reference and CUDA parity harness:
  - `cost_draft_tree_score_hls.hpp`
  - `cost_draft_tree_score_tb.cpp`
  - `dump_cost_draft_tree_score_case.py`

Use this as the template for all next kernels: **CUDA one-shot dump -> HLS C-sim compare**.

## 3) Migration order (recommended)

### Phase P0: End-to-end correctness with minimal kernel set
Implement in this order:
1. `draft_tree_layer_score_index_gen_op`
2. `update_cumu_draft_state`
3. `calculate_work_deltas`
4. `fused_tree_mask_operator`
5. `convert_logits_to_probas`
6. `draft_tree_reject_sample`
7. `update_verify_result`

This is the minimum set to preserve draft/verify behavior and dynamic width control.

### Phase P1: Performance support kernels
8. `compute_kv_indices`
9. `calculate_verify_kv_qo_ms`
10. `assign_draft_cache_locs_operator`, `assign_req_to_token_pool_operator`, `update_node_data`

### Phase P2: Graph replay / launch overhead reduction
11. `draft_replay_input_operator`
12. `extend_replay_input_operator`

## 4) Kernel interface contracts (HLS-focused)

### K1) `draft_tree_layer_score_index_gen_op`
- Role: per-layer score multiply + sort + parent index + hidden gather.
- Core shapes:
  - `topk_probas_sampling`: `[B, tree_width*topk]` float32
  - `last_layer_scores`: `[B, tree_width]` float32
  - `input_hidden_states`: `[B*tree_width, H]` fp16/bf16
  - outputs: sorted scores/indices + `[B, topk, H]` hidden gather
- Notes:
  - fixed-width bitonic sort (`<=64`) maps cleanly to pipelined compare/swap network.
  - hidden gather dominates traffic.

### K2) `update_cumu_draft_state`
- Role: update tree state buffers (`cumu_tokens/scores/deltas`, parent-child links), update output tokens/scores, merge-sort-like score maintenance.
- Core shapes:
  - state tensors `[B, MAX_Node_Count]`, score work buffers `[B, MAX_Verify_Num + topk]`
  - tree mask update into `[B, topk, MAX_Input_Size+1]`
- Notes:
  - this is a stateful bookkeeping kernel; correctness first, then optimize memory layout.
  - shared-memory insertion sort in CUDA should become deterministic local sort for short windows.

### K3) `calculate_work_deltas`
- Role: compute marginal gain for dynamic width decision.
- Core shapes:
  - `work_scores`: `[B, node_count + topk]` float32 (two sorted segments)
  - `work_deltas`: `[topk+1]` float32
- Notes:
  - easy HLS target (merge-sum + reduction), low risk.

### K4) `fused_tree_mask_operator`
- Role: KV index fill + bool mask fill + pack to bytes.
- Notes:
  - pure memory/bit operations; major kernel-launch reduction candidate.
  - define byte-packing compatibility tests carefully.

### K5) `convert_logits_to_probas`
- Role: softmax + temperature path (including greedy fast path).
- Notes:
  - this can be expensive if full vocab is included.
  - if offloaded, keep numerical behavior equivalent enough for acceptance decisions.

### K6) `draft_tree_reject_sample`
- Role: multi-phase rejection sampling over tree links.
- Notes:
  - RNG source must be deterministic and controllable (host-fed random coins first).

### K7) `update_verify_result`
- Role: post-accept compaction/scatter, prefix sums, next-step input generation.
- Notes:
  - heavy pointer/index logic; test edge cases (`unfinished_batch_count==0`, variable accept lengths).

## 5) Test/validation strategy (mandatory)
For each migrated kernel:
1. Add CUDA one-shot dump script for real inputs/outputs.
2. Add HLS TB loader for same dump.
3. Compare outputs with strict tolerance and shape checks.
4. Add adversarial cases:
   - min/max batch
   - width transitions
   - zero/one accepted tokens
   - random seeds fixed

Pass criteria: deterministic parity on representative traces before optimization.

## 6) Throughput-focused architecture guidance
- Keep control flow on host; offload only stable kernels first.
- Use HBM as primary weight/state store; use URAM/BRAM for hot tiles and short-lived buffers.
- Use double buffering for stream/gather-heavy kernels.
- Prioritize memory layout stability over aggressive loop unrolling until parity is locked.

## 7) On-chip weight feasibility (your device)
Given resources:
- BRAM: `132 Mb` = `16.5 MB`
- URAM: `541 Mb` = `67.6 MB`
- Total on-chip SRAM: `~84.1 MB`
- HBM: `32 GB`, `820 GB/s peak`

From `gptq_model-4bit-128g-002_weights.csv`:
- `embed_tokens.weight` = `1.0507 GB` (F16)
- `lm_head.weight` = `1.0507 GB` (F16)
- `fc.weight` = `100.66 MB` (F16)
- quantized self-attn group (aggregate) ≈ `61 MB`
- quantized mlp group (aggregate) ≈ `122 MB`

### Conclusion: is “on-chip weights” still possible?
- **Full model on-chip**: No.
- **All major drafter weights on-chip**: No (fc+mlp exceed ~84 MB).
- **Partial on-chip**: Yes.
  - self-attn quantized block (~61 MB) is plausible on-chip if budgeted carefully.
  - norms/maps/small metadata are easy on-chip.
- **Best practical strategy**:
  - keep full weights in HBM,
  - keep currently active tiles + metadata in URAM/BRAM,
  - overlap transfer/compute.

### Implication for design decisions
- Do not architect around “zero weight movement”.
- Architect around **predictable tiled movement from HBM** + **on-chip reuse windows**.

## 8) Execution plan for the expert team
1. Freeze kernel ABI (Python wrapper signatures unchanged).
2. Finish P0 parity for K1-K7 with dump-based tests.
3. Profile per-kernel runtime share on real traces (batch 1/8/32, verify 8/32/64).
4. Optimize top 2 kernels by measured share only.
5. Add replay-input kernels (P2) only after compute kernels are stable.
6. Re-evaluate precision downgrades (FP32->BF16/FP16) after end-to-end acceptance metrics are collected.

## 9) Hard risks to track
- Acceptance-length drift from softmax/sampling numeric mismatch.
- Tree-mask packing mismatch causing silent attention errors.
- Prefix-sum/scatter edge cases under dynamic batch compaction.
- Over-unrolling causing routing/fmax collapse.

## 10) Definition of done
- End-to-end run executes with FPGA backend enabled for P0 kernels.
- Acceptance length and throughput regressions are quantified vs CUDA baseline.
- Kernel-level parity artifacts exist and are reproducible in CI-like scripted flow.
