# EAGLE4 FPGA Performance Budget + H100 Strategy

This document is the execution guide for driving the FPGA implementation toward H100-class performance on the **specific EAGLE4 deployment workload**.

## 1) Performance target must be explicit

Do not use a vague target like "faster than H100". Define at least one of:
- **TTFT** (time-to-first-token)
- **ITL** (inter-token latency, p50/p99)
- **Throughput** (tokens/s) at fixed `(batch, prompt_len, output_len)`
- **Energy/token**

For the same scenario, fix:
- model pair (target + draft checkpoint)
- decoding params (`node_top_k`, `tree_depth`, `verify_num`, temperature)
- batch distribution over time

## 2) Reality check vs H100 (roofline-level)

Given hardware:
- FPGA HBM peak: `820 GB/s`
- H100 HBM peak (typical SXM): `~3.35 TB/s`

Bandwidth ratio:
- `820 / 3350 ~= 0.245`

Interpretation:
- If workload is memory-bound and algorithmically identical, FPGA is unlikely to beat H100.
- To beat H100 latency/throughput, FPGA must deliver a large **algorithmic byte reduction** and/or remove major software overhead.

Required byte reduction factor for parity:
- `R >= 3350 / 820 ~= 4.1x`

For 20% better than H100:
- `R >= 1.2 * 3350 / 820 ~= 4.9x`

Therefore, winning requires more than kernel translation; it requires **system-level specialization**.

## 3) Cycle budget template (per kernel)

Use this template for each migrated kernel.

## 3.1 Metrics to log
- `B_in` / `B_out` bytes from/to HBM
- `B_onchip` bytes BRAM/URAM traffic
- `N_ops` arithmetic ops (or MACs)
- `II` achieved in HLS report
- `fclk` achieved MHz
- `C_total` cycles
- `T_us = C_total / fclk_MHz`

## 3.2 Budget formulas
- `C_mem_in  = ceil(B_in  / BW_in_bytes_per_cycle)`
- `C_mem_out = ceil(B_out / BW_out_bytes_per_cycle)`
- `C_comp    = ceil(N_ops / ops_per_cycle)`
- `C_total   = max(C_mem_in + C_mem_out, C_comp) + C_ctrl`
- `T_us      = C_total / fclk_MHz`

Where `BW_*` are **effective** bandwidths (not peak), measured from hardware counters.

## 3.3 Kernel budget sheet (fill this)

| Kernel | B_in | B_out | N_ops | II/fclk | C_total | T_us | Notes |
|---|---:|---:|---:|---|---:|---:|---|
| `draft_tree_layer_score_index_gen_op` |  |  |  |  |  |  | hidden gather dominated |
| `update_cumu_draft_state` |  |  |  |  |  |  | state+mask bookkeeping |
| `calculate_work_deltas` |  |  |  |  |  |  | merge/reduction |
| `fused_tree_mask_operator` |  |  |  |  |  |  | bit-pack + index fill |
| `convert_logits_to_probas` |  |  |  |  |  |  | vocab softmax path |
| `draft_tree_reject_sample` |  |  |  |  |  |  | RNG + tree traversal |
| `update_verify_result` |  |  |  |  |  |  | compaction/scatter |

## 3.4 Example (already measured-style estimate)
For `draft_tree_layer_score_index_gen_op` with `B=1, tree_width=8, topk=8, H=4096, fp16 hidden`:
- hidden input ~`64 KB`, hidden output ~`64 KB`
- plus small score/index tensors
- arithmetic is small; memory dominates

This kernel can be very fast on FPGA **if** hidden gather stays on-chip between adjacent stages.

## 4) Single-model deployment advantages (how to cash them in)

This is your strongest differentiator. Exploit all of these:

### A) Compile-time constant specialization
- Fix and hardwire common values (`node_top_k`, max `tree_width`, max `verify_num`, `hidden_size`).
- Remove all generic branches from datapath.
- Pre-size all buffers statically; no dynamic allocation.

### B) Precomputed tables
- Precompute and store:
  - rope/position coefficient tables
  - mask templates by `(tree_width, input_count)`
  - index remap helpers for tree parent/child/side transitions
  - fast softmax LUT/PWL tables (with acceptance-quality validation)

### C) Weight prepacking/offline transforms
- Reorder and pack weights exactly for your systolic/dataflow order.
- Precompute dequant helper terms where possible (`scale`, `zero` folds).
- Align memory layout to HBM bank-stripe strategy used by kernel readers.

### D) Cross-kernel fusion (critical)
- Avoid round-trips between kernels:
  - score -> topk -> parent gather should stream, not materialize repeatedly in HBM.
  - state updates should consume upstream streams directly where possible.
- Build a persistent dataflow region instead of many short host-launched kernels.

### E) Runtime determinism
- Fixed micro-batch envelopes and static command queues.
- Deterministic RNG path (host-fed or hardware PRNG with fixed seed behavior).
- Eliminate framework-level launch overhead from critical loop.

## 5) On-chip weights: what is and is not possible

From current weight inventory (`gptq_model-4bit-128g-002_weights.csv`):
- `embed_tokens.weight` ~`1.05 GB`
- `lm_head.weight` ~`1.05 GB`
- `fc.weight` ~`100.66 MB`
- quantized self-attn block aggregate ~`61 MB`
- quantized mlp block aggregate ~`122 MB`

On-chip SRAM available:
- BRAM `16.5 MB` + URAM `67.6 MB` = `~84.1 MB`

Conclusion:
- Full on-chip model: **No**.
- Full heavy-path on-chip: **No**.
- Strategic partial on-chip: **Yes**.
  - self-attn quantized block is plausible if tightly budgeted.
  - small metadata/tables/state should be on-chip.

Best architecture:
- HBM as source of truth.
- URAM/BRAM as rolling tile cache + state scratchpad.
- Double-buffered prefetch/compute/evict pipeline.

## 6) Concrete ways to beat a GPU in this workload

You can win if you do several together:

1. **Algorithmic byte reduction (must-have)**
- Reduce effective bytes/token by ~`4-5x` vs naive GPU path.
- Main lever: avoid full-vocab heavy paths in drafter and reduce repeated state traffic.

2. **Aggressive fusion and persistent scheduling**
- Convert multi-kernel launch chain into one or few persistent pipelines.
- Remove Python/CUDA launch overhead in tight loops.

3. **Deployment-specialized acceptance-quality tuning**
- Use controlled approximations (LUT softmax, quantized accumulators) only where acceptance metrics stay stable.
- Validate against acceptance length and final token equivalence metrics.

4. **HBM bank-aware layout + deterministic DMA**
- Bank-strip weight tiles and hot state to minimize bank conflicts.
- Build fixed burst schedules.

5. **Mixed precision where mathematically safe**
- Keep critical control scores stable (float32 where needed).
- Use lower precision for hidden transport and non-critical intermediate paths.

## 7) Execution milestones with kill criteria

### M0: Baseline and target definition
- Measure H100 for exact production scenario.
- Lock KPI target values.

### M1: P0 kernel parity on FPGA backend
- Full correctness parity (trace-level) on representative sets.
- Kill criterion: any unresolved nondeterministic mismatch.

### M2: Memory-traffic minimization pass
- Add fusion and stream chaining.
- Quantify bytes/token drop from counters.
- Kill criterion: <`2x` byte reduction after fusion pass.

### M3: H100 gap closure
- Compare TTFT/ITL/tokens-s vs H100 baseline.
- If behind, prioritize top 2 kernels by wall time and re-architect; do not micro-opt low-share kernels.

### M4: Production hardening
- Robustness under variable batch/sequence mix.
- p99 latency and long-run stability.

## 8) What to do next (immediate)
1. Create a shared perf sheet from the table in §3 and fill it with current CUDA numbers.
2. Add hardware counters in HLS wrappers for bytes/cycle and stall reasons.
3. Start with P0 kernels from `EAGLE4_FPGA_MIGRATION_PLAYBOOK.md` and track bytes/token after each fusion step.
4. Revisit `lm_head` path early; it is a dominant bandwidth risk.

---

If desired, add a companion script to auto-compute `C_total` from measured counters and produce per-kernel roofline plots.
