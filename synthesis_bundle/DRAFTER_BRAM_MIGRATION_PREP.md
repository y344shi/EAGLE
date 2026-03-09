# EAGLE4 Drafter BRAM Migration Prep

Scope: this note covers the active `eagle4_draft` synthesis path rooted at
`cost_draft_tree_fused_wiring_hls.cpp`, with supporting compute in:

- `eagle_tier1_top.cpp`
- `eagle_tier1_lm_top.cpp`
- `eagle4_lm_head_hls.cpp`
- `lm_head_8way_top.cpp`
- `lm_head_engine.cpp`

The goal is not to estimate total FPGA utilization precisely. The goal is to
pin down tensor widths, per-stage transaction sizes, and the explicit on-chip
arrays that matter for BRAM/URAM migration planning.

## 1. Fixed Dimensions

Taken from `tmac_utils.hpp`, `eagle_tier1_top.hpp`, and `eagle4_lm_head_hls.hpp`:

- `VEC_W = 16`
- `TREE_WIDTH = 4`
- `HIDDEN = 4096`
- `INTERMEDIATE = 14336`
- `HEAD_DIM = 128`
- `NUM_HEADS = 32`
- `NUM_KV_HEADS = 32`
- `QKV_INPUT = 8192` (`2 * HIDDEN`)
- `DOWN_OUTPUT = 8192` (`2 * HIDDEN`)
- `kEagle4LmRankMax = 256`
- `kEagle4LmTopKMax = 1024`
- `kCdtFusedMaxBatch = 128`
- `kCdtFusedMaxNodeTopK = 16`
- `kHlsMaxNodeCount = 4032`
- `kMaxDraftDepth = 16`

Assumed scalar sizes:

- `float = 4 B`
- `int64_t = 8 B`
- `int32_t = 4 B`
- `uint16_t = 2 B`
- `vec_t<16> = 16 * float = 64 B`
- `pack512 = 64 B`

## 2. Basic Payload Units

Per token:

- Hidden state (`HIDDEN`): `4096 * 4 B = 16 KiB`
- Concat / down-proj output (`2H`): `8192 * 4 B = 32 KiB`
- FFN intermediate: `14336 * 4 B = 56 KiB`
- One K or V token: `4096 * 4 B = 16 KiB`
- One KV pair: `32 KiB`

Per frontier call at fixed `TREE_WIDTH = 4`:

- Hidden payload: `4 * 16 KiB = 64 KiB`
- `2H` payload: `4 * 32 KiB = 128 KiB`
- FFN intermediate payload: `4 * 56 KiB = 224 KiB`
- New K or V frontier payload: `4 * 16 KiB = 64 KiB`
- New KV pair payload: `128 KiB`

Useful vector counts:

- Hidden vectors per token: `4096 / 16 = 256`
- `2H` vectors per token: `8192 / 16 = 512`
- Intermediate vectors per token: `14336 / 16 = 896`
- KV vectors per token, per K or V: `(32 * 128) / 16 = 256`

## 3. Drafter Transformer Stage Traffic

This is the recurrent SLM path in `eagle_tier1_top_eagle4_l0(...)`.
Numbers below are logical payload sizes for one frontier call with `TREE_WIDTH = 4`.

### Stage 0. Hidden duplication

- Input: hidden stream = `64 KiB`
- Output:
  - hidden norm branch = `64 KiB`
  - residual branch = `64 KiB`

### Stage 1. RMSNorm

- Hidden RMSNorm input/output = `64 KiB`
- Embed RMSNorm input/output = `64 KiB`
- Gamma weights per norm = `4096 * 4 B = 16 KiB`

### Stage 2. Concat `[embed_norm || hidden_norm]`

- Read = `64 KiB + 64 KiB`
- Output `2H` stream = `128 KiB`

### Stage 3. Triplicate for Q/K/V

- Read = `128 KiB`
- Outputs:
  - Q input = `128 KiB`
  - K input = `128 KiB`
  - V input = `128 KiB`

### Stage 4. Q/K/V projections

Each projection consumes `2H = 8192` scalars per token.

Activation traffic per frontier:

- Q projection input `128 KiB`, output `64 KiB`
- K projection input `128 KiB`, output `64 KiB`
- V projection input `128 KiB`, output `64 KiB`

Packed weight and scale traffic per projection:

- Formula:
  - int4 weights = `INPUT_DIM * OUT_DIM / 2` bytes
  - float scales, group size 128 = `(INPUT_DIM / 128) * OUT_DIM * 4` bytes

Current compiled sizes:

- `q_proj`: `8192 x 4096`
  - weights = `16 MiB`
  - scales = `1 MiB`
- `k_proj`: same as `q_proj`
  - weights = `16 MiB`
  - scales = `1 MiB`
- `v_proj`: same as `q_proj`
  - weights = `16 MiB`
  - scales = `1 MiB`

### Stage 5. RoPE on Q/K

- Q input/output = `64 KiB`
- K input/output = `64 KiB`
- RoPE tables:
  - cos table = `(128 / 2) * 4 B = 256 B`
  - sin table = `256 B`

### Stage 6. Contiguous KV write + gather

Per new frontier:

- K write to HBM = `64 KiB`
- V write to HBM = `64 KiB`
- Total write = `128 KiB`

Gather history:

- One query sees `hist_len = prefix_len + current_depth + 1` tokens
- One query KV history payload = `hist_len * 32 KiB`
- All 4 queries logical payload = `hist_len * 128 KiB`

HBM reads are asymmetric:

- Prefix is read once and broadcast:
  - `prefix_len * 32 KiB`
- Ancestors are read per query:
  - `current_depth * 4 * 32 KiB`
- Self tokens are read per query:
  - `4 * 32 KiB`

So total HBM KV read volume per frontier call is:

- `prefix_len * 32 KiB + (current_depth + 1) * 128 KiB`

### Stage 7. Grouped-query attention

Logical external payload to attention core:

- Q input = `64 KiB`
- K history input = `hist_len * 64 KiB`
- V history input = `hist_len * 64 KiB`
- Context output = `64 KiB`

Important on-chip local state inside `fused_online_attention_pwl<128>`:

- `q_buffer[128]` = `512 B`
- `ctx_acc[128]` = `512 B`
- `v_local[128]` per token = `512 B`

These are small. The dominant concern is history bandwidth, not local BRAM.

### Stage 8. Output projection

- Activation input = `64 KiB`
- Activation output = `64 KiB`
- Weights (`4096 x 4096`) = `8 MiB`
- Scales = `512 KiB`

### Stage 9. Residual add + RMSNorm

- Residual add inputs = `64 KiB + 64 KiB`
- Residual output = `64 KiB`
- Duplicated into:
  - norm input = `64 KiB`
  - residual-for-add = `64 KiB`
- Post-attn norm output = `64 KiB`
- Gamma weights = `16 KiB`

### Stage 10. FFN gate/up + SiLU

- Input after duplication:
  - gate path input = `64 KiB`
  - up path input = `64 KiB`
- Gate output = `224 KiB`
- Up output = `224 KiB`
- SiLU-mul output = `224 KiB`

Projection weights:

- `gate_proj`: `4096 x 14336`
  - weights = `28 MiB`
  - scales = `1.75 MiB`
- `up_proj`: same
  - weights = `28 MiB`
  - scales = `1.75 MiB`

### Stage 11. Down projection + split

- Input = `224 KiB`
- Output before split (`2H`) = `128 KiB`
- Split outputs:
  - logits path = `64 KiB`
  - reasoning path = `64 KiB`

Projection weights:

- `down_proj`: `14336 x 8192`
  - weights = `56 MiB`
  - scales = `3.5 MiB`

### Stage 12. Final norm + reasoning residual

- Final norm input/output = `64 KiB`
- Reasoning residual add inputs = `64 KiB + 64 KiB`
- Reasoning output = `64 KiB`
- Gamma weights = `16 KiB`

Outputs consumed by the drafter wrapper:

- `reasoning_state_out` = `TREE_WIDTH * HIDDEN = 64 KiB`
- `logits_norm_out_stream` = `64 KiB`

## 4. Efficient LM-Head Traffic

This is the active path in `eagle_tier1_lm_top_eagle4(...)`.

### 4.1 Local scratch in the wrapper

Compile-time local arrays:

- `logits_hidden[4][4096]` = `64 KiB`
- `low_rank[4][256]` = `4 KiB`
- `candidate_indices[4][1024]` = `16 KiB`
- `candidate_scores[4][1024]` = `16 KiB`
- `gathered_logits[4][1024]` = `16 KiB`
- `topk_tokens[4][1024]` = `16 KiB`
- `topk_probas[4][1024]` = `16 KiB`

Note: runtime `node_top_k` is bounded by `16` in `eagle4_draft_impl`, but the wrapper
still reserves scratch to `kEagle4LmTopKMax = 1024`.

### 4.2 Down projection

`eagle4_lm_down_project(...)`:

- Input hidden = `64 KiB`
- Output low-rank = `TREE_WIDTH * rank * 4 B`
- Weight traffic formula = `rank * HIDDEN * 2` bytes

Example at `rank = 256`:

- low-rank output = `4 * 256 * 4 B = 4 KiB`
- weight traffic = `256 * 4096 * 2 = 2 MiB`

### 4.3 Candidate logits over GPTQ row-major

`eagle4_lm_candidate_logits_row4(...)`:

- Input low-rank = `4 * rank * 4 B`
- Output top-k candidate IDs and scores = `4 * topk * (4 B + 4 B)`

Weight-like traffic formulas:

- qweight = `vocab * rank / 2` bytes
- scales = `(rank / group_size) * vocab * 2` bytes
- qzeros optional = `(rank / group_size) * ceil(vocab / 8) * 4` bytes

Current group size is hard-coded to `128`.

Example at `rank = 256`, `vocab = 32000`:

- qweight = `32000 * 256 / 2 = 3.91 MiB`
- scales = `(256 / 128) * 32000 * 2 = 125 KiB`
- qzeros optional = `(256 / 128) * ceil(32000 / 8) * 4 = 31.25 KiB`

### 4.4 Gather-dot over selected full LM-head rows

`eagle4_lm_gather_dot_fp16(...)`:

- Reads `topk` full-fp16 rows for each of 4 tree tokens
- One candidate row = `4096 * 2 B = 8 KiB`
- Total row traffic = `4 * topk * 8 KiB`

Example at `topk = 16`:

- traffic = `512 KiB`

### 4.5 Candidate softmax

`eagle4_lm_softmax_topk(...)`:

- Input logits = `4 * topk * 4 B`
- Output probs = `4 * topk * 4 B`
- Output token IDs = `4 * topk * 4 B`

At the active drafter bound `topk = node_top_k <= 16`, this stage is small.

## 5. Draft-State BRAM / URAM Staging in `eagle4_draft_impl`

These are explicit on-chip arrays already declared in the top implementation.
They matter more for BRAM migration planning than the stream payload sizes.

### 5.1 BRAM-resident top-level staging

- `parent_indices_accum[16 * 4]` = `0.25 KiB`
- `s_parent_scratch[128 * 16]` = `16 KiB`
- `s_initial_topk_probas[128 * 16]` = `8 KiB`
- `s_initial_topk_tokens[128 * 16]` = `16 KiB`
- `bram_step_input_tokens[128 * 4]` = `4 KiB`
- `bram_step_last_layer_scores[128 * 4]` = `2 KiB`
- `bram_step_topk_indexs_prev[128 * 4]` = `4 KiB`
- `bram_step_topk_probas_sampling[128 * 4 * 16]` = `32 KiB`
- `bram_step_topk_tokens_sampling[128 * 4 * 16]` = `64 KiB`
- `bram_output_scores[128 * 16]` = `8 KiB`
- `bram_output_tokens[128 * 16]` = `16 KiB`
- `bram_work_scores[128 * (128 + 16)]` = `72 KiB`
- `bram_sort_scores[128 * 128]` = `64 KiB`
- `bram_cache_topk_indices[128 * 16]` = `16 KiB`
- `bram_rope_cos_vals[64]` = `0.25 KiB`
- `bram_rope_sin_vals[64]` = `0.25 KiB`
- `bram_step_input_hidden_states[1 * 4 * 4096]` = `64 KiB`
- `bram_output_hidden_states[1 * 16 * 4096]` = `256 KiB`

### 5.2 BRAM inside `e4d_fused_step(...)`

- `s_curr_layer_scores[128 * 64]` = `32 KiB`
- `s_sort_layer_scores[128 * 64]` = `32 KiB`
- `s_sort_layer_indices[128 * 64]` = `64 KiB`
- `s_parent_indices_in_layer[128 * 16]` = `16 KiB`
- `s_remapped_topk_tokens[128 * 64]` = `64 KiB`

### 5.3 BRAM inside `e4d_slm_topk(...)`

- `reasoning_state[4 * 4096]` = `64 KiB`
- `candidate_indices[4 * 16]` = `0.25 KiB`
- `gathered_logits[4 * 16]` = `0.25 KiB`

### 5.4 URAM-resident persistent draft state

- `uram_cumu_tokens[128 * 4032]` = `4032 KiB`
- `uram_cumu_scores[128 * 4032]` = `2016 KiB`
- `uram_cumu_deltas[128 * 4032]` = `4032 KiB`
- `uram_prev_indexs[128 * 4032]` = `4032 KiB`
- `uram_next_indexs[128 * 4032]` = `4032 KiB`
- `uram_side_indexs[128 * 4032]` = `4032 KiB`

Explicit persistent URAM total:

- `22176 KiB` = `21.66 MiB`

Explicit BRAM-resident local arrays counted above:

- about `915.25 KiB`

That total does not include stream FIFOs inserted by HLS.

## 6. What This Means for BRAM Migration

The main observations are:

- The transformer activation payloads themselves are moderate.
  - Hidden frontier = `64 KiB`
  - `2H` frontier = `128 KiB`
  - FFN intermediate frontier = `224 KiB`
- The dominant persistent state is not the transformer core.
  - It is the cumulative draft graph state already mapped to URAM.
- The biggest explicit BRAM blocks in the top are:
  - `bram_output_hidden_states` = `256 KiB`
  - `bram_step_input_hidden_states` = `64 KiB`
  - `bram_work_scores` = `72 KiB`
  - `bram_sort_scores` = `64 KiB`
  - `bram_step_topk_tokens_sampling` = `64 KiB`
  - fused-step scratch = `208 KiB` total across its local score/sort buffers
- The biggest external bandwidth pressure is still projection weights and KV history:
  - Q/K/V each pull `16 MiB` weights per call
  - gate/up each pull `28 MiB`
  - down pulls `56 MiB`
  - KV gather reads grow with `prefix_len` and `current_depth`

## 7. Practical Migration Priorities

If the goal is to move high-value runtime traffic onto BRAM/URAM first, the order should be:

1. Keep the explicit draft-state arrays (`work_scores`, `sort_scores`, hidden recurrence buffers) on BRAM as they already are.
2. Keep cumulative graph state (`cumu_*`, `prev/next/side`) on URAM; this is the correct large-state tier.
3. Treat KV history as a bandwidth problem, not a BRAM-capacity problem, unless prefix windows are aggressively bounded.
4. Do not target full projection weight residency in BRAM.
   - Q/K/V/O/FFN packed weights are much larger than the local staging arrays.
5. If one extra compute-side BRAM optimization is needed, the highest-value candidate is the hidden recurrence path:
   - `bram_step_input_hidden_states`
   - `bram_output_hidden_states`
   - `reasoning_state`

## 8. Assumptions / Caveats

- This note uses compile-time maxima where arrays are statically allocated.
- Runtime `node_top_k` is capped at `16` by `eagle4_draft_impl`, even though some LM-head scratch arrays reserve up to `1024`.
- The `eagle_tier1_top_eagle4_l0(...)` path processes one batch entry at a time through the transformer core; some top-level buffers still reserve for `batch_size <= 128`.
- Stream FIFO depths are not included here because they are mostly implicit HLS implementation choices.
