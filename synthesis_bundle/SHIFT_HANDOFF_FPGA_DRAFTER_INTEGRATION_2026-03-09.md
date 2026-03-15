# SHIFT HANDOFF: EAGLE4 Drafter FPGA Integration (2026-03-09)

## 1) Purpose
This document is a zero-memory handoff for continuing the EAGLE4 drafter offload work.  
If a future engineer/agent starts from scratch, this file should be enough to re-implement and validate the current shift goals.

## 2) Shift Goal (must preserve)
Replace the effective CostDraftTree drafter loop path with FPGA execution (device-memory handoff via CUDA memcpy / extension bridge), while preserving runtime behavior against SGLang golden captures.

Required behavior target:
- Input captures from real SGLang runtime, including mid/low/high compute-intensity profiles.
- Final candidate behavior parity across profiles.
- Intermediate checkpoint parity across profiles and against captured goldens.
- End-to-end draft->verify contract compatibility in SGLang.

## 3) Core Decisions Made In This Shift
- Focus scope: `bs=1` transition path first.
- Use fixed tree-policy mode first (no per-depth dynamic policy arrays in transition entry).
- Keep full intermediate outputs/checkpoints exposed (do not collapse outputs to only final token).
- Keep rerank path included in FPGA-side drafter flow; do not split rerank back to host/GPU for this integration target.

## 4) Current Local Code Additions
Added a new C++ entry point for transition-mode integration:
- Declaration: `cost_draft_tree_fused_wiring_hls.hpp` -> `eagle4_draft_bs1_fixed(...)`
- Definition: `cost_draft_tree_fused_wiring_hls.cpp` -> `eagle4_draft_bs1_fixed(...)`

Associated fixed transition constants (in header):
- `kE4dTransitionBatchSize = 1`
- `kE4dTransitionNodeTopK = 8`
- `kE4dTransitionHiddenSize = HIDDEN`
- `kE4dTransitionMaxTreeWidth = TREE_WIDTH`
- `kE4dTransitionMaxNodeCount = kHlsMaxNodeCount`
- `kE4dTransitionMaxVerifyNum = kCdtFusedMaxBatch`

Transition function behavior:
- Internally clamps `fixed_tree_width` and `fixed_verify_num`.
- Disables scheduled policy arrays (`use_policy_schedule=false`, null policy pointers).
- Calls `eagle4_draft_impl(...)` with fixed bs/topk/hidden/capacities.
- Returns `io_cumu_count`, optional `final_tree_width`, optional `final_verify_num`.
- Preserves all state/debug output buffers needed for checkpoint validation.

Compile check already done:
- `g++ -std=c++17 -O2 -I. -c cost_draft_tree_fused_wiring_hls.cpp -o /tmp/cost_draft_tree_fused_wiring_hls.o`

## 5) Source-of-Truth Entry Points

### 5.1 HLS side (this repo)
- Full draft top: `cost_draft_tree_fused_wiring_hls.cpp` -> `eagle4_draft(...)`
- New transition top (bs1 fixed): `cost_draft_tree_fused_wiring_hls.cpp` -> `eagle4_draft_bs1_fixed(...)`
- Tier1 layer0 SLM path: `eagle_tier1_top.cpp` -> `eagle_tier1_top_eagle4_l0(...)`
- Tier1+LM path: `eagle_tier1_lm_top.cpp` -> `eagle_tier1_lm_top_eagle4(...)`

### 5.2 SGLang side (external repo)
- Model path file:  
  `/home/y344shi/workspace/amdv80/eagle4_hardware_adaptation/sglang-v0.5.6/python/sglang/srt/models/llama_eagle4.py`
- Draft/verify orchestration:  
  `/home/y344shi/workspace/amdv80/eagle4_hardware_adaptation/sglang-v0.5.6/python/sglang/srt/speculative/cost_draft_tree_main.py`

## 6) Critical SGLang Contract To Preserve
In `cost_draft_tree_main.py`:
- `draft_Finalization()` writes:
  - `batch.spec_info.custom_draft_attention = False`
  - `batch.spec_info.func_info = (cumu_tokens, cumu_scores, cumu_deltas, prev_indexs, next_indexs, side_indexs, cumu_count)`
- `verify_Initialization_0()` reads exactly the above tuple order.
- Verify init also requires these pre-populated fields on `batch.spec_info`:
  - `seq_lens_int32_cpu`
  - `seq_lens_int64_cpu`
  - `seq_lens_int32_gpu`

Integration implication:
- Any FPGA transition path must return tensors in the same semantic shapes/order as this tuple.
- Must keep `seq_lens_*` fields valid before verify phase.

## 7) Golden Capture And E2E Requirements

### 7.1 Required true E2E artifacts per profile
- `cost_draft_tree_draft_e2e_case.txt`
- `cost_draft_tree_draft_e2e_prefix_k_layer0.fp16.bin`
- `cost_draft_tree_draft_e2e_prefix_v_layer0.fp16.bin`

Expected profile directories:
- mid/default: `<case_dir>/`
- low: `<case_dir>/feature_low/`
- high: `<case_dir>/feature_high/`

### 7.2 Required profile consistency checks
Across mid/low/high, must satisfy:
- `expected_output_tokens` exact match
- `recurrent_topk_tokens` exact match
- `expected_output_scores` max abs delta <= `1e-4`

Implemented in:
- `run_cost_draft_tree_e2e_flow.sh`
- `check_cost_draft_tree_goldens.sh --require-e2e`

### 7.3 Stage-checkpoint parity requirement
`capture_all_goldens.sh` verifies required stage tensors across feature profiles.
Do not declare success unless stage checkpoints exist and compare cleanly.
Key stage tensor families include:
- Core: `tensor_006`, `tensor_007`, `tensor_011`, `tensor_014`, `tensor_110`
- Full stage set when `REQUIRE_STAGE_GOLDENS=1`: `tensor_101..124`, `tensor_130..134`

## 8) How To Re-run Full Capture On H100
Run from:
- `/home/y344shi/workspace/amdv80/eagle4_hardware_adaptation/eagle-project/eagle4`

Recommended command pattern:
```bash
bash capture_all_goldens.sh \
  --sglang-root /home/y344shi/workspace/amdv80/eagle4_hardware_adaptation \
  --synthesis-bundle-dir /home/y344shi/workspace/amdv80/hls_eagle4_synthesis/EAGLE/synthesis_bundle \
  --capture-draft-e2e-dir /home/y344shi/workspace/amdv80/eagle4_hardware_adaptation/eagle-project/eagle4/capture/cases \
  --capture-draft-e2e-source eagle4_classic \
  --compute-intensity-levels low_intensity,undefined,high_intensity \
  --draft-tree-width 8 \
  --verify-tokens-num 64 \
  --max-running-requests 1 \
  --max-requests-num 1 \
  --max-new-tokens 1
```

Notes:
- If profile sweep is disabled, you lose low/high parity coverage.
- If `--skip-capture` is used, runtime stage tensor parity is not guaranteed.

## 9) How To Re-run HLS-side E2E Validation
Run from this repo:
- `/home/y344shi/workspace/amdv80/hls_eagle4_synthesis/EAGLE/synthesis_bundle`

Primary flow:
```bash
bash run_cost_draft_tree_e2e_flow.sh \
  --case-dir /path/to/cases \
  --feature-profiles mid,low,high \
  --require-e2e \
  --strict
```

Local smoke + optional E2E:
```bash
bash run_smoke_suite.sh --with-cdt-e2e --cdt-case-dir /path/to/cases
```

## 10) Reimplementation Plan (from zero)

1. Re-capture runtime goldens on H100 with feature sweep enabled.
2. Ensure true E2E sidecars exist for mid/low/high profiles.
3. Regenerate CostDraftTree case files from runtime E2E baseline:
   - `dump_all_cost_draft_tree_cases.sh`
   - `dump_cost_draft_tree_multilayer_orchestrator_case.py --e2e-case ...`
4. Run strict case checks:
   - `check_cost_draft_tree_goldens.sh --require-e2e <profile_case_dir>`
5. Keep/verify FPGA entrypoint contract:
   - preserve `func_info` tuple semantics
   - preserve `seq_lens_*` spec_info fields
6. Implement SGLang bridge (CUDA extension / torch op):
   - input pointers from SGLang tensors
   - call FPGA entry (`eagle4_draft_bs1_fixed` first)
   - write output tensors back in-place
7. Wire draft path in SGLang to choose FPGA bridge for bs1 fixed-policy mode.
8. Verify draft->verify flow end-to-end in live run and in case-based replay.

## 11) Required Data Mapping For Bridge
Minimum mapping from SGLang runtime buffers to HLS entry:
- Recurrent step buffers:
  - `step_input_tokens`
  - `step_input_hidden_states`
  - `step_last_layer_scores`
  - `step_topk_indexs_prev`
  - `step_topk_probas_sampling`
  - `step_topk_tokens_sampling`
- Initial loop inputs (if enabled):
  - `initial_hidden_states`
  - `initial_topk_probas` / `initial_topk_tokens` or `initial_logits`
- Persistent legacy outputs required by verify:
  - `cumu_tokens`, `cumu_scores`, `cumu_deltas`, `prev_indexs`, `next_indexs`, `side_indexs`, `cumu_count`
- Optional but required for checkpoint debugging:
  - debug score/sort/parent/remap buffers

## 12) Definition of Done For This Workstream
All items below must pass:
- Full runtime capture succeeds on H100 for mid/low/high.
- Required E2E case and prefix sidecars exist for all profiles.
- Cross-profile final candidate checks pass (`tokens`, `recurrent_topk_tokens`, score delta threshold).
- Stage checkpoint files are present and verified across profiles.
- HLS case TB checks pass in strict mode.
- SGLang draft->verify runs via FPGA path without breaking `func_info` contract.

## 13) Known Remaining Gaps
- The new `eagle4_draft_bs1_fixed` entry exists in C++ but is not yet wired into a live CUDA/torch bridge call path.
- Dynamic policy mode is intentionally bypassed in this transition entry; this is expected for the current bs1 fixed-policy milestone.
- Final integration still needs explicit host-side memory bridge and runtime dispatch in SGLang.

## 14) Fast Restart Checklist
- Verify paths exist:
  - SGLang root
  - synthesis_bundle
  - capture case roots
- Re-run capture on H100.
- Re-run `run_cost_draft_tree_e2e_flow.sh --require-e2e --strict`.
- Confirm `eagle4_draft_bs1_fixed` still compiles.
- Start bridge implementation from Section 11 mapping.
