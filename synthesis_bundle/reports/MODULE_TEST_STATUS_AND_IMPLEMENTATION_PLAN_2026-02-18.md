# EAGLE4 HLS Module Test Status and Implementation Plan

Date: 2026-02-18  
Owner: HLS adaptation track  
Scope: CostDraftTree modules, attention kernels, and Tier1 end-to-end harness

## 1) Executive Summary

- CostDraftTree module-level verification is in good shape: all file-driven case tests pass against captured vectors.
- Smoke regression across synthesis bundle passes.
- Full Tier1 harness (`test_eagle_top`, non-smoke) runs to completion, but has a large numerical mismatch vs golden output.
- Primary current blocker is functional parity at full Tier1 integration, not missing infrastructure.

## 2) What Was Tested (Latest Run)

Commands executed from `hardware/EAGLE/eagle/hls_hw/synthesis_bundle` and `capture/`:

- `./check_cost_draft_tree_goldens.sh /home/y344shi/workspace/eagle4_adaptation/capture`
- File-driven TB replay:
  - `cost_draft_tree_score_tb` (normal + hot-token case)
  - `cost_draft_tree_update_tb --case-file ...`
  - `cost_draft_tree_controller_tb --case-file ...`
  - `cost_draft_tree_fused_wiring_tb --case-file ...`
- `./run_smoke_suite.sh`
- Full non-smoke run:
  - `g++ -std=c++17 -I. test_eagle_top.cpp eagle_tier1_top.cpp -o /tmp/test_eagle_top_full`
  - `/tmp/test_eagle_top_full`

## 3) Module Status Matrix

| Module | Test Type | Status | Evidence | Open Issues |
|---|---|---|---|---|
| `cost_draft_tree_score_hls.hpp` | file-driven golden replay | PASS | zero diffs, zero index/token mismatches for both `score_case` and `score_case_hot` | none blocking |
| `cost_draft_tree_update_hls.hpp` | file-driven golden replay | PASS | all buffer mismatches = 0; float diff max `5.96046e-08` | none blocking |
| `cost_draft_tree_controller_hls.hpp` | file-driven case replay | PASS | controller case-file check pass | none blocking |
| `cost_draft_tree_fused_wiring_hls.hpp` | file-driven case replay | PASS | hidden/scores/tokens/KV indices/masks all matched | none blocking |
| `dump_all_cost_draft_tree_cases.sh` + checkers | generation + self-check | PASS | expected names/order verified, dry-run parse clean | none blocking |
| `fused_online_attention_pwl.hpp` | C-sim smoke | PASS | `max|HW - online(PWL)| = 7e-8` | still a smoke-only gate today |
| `deep_pipeline_lutmac.hpp` | C-sim smoke | PASS | `[smoke_scaled] PASS (T-MAC vs DSP match)` | still a smoke-only gate today |
| `eagle_tier1_top.cpp` + `test_eagle_top.cpp --smoke` | end-to-end smoke | PASS | finite, non-degenerate output for default and KV8 profile | smoke only; not golden-parity proof |
| `eagle_tier1_top.cpp` + `test_eagle_top.cpp` (non-smoke) | full golden harness | RUNS, **NOT MATCHING** | completed with `Max diff vs golden: 51.1367` and large local deviations | major parity blocker |

## 4) Golden Asset Status

Core required files for Tier1 harness are present under `capture/` (weights, scales, layernorms, core tensors).  
CostDraftTree case files are complete and ordered (5/5 present).

Known capture inconsistency:

- `input_norm1.fp16.bin` missing
- `input_norm2.fp16.bin` missing

Current harness behavior:

- Fallback is implemented in `test_eagle_top.cpp` to use:
  - `input_layernorm.fp16.bin` when `input_norm1` is absent
  - `post_attention_layernorm.fp16.bin` when `input_norm2` is absent
- This prevents segfault and keeps verification runnable.

Additional note:

- LM head check is currently skipped due hardcoded `VOCAB=73448` in harness while captured tensors indicate vocab 128256:
  - `tensor_005_BASE_Logits.bin` count = 128256 FP16
  - `embed_tokens.fp16.bin` implies `128256 x 4096`

## 5) Current Technical Gaps

### P0 (blocking full parity)

1. Full Tier1 numerical mismatch remains large (`Max diff vs golden: 51.1367`).
2. Harness does not fail CI on golden mismatch (it reports diff but returns success).
3. Vocab constant mismatch disables LM head parity check (`73448` vs `128256`).

### P1 (stability + correctness hardening)

1. Norm file naming mismatch in capture (`input_norm1/2` absent).
2. Need stage-level parity checkpoints to localize Tier1 drift (Q/K/V, RoPE, attention output, residual, MLP output).

### P2 (integration roadmap)

1. Integrate candidate-tree controller path into larger EAGLE4 multi-candidate drafting flow.
2. Wire parent-only visibility masks/lists directly into KV-attention path in a dataflow-safe way.
3. Add cycle/resource reporting gates after correctness lock.

## 6) Implementation Plan for Next Sprint

### Phase A: Make Tier1 verification strict (1-2 days)

1. Add explicit pass/fail thresholds in `test_eagle_top.cpp` for non-smoke mode.
2. Exit non-zero when max diff exceeds threshold.
3. Parameterize thresholds by stage and final output.

Deliverable:

- deterministic red/green Tier1 golden gate for CI and local runs.

### Phase B: Close parity gap (2-4 days)

1. Update harness constants from capture metadata (including vocab = 128256).
2. Add per-stage compare dumps and checks:
   - norm1
   - q/k/v projection
   - RoPE outputs
   - attention context
   - post-attn residual
   - MLP output
3. Capture any missing stage goldens required for localization.

Deliverable:

- full Tier1 parity within agreed tolerance.

### Phase C: Multi-candidate tree integration (3-5 days)

1. Connect `cost_draft_tree_fused_step_hls` outputs into next-stage attention scheduling inputs.
2. Enforce parent-only KV visibility contract through controller-generated indices/masks.
3. Keep dataflow boundaries explicit to preserve II targets and avoid stream deadlock.

Deliverable:

- integrated candidate-tree drafting control path with deterministic TB coverage.

## 7) Decisions Needed in Group Meeting

1. Confirm parity acceptance thresholds (stage-level and final output).
2. Decide whether to regenerate `input_norm1/2` in capture pipeline or keep fallback as permanent compatibility path.
3. Confirm target vocab and model metadata contract for all harnesses (currently 128256 in captured assets).
4. Prioritize schedule split between parity closure and performance tuning (no perf claims before parity lock).

## 8) Immediate Action Items

1. Convert non-smoke Tier1 harness into hard fail on mismatch.
2. Align vocab/config constants with captured model artifacts.
3. Add a stage-by-stage diff report file artifact per run.
4. Start parity root-cause session focused on attention/residual transition where large outlier appears.
