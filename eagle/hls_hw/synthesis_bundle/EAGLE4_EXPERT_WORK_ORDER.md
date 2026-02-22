# EAGLE4 FPGA Program — Expert Work Order (Dispatch Version)

## 0) Mission
Deliver an FPGA implementation of EAGLE4 speculative decoding kernels with measured evidence of competitive performance versus H100 on the **same workload and decoding settings**.

This is a results contract, not a research note. All milestones are pass/fail.

---

## 1) Scope

### In scope
- CostDraftTree kernel migration and optimization from:
  - `sglang-v0.5.6/python/sglang/srt/speculative/cost_draft_tree_kernel.cu`
- HLS implementation in:
  - `hardware/EAGLE/eagle/hls_hw/synthesis_bundle/`
- End-to-end correctness + performance comparison against GPU baseline.

### Out of scope (for this order)
- Full framework redesign.
- New model architecture training.
- Non-EAGLE4 algorithm variants.

---

## 2) Fixed success criteria (program-level)

All must be met:
1. **Reproducible H100 baseline** on exact workload (same prompts, decode params, batch profile).
2. **Kernel-level parity** for P0 kernel set with dump-based harnesses.
3. **Measured effective byte reduction**:
   - Early gate: >= 2.0x vs initial unfused FPGA backend.
   - Final gate: >= 4.0x target, stretch 4.5–5.0x.
4. **Latency/throughput evidence** against H100 in at least one production-relevant operating point.
5. Full artifact pack delivered (scripts, logs, reports, commit hashes).

If any item fails, project is not marked complete.

---

## 3) Work packages and strict fulfilment targets

## WP0 — Reproducible Benchmark Contract

### Tasks
- Define one canonical benchmark contract:
  - model paths, draft/target pair
  - decoding params (`node_top_k`, `tree_depth`, `verify_num`, temperatures)
  - batch distribution and prompt/output lengths
- Freeze software revisions (`git rev-parse HEAD`) and driver/runtime versions.

### Deliverables
- `reports/benchmark_contract.md`
- `reports/env_manifest.txt` (CUDA, driver, compiler, board, clocks)

### Acceptance (pass/fail)
- Contract can be replayed by another engineer with no manual edits.

---

## WP1 — H100 Baseline (Exact Workload)

### Tasks
- Run baseline on H100 using same decode settings as FPGA target comparison.
- Collect: TTFT p50/p99, ITL p50/p99, tokens/s, avg acceptance length.

### Deliverables
- `reports/h100_baseline_raw.csv`
- `reports/h100_baseline_summary.md`
- run scripts under `scripts/bench_h100_*.sh`

### Acceptance (pass/fail)
- At least 5 independent runs per operating point.
- CoV (tokens/s) <= 5% per point.

---

## WP2 — Current Flow Profiling (Ground Truth)

### Tasks
- Measure kernel time share and bytes/token on current CUDA flow.
- Identify top-3 kernels by wall-time and top-3 by bytes moved.

### Deliverables
- `reports/cuda_kernel_times.csv`
- `reports/cuda_bytes_per_token.csv`
- `reports/hotspots_ranked.md`

### Acceptance (pass/fail)
- Includes both per-kernel time and traffic estimates for the same workload as WP1.
- Top hotspots account for >= 70% of end-to-end latency.

---

## WP3 — P0 Kernel Parity Completion

### P0 kernel set (mandatory)
1. `draft_tree_layer_score_index_gen_op`
2. `update_cumu_draft_state`
3. `calculate_work_deltas`
4. `fused_tree_mask_operator`
5. `convert_logits_to_probas`
6. `draft_tree_reject_sample`
7. `update_verify_result`

### Tasks
- For each kernel: implement HLS version + one-shot CUDA dump + C-sim compare.
- Use deterministic seeds and edge-case vectors.

### Deliverables
- Kernel source + TB + dump script per kernel.
- `reports/p0_parity_matrix.md` with case-by-case pass table.

### Acceptance (pass/fail)
- 100% parity pass on all mandatory test vectors.
- No unresolved nondeterministic mismatch.

---

## WP4 — Fusion + Dataflow Optimization (First Performance Gate)

### Tasks
- Fuse highest-share kernels first (from WP2 evidence).
- Introduce persistent/streaming dataflow to eliminate intermediate materialization.
- Keep host control plane minimal.

### Deliverables
- `reports/fusion_design_v1.md`
- before/after traffic and timing reports.

### Acceptance (pass/fail)
- **Early gate:** effective bytes/token reduction >= 2.0x.
- If < 2.0x, mandatory redesign review within 48 hours.

---

## WP5 — HLS Synthesis Closure + Counter Instrumentation

### Tasks
- Add HLS build scripts/TCL for reproducible synth runs.
- Capture latency/resource/Fmax for critical kernels.
- Add hardware counters for bytes/cycle and stall classification.

### Deliverables
- `hls/` scripts + `reports/synth/*.rpt`
- `reports/hw_counters.csv`

### Acceptance (pass/fail)
- Synth reports generated automatically from one command.
- Counters map to budget model in `EAGLE4_FPGA_PERF_AND_H100_STRATEGY.md`.

---

## WP6 — H100 Competition Gate (Final Performance Gate)

### Tasks
- Re-run benchmark contract with FPGA backend enabled.
- Compare directly against WP1 metrics.

### Deliverables
- `reports/fpga_vs_h100_summary.md`
- `reports/fpga_vs_h100_raw.csv`

### Acceptance (pass/fail)
- At least one target operating point meets one of:
  - better ITL p99 than H100, or
  - better tokens/s at same quality constraints.
- Effective byte reduction target:
  - minimum >= 4.0x,
  - stretch >= 4.5–5.0x.

If target is not met, deliver a quantified gap report and next-iteration plan with top-2 bottlenecks.

---

## WP7 — Handover Pack

### Deliverables
- Code + scripts + docs + runbooks + exact commit hashes.
- One-page operator guide: how to run baseline, profiling, parity, and competition tests.

### Acceptance (pass/fail)
- Fresh machine replay succeeds with documented commands only.

---

## 4) Mandatory artifact structure

Create:
- `reports/`
  - `benchmark_contract.md`
  - `env_manifest.txt`
  - `h100_baseline_*`
  - `cuda_*`
  - `p0_parity_matrix.md`
  - `fusion_design_v1.md`
  - `synth/*.rpt`
  - `fpga_vs_h100_*`
- `scripts/`
  - benchmark, profiling, parity, synth orchestration scripts.

No milestone is accepted without artifacts committed.

---

## 5) Quality guardrails

- No metric claims without raw CSV attached.
- No “improved” claim without before/after numbers on same contract.
- Any approximation change must include acceptance-length impact report.
- Any missing parity in P0 blocks performance sign-off.

---

## 6) Program timeline (recommended)

- Week 1: WP0, WP1, WP2 complete.
- Week 2: WP3 complete.
- Week 3: WP4 + WP5 early reports.
- Week 4: WP6 gate + WP7 handover.

Escalation rule: if WP4 early gate (<2x bytes/token) fails, freeze feature work and redesign immediately.

---

## 7) Dispatch summary (for assignment)

Assign an expert owner with authority over:
- kernel implementation,
- memory/dataflow architecture,
- benchmarking and evidence generation.

Completion definition: all WP acceptance checks pass with artifacts.
