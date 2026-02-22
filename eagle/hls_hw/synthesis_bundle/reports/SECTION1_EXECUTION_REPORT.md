# Section 1 Execution Report (Plan v1.0)

Date: 2026-02-16
Location: `hardware/EAGLE/eagle/hls_hw/synthesis_bundle`

## Scope executed
- 1.1 QDMA/HBM transfer measurements (executed with available hardware)
- 1.2 LM Head B FP16 microbenchmark + stage breakdown
- 1.3 End-to-end FPGA token generation check attempt

## 1.1 QDMA and HBM Bandwidth

### Environment status
- FPGA QDMA device nodes: **not present** (`/dev/qdma*`, `xocl`, `xclmgmt` not found)
- Proxy measurement used: host -> CUDA device transfer on available GPU (V100 PCIe)

### Measured table (requested 4 transfer points)
Source: `reports/section1_qdma_hbm_proxy.csv`

| Transfer | Mean latency (us) | Std (us) | Effective GB/s |
|---|---:|---:|---:|
| 4KB | 7.882 | 1.240 | 0.520 |
| 64KB | 26.231 | 1.078 | 2.498 |
| 1MB | 323.008 | 1.127 | 3.246 |
| 16MB | 5055.380 | 5.333 | 3.319 |

Notes:
- This is **not FPGA QDMA**. It is a host->GPU PCIe proxy, useful only as interim transport baseline.

## 1.2 LM Head B Microbenchmark (FP16 baseline)

Source: `reports/section1_lm_head_benchmark.csv`

### Mandatory B-only benchmark (128 x 128k projection)
- B-only latency: **51.960 us** (std 0.658 us)
- HBM read bytes/token (B stage): **32,833,792 bytes**
- Effective bandwidth: **631.910 GB/s**

### Stage latency breakdown (proxy decode path)

| Stage | Mean latency (us) |
|---|---:|
| Embedding | 15.949 |
| Transformer layers (proxy) | 494.450 |
| LM head A | 33.291 |
| LM head B | 54.079 |
| LM head C | 86.146 |
| Top-K | 14.362 |
| Total | 698.278 |

- LM head B share of total: **7.74%**

Notes:
- Breakdown is a controlled GPU proxy benchmark, not FPGA kernel timing.
- It is valid for bottleneck triage and relative stage sizing before FPGA integration.

## 1.3 End-to-End Token Generation Verification

### Result: **Blocked (not completed)**
- Attempted harness: `./run_eagle_sim`
- Current output: `Input tensor missing or too small; got 0`
- Blocker: required reference tensors under `../eagle_verified_pipeline_4bit/...` are missing in current workspace.

### Required to unblock
1. Provide reference tensor package expected by `test_eagle_top.cpp` / `run_eagle_sim`.
2. Re-run harness and capture:
   - FPGA output token(s)
   - CPU reference token(s)
   - mismatch count / max diff

## Artifacts generated
- `scripts/measure_host_to_hbm_proxy.py`
- `scripts/benchmark_lm_head_b_fp16.py`
- `reports/section1_qdma_hbm_proxy.csv`
- `reports/section1_lm_head_benchmark.csv`
- `reports/SECTION1_EXECUTION_REPORT.md`

## Immediate next action (strict)
- Provide missing `eagle_verified_pipeline_4bit` tensor bundle, then execute 1.3 to close Section 1.
