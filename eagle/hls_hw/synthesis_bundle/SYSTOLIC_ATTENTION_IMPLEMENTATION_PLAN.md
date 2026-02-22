# SystolicAttention HLS Implementation Plan

Source paper: `SystolicAttention: Fused Online Softmax and Matrix Multiplication Design for LLM Acceleration on FPGAs` (MLSys 2025 / arXiv `2507.11331v3`).

## Scope
- Implement a synthesizable decode-time single-head attention kernel that follows the paper's core technique:
  - one-pass online softmax fused with `QK^T` and `P*V`
  - base-2 exponential path with piecewise linear (`exp2`) approximation
- Keep it drop-in compatible with existing stream/vector utilities in this repo (`tmac_utils.hpp`).

## Phase Plan
1. Spec extraction and mapping
- Map paper concepts to this codebase:
  - `Q` stationary in on-chip buffer
  - `K/V` streamed from history
  - online `(m, d)` update state to avoid storing full score/probability matrices
- Done in this iteration.

2. First synthesizable kernel
- Add `fused_online_attention_pwl.hpp` with:
  - AXIS-like stream interfaces
  - online softmax recurrence
  - `exp2` piecewise approximation helper
  - padded length masking support (for decode alignment)
- Done in this iteration.

3. Testable C-simulation
- Add `fused_online_attention_pwl_tb.cpp`:
  - deterministic random vectors
  - software references:
    - exact softmax reference
    - online + same `exp2` PWL reference
  - pass/fail thresholds for:
    - implementation correctness (`HW` vs online-PWL reference)
    - approximation quality (`HW` vs exact softmax)
- Done in this iteration.

4. Integration and synthesis follow-up
- Optional drop-in replacement path in `eagle_tier1_top.cpp` (feature flag).
- Add Vitis HLS TCL and report collection (latency/resource estimates).
- Calibrate PWL segment coefficients with quantized fixed-point sweep from the paper's numeric regime.

## Current Assumptions
- The HTML rendering of the paper does not expose all Eq. (19/20) coefficient values directly.
- This implementation uses a parameterizable 4-segment `exp2` linear approximation on the fractional domain `[0, 1)`, ready for coefficient retuning.

## How To Run C-Sim
```bash
cd hardware/EAGLE/eagle/hls_hw/synthesis_bundle
g++ -std=c++17 -I. fused_online_attention_pwl_tb.cpp -o /tmp/fused_online_attention_pwl_tb
/tmp/fused_online_attention_pwl_tb
```

## PyTorch Cross-Check
Compare the C++ HLS kernel output against PyTorch SDPA/Flash on identical random tensors:

```bash
cd hardware/EAGLE/eagle/hls_hw/synthesis_bundle
python test_vs_pytorch_sdpa.py --cases 20 --backend math --device cuda --dtype float16
```

For true FlashAttention comparison, run on `sm80+` GPUs (A100/H100 class):

```bash
python test_vs_pytorch_sdpa.py --cases 20 --backend flash --device cuda --dtype float16
```
