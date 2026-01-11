# Deep Pipeline + LUT-MAC (TileRT-Inspired, Spatial Execution)

This subpackage explores a structure-based execution model: stream I/O, hard-wired topology, LUT-based MACs for INT4 weights, and template-folded scales (power-of-two). It is self-contained for experimentation and C-sim.

Files
- `deep_pipeline_lutmac.hpp`: Streaming kernel sketch with LUT-MAC, template scale folding, and lightweight host fallbacks for hls::stream/ap_int.
- `deep_pipeline_lutmac_tb.cpp`: CPU testbench that feeds one tile through the stream pipeline, compares against a golden LUT-MAC path, and prints PASS/FAIL.
- `design.md`: Rationale and application notes for spatial deep pipelines.

Quick test (C-sim)
```bash
g++ -std=c++17 -I.. deep_pipeline_lutmac_tb.cpp -o /tmp/dp_lutmac_tb && /tmp/dp_lutmac_tb
# Expect: PASS
```

Notes
- Interfaces are AXIS-style streams for activations/output; weights are read from a packed array (models HBM).
- INT4 weights assumed in range [-8, 7]; scales are powers-of-two via template exponent to enable constant folding (shift-based).
- This is a conceptual scaffold; synthesize in Vitis HLS to map to real AXIS/HBM and refine packing.***
