# Deep Pipeline + LUT-MAC (Spatial, Broadcast for Dense)

This design turns a generic MAC into a model-shaped, stream-through layer: AXIS in/out, packed INT4 weights, LUT-based multiply (no DSP), and template-folded power-of-two scales. It now supports a true dense projection by broadcasting one input scalar to 128 weights and accumulating across the input dimension.

## Highlights
- Streamed activations/output (`hls::stream<vec_t<16>>`); weights on AXI-MM (HBM) packed as one `pack512` per input scalar (128 INT4 weights per packet).
- LUT-MAC: replace DSP mult with precompute+select; `SCALE_EXP` template folds scaling into shifts.
- Broadcast matvec: for each input scalar, fetch its packet, broadcast to 128 outputs, accumulate into stationary accumulators.
- Interleaved accumulators: 4 banks rotate updates to hide FP add latency and sustain II=1.
- Host/synth dual-path: fallbacks for `hls_stream`, `ap_uint`, `vec_t` allow C-sim without HLS.

## Dataflow
```mermaid
flowchart LR
    subgraph Host (Prepare)
        Q[Quantize weights to INT4]
        S[Snap scales to pow2 => SCALE_EXP]
        P[Pack weights: one packet per input scalar,\n128 nybbles each]
    end
    subgraph Kernel (Run)
        A[Read input vector chunk\n(vec16)]
        R[For each scalar in chunk:\nload packet, LUT-MAC broadcast\naccumulate 128 outputs]
        O[Write output vec16 stream\n(8 chunks for 128 outputs)]
    end
    Q --> S --> P --> R
    A --> R --> O
```

## Weight Layout (Broadcast)
- `weights[k]` holds 128 signed 4-bit weights: connections from input scalar `k` to outputs 0..127.
- Input vector arrives as `vec_t<16>` chunks; iterate over input_dim/16 chunks and packets accordingly.

## Host Packing Sketch (Python)
```python
def pack_weights_broadcast(w_mat):  # shape: input_dim x 128, int4 in [-8,7]
    pkts = []
    for k in range(w_mat.shape[0]):
        nyb = (w_mat[k].clip(-8,7).astype(np.int8) & 0xF).astype(np.uint8)
        pkt = np.zeros(64, dtype=np.uint8)
        for j in range(0, 128, 2):
            pkt[j//2] = nyb[j] | (nyb[j+1] << 4)
        pkts.append(pkt)  # 64 bytes per packet
    return pkts  # write contiguously to HBM
```

## Test (C-sim)
```bash
g++ -std=c++17 -I.. deep_pipeline_lutmac_tb.cpp -o /tmp/dp_lutmac_tb && /tmp/dp_lutmac_tb
# Expect: PASS
```

## HLS Notes
- Top: `dense_projection_layer<SCALE_EXP, INPUT_DIM, OUT_W=128>` with AXIS for activations/output, AXI-MM for weights.
- Keep `SCALE_EXP` templated to force shift-only scaling; INT4 weights assumed; `INPUT_DIM` multiple of 16 required.
- For larger dimensions: read multiple input chunks and iterate packets accordingly; size streams/FIFOs to cover pipeline latency if chained.***
