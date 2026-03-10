# CostDraftTree 32-PC Packed HBM Layout

This document describes the new packed entrypoint:

- `eagle4_draft_packed_32pc(...)`
- Declared in `cost_draft_tree_hbm_packed.hpp`
- Defined in `cost_draft_tree_hbm_packed.cpp`

The packed entrypoint keeps the existing compute core (`eagle4_draft_impl`) unchanged and remaps
all large runtime buffers onto 32 HBM pseudo-channel base pointers (`pc00..pc31`) with
offset-based addressing.

## Port Assignment (Current Revision)

- `pc00..pc06`: Q/K/V/O/Gate/Up/Down weights + scales
- `pc07`: norm gamma tensors + rope config table
- `pc08`: KV cache `hbm_k` base
- `pc09`: KV cache `hbm_v` base
- `pc10`: efficient LM down-proj + qweight
- `pc11`: efficient LM scales + qzeros + g_idx
- `pc12`: `lm_head_weight`
- `pc13`: `draft_embed_tokens_weight`
- `pc14`: recurrent step token/score/topk streams
- `pc15`: recurrent hidden stream + output hidden stream
- `pc16..pc21`: cumulative arrays (`cumu_*`, `prev/next/side`)
- `pc22`: output/work/sort/cache_topk arrays
- `pc23`: node mapping + accepted ids + IO/debug scalars
- `pc24`: `initial_logits`
- `pc25`: initial candidate/topk/hidden buffers
- `pc26`: prefill hidden(3H) + prefill embed(H)
- `pc27`: prefill FC weight + prefill FC scales
- `pc28`: debug float buffers
- `pc29`: debug int64 buffers
- `pc30`: policy arrays + hot token table
- `pc31`: reserved

## Offset Contract

Offsets are computed in 256-bit words (`32 bytes`) and aligned per region in
`build_layout(...)` inside `cost_draft_tree_hbm_packed.cpp`.

Host-side packers must mirror those formulas exactly:

- `bytes_to_hbm_words(bytes) = ceil(bytes / 32)`
- region offset = cumulative word count of prior regions in that PC

The current implementation uses dynamic dimensions from AXI-lite scalars:

- `batch_size`
- `node_top_k`
- `hidden_size`
- `max_node_count`
- `max_verify_num`
- `max_tree_width`
- `policy_depth`
- `initial_logits_width`
- `hot_token_vocab_size`

## Notes

- This revision is additive and non-breaking. Existing `eagle4_draft(...)` remains intact.
- `eagle4_draft_packed_32pc(...)` is the integration point for CUDA memcpy / GPUDirect-RDMA packing.
- Next optimization step is replacing sequential prefetch/writeback with true per-PC DATAFLOW burst stages.
