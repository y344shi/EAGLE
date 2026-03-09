#ifndef TMAC_CONTIGUOUS_KV_HLS_HPP
#define TMAC_CONTIGUOUS_KV_HLS_HPP

// Contiguous KV cache: write new tokens to HBM, gather prefix + ancestors + self.
// Replaces kv_cache_manager (URAM/HBM hot-cold tiering) and paged-attention indirection.
//
// HBM layout (batch_size=1):
//   [0 .. prefix_len-1]                          : prefix KV from target model
//   [prefix_len + layer * max_tree_width + t]     : draft layer `layer`, tree slot `t`
//
// max_tree_width is the fixed stride per layer (= compile-time kContiguousKvTreeWidth for now).
// Each layer may use fewer than max_tree_width slots; unused slots are padding.
//
// Each token occupies VECS_PER_KV_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W vectors.

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

constexpr int kMaxDraftDepth = 16;
constexpr int kContiguousKvTreeWidth = 4;
constexpr int kContiguousKvMaxAccepted = 64;

// ---------------------------------------------------------------------------
// Write kContiguousKvTreeWidth new K/V tokens to contiguous HBM.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS>
void contiguous_kv_write(
    hls_stream<vec_t<VEC_W>>& k_in,   // post-RoPE K, kContiguousKvTreeWidth tokens token-major
    hls_stream<vec_t<VEC_W>>& v_in,   // V projection, kContiguousKvTreeWidth tokens token-major
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int write_base_token               // = prefix_len + current_depth * max_tree_width
) {
#pragma HLS INLINE off
    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0, "KV width must align to VEC_W");

write_token_loop:
    for (int t = 0; t < kContiguousKvTreeWidth; ++t) {
#pragma HLS loop_tripcount min=kContiguousKvTreeWidth max=kContiguousKvTreeWidth avg=kContiguousKvTreeWidth
        const int base = (write_base_token + t) * VECS_PER_TOKEN;
    write_vec_loop:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS loop_tripcount min=NUM_KV_HEADS*HEAD_DIM/VEC_W max=NUM_KV_HEADS*HEAD_DIM/VEC_W avg=NUM_KV_HEADS*HEAD_DIM/VEC_W
#pragma HLS PIPELINE II=1
            vec_t<VEC_W> kv = k_in.read();
            vec_t<VEC_W> vv = v_in.read();
            hbm_k[base + v] = kv;
            hbm_v[base + v] = vv;
        }
    }
}

// ---------------------------------------------------------------------------
// Compact accepted draft rows into prefix tail:
//   dst_slot = prefix_len + i
//   src_slot = accepted_draft_kv_indices[i]
//
// accepted_draft_kv_indices must be contiguous-HBM slot IDs (not paged-KV IDs).
// A local staging buffer is used to guarantee overlap-safe permutation copy.
// Returns false when any source/destination slot is invalid.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_ACCEPTED>
bool contiguous_kv_compact_accepted(
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int prefix_len,
    const int64_t* accepted_draft_kv_indices,
    int accepted_draft_kv_count,
    int max_hbm_token_count
) {
#pragma HLS INLINE off
    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0, "KV width must align to VEC_W");

    if (hbm_k == nullptr || hbm_v == nullptr) {
        return false;
    }
    if (accepted_draft_kv_count <= 0) {
        return true;
    }
    if (accepted_draft_kv_indices == nullptr || max_hbm_token_count <= 0) {
        return false;
    }
    if (accepted_draft_kv_count > MAX_ACCEPTED) {
        return false;
    }

    vec_t<VEC_W> src_k[MAX_ACCEPTED * VECS_PER_TOKEN];
    vec_t<VEC_W> src_v[MAX_ACCEPTED * VECS_PER_TOKEN];
#pragma HLS BIND_STORAGE variable=src_k type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=src_v type=ram_2p impl=bram

read_src_tokens:
    for (int i = 0; i < MAX_ACCEPTED; ++i) {
#pragma HLS loop_tripcount min=1 max=MAX_ACCEPTED avg=MAX_ACCEPTED/2
        if (i >= accepted_draft_kv_count) {
            break;
        }
        const int64_t src_slot = accepted_draft_kv_indices[i];
        if (src_slot < 0 || src_slot >= max_hbm_token_count) {
            return false;
        }
        const int src_base = static_cast<int>(src_slot) * VECS_PER_TOKEN;
    read_src_vecs:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS loop_tripcount min=NUM_KV_HEADS*HEAD_DIM/VEC_W max=NUM_KV_HEADS*HEAD_DIM/VEC_W avg=NUM_KV_HEADS*HEAD_DIM/VEC_W
#pragma HLS PIPELINE II=1
            src_k[i * VECS_PER_TOKEN + v] = hbm_k[src_base + v];
            src_v[i * VECS_PER_TOKEN + v] = hbm_v[src_base + v];
        }
    }

write_dst_tokens:
    for (int i = 0; i < MAX_ACCEPTED; ++i) {
#pragma HLS loop_tripcount min=1 max=MAX_ACCEPTED avg=MAX_ACCEPTED/2
        if (i >= accepted_draft_kv_count) {
            break;
        }
        const int dst_slot = prefix_len + i;
        if (dst_slot < 0 || dst_slot >= max_hbm_token_count) {
            return false;
        }
        const int dst_base = dst_slot * VECS_PER_TOKEN;
    write_dst_vecs:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS loop_tripcount min=NUM_KV_HEADS*HEAD_DIM/VEC_W max=NUM_KV_HEADS*HEAD_DIM/VEC_W avg=NUM_KV_HEADS*HEAD_DIM/VEC_W
#pragma HLS PIPELINE II=1
            hbm_k[dst_base + v] = src_k[i * VECS_PER_TOKEN + v];
            hbm_v[dst_base + v] = src_v[i * VECS_PER_TOKEN + v];
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Gather prefix + ancestor + self KV from contiguous HBM.
//
// Traces each query's ancestor chain internally using parent_indices_per_layer:
//   parent_indices_per_layer[l * max_tree_width + t] = slot in layer l that is
//   the parent of slot t at layer l+1.
//
// max_tree_width is the fixed stride in both the HBM layout and parent_indices array.
// Output per query: (prefix_len + current_depth + 1) tokens of KV history.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_DEPTH>
void contiguous_kv_gather(
    const vec_t<VEC_W>* hbm_k,
    const vec_t<VEC_W>* hbm_v,
    int prefix_len,
    int current_depth,
    int max_tree_width,
    const int* parent_indices_per_layer,  // [MAX_DEPTH * max_tree_width]
    hls_stream<vec_t<VEC_W>> k_out[kContiguousKvTreeWidth],
    hls_stream<vec_t<VEC_W>> v_out[kContiguousKvTreeWidth]
) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=parent_indices_per_layer type=ram_2p impl=bram
    constexpr int VECS_PER_TOKEN = (NUM_KV_HEADS * HEAD_DIM) / VEC_W;
    static_assert((NUM_KV_HEADS * HEAD_DIM) % VEC_W == 0, "KV width must align to VEC_W");

    // Phase 1: Stream prefix tokens (shared across all queries, broadcast).
prefix_token_loop:
    for (int p = 0; p < prefix_len; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=1 avg=1024 max=2048
        const int base = p * VECS_PER_TOKEN;
    prefix_vec_loop:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS loop_tripcount min=NUM_KV_HEADS*HEAD_DIM/VEC_W max=NUM_KV_HEADS*HEAD_DIM/VEC_W avg=NUM_KV_HEADS*HEAD_DIM/VEC_W
#pragma HLS PIPELINE II=1
            vec_t<VEC_W> kv = hbm_k[base + v];
            vec_t<VEC_W> vv = hbm_v[base + v];
        prefix_broadcast_loop:
            for (int t = 0; t < kContiguousKvTreeWidth; ++t) {
#pragma HLS UNROLL
                k_out[t].write(kv);
                v_out[t].write(vv);
            }
        }
    }

    // Phase 2: Stream ancestor tokens in chronological depth order.
    // We first trace parent slots backward (d-1 -> 0), then emit forward (0 -> d-1).
ancestor_query_loop:
    for (int t = 0; t < kContiguousKvTreeWidth; ++t) {
#pragma HLS loop_tripcount min=kContiguousKvTreeWidth max=kContiguousKvTreeWidth avg=kContiguousKvTreeWidth
        int slot = t;
        int ancestor_slots[MAX_DEPTH];
#pragma HLS BIND_STORAGE variable=ancestor_slots type=ram_2p impl=bram
    ancestor_slots_init_loop:
        for (int l = 0; l < MAX_DEPTH; ++l) {
#pragma HLS LOOP_TRIPCOUNT min=1 avg=4 max=MAX_DEPTH
#pragma HLS PIPELINE II=1
            ancestor_slots[l] = 0;
        }
    ancestor_trace_loop:
        for (int l = current_depth - 1; l >= 0; --l) {
#pragma HLS LOOP_TRIPCOUNT min=0 avg=2 max=16
            int parent_slot = parent_indices_per_layer[l * max_tree_width + slot];
            ancestor_slots[l] = parent_slot;
            slot = parent_slot;
        }
    ancestor_emit_loop:
        for (int l = 0; l < current_depth; ++l) {
#pragma HLS LOOP_TRIPCOUNT min=0 avg=2 max=16
            int parent_slot = ancestor_slots[l];
            int token_idx = prefix_len + l * max_tree_width + parent_slot;
            int base = token_idx * VECS_PER_TOKEN;
        ancestor_vec_loop:
            for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS loop_tripcount min=NUM_KV_HEADS*HEAD_DIM/VEC_W max=NUM_KV_HEADS*HEAD_DIM/VEC_W avg=NUM_KV_HEADS*HEAD_DIM/VEC_W
#pragma HLS PIPELINE II=1
                k_out[t].write(hbm_k[base + v]);
                v_out[t].write(hbm_v[base + v]);
            }
        }
    }

    // Phase 3: Stream self tokens (one per query, just written by contiguous_kv_write).
self_token_loop:
    for (int t = 0; t < kContiguousKvTreeWidth; ++t) {
#pragma HLS loop_tripcount min=kContiguousKvTreeWidth max=kContiguousKvTreeWidth avg=kContiguousKvTreeWidth
        int token_idx = prefix_len + current_depth * max_tree_width + t;
        int base = token_idx * VECS_PER_TOKEN;
    self_vec_loop:
        for (int v = 0; v < VECS_PER_TOKEN; ++v) {
#pragma HLS loop_tripcount min=NUM_KV_HEADS*HEAD_DIM/VEC_W max=NUM_KV_HEADS*HEAD_DIM/VEC_W avg=NUM_KV_HEADS*HEAD_DIM/VEC_W
#pragma HLS PIPELINE II=1
            k_out[t].write(hbm_k[base + v]);
            v_out[t].write(hbm_v[base + v]);
        }
    }
}

// ---------------------------------------------------------------------------
// Combined write + gather: single DATAFLOW stage that writes new K/V to HBM
// then gathers prefix + ancestors + self for all kContiguousKvTreeWidth queries.
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int NUM_KV_HEADS, int MAX_DEPTH>
void contiguous_kv_write_and_gather(
    hls_stream<vec_t<VEC_W>>& k_in,
    hls_stream<vec_t<VEC_W>>& v_in,
    vec_t<VEC_W>* hbm_k,
    vec_t<VEC_W>* hbm_v,
    int prefix_len,
    int current_depth,
    int max_tree_width,
    const int* parent_indices_per_layer,  // [MAX_DEPTH * max_tree_width]
    hls_stream<vec_t<VEC_W>> k_out[kContiguousKvTreeWidth],
    hls_stream<vec_t<VEC_W>> v_out[kContiguousKvTreeWidth]
) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=parent_indices_per_layer type=ram_2p impl=bram
    // Write first (so self-token is in HBM for gather phase 3).
    contiguous_kv_write<HEAD_DIM, NUM_KV_HEADS>(
        k_in, v_in, hbm_k, hbm_v,
        prefix_len + current_depth * max_tree_width);

    // Then gather.
    contiguous_kv_gather<HEAD_DIM, NUM_KV_HEADS, MAX_DEPTH>(
        hbm_k, hbm_v, prefix_len, current_depth,
        max_tree_width, parent_indices_per_layer, k_out, v_out);
}

} // namespace hls
} // namespace tmac

#endif // TMAC_CONTIGUOUS_KV_HLS_HPP
