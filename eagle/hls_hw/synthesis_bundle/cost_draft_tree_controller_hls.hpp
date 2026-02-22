#ifndef TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP
#define TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP

#include <cstdint>

namespace tmac {
namespace hls {

constexpr int kCdtControllerMaxDepth = 64;

inline int cdt_clamp_int(int x, int low, int high) {
#pragma HLS INLINE
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

inline int64_t cdt_clamp_i64(int64_t x, int64_t low, int64_t high) {
#pragma HLS INLINE
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

inline void cdt_controller_reset(
    int batch_size,
    int max_tree_width,
    int max_node_count,
    int* node_count,                    // [batch]
    int64_t* frontier_node_ids,         // [batch, max_tree_width]
    int64_t* node_token_ids,            // [batch, max_node_count]
    int64_t* node_parent_ids,           // [batch, max_node_count]
    int64_t* node_first_child_ids,      // [batch, max_node_count]
    int64_t* node_last_child_ids,       // [batch, max_node_count]
    int64_t* node_next_sibling_ids,     // [batch, max_node_count]
    int64_t* node_depths,               // [batch, max_node_count]
    int32_t* node_cache_locs            // [batch, max_node_count]
) {
#pragma HLS INLINE off
reset_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        node_count[b] = 0;
    reset_frontier_loop:
        for (int i = 0; i < max_tree_width; ++i) {
#pragma HLS PIPELINE II = 1
            frontier_node_ids[b * max_tree_width + i] = -1;
        }

        const int base = b * max_node_count;
    reset_node_loop:
        for (int n = 0; n < max_node_count; ++n) {
#pragma HLS PIPELINE II = 1
            node_token_ids[base + n] = -1;
            node_parent_ids[base + n] = -1;
            node_first_child_ids[base + n] = -1;
            node_last_child_ids[base + n] = -1;
            node_next_sibling_ids[base + n] = -1;
            node_depths[base + n] = -1;
            node_cache_locs[base + n] = -1;
        }
    }
}

inline void cdt_controller_seed_frontier(
    const int64_t* seed_tokens,         // [batch, width]
    const int32_t* seed_cache_locs,     // [batch, width]
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    int* node_count,                    // [batch] in/out
    int64_t* frontier_node_ids,         // [batch, max_tree_width] out
    int64_t* node_token_ids,            // [batch, max_node_count] in/out
    int64_t* node_parent_ids,           // [batch, max_node_count] in/out
    int64_t* node_first_child_ids,      // [batch, max_node_count] in/out
    int64_t* node_last_child_ids,       // [batch, max_node_count] in/out
    int64_t* node_next_sibling_ids,     // [batch, max_node_count] in/out
    int64_t* node_depths,               // [batch, max_node_count] in/out
    int32_t* node_cache_locs            // [batch, max_node_count] in/out
) {
#pragma HLS INLINE off
    const int use_width = cdt_clamp_int(width, 0, max_tree_width);

seed_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        int64_t prev_seed_id = -1;
        const int base = b * max_node_count;

    seed_node_loop:
        for (int i = 0; i < use_width; ++i) {
#pragma HLS PIPELINE II = 1
            int nid = node_count[b];
            if (nid >= max_node_count) {
                frontier_node_ids[b * max_tree_width + i] = -1;
                continue;
            }
            node_count[b] = nid + 1;

            const int widx = b * use_width + i;
            frontier_node_ids[b * max_tree_width + i] = nid;

            node_token_ids[base + nid] = seed_tokens[widx];
            node_parent_ids[base + nid] = -1;
            node_first_child_ids[base + nid] = -1;
            node_last_child_ids[base + nid] = -1;
            node_next_sibling_ids[base + nid] = -1;
            node_depths[base + nid] = 0;
            node_cache_locs[base + nid] = seed_cache_locs[widx];

            // Keep deterministic sibling order under virtual root.
            if (prev_seed_id >= 0) {
                node_next_sibling_ids[base + prev_seed_id] = nid;
            }
            prev_seed_id = nid;
        }

    seed_clear_frontier_loop:
        for (int i = use_width; i < max_tree_width; ++i) {
#pragma HLS PIPELINE II = 1
            frontier_node_ids[b * max_tree_width + i] = -1;
        }
    }
}

inline void cdt_controller_expand_frontier(
    const int64_t* parent_frontier_node_ids, // [batch, max_tree_width]
    const int64_t* parent_slots,             // [batch, width], each in [0, parent_width)
    const int64_t* child_tokens,             // [batch, width]
    const int32_t* child_cache_locs,         // [batch, width]
    int batch_size,
    int parent_width,
    int width,
    int max_tree_width,
    int max_node_count,
    int* node_count,                         // [batch] in/out
    int64_t* next_frontier_node_ids,         // [batch, max_tree_width] out
    int64_t* node_token_ids,                 // [batch, max_node_count] in/out
    int64_t* node_parent_ids,                // [batch, max_node_count] in/out
    int64_t* node_first_child_ids,           // [batch, max_node_count] in/out
    int64_t* node_last_child_ids,            // [batch, max_node_count] in/out
    int64_t* node_next_sibling_ids,          // [batch, max_node_count] in/out
    int64_t* node_depths,                    // [batch, max_node_count] in/out
    int32_t* node_cache_locs                 // [batch, max_node_count] in/out
) {
#pragma HLS INLINE off
    const int use_width = cdt_clamp_int(width, 0, max_tree_width);
    const int use_parent_width = cdt_clamp_int(parent_width, 1, max_tree_width);

expand_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        const int base = b * max_node_count;

    expand_child_loop:
        for (int i = 0; i < use_width; ++i) {
#pragma HLS PIPELINE II = 1
            const int in_idx = b * use_width + i;

            int64_t slot = parent_slots[in_idx];
            slot = cdt_clamp_i64(slot, 0, use_parent_width - 1);
            int64_t parent_nid = parent_frontier_node_ids[b * max_tree_width + slot];
            if (parent_nid < 0 || parent_nid >= max_node_count) {
                parent_nid = -1;
            }

            int nid = node_count[b];
            if (nid >= max_node_count) {
                next_frontier_node_ids[b * max_tree_width + i] = -1;
                continue;
            }
            node_count[b] = nid + 1;
            next_frontier_node_ids[b * max_tree_width + i] = nid;

            node_token_ids[base + nid] = child_tokens[in_idx];
            node_parent_ids[base + nid] = parent_nid;
            node_first_child_ids[base + nid] = -1;
            node_last_child_ids[base + nid] = -1;
            node_next_sibling_ids[base + nid] = -1;
            node_cache_locs[base + nid] = child_cache_locs[in_idx];

            int64_t depth = 0;
            if (parent_nid >= 0) {
                depth = node_depths[base + parent_nid] + 1;
            }
            node_depths[base + nid] = depth;

            // Link under parent: O(1) append via node_last_child_ids.
            if (parent_nid >= 0) {
                const int pidx = base + static_cast<int>(parent_nid);
                const int64_t last_child = node_last_child_ids[pidx];
                if (last_child < 0) {
                    node_first_child_ids[pidx] = nid;
                } else if (last_child < max_node_count) {
                    node_next_sibling_ids[base + static_cast<int>(last_child)] = nid;
                }
                node_last_child_ids[pidx] = nid;
            }
        }

    expand_clear_frontier_loop:
        for (int i = use_width; i < max_tree_width; ++i) {
#pragma HLS PIPELINE II = 1
            next_frontier_node_ids[b * max_tree_width + i] = -1;
        }
    }
}

inline void cdt_controller_export_frontier(
    const int64_t* frontier_node_ids,   // [batch, max_tree_width]
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    const int64_t* node_token_ids,      // [batch, max_node_count]
    const int64_t* node_parent_ids,     // [batch, max_node_count]
    const int64_t* node_depths,         // [batch, max_node_count]
    const int32_t* node_cache_locs,     // [batch, max_node_count]
    int64_t* frontier_tokens,           // [batch, max_tree_width]
    int64_t* frontier_parent_ids,       // [batch, max_tree_width]
    int64_t* frontier_depths,           // [batch, max_tree_width]
    int32_t* frontier_cache_locs        // [batch, max_tree_width]
) {
#pragma HLS INLINE off
    const int use_width = cdt_clamp_int(width, 0, max_tree_width);

export_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        const int base = b * max_node_count;
    export_slot_loop:
        for (int i = 0; i < max_tree_width; ++i) {
#pragma HLS PIPELINE II = 1
            if (i >= use_width) {
                frontier_tokens[b * max_tree_width + i] = -1;
                frontier_parent_ids[b * max_tree_width + i] = -1;
                frontier_depths[b * max_tree_width + i] = -1;
                frontier_cache_locs[b * max_tree_width + i] = -1;
                continue;
            }

            const int64_t nid = frontier_node_ids[b * max_tree_width + i];
            if (nid < 0 || nid >= max_node_count) {
                frontier_tokens[b * max_tree_width + i] = -1;
                frontier_parent_ids[b * max_tree_width + i] = -1;
                frontier_depths[b * max_tree_width + i] = -1;
                frontier_cache_locs[b * max_tree_width + i] = -1;
                continue;
            }

            frontier_tokens[b * max_tree_width + i] = node_token_ids[base + nid];
            frontier_parent_ids[b * max_tree_width + i] = node_parent_ids[base + nid];
            frontier_depths[b * max_tree_width + i] = node_depths[base + nid];
            frontier_cache_locs[b * max_tree_width + i] = node_cache_locs[base + nid];
        }
    }
}

// Build strict parent-visible KV listings and a tree-shaped mask field.
// Visibility rule per candidate query:
//   visible = prefix tokens + ancestor chain(root->...->parent->self)
// No sibling or cousin visibility is included.
inline void cdt_controller_build_parent_visible_kv(
    const int64_t* frontier_node_ids,   // [batch, max_tree_width]
    const int32_t* prefix_kv_locs,      // [batch, max_prefix_len]
    const int* prefix_lens,             // [batch]
    int batch_size,
    int width,
    int max_tree_width,
    int max_prefix_len,
    int max_input_size,
    int max_node_count,
    const int64_t* node_parent_ids,     // [batch, max_node_count]
    const int32_t* node_cache_locs,     // [batch, max_node_count]
    int32_t* kv_indices,                // [batch, max_tree_width, max_input_size]
    bool* kv_mask,                      // [batch, max_tree_width, max_input_size]
    int* kv_lens,                       // [batch, max_tree_width]
    int64_t* ancestor_node_ids          // [batch, max_tree_width, kCdtControllerMaxDepth] (optional)
) {
#pragma HLS INLINE off
    const int use_width = cdt_clamp_int(width, 0, max_tree_width);

kv_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        const int base = b * max_node_count;
        int prefix_len = prefix_lens[b];
        prefix_len = cdt_clamp_int(prefix_len, 0, max_prefix_len);

    kv_query_loop:
        for (int q = 0; q < max_tree_width; ++q) {
            const int out_list_base = (b * max_tree_width + q) * max_input_size;
            const int anc_base = (b * max_tree_width + q) * kCdtControllerMaxDepth;

        kv_zero_loop:
            for (int i = 0; i < max_input_size; ++i) {
#pragma HLS PIPELINE II = 1
                kv_indices[out_list_base + i] = -1;
                kv_mask[out_list_base + i] = false;
            }
            kv_lens[b * max_tree_width + q] = 0;
            if (ancestor_node_ids != nullptr) {
            kv_zero_anc_loop:
                for (int i = 0; i < kCdtControllerMaxDepth; ++i) {
#pragma HLS PIPELINE II = 1
                    ancestor_node_ids[anc_base + i] = -1;
                }
            }

            if (q >= use_width) {
                continue;
            }

            int64_t nid = frontier_node_ids[b * max_tree_width + q];
            if (nid < 0 || nid >= max_node_count) {
                continue;
            }

            // Build self->parent->... chain first.
            int64_t chain[kCdtControllerMaxDepth];
#pragma HLS ARRAY_PARTITION variable = chain cyclic factor = 8
            int chain_len = 0;

        trace_parent_loop:
            for (int d = 0; d < kCdtControllerMaxDepth; ++d) {
#pragma HLS PIPELINE II = 1
                if (nid < 0 || nid >= max_node_count) {
                    break;
                }
                chain[chain_len++] = nid;
                nid = node_parent_ids[base + static_cast<int>(nid)];
            }

            int out_len = 0;

        write_prefix_loop:
            for (int i = 0; i < prefix_len && out_len < max_input_size; ++i) {
#pragma HLS PIPELINE II = 1
                kv_indices[out_list_base + out_len] = prefix_kv_locs[b * max_prefix_len + i];
                kv_mask[out_list_base + out_len] = true;
                ++out_len;
            }

            // Reverse chain => root->...->self order.
        write_chain_loop:
            for (int i = chain_len - 1; i >= 0 && out_len < max_input_size; --i) {
#pragma HLS PIPELINE II = 1
                const int64_t node_id = chain[i];
                kv_indices[out_list_base + out_len] =
                    node_cache_locs[base + static_cast<int>(node_id)];
                kv_mask[out_list_base + out_len] = true;
                if (ancestor_node_ids != nullptr && out_len - prefix_len < kCdtControllerMaxDepth) {
                    ancestor_node_ids[anc_base + (out_len - prefix_len)] = node_id;
                }
                ++out_len;
            }

            kv_lens[b * max_tree_width + q] = out_len;
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP
