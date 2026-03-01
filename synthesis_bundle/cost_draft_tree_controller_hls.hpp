#ifndef TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP
#define TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP

#include <cstdint>

namespace tmac {
namespace hls {

constexpr int kCdtControllerMaxDepth = 64;

inline int e4d_clamp_int(int x, int low, int high) {
#pragma HLS INLINE
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

inline int64_t e4d_clamp_i64(int64_t x, int64_t low, int64_t high) {
#pragma HLS INLINE
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

void e4d_ctrl_reset(
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
    int64_t* node_depths               // [batch, max_node_count]
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
        }
    }
}

void e4d_ctrl_seed(
    const int64_t* seed_tokens,         // [batch, width]
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
    int64_t* node_depths               // [batch, max_node_count] in/out
) {
#pragma HLS INLINE off
    const int use_width = e4d_clamp_int(width, 0, max_tree_width);

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

void e4d_ctrl_expand(
    const int64_t* parent_frontier_node_ids, // [batch, max_tree_width]
    const int64_t* parent_slots,             // [batch, width], each in [0, parent_width)
    const int64_t* child_tokens,             // [batch, width]
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
    int64_t* node_depths                     // [batch, max_node_count] in/out
) {
#pragma HLS INLINE off
    const int use_width = e4d_clamp_int(width, 0, max_tree_width);
    const int use_parent_width = e4d_clamp_int(parent_width, 1, max_tree_width);

expand_batch_loop:
    for (int b = 0; b < batch_size; ++b) {
        const int base = b * max_node_count;

    expand_child_loop:
        for (int i = 0; i < use_width; ++i) {
#pragma HLS PIPELINE II = 1
            const int in_idx = b * use_width + i;

            int64_t slot = parent_slots[in_idx];
            slot = e4d_clamp_i64(slot, 0, use_parent_width - 1);
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

void e4d_ctrl_export(
    const int64_t* frontier_node_ids,   // [batch, max_tree_width]
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    const int64_t* node_token_ids,      // [batch, max_node_count]
    const int64_t* node_parent_ids,     // [batch, max_node_count]
    const int64_t* node_depths,         // [batch, max_node_count]
    int64_t* frontier_tokens,           // [batch, max_tree_width]
    int64_t* frontier_parent_ids,       // [batch, max_tree_width]
    int64_t* frontier_depths            // [batch, max_tree_width]
) {
#pragma HLS INLINE off
    const int use_width = e4d_clamp_int(width, 0, max_tree_width);

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
                continue;
            }

            const int64_t nid = frontier_node_ids[b * max_tree_width + i];
            if (nid < 0 || nid >= max_node_count) {
                frontier_tokens[b * max_tree_width + i] = -1;
                frontier_parent_ids[b * max_tree_width + i] = -1;
                frontier_depths[b * max_tree_width + i] = -1;
                continue;
            }

            frontier_tokens[b * max_tree_width + i] = node_token_ids[base + nid];
            frontier_parent_ids[b * max_tree_width + i] = node_parent_ids[base + nid];
            frontier_depths[b * max_tree_width + i] = node_depths[base + nid];
        }
    }
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP
