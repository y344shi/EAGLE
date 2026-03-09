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
);

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
);

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
);

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
);

// Backward-compatible wrappers for legacy TB entrypoints.
inline void cdt_controller_reset(
    int batch_size,
    int max_tree_width,
    int max_node_count,
    int* node_count,
    int64_t* frontier_node_ids,
    int64_t* node_token_ids,
    int64_t* node_parent_ids,
    int64_t* node_first_child_ids,
    int64_t* node_last_child_ids,
    int64_t* node_next_sibling_ids,
    int64_t* node_depths) {
#pragma HLS INLINE
    e4d_ctrl_reset(
        batch_size,
        max_tree_width,
        max_node_count,
        node_count,
        frontier_node_ids,
        node_token_ids,
        node_parent_ids,
        node_first_child_ids,
        node_last_child_ids,
        node_next_sibling_ids,
        node_depths);
}

inline void cdt_controller_seed_frontier(
    const int64_t* seed_tokens,
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    int* node_count,
    int64_t* frontier_node_ids,
    int64_t* node_token_ids,
    int64_t* node_parent_ids,
    int64_t* node_first_child_ids,
    int64_t* node_last_child_ids,
    int64_t* node_next_sibling_ids,
    int64_t* node_depths) {
#pragma HLS INLINE
    e4d_ctrl_seed(
        seed_tokens,
        batch_size,
        width,
        max_tree_width,
        max_node_count,
        node_count,
        frontier_node_ids,
        node_token_ids,
        node_parent_ids,
        node_first_child_ids,
        node_last_child_ids,
        node_next_sibling_ids,
        node_depths);
}

inline void cdt_controller_expand_frontier(
    const int64_t* parent_frontier_node_ids,
    const int64_t* parent_slots,
    const int64_t* child_tokens,
    int batch_size,
    int parent_width,
    int width,
    int max_tree_width,
    int max_node_count,
    int* node_count,
    int64_t* next_frontier_node_ids,
    int64_t* node_token_ids,
    int64_t* node_parent_ids,
    int64_t* node_first_child_ids,
    int64_t* node_last_child_ids,
    int64_t* node_next_sibling_ids,
    int64_t* node_depths) {
#pragma HLS INLINE
    e4d_ctrl_expand(
        parent_frontier_node_ids,
        parent_slots,
        child_tokens,
        batch_size,
        parent_width,
        width,
        max_tree_width,
        max_node_count,
        node_count,
        next_frontier_node_ids,
        node_token_ids,
        node_parent_ids,
        node_first_child_ids,
        node_last_child_ids,
        node_next_sibling_ids,
        node_depths);
}

inline void cdt_controller_export_frontier(
    const int64_t* frontier_node_ids,
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    const int64_t* node_token_ids,
    const int64_t* node_parent_ids,
    const int64_t* node_depths,
    int64_t* frontier_tokens,
    int64_t* frontier_parent_ids,
    int64_t* frontier_depths) {
#pragma HLS INLINE
    e4d_ctrl_export(
        frontier_node_ids,
        batch_size,
        width,
        max_tree_width,
        max_node_count,
        node_token_ids,
        node_parent_ids,
        node_depths,
        frontier_tokens,
        frontier_parent_ids,
        frontier_depths);
}

// Backward-compatible aliases used by existing TBs/scripts.
inline void cdt_controller_reset(
    int batch_size,
    int max_tree_width,
    int max_node_count,
    int* node_count,
    int64_t* frontier_node_ids,
    int64_t* node_token_ids,
    int64_t* node_parent_ids,
    int64_t* node_first_child_ids,
    int64_t* node_last_child_ids,
    int64_t* node_next_sibling_ids,
    int64_t* node_depths) {
    e4d_ctrl_reset(
        batch_size,
        max_tree_width,
        max_node_count,
        node_count,
        frontier_node_ids,
        node_token_ids,
        node_parent_ids,
        node_first_child_ids,
        node_last_child_ids,
        node_next_sibling_ids,
        node_depths);
}

inline void cdt_controller_seed_frontier(
    const int64_t* seed_tokens,
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    int* node_count,
    int64_t* frontier_node_ids,
    int64_t* node_token_ids,
    int64_t* node_parent_ids,
    int64_t* node_first_child_ids,
    int64_t* node_last_child_ids,
    int64_t* node_next_sibling_ids,
    int64_t* node_depths) {
    e4d_ctrl_seed(
        seed_tokens,
        batch_size,
        width,
        max_tree_width,
        max_node_count,
        node_count,
        frontier_node_ids,
        node_token_ids,
        node_parent_ids,
        node_first_child_ids,
        node_last_child_ids,
        node_next_sibling_ids,
        node_depths);
}

inline void cdt_controller_expand_frontier(
    const int64_t* parent_frontier_node_ids,
    const int64_t* parent_slots,
    const int64_t* child_tokens,
    int batch_size,
    int parent_width,
    int width,
    int max_tree_width,
    int max_node_count,
    int* node_count,
    int64_t* next_frontier_node_ids,
    int64_t* node_token_ids,
    int64_t* node_parent_ids,
    int64_t* node_first_child_ids,
    int64_t* node_last_child_ids,
    int64_t* node_next_sibling_ids,
    int64_t* node_depths) {
    e4d_ctrl_expand(
        parent_frontier_node_ids,
        parent_slots,
        child_tokens,
        batch_size,
        parent_width,
        width,
        max_tree_width,
        max_node_count,
        node_count,
        next_frontier_node_ids,
        node_token_ids,
        node_parent_ids,
        node_first_child_ids,
        node_last_child_ids,
        node_next_sibling_ids,
        node_depths);
}

inline void cdt_controller_export_frontier(
    const int64_t* frontier_node_ids,
    int batch_size,
    int width,
    int max_tree_width,
    int max_node_count,
    const int64_t* node_token_ids,
    const int64_t* node_parent_ids,
    const int64_t* node_depths,
    int64_t* frontier_tokens,
    int64_t* frontier_parent_ids,
    int64_t* frontier_depths) {
    e4d_ctrl_export(
        frontier_node_ids,
        batch_size,
        width,
        max_tree_width,
        max_node_count,
        node_token_ids,
        node_parent_ids,
        node_depths,
        frontier_tokens,
        frontier_parent_ids,
        frontier_depths);
}

} // namespace hls
} // namespace tmac

#endif // TMAC_COST_DRAFT_TREE_CONTROLLER_HLS_HPP
