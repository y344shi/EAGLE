#include "cost_draft_tree_controller_hls.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

static bool check_eq_i64(const char* name, int64_t got, int64_t exp) {
    if (got != exp) {
        std::cerr << "[FAIL] " << name << ": got=" << got << " expected=" << exp << "\n";
        return false;
    }
    return true;
}

static bool check_eq_i32(const char* name, int32_t got, int32_t exp) {
    if (got != exp) {
        std::cerr << "[FAIL] " << name << ": got=" << got << " expected=" << exp << "\n";
        return false;
    }
    return true;
}

static int run_synthetic_smoke() {
    constexpr int B = 1;
    constexpr int MAX_TREE_WIDTH = 4;
    constexpr int MAX_NODE_COUNT = 64;

    std::vector<int> node_count(B, 0);
    std::vector<int64_t> frontier(B * MAX_TREE_WIDTH, -1);
    std::vector<int64_t> frontier_next(B * MAX_TREE_WIDTH, -1);

    std::vector<int64_t> node_token(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_parent(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_first_child(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_last_child(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_next_sibling(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_depth(B * MAX_NODE_COUNT, -1);

    tmac::hls::cdt_controller_reset(
        B,
        MAX_TREE_WIDTH,
        MAX_NODE_COUNT,
        node_count.data(),
        frontier.data(),
        node_token.data(),
        node_parent.data(),
        node_first_child.data(),
        node_last_child.data(),
        node_next_sibling.data(),
        node_depth.data());

    bool ok = true;

    // Layer 1 seed.
    const int width1 = 4;
    const std::vector<int64_t> seed_tokens = {101, 102, 103, 104};

    tmac::hls::cdt_controller_seed_frontier(
        seed_tokens.data(),
        B,
        width1,
        MAX_TREE_WIDTH,
        MAX_NODE_COUNT,
        node_count.data(),
        frontier.data(),
        node_token.data(),
        node_parent.data(),
        node_first_child.data(),
        node_last_child.data(),
        node_next_sibling.data(),
        node_depth.data());

    ok &= check_eq_i32("node_count_after_seed", node_count[0], 4);

    for (int i = 0; i < width1; ++i) {
        ok &= check_eq_i64("seed_frontier_id", frontier[i], i);
        ok &= check_eq_i64("seed_token", node_token[i], seed_tokens[static_cast<size_t>(i)]);
        ok &= check_eq_i64("seed_parent", node_parent[i], -1);
        ok &= check_eq_i64("seed_depth", node_depth[i], 0);
    }

    // Layer 2 expansion.
    const int width2 = 4;
    const int parent_width2 = 4;
    const std::vector<int64_t> l2_parent_slots = {0, 0, 2, 1};
    const std::vector<int64_t> l2_tokens = {201, 202, 203, 204};

    tmac::hls::cdt_controller_expand_frontier(
        frontier.data(),
        l2_parent_slots.data(),
        l2_tokens.data(),
        B,
        parent_width2,
        width2,
        MAX_TREE_WIDTH,
        MAX_NODE_COUNT,
        node_count.data(),
        frontier_next.data(),
        node_token.data(),
        node_parent.data(),
        node_first_child.data(),
        node_last_child.data(),
        node_next_sibling.data(),
        node_depth.data());

    ok &= check_eq_i32("node_count_after_l2", node_count[0], 8);

    const std::vector<int64_t> expected_l2_ids = {4, 5, 6, 7};
    const std::vector<int64_t> expected_l2_parents = {0, 0, 2, 1};

    for (int i = 0; i < width2; ++i) {
        ok &= check_eq_i64("l2_frontier_id", frontier_next[i], expected_l2_ids[static_cast<size_t>(i)]);
        const int nid = static_cast<int>(frontier_next[i]);
        ok &= check_eq_i64("l2_token", node_token[nid], l2_tokens[static_cast<size_t>(i)]);
        ok &= check_eq_i64("l2_parent", node_parent[nid], expected_l2_parents[static_cast<size_t>(i)]);
        ok &= check_eq_i64("l2_depth", node_depth[nid], 1);
    }

    // Validate child linking.
    ok &= check_eq_i64("node0_first_child", node_first_child[0], 4);
    ok &= check_eq_i64("node0_last_child", node_last_child[0], 5);
    ok &= check_eq_i64("node4_next_sibling", node_next_sibling[4], 5);
    ok &= check_eq_i64("node2_first_child", node_first_child[2], 6);
    ok &= check_eq_i64("node1_first_child", node_first_child[1], 7);

    // Export layer 2 frontier.
    std::vector<int64_t> export_tokens(B * MAX_TREE_WIDTH, -1);
    std::vector<int64_t> export_parents(B * MAX_TREE_WIDTH, -1);
    std::vector<int64_t> export_depths(B * MAX_TREE_WIDTH, -1);

    tmac::hls::cdt_controller_export_frontier(
        frontier_next.data(),
        B,
        width2,
        MAX_TREE_WIDTH,
        MAX_NODE_COUNT,
        node_token.data(),
        node_parent.data(),
        node_depth.data(),
        export_tokens.data(),
        export_parents.data(),
        export_depths.data());

    for (int i = 0; i < width2; ++i) {
        ok &= check_eq_i64("export_token", export_tokens[i], l2_tokens[static_cast<size_t>(i)]);
        ok &= check_eq_i64("export_parent", export_parents[i], expected_l2_parents[static_cast<size_t>(i)]);
        ok &= check_eq_i64("export_depth", export_depths[i], 1);
    }

    // Layer 3 expansion from layer2 frontier.
    const std::vector<int64_t> l3_parent_slots = {1, 1, 3, 0};
    const std::vector<int64_t> l3_tokens = {301, 302, 303, 304};

    std::vector<int64_t> frontier_l3(B * MAX_TREE_WIDTH, -1);

    tmac::hls::cdt_controller_expand_frontier(
        frontier_next.data(),
        l3_parent_slots.data(),
        l3_tokens.data(),
        B,
        width2,
        width2,
        MAX_TREE_WIDTH,
        MAX_NODE_COUNT,
        node_count.data(),
        frontier_l3.data(),
        node_token.data(),
        node_parent.data(),
        node_first_child.data(),
        node_last_child.data(),
        node_next_sibling.data(),
        node_depth.data());

    ok &= check_eq_i32("node_count_after_l3", node_count[0], 12);

    for (int i = 0; i < width2; ++i) {
        const int nid = static_cast<int>(frontier_l3[i]);
        ok &= check_eq_i64("l3_token", node_token[nid], l3_tokens[static_cast<size_t>(i)]);
        ok &= check_eq_i64("l3_depth", node_depth[nid], 2);
    }

    if (!ok) {
        std::cerr << "[FAIL] cost_draft_tree controller HLS smoke failed.\n";
        return 1;
    }

    std::cout << "[PASS] cost_draft_tree controller HLS smoke passed.\n";
    return 0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    return run_synthetic_smoke();
}
