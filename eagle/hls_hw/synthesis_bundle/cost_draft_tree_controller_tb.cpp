#include "cost_draft_tree_controller_hls.hpp"
#include "cost_draft_tree_tb_case_io.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct CliOptions {
    std::string case_file;
    bool dry_run = false;
};

struct BoolBuffer {
    BoolBuffer() = default;

    explicit BoolBuffer(size_t n)
        : size_(n), data_(n ? new bool[n] : nullptr) {
        for (size_t i = 0; i < size_; ++i) data_[i] = false;
    }

    BoolBuffer(const BoolBuffer& other)
        : size_(other.size_), data_(other.size_ ? new bool[other.size_] : nullptr) {
        for (size_t i = 0; i < size_; ++i) data_[i] = other.data_[i];
    }

    BoolBuffer& operator=(const BoolBuffer& other) {
        if (this == &other) return *this;
        BoolBuffer tmp(other);
        swap(tmp);
        return *this;
    }

    void swap(BoolBuffer& other) {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

    size_t size() const { return size_; }
    bool* data() { return data_.get(); }
    const bool* data() const { return data_.get(); }

    bool& operator[](size_t i) { return data_[i]; }
    const bool& operator[](size_t i) const { return data_[i]; }

private:
    size_t size_ = 0;
    std::unique_ptr<bool[]> data_;
};

static bool check_eq_i64(const std::string& name, int64_t got, int64_t exp) {
    if (got != exp) {
        std::cerr << "[FAIL] " << name << ": got=" << got << " expected=" << exp << "\n";
        return false;
    }
    return true;
}

static bool check_eq_i32(const std::string& name, int32_t got, int32_t exp) {
    if (got != exp) {
        std::cerr << "[FAIL] " << name << ": got=" << got << " expected=" << exp << "\n";
        return false;
    }
    return true;
}

static bool check_list(const std::string& name,
                       const int32_t* got,
                       const bool* mask,
                       int len,
                       const std::vector<int32_t>& exp,
                       int max_input_size) {
    bool ok = true;
    if (len != static_cast<int>(exp.size())) {
        std::cerr << "[FAIL] " << name << " length: got=" << len
                  << " expected=" << exp.size() << "\n";
        ok = false;
    }

    const int check_n = (len < static_cast<int>(exp.size())) ? len : static_cast<int>(exp.size());
    for (int i = 0; i < check_n; ++i) {
        if (got[i] != exp[static_cast<size_t>(i)]) {
            std::cerr << "[FAIL] " << name << " value[" << i << "]: got=" << got[i]
                      << " expected=" << exp[static_cast<size_t>(i)] << "\n";
            ok = false;
        }
        if (!mask[i]) {
            std::cerr << "[FAIL] " << name << " mask[" << i << "] should be true\n";
            ok = false;
        }
    }

    for (int i = len; i < max_input_size; ++i) {
        if (mask[i]) {
            std::cerr << "[FAIL] " << name << " mask tail[" << i << "] should be false\n";
            ok = false;
        }
    }

    return ok;
}

static bool parse_cli(int argc, char** argv, CliOptions* opts, std::string* err_msg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--case-file") {
            if (i + 1 >= argc) {
                *err_msg = "--case-file requires a path";
                return false;
            }
            opts->case_file = argv[++i];
        } else if (arg == "--dry-run") {
            opts->dry_run = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: cost_draft_tree_controller_tb [--case-file <path>] [--dry-run]\n";
            return false;
        } else {
            *err_msg = "unknown argument: " + arg;
            return false;
        }
    }
    return true;
}

struct ControllerKvCase {
    int batch_size = 0;
    int width = 0;
    int max_tree_width = 0;
    int max_prefix_len = 0;
    int max_input_size = 0;
    int max_node_count = 0;

    std::vector<int64_t> frontier_node_ids;
    std::vector<int32_t> prefix_kv_locs;
    std::vector<int> prefix_lens;
    std::vector<int64_t> node_parent_ids;
    std::vector<int32_t> node_cache_locs;

    std::vector<int32_t> expected_kv_indices;
    BoolBuffer expected_kv_mask;
    std::vector<int> expected_kv_lens;
    std::vector<int64_t> expected_ancestor_node_ids;
};

static bool load_controller_kv_case(const std::string& path,
                                    ControllerKvCase* tc,
                                    std::string* err_msg) {
    using namespace tmac::hls::tb_case_io;
    RawCaseMap kv;
    if (!parse_key_count_file(path, &kv, err_msg)) {
        return false;
    }

    if (!read_scalar_int(kv, "batch_size", &tc->batch_size, err_msg, true) ||
        !read_scalar_int(kv, "width", &tc->width, err_msg, true) ||
        !read_scalar_int(kv, "max_tree_width", &tc->max_tree_width, err_msg, true) ||
        !read_scalar_int(kv, "max_prefix_len", &tc->max_prefix_len, err_msg, true) ||
        !read_scalar_int(kv, "max_input_size", &tc->max_input_size, err_msg, true) ||
        !read_scalar_int(kv, "max_node_count", &tc->max_node_count, err_msg, true)) {
        return false;
    }

    if (tc->batch_size <= 0 || tc->width <= 0 || tc->max_tree_width <= 0 ||
        tc->max_prefix_len <= 0 || tc->max_input_size <= 0 || tc->max_node_count <= 0 ||
        tc->width > tc->max_tree_width) {
        *err_msg = "invalid scalar dimensions in controller case";
        return false;
    }

    const size_t frontier_n = static_cast<size_t>(tc->batch_size) * tc->max_tree_width;
    const size_t prefix_n = static_cast<size_t>(tc->batch_size) * tc->max_prefix_len;
    const size_t node_n = static_cast<size_t>(tc->batch_size) * tc->max_node_count;
    const size_t kv_n = static_cast<size_t>(tc->batch_size) * tc->max_tree_width * tc->max_input_size;
    const size_t kv_lens_n = static_cast<size_t>(tc->batch_size) * tc->max_tree_width;

    if (!read_i64_array(kv, "frontier_node_ids", frontier_n, &tc->frontier_node_ids, err_msg, true)) {
        return false;
    }
    if (!read_i32_array(kv, "prefix_kv_locs", prefix_n, &tc->prefix_kv_locs, err_msg, true)) {
        return false;
    }
    if (!read_int_array(kv, "prefix_lens", tc->batch_size, &tc->prefix_lens, err_msg, true)) {
        return false;
    }
    if (!read_i64_array(kv, "node_parent_ids", node_n, &tc->node_parent_ids, err_msg, true)) {
        return false;
    }
    if (!read_i32_array(kv, "node_cache_locs", node_n, &tc->node_cache_locs, err_msg, true)) {
        return false;
    }

    if (!read_i32_array(kv, "expected_kv_indices", kv_n, &tc->expected_kv_indices, err_msg, true)) {
        return false;
    }
    std::vector<bool> mask_tmp;
    if (!read_bool_array(kv, "expected_kv_mask", kv_n, &mask_tmp, err_msg, true)) {
        return false;
    }
    tc->expected_kv_mask = BoolBuffer(kv_n);
    for (size_t i = 0; i < kv_n; ++i) {
        tc->expected_kv_mask[i] = mask_tmp[i];
    }
    if (!read_int_array(kv, "expected_kv_lens", kv_lens_n, &tc->expected_kv_lens, err_msg, true)) {
        return false;
    }

    // Optional ancestor golden.
    if (has_key(kv, "expected_ancestor_node_ids")) {
        const size_t anc_n = static_cast<size_t>(tc->batch_size) * tc->max_tree_width *
                             tmac::hls::kCdtControllerMaxDepth;
        if (!read_i64_array(kv, "expected_ancestor_node_ids", anc_n, &tc->expected_ancestor_node_ids,
                            err_msg, true)) {
            return false;
        }
    }

    return true;
}

static bool run_file_case(const ControllerKvCase& tc, bool dry_run) {
    const size_t kv_n = static_cast<size_t>(tc.batch_size) * tc.max_tree_width * tc.max_input_size;
    const size_t kv_lens_n = static_cast<size_t>(tc.batch_size) * tc.max_tree_width;
    std::vector<int32_t> kv_indices(kv_n, -1);
    BoolBuffer kv_mask(kv_n);
    std::vector<int> kv_lens(kv_lens_n, 0);

    std::vector<int64_t> ancestor;
    int64_t* anc_ptr = nullptr;
    if (!tc.expected_ancestor_node_ids.empty()) {
        ancestor.assign(static_cast<size_t>(tc.batch_size) * tc.max_tree_width *
                            tmac::hls::kCdtControllerMaxDepth,
                        -1);
        anc_ptr = ancestor.data();
    }

    if (dry_run) {
        std::cout << "[DRY-RUN] Parsed controller case successfully.\n";
        std::cout << "[DRY-RUN] dims(B,width,max_tree,max_input,max_node)= " << tc.batch_size
                  << "," << tc.width << "," << tc.max_tree_width << "," << tc.max_input_size
                  << "," << tc.max_node_count << "\n";
        return true;
    }

    tmac::hls::cdt_controller_build_parent_visible_kv(
        tc.frontier_node_ids.data(),
        tc.prefix_kv_locs.data(),
        tc.prefix_lens.data(),
        tc.batch_size,
        tc.width,
        tc.max_tree_width,
        tc.max_prefix_len,
        tc.max_input_size,
        tc.max_node_count,
        tc.node_parent_ids.data(),
        tc.node_cache_locs.data(),
        kv_indices.data(),
        kv_mask.data(),
        kv_lens.data(),
        anc_ptr);

    bool ok = true;
    for (size_t i = 0; i < kv_n; ++i) {
        if (kv_indices[i] != tc.expected_kv_indices[i]) {
            std::cerr << "[FAIL] kv_indices[" << i << "]: got=" << kv_indices[i]
                      << " expected=" << tc.expected_kv_indices[i] << "\n";
            ok = false;
            break;
        }
    }
    for (size_t i = 0; i < kv_n; ++i) {
        if (kv_mask[i] != tc.expected_kv_mask[i]) {
            std::cerr << "[FAIL] kv_mask[" << i << "]: got=" << static_cast<int>(kv_mask[i])
                      << " expected=" << static_cast<int>(tc.expected_kv_mask[i]) << "\n";
            ok = false;
            break;
        }
    }
    for (size_t i = 0; i < kv_lens_n; ++i) {
        if (kv_lens[i] != tc.expected_kv_lens[i]) {
            std::cerr << "[FAIL] kv_lens[" << i << "]: got=" << kv_lens[i]
                      << " expected=" << tc.expected_kv_lens[i] << "\n";
            ok = false;
            break;
        }
    }

    if (ok && !tc.expected_ancestor_node_ids.empty()) {
        for (size_t i = 0; i < tc.expected_ancestor_node_ids.size(); ++i) {
            if (ancestor[i] != tc.expected_ancestor_node_ids[i]) {
                std::cerr << "[FAIL] ancestor_node_ids[" << i << "]: got=" << ancestor[i]
                          << " expected=" << tc.expected_ancestor_node_ids[i] << "\n";
                ok = false;
                break;
            }
        }
    }

    if (!ok) {
        return false;
    }

    std::cout << "[PASS] cost_draft_tree controller case-file check passed.\n";
    return true;
}

static int run_synthetic_smoke() {
    constexpr int B = 1;
    constexpr int MAX_TREE_WIDTH = 4;
    constexpr int MAX_NODE_COUNT = 64;
    constexpr int MAX_PREFIX = 8;
    constexpr int MAX_INPUT = 16;

    std::vector<int> node_count(B, 0);
    std::vector<int64_t> frontier(B * MAX_TREE_WIDTH, -1);
    std::vector<int64_t> frontier_next(B * MAX_TREE_WIDTH, -1);

    std::vector<int64_t> node_token(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_parent(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_first_child(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_last_child(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_next_sibling(B * MAX_NODE_COUNT, -1);
    std::vector<int64_t> node_depth(B * MAX_NODE_COUNT, -1);
    std::vector<int32_t> node_cache_loc(B * MAX_NODE_COUNT, -1);

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
        node_depth.data(),
        node_cache_loc.data());

    bool ok = true;

    // Layer 1 seed.
    const int width1 = 4;
    const std::vector<int64_t> seed_tokens = {101, 102, 103, 104};
    const std::vector<int32_t> seed_caches = {1001, 1002, 1003, 1004};

    tmac::hls::cdt_controller_seed_frontier(
        seed_tokens.data(),
        seed_caches.data(),
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
        node_depth.data(),
        node_cache_loc.data());

    ok &= check_eq_i32("node_count_after_seed", node_count[0], 4);

    for (int i = 0; i < width1; ++i) {
        ok &= check_eq_i64("seed_frontier_id", frontier[i], i);
        ok &= check_eq_i64("seed_token", node_token[i], seed_tokens[static_cast<size_t>(i)]);
        ok &= check_eq_i64("seed_parent", node_parent[i], -1);
        ok &= check_eq_i64("seed_depth", node_depth[i], 0);
        ok &= check_eq_i32("seed_cache", node_cache_loc[i], seed_caches[static_cast<size_t>(i)]);
    }

    // Layer 2 expansion.
    const int width2 = 4;
    const int parent_width2 = 4;
    const std::vector<int64_t> l2_parent_slots = {0, 0, 2, 1};
    const std::vector<int64_t> l2_tokens = {201, 202, 203, 204};
    const std::vector<int32_t> l2_caches = {1101, 1102, 1103, 1104};

    tmac::hls::cdt_controller_expand_frontier(
        frontier.data(),
        l2_parent_slots.data(),
        l2_tokens.data(),
        l2_caches.data(),
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
        node_depth.data(),
        node_cache_loc.data());

    ok &= check_eq_i32("node_count_after_l2", node_count[0], 8);

    const std::vector<int64_t> expected_l2_ids = {4, 5, 6, 7};
    const std::vector<int64_t> expected_l2_parents = {0, 0, 2, 1};

    for (int i = 0; i < width2; ++i) {
        ok &= check_eq_i64("l2_frontier_id", frontier_next[i], expected_l2_ids[static_cast<size_t>(i)]);
        const int nid = static_cast<int>(frontier_next[i]);
        ok &= check_eq_i64("l2_token", node_token[nid], l2_tokens[static_cast<size_t>(i)]);
        ok &= check_eq_i64("l2_parent", node_parent[nid], expected_l2_parents[static_cast<size_t>(i)]);
        ok &= check_eq_i64("l2_depth", node_depth[nid], 1);
        ok &= check_eq_i32("l2_cache", node_cache_loc[nid], l2_caches[static_cast<size_t>(i)]);
    }

    // Validate child linking.
    ok &= check_eq_i64("node0_first_child", node_first_child[0], 4);
    ok &= check_eq_i64("node0_last_child", node_last_child[0], 5);
    ok &= check_eq_i64("node4_next_sibling", node_next_sibling[4], 5);
    ok &= check_eq_i64("node2_first_child", node_first_child[2], 6);
    ok &= check_eq_i64("node1_first_child", node_first_child[1], 7);

    // Build parent-visible KV lists for layer2 frontier.
    const std::vector<int32_t> prefix_kv = {11, 12, 13, 0, 0, 0, 0, 0};
    std::vector<int> prefix_lens = {3};

    std::vector<int32_t> kv_indices(B * MAX_TREE_WIDTH * MAX_INPUT, -1);
    BoolBuffer kv_mask(B * MAX_TREE_WIDTH * MAX_INPUT);
    std::vector<int> kv_lens(B * MAX_TREE_WIDTH, 0);
    std::vector<int64_t> ancestor_ids(B * MAX_TREE_WIDTH * tmac::hls::kCdtControllerMaxDepth, -1);

    tmac::hls::cdt_controller_build_parent_visible_kv(
        frontier_next.data(),
        prefix_kv.data(),
        prefix_lens.data(),
        B,
        width2,
        MAX_TREE_WIDTH,
        MAX_PREFIX,
        MAX_INPUT,
        MAX_NODE_COUNT,
        node_parent.data(),
        node_cache_loc.data(),
        kv_indices.data(),
        kv_mask.data(),
        kv_lens.data(),
        ancestor_ids.data());

    ok &= check_list("l2_q0", kv_indices.data() + 0 * MAX_INPUT, kv_mask.data() + 0 * MAX_INPUT,
                     kv_lens[0], {11, 12, 13, 1001, 1101}, MAX_INPUT);
    ok &= check_list("l2_q1", kv_indices.data() + 1 * MAX_INPUT, kv_mask.data() + 1 * MAX_INPUT,
                     kv_lens[1], {11, 12, 13, 1001, 1102}, MAX_INPUT);
    ok &= check_list("l2_q2", kv_indices.data() + 2 * MAX_INPUT, kv_mask.data() + 2 * MAX_INPUT,
                     kv_lens[2], {11, 12, 13, 1003, 1103}, MAX_INPUT);
    ok &= check_list("l2_q3", kv_indices.data() + 3 * MAX_INPUT, kv_mask.data() + 3 * MAX_INPUT,
                     kv_lens[3], {11, 12, 13, 1002, 1104}, MAX_INPUT);

    // Layer 3 expansion from layer2 frontier.
    const std::vector<int64_t> l3_parent_slots = {1, 1, 3, 0};
    const std::vector<int64_t> l3_tokens = {301, 302, 303, 304};
    const std::vector<int32_t> l3_caches = {1201, 1202, 1203, 1204};

    std::vector<int64_t> frontier_l3(B * MAX_TREE_WIDTH, -1);

    tmac::hls::cdt_controller_expand_frontier(
        frontier_next.data(),
        l3_parent_slots.data(),
        l3_tokens.data(),
        l3_caches.data(),
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
        node_depth.data(),
        node_cache_loc.data());

    ok &= check_eq_i32("node_count_after_l3", node_count[0], 12);

    // Build parent-visible KV lists for layer3 frontier.
    std::fill(kv_indices.begin(), kv_indices.end(), -1);
    for (size_t i = 0; i < kv_mask.size(); ++i) kv_mask[i] = false;
    std::fill(kv_lens.begin(), kv_lens.end(), 0);

    tmac::hls::cdt_controller_build_parent_visible_kv(
        frontier_l3.data(),
        prefix_kv.data(),
        prefix_lens.data(),
        B,
        width2,
        MAX_TREE_WIDTH,
        MAX_PREFIX,
        MAX_INPUT,
        MAX_NODE_COUNT,
        node_parent.data(),
        node_cache_loc.data(),
        kv_indices.data(),
        kv_mask.data(),
        kv_lens.data(),
        nullptr);

    ok &= check_list("l3_q0", kv_indices.data() + 0 * MAX_INPUT, kv_mask.data() + 0 * MAX_INPUT,
                     kv_lens[0], {11, 12, 13, 1001, 1102, 1201}, MAX_INPUT);
    ok &= check_list("l3_q1", kv_indices.data() + 1 * MAX_INPUT, kv_mask.data() + 1 * MAX_INPUT,
                     kv_lens[1], {11, 12, 13, 1001, 1102, 1202}, MAX_INPUT);
    ok &= check_list("l3_q2", kv_indices.data() + 2 * MAX_INPUT, kv_mask.data() + 2 * MAX_INPUT,
                     kv_lens[2], {11, 12, 13, 1002, 1104, 1203}, MAX_INPUT);
    ok &= check_list("l3_q3", kv_indices.data() + 3 * MAX_INPUT, kv_mask.data() + 3 * MAX_INPUT,
                     kv_lens[3], {11, 12, 13, 1001, 1101, 1204}, MAX_INPUT);

    if (!ok) {
        std::cerr << "[FAIL] cost_draft_tree controller HLS smoke failed.\n";
        return 1;
    }

    std::cout << "[PASS] cost_draft_tree controller HLS smoke passed.\n";
    return 0;
}

int main(int argc, char** argv) {
    CliOptions opts;
    std::string err_msg;
    if (!parse_cli(argc, argv, &opts, &err_msg)) {
        if (!err_msg.empty()) {
            std::cerr << "[FAIL] " << err_msg << "\n";
            return 1;
        }
        return 0;
    }

    if (opts.case_file.empty()) {
        return run_synthetic_smoke();
    }

    ControllerKvCase tc;
    if (!load_controller_kv_case(opts.case_file, &tc, &err_msg)) {
        std::cerr << "[FAIL] " << err_msg << "\n";
        return 1;
    }

    if (!run_file_case(tc, opts.dry_run)) {
        std::cerr << "[FAIL] cost_draft_tree controller case-file check failed.\n";
        return 1;
    }
    return 0;
}
