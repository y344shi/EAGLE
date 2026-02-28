#include "cost_draft_tree_fused_wiring_hls.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

constexpr int kHeadDim = 16;
constexpr int kNumKvHeads = 2;
constexpr int kMaxTreeWidth = 4;
constexpr int kMaxInputSize = 8;
constexpr int kBatch = 2;
constexpr int kKvCacheTokens = 32;
constexpr int kVecsPerToken = (kNumKvHeads * kHeadDim) / tmac::hls::VEC_W;

struct BoolBuffer {
    explicit BoolBuffer(size_t n)
        : size(n), data(n ? new bool[n] : nullptr) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = false;
        }
    }

    size_t size = 0;
    std::unique_ptr<bool[]> data;
};

void fill_cache_buffer(std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>>& buf, float base_offset) {
    for (int t = 0; t < kKvCacheTokens; ++t) {
        for (int v = 0; v < kVecsPerToken; ++v) {
            tmac::hls::vec_t<tmac::hls::VEC_W> x;
            for (int lane = 0; lane < tmac::hls::VEC_W; ++lane) {
                x[lane] = base_offset + static_cast<float>(t * 1000 + v * 100 + lane);
            }
            buf[static_cast<size_t>(t * kVecsPerToken + v)] = x;
        }
    }
}

bool read_and_check_token(
    tmac::hls::hls_stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& s,
    int token_idx,
    float base_offset,
    const char* stream_name,
    int query_idx) {
    for (int v = 0; v < kVecsPerToken; ++v) {
        if (s.empty()) {
            std::cerr << "[FAIL] " << stream_name << " q" << query_idx
                      << " ended early while reading token " << token_idx << " vec " << v
                      << "\n";
            return false;
        }
        tmac::hls::vec_t<tmac::hls::VEC_W> got = s.read();
        for (int lane = 0; lane < tmac::hls::VEC_W; ++lane) {
            const float exp = base_offset + static_cast<float>(token_idx * 1000 + v * 100 + lane);
            const float err = std::fabs(got[lane] - exp);
            if (err > 1e-6f) {
                std::cerr << "[FAIL] " << stream_name << " q" << query_idx
                          << " token=" << token_idx << " vec=" << v << " lane=" << lane
                          << " got=" << got[lane] << " exp=" << exp << "\n";
                return false;
            }
        }
    }
    return true;
}

bool check_stream_empty(
    const tmac::hls::hls_stream<tmac::hls::vec_t<tmac::hls::VEC_W>>& s,
    const char* stream_name,
    int query_idx) {
    if (!s.empty()) {
        std::cerr << "[FAIL] " << stream_name << " q" << query_idx << " has unexpected tail data\n";
        return false;
    }
    return true;
}

void set_kv_entry(std::vector<int32_t>* kv_indices,
                  BoolBuffer* kv_mask,
                  int b,
                  int q,
                  int p,
                  int tok,
                  bool valid) {
    const int flat = ((b * kMaxTreeWidth + q) * kMaxInputSize + p);
    (*kv_indices)[static_cast<size_t>(flat)] = static_cast<int32_t>(tok);
    kv_mask->data[static_cast<size_t>(flat)] = valid;
}

bool run_batch_test(
    int batch_idx,
    int tree_width,
    const std::vector<int32_t>& kv_indices,
    const BoolBuffer& kv_mask,
    const std::vector<int>& kv_lens,
    const std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>>& hbm_k,
    const std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>>& hbm_v,
    const std::vector<std::vector<int>>& expected_tokens,
    const std::vector<int>& expected_lens) {

    tmac::hls::hls_stream<tmac::hls::vec_t<tmac::hls::VEC_W>> k_streams[kMaxTreeWidth];
    tmac::hls::hls_stream<tmac::hls::vec_t<tmac::hls::VEC_W>> v_streams[kMaxTreeWidth];
    int query_seq_lens[kMaxTreeWidth] = {0, 0, 0, 0};

    tmac::hls::cost_draft_tree_tree_kv_cache_gather_hls<
        kHeadDim, kNumKvHeads, kMaxTreeWidth, kMaxInputSize>(
        hbm_k.data(),
        hbm_v.data(),
        kv_indices.data(),
        kv_mask.data.get(),
        kv_lens.data(),
        batch_idx,
        kBatch,
        tree_width,
        kMaxTreeWidth,
        kMaxInputSize,
        kKvCacheTokens,
        k_streams,
        v_streams,
        query_seq_lens);

    for (int q = 0; q < kMaxTreeWidth; ++q) {
        if (query_seq_lens[q] != expected_lens[static_cast<size_t>(q)]) {
            std::cerr << "[FAIL] batch=" << batch_idx << " q" << q
                      << " seq_len got=" << query_seq_lens[q]
                      << " exp=" << expected_lens[static_cast<size_t>(q)] << "\n";
            return false;
        }

        for (int tok : expected_tokens[static_cast<size_t>(q)]) {
            if (!read_and_check_token(k_streams[q], tok, 0.0f, "K", q)) {
                return false;
            }
            if (!read_and_check_token(v_streams[q], tok, 50000.0f, "V", q)) {
                return false;
            }
        }

        if (!check_stream_empty(k_streams[q], "K", q)) {
            return false;
        }
        if (!check_stream_empty(v_streams[q], "V", q)) {
            return false;
        }
    }

    return true;
}

bool run_zero_batch_test(
    const std::vector<int32_t>& kv_indices,
    const BoolBuffer& kv_mask,
    const std::vector<int>& kv_lens,
    const std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>>& hbm_k,
    const std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>>& hbm_v) {

    tmac::hls::hls_stream<tmac::hls::vec_t<tmac::hls::VEC_W>> k_streams[kMaxTreeWidth];
    tmac::hls::hls_stream<tmac::hls::vec_t<tmac::hls::VEC_W>> v_streams[kMaxTreeWidth];
    int query_seq_lens[kMaxTreeWidth] = {7, 7, 7, 7};

    tmac::hls::cost_draft_tree_tree_kv_cache_gather_hls<
        kHeadDim, kNumKvHeads, kMaxTreeWidth, kMaxInputSize>(
        hbm_k.data(),
        hbm_v.data(),
        kv_indices.data(),
        kv_mask.data.get(),
        kv_lens.data(),
        0,
        0,
        kMaxTreeWidth,
        kMaxTreeWidth,
        kMaxInputSize,
        kKvCacheTokens,
        k_streams,
        v_streams,
        query_seq_lens);

    for (int q = 0; q < kMaxTreeWidth; ++q) {
        if (query_seq_lens[q] != 0) {
            std::cerr << "[FAIL] zero-batch q" << q << " seq_len got=" << query_seq_lens[q]
                      << " exp=0\n";
            return false;
        }
        if (!k_streams[q].empty() || !v_streams[q].empty()) {
            std::cerr << "[FAIL] zero-batch q" << q << " produced stream data\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const size_t cache_n = static_cast<size_t>(kKvCacheTokens) * kVecsPerToken;
    std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>> hbm_k(cache_n);
    std::vector<tmac::hls::vec_t<tmac::hls::VEC_W>> hbm_v(cache_n);

    fill_cache_buffer(hbm_k, 0.0f);
    fill_cache_buffer(hbm_v, 50000.0f);

    const size_t kv_n = static_cast<size_t>(kBatch) * kMaxTreeWidth * kMaxInputSize;
    std::vector<int32_t> kv_indices(kv_n, -1);
    BoolBuffer kv_mask(kv_n);
    std::vector<int> kv_lens(static_cast<size_t>(kBatch) * kMaxTreeWidth, 0);

    // Batch 0 setup.
    kv_lens[0 * kMaxTreeWidth + 0] = 5;
    set_kv_entry(&kv_indices, &kv_mask, 0, 0, 0, 2, true);
    set_kv_entry(&kv_indices, &kv_mask, 0, 0, 1, 4, false);
    set_kv_entry(&kv_indices, &kv_mask, 0, 0, 2, 6, true);
    set_kv_entry(&kv_indices, &kv_mask, 0, 0, 3, 33, true); // out-of-range -> dropped
    set_kv_entry(&kv_indices, &kv_mask, 0, 0, 4, 8, true);

    kv_lens[0 * kMaxTreeWidth + 1] = 4;
    set_kv_entry(&kv_indices, &kv_mask, 0, 1, 0, 1, true);
    set_kv_entry(&kv_indices, &kv_mask, 0, 1, 1, 3, true);
    set_kv_entry(&kv_indices, &kv_mask, 0, 1, 2, 5, true);
    set_kv_entry(&kv_indices, &kv_mask, 0, 1, 3, 7, true);

    kv_lens[0 * kMaxTreeWidth + 2] = 2;
    set_kv_entry(&kv_indices, &kv_mask, 0, 2, 0, 9, false);
    set_kv_entry(&kv_indices, &kv_mask, 0, 2, 1, 10, false);

    // Batch 1 setup.
    kv_lens[1 * kMaxTreeWidth + 0] = 3;
    set_kv_entry(&kv_indices, &kv_mask, 1, 0, 0, 11, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 0, 1, 12, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 0, 2, 13, true);

    kv_lens[1 * kMaxTreeWidth + 1] = 5;
    set_kv_entry(&kv_indices, &kv_mask, 1, 1, 0, 14, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 1, 1, 15, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 1, 2, 16, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 1, 3, 17, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 1, 4, 18, true);

    kv_lens[1 * kMaxTreeWidth + 2] = 1;
    set_kv_entry(&kv_indices, &kv_mask, 1, 2, 0, 19, true);

    kv_lens[1 * kMaxTreeWidth + 3] = 2;
    set_kv_entry(&kv_indices, &kv_mask, 1, 3, 0, -1, true);
    set_kv_entry(&kv_indices, &kv_mask, 1, 3, 1, 20, true);

    const std::vector<std::vector<int>> exp_tokens_b0 = {
        {2, 6, 8},
        {1, 3, 5, 7},
        {},
        {},
    };
    const std::vector<int> exp_lens_b0 = {3, 4, 0, 0};

    const std::vector<std::vector<int>> exp_tokens_b1 = {
        {11, 12, 13},
        {14, 15, 16, 17, 18},
        {19},
        {20},
    };
    const std::vector<int> exp_lens_b1 = {3, 5, 1, 1};

    if (!run_batch_test(0,
                        3,
                        kv_indices,
                        kv_mask,
                        kv_lens,
                        hbm_k,
                        hbm_v,
                        exp_tokens_b0,
                        exp_lens_b0)) {
        return 1;
    }

    if (!run_batch_test(1,
                        4,
                        kv_indices,
                        kv_mask,
                        kv_lens,
                        hbm_k,
                        hbm_v,
                        exp_tokens_b1,
                        exp_lens_b1)) {
        return 1;
    }

    if (!run_zero_batch_test(kv_indices, kv_mask, kv_lens, hbm_k, hbm_v)) {
        return 1;
    }

    std::cout << "[PASS] cost_draft_tree_kv_cache_tb\n";
    return 0;
}
