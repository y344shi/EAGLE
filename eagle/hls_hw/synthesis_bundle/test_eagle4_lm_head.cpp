#include "eagle4_lm_head_hls.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

template <typename T>
std::vector<T> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> out(static_cast<size_t>(sz / static_cast<std::streamsize>(sizeof(T))));
    if (!f.read(reinterpret_cast<char*>(out.data()), sz)) return {};
    return out;
}

size_t file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return static_cast<size_t>(f.tellg());
}

void print_required(const std::string& tensor_dir, const std::string& lm_dir) {
    std::cout << "Required files for test_eagle4_lm_head:\n";
    std::cout << "  " << tensor_dir << "/tensor_130_EAGLE_LM_input_hidden.bin\n";
    std::cout << "  " << tensor_dir << "/tensor_131_EAGLE_LM_low_rank.bin\n";
    std::cout << "  " << tensor_dir << "/tensor_132_EAGLE_LM_candidate_logits.bin\n";
    std::cout << "  " << tensor_dir << "/tensor_133_EAGLE_LM_candidate_indices.bin\n";
    std::cout << "  " << tensor_dir << "/tensor_134_EAGLE_LM_gathered_logits.bin\n";
    std::cout << "  " << lm_dir << "/efficient_lm_head_down_proj_weight.fp16.bin\n";
    std::cout << "  " << lm_dir << "/efficient_lm_head_qweight_row_major.bin\n";
    std::cout << "  " << lm_dir << "/efficient_lm_head_scales_row_major.bin\n";
    std::cout << "  " << lm_dir << "/efficient_lm_head_qzeros.bin (optional; uses zero-point=8 if missing)\n";
    std::cout << "  " << lm_dir << "/efficient_lm_head_g_idx.bin (optional; uses contiguous groups if missing)\n";
    std::cout << "  " << lm_dir << "/lm_head_weight.fp16.bin\n";
}

void compute_diff(const std::vector<float>& got, const std::vector<float>& ref, float* max_abs, float* mean_abs) {
    *max_abs = 0.0f;
    *mean_abs = 0.0f;
    if (got.empty() || ref.empty()) return;
    const size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        if (d > *max_abs) *max_abs = d;
        *mean_abs += d;
    }
    *mean_abs /= static_cast<float>(n);
}

size_t set_mismatch_count(const std::vector<int>& a, const std::vector<int>& b) {
    std::unordered_set<int> sa(a.begin(), a.end());
    std::unordered_set<int> sb(b.begin(), b.end());
    size_t miss = 0;
    for (int x : sa) {
        if (sb.find(x) == sb.end()) miss++;
    }
    for (int x : sb) {
        if (sa.find(x) == sa.end()) miss++;
    }
    return miss;
}

bool gather_dot_from_lm_head_file(
    const std::string& lm_head_path,
    const std::vector<float>& hidden,           // [hidden_dim]
    const std::vector<int>& candidate_indices,  // [num_candidates]
    int hidden_dim,
    int vocab,
    std::vector<float>* gathered_out) {
    std::ifstream f(lm_head_path, std::ios::binary);
    if (!f) return false;

    std::vector<uint16_t> row_fp16(static_cast<size_t>(hidden_dim));
    gathered_out->assign(candidate_indices.size(), 0.0f);

    for (size_t i = 0; i < candidate_indices.size(); ++i) {
        const int tok = candidate_indices[i];
        if (tok < 0 || tok >= vocab) return false;
        const size_t off = static_cast<size_t>(tok) * static_cast<size_t>(hidden_dim) * sizeof(uint16_t);
        f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        if (!f.read(reinterpret_cast<char*>(row_fp16.data()),
                    static_cast<std::streamsize>(row_fp16.size() * sizeof(uint16_t)))) {
            return false;
        }
        float acc = 0.0f;
        for (int h = 0; h < hidden_dim; ++h) {
            acc += hidden[h] * tmac::hls::eagle4_fp16_to_float(row_fp16[static_cast<size_t>(h)]);
        }
        (*gathered_out)[i] = acc;
    }
    return true;
}

int run_smoke(int seed) {
    std::mt19937 rng(seed);
    constexpr int hidden = 64;
    constexpr int rank = 32;
    constexpr int vocab = 256;
    constexpr int topk = 16;
    constexpr int group = 16;

    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::uniform_int_distribution<int> qdist(0, 15);

    std::vector<float> hidden_v(hidden);
    for (float& x : hidden_v) x = dist(rng);

    std::vector<uint16_t> down_proj(static_cast<size_t>(rank) * hidden);
    for (uint16_t& x : down_proj) {
        const float v = dist(rng);
        uint32_t f32;
        std::memcpy(&f32, &v, sizeof(float));
        const uint32_t sign = (f32 >> 31) & 0x1u;
        int exp = static_cast<int>((f32 >> 23) & 0xFFu) - 127 + 15;
        uint32_t mant = (f32 >> 13) & 0x3FFu;
        if (exp <= 0) {
            x = static_cast<uint16_t>(sign << 15);
        } else if (exp >= 31) {
            x = static_cast<uint16_t>((sign << 15) | (31u << 10));
        } else {
            x = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(exp) << 10) | mant);
        }
    }

    const int packs = rank / 8;
    const int groups = rank / group;
    const int vocab_pack = vocab / 8;
    std::vector<int32_t> qweight(static_cast<size_t>(vocab) * packs, 0);
    std::vector<uint16_t> scales(static_cast<size_t>(groups) * vocab, 0);
    std::vector<int32_t> qzeros(static_cast<size_t>(groups) * vocab_pack, 0);

    for (int g = 0; g < groups; ++g) {
        for (int o = 0; o < vocab; ++o) {
            const float s = std::fabs(dist(rng)) + 1e-3f;
            uint32_t f32;
            std::memcpy(&f32, &s, sizeof(float));
            const uint16_t h =
                static_cast<uint16_t>(((f32 >> 31) & 0x1u) << 15 | ((((f32 >> 23) & 0xFFu) - 127 + 15) << 10) |
                                      ((f32 >> 13) & 0x3FFu));
            scales[static_cast<size_t>(g) * vocab + o] = h;
        }
    }
    for (size_t i = 0; i < qweight.size(); ++i) {
        int32_t pack = 0;
        for (int j = 0; j < 8; ++j) {
            pack |= (qdist(rng) & 0xF) << (j * 4);
        }
        qweight[i] = pack;
    }
    for (size_t i = 0; i < qzeros.size(); ++i) {
        int32_t pack = 0;
        for (int j = 0; j < 8; ++j) {
            pack |= (8 & 0xF) << (j * 4);
        }
        qzeros[i] = pack;
    }

    std::vector<float> low_rank(rank, 0.0f);
    std::vector<float> logits(vocab, 0.0f);
    std::vector<int> topk_idx(topk, -1);
    std::vector<float> topk_scores(topk, -std::numeric_limits<float>::infinity());
    tmac::hls::eagle4_lm_down_project(hidden_v.data(), down_proj.data(), low_rank.data(), hidden, rank);
    tmac::hls::eagle4_lm_candidate_logits_row4(
        low_rank.data(), qweight.data(), scales.data(), qzeros.data(), nullptr, rank, vocab, group, logits.data(), topk,
        topk_idx.data(), topk_scores.data());

    bool ok = true;
    for (float x : low_rank) ok &= std::isfinite(x);
    for (float x : logits) ok &= std::isfinite(x);
    for (int x : topk_idx) ok &= (x >= 0 && x < vocab);
    if (!ok) {
        std::cout << "[smoke] FAIL: non-finite or invalid index.\n";
        return 1;
    }
    std::cout << "[smoke] PASS (EAGLE4 LM-head core kernels emitted finite outputs).\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string tensor_dir = "../eagle_verified_pipeline_4bit/cpmcu_tensors";
    std::string lm_dir = "../eagle_verified_pipeline_4bit/hls_4bit/lm_head";
    int hidden_dim = 4096;
    int group_size = 128;
    int token_idx = -1;
    bool list_required = false;
    bool smoke = false;
    int smoke_seed = 7;
    float tol_low_rank = 0.02f;
    float tol_candidate = 0.2f;
    float tol_gather = 0.1f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tensor-dir" && i + 1 < argc) tensor_dir = argv[++i];
        else if (arg == "--lm-dir" && i + 1 < argc) lm_dir = argv[++i];
        else if (arg == "--hidden-dim" && i + 1 < argc) hidden_dim = std::atoi(argv[++i]);
        else if (arg == "--group-size" && i + 1 < argc) group_size = std::atoi(argv[++i]);
        else if (arg == "--token-idx" && i + 1 < argc) token_idx = std::atoi(argv[++i]);
        else if (arg == "--tol-low-rank" && i + 1 < argc) tol_low_rank = std::atof(argv[++i]);
        else if (arg == "--tol-candidate" && i + 1 < argc) tol_candidate = std::atof(argv[++i]);
        else if (arg == "--tol-gather" && i + 1 < argc) tol_gather = std::atof(argv[++i]);
        else if (arg == "--smoke") smoke = true;
        else if (arg == "--smoke-seed" && i + 1 < argc) smoke_seed = std::atoi(argv[++i]);
        else if (arg == "--list-required-goldens") list_required = true;
        else if (arg.rfind("--tensor-dir=", 0) == 0) tensor_dir = arg.substr(13);
        else if (arg.rfind("--lm-dir=", 0) == 0) lm_dir = arg.substr(9);
    }

    if (list_required) {
        print_required(tensor_dir, lm_dir);
        if (!smoke) return 0;
    }

    if (smoke) return run_smoke(smoke_seed);

    const std::string t130 = tensor_dir + "/tensor_130_EAGLE_LM_input_hidden.bin";
    const std::string t131 = tensor_dir + "/tensor_131_EAGLE_LM_low_rank.bin";
    const std::string t132 = tensor_dir + "/tensor_132_EAGLE_LM_candidate_logits.bin";
    const std::string t133 = tensor_dir + "/tensor_133_EAGLE_LM_candidate_indices.bin";
    const std::string t134 = tensor_dir + "/tensor_134_EAGLE_LM_gathered_logits.bin";

    const std::string w_down = lm_dir + "/efficient_lm_head_down_proj_weight.fp16.bin";
    const std::string w_qrow = lm_dir + "/efficient_lm_head_qweight_row_major.bin";
    const std::string w_srow = lm_dir + "/efficient_lm_head_scales_row_major.bin";
    const std::string w_qzeros = lm_dir + "/efficient_lm_head_qzeros.bin";
    const std::string w_gidx = lm_dir + "/efficient_lm_head_g_idx.bin";
    const std::string w_lm = lm_dir + "/lm_head_weight.fp16.bin";

    auto hidden_fp16 = load_bin<uint16_t>(t130);
    auto low_rank_ref_fp16 = load_bin<uint16_t>(t131);
    auto candidate_ref_fp16 = load_bin<uint16_t>(t132);
    auto candidate_idx_ref = load_bin<int32_t>(t133);
    auto gathered_ref_fp16 = load_bin<uint16_t>(t134);

    auto down_proj_fp16 = load_bin<uint16_t>(w_down);
    auto qweight_row = load_bin<int32_t>(w_qrow);
    auto scales_row_fp16 = load_bin<uint16_t>(w_srow);
    auto qzeros = load_bin<int32_t>(w_qzeros);
    auto g_idx = load_bin<int32_t>(w_gidx);

    if (hidden_fp16.empty() || low_rank_ref_fp16.empty() || candidate_ref_fp16.empty() || candidate_idx_ref.empty() ||
        gathered_ref_fp16.empty() || down_proj_fp16.empty() || qweight_row.empty() || scales_row_fp16.empty()) {
        std::cout << "[FAIL] Missing required tensors/weights.\n";
        print_required(tensor_dir, lm_dir);
        return 1;
    }

    if (hidden_dim <= 0 || (hidden_fp16.size() % static_cast<size_t>(hidden_dim)) != 0) {
        std::cout << "[FAIL] tensor_130 shape incompatible with hidden_dim=" << hidden_dim << "\n";
        return 1;
    }

    const int tokens = static_cast<int>(hidden_fp16.size() / static_cast<size_t>(hidden_dim));
    const int rank = static_cast<int>(low_rank_ref_fp16.size() / static_cast<size_t>(tokens));
    const int vocab = static_cast<int>(candidate_ref_fp16.size() / static_cast<size_t>(tokens));
    const int num_candidates = static_cast<int>(candidate_idx_ref.size() / static_cast<size_t>(tokens));

    if (rank <= 0 || vocab <= 0 || num_candidates <= 0) {
        std::cout << "[FAIL] Invalid inferred LM dimensions.\n";
        return 1;
    }
    if (static_cast<size_t>(rank) * static_cast<size_t>(hidden_dim) != down_proj_fp16.size()) {
        std::cout << "[FAIL] down_proj weight shape mismatch.\n";
        return 1;
    }
    if ((rank % 8) != 0 || (rank % group_size) != 0) {
        std::cout << "[FAIL] rank must be divisible by 8 and group_size.\n";
        return 1;
    }
    const int rank_packs = rank / 8;
    const int rank_groups = rank / group_size;
    if (qweight_row.size() != static_cast<size_t>(vocab) * static_cast<size_t>(rank_packs)) {
        std::cout << "[FAIL] qweight row-major size mismatch.\n";
        return 1;
    }
    if (scales_row_fp16.size() != static_cast<size_t>(rank_groups) * static_cast<size_t>(vocab)) {
        std::cout << "[FAIL] scales row-major size mismatch.\n";
        return 1;
    }
    if (gathered_ref_fp16.size() != static_cast<size_t>(tokens) * static_cast<size_t>(num_candidates)) {
        std::cout << "[FAIL] gathered logits tensor size mismatch.\n";
        return 1;
    }

    const size_t lm_head_bytes = file_bytes(w_lm);
    if (lm_head_bytes == 0 || (lm_head_bytes % (static_cast<size_t>(hidden_dim) * sizeof(uint16_t))) != 0) {
        std::cout << "[FAIL] lm_head_weight file size mismatch.\n";
        return 1;
    }
    const int lm_vocab = static_cast<int>(lm_head_bytes / (static_cast<size_t>(hidden_dim) * sizeof(uint16_t)));
    if (lm_vocab != vocab) {
        std::cout << "[warn] vocab mismatch: tensor_132=" << vocab << " lm_head=" << lm_vocab << "\n";
    }

    const int use_tok = (token_idx >= 0) ? token_idx : (tokens - 1);
    if (use_tok < 0 || use_tok >= tokens) {
        std::cout << "[FAIL] token index out of range.\n";
        return 1;
    }

    std::vector<float> hidden(hidden_dim, 0.0f);
    for (int i = 0; i < hidden_dim; ++i) {
        hidden[i] = tmac::hls::eagle4_fp16_to_float(hidden_fp16[static_cast<size_t>(use_tok) * hidden_dim + i]);
    }

    std::vector<float> low_rank_ref(rank, 0.0f);
    for (int i = 0; i < rank; ++i) {
        low_rank_ref[i] =
            tmac::hls::eagle4_fp16_to_float(low_rank_ref_fp16[static_cast<size_t>(use_tok) * rank + i]);
    }

    std::vector<float> candidate_ref(vocab, 0.0f);
    for (int i = 0; i < vocab; ++i) {
        candidate_ref[i] =
            tmac::hls::eagle4_fp16_to_float(candidate_ref_fp16[static_cast<size_t>(use_tok) * vocab + i]);
    }

    std::vector<int> candidate_idx_gold(num_candidates, 0);
    for (int i = 0; i < num_candidates; ++i) {
        candidate_idx_gold[i] = candidate_idx_ref[static_cast<size_t>(use_tok) * num_candidates + i];
    }

    std::vector<float> gathered_ref(num_candidates, 0.0f);
    for (int i = 0; i < num_candidates; ++i) {
        gathered_ref[i] =
            tmac::hls::eagle4_fp16_to_float(gathered_ref_fp16[static_cast<size_t>(use_tok) * num_candidates + i]);
    }

    const int vocab_packed = (vocab + 7) / 8;
    const bool has_qzeros = (qzeros.size() == static_cast<size_t>(rank_groups) * static_cast<size_t>(vocab_packed));
    const int32_t* qzeros_ptr = has_qzeros ? qzeros.data() : nullptr;
    const bool has_gidx = (g_idx.size() == static_cast<size_t>(rank));
    const int32_t* gidx_ptr = has_gidx ? g_idx.data() : nullptr;

    std::vector<float> low_rank(rank, 0.0f);
    std::vector<float> candidate_logits(vocab, 0.0f);
    std::vector<int> topk_idx(num_candidates, -1);
    std::vector<float> topk_scores(num_candidates, -std::numeric_limits<float>::infinity());

    tmac::hls::eagle4_lm_down_project(hidden.data(), down_proj_fp16.data(), low_rank.data(), hidden_dim, rank);
    tmac::hls::eagle4_lm_candidate_logits_row4(
        low_rank.data(), qweight_row.data(), scales_row_fp16.data(), qzeros_ptr, gidx_ptr, rank, vocab, group_size,
        candidate_logits.data(), num_candidates, topk_idx.data(), topk_scores.data());

    std::vector<float> gathered_from_file;
    if (!gather_dot_from_lm_head_file(
            w_lm, hidden, candidate_idx_gold, hidden_dim, lm_vocab, &gathered_from_file)) {
        std::cout << "[FAIL] failed gather-dot from lm_head file.\n";
        return 1;
    }

    float low_rank_max = 0.0f, low_rank_mean = 0.0f;
    float cand_max = 0.0f, cand_mean = 0.0f;
    float gather_max = 0.0f, gather_mean = 0.0f;
    compute_diff(low_rank, low_rank_ref, &low_rank_max, &low_rank_mean);
    compute_diff(candidate_logits, candidate_ref, &cand_max, &cand_mean);
    compute_diff(gathered_from_file, gathered_ref, &gather_max, &gather_mean);

    std::cout << "[result] token_idx=" << use_tok << " hidden=" << hidden_dim << " rank=" << rank
              << " vocab=" << vocab << " topk=" << num_candidates << "\n";
    std::cout << "[result] low_rank  max_abs=" << low_rank_max << " mean_abs=" << low_rank_mean << "\n";
    std::cout << "[result] cand_full max_abs=" << cand_max << " mean_abs=" << cand_mean << "\n";
    std::cout << "[result] gather    max_abs=" << gather_max << " mean_abs=" << gather_mean << "\n";

    const size_t topk_set_miss = set_mismatch_count(topk_idx, candidate_idx_gold);
    std::cout << "[result] topk_set_symmetric_diff=" << topk_set_miss << "\n";
    if (!has_qzeros) {
        std::cout << "[warn] efficient_lm_head_qzeros.bin missing or shape-mismatched; used zero-point=8.\n";
    }
    if (!has_gidx) {
        std::cout << "[warn] efficient_lm_head_g_idx.bin missing or shape-mismatched; used contiguous groups.\n";
    }

    bool pass = true;
    if (low_rank_max > tol_low_rank) {
        std::cout << "[FAIL] low_rank exceeds tolerance " << tol_low_rank << "\n";
        pass = false;
    }
    if (cand_max > tol_candidate) {
        std::cout << "[FAIL] candidate logits exceed tolerance " << tol_candidate << "\n";
        pass = false;
    }
    if (gather_max > tol_gather) {
        std::cout << "[FAIL] gathered logits exceed tolerance " << tol_gather << "\n";
        pass = false;
    }

    if (!pass) return 1;
    std::cout << "[PASS] EAGLE4 LM-head file-driven checks passed.\n";
    return 0;
}
