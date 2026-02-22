#include "fused_online_attention_pwl.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using tmac::hls::VEC_W;
using tmac::hls::hls_stream;
using tmac::hls::fused_online_attention_pwl;
using tmac::hls::vec_t;

namespace {

constexpr uint32_t kMagic = 0x53415454u; // "SATT"

struct InputHeader {
    uint32_t magic;
    int32_t head_dim;
    int32_t seq_len;
    int32_t padded_len;
};

bool read_exact(std::ifstream& ifs, void* dst, std::size_t bytes) {
    ifs.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<std::size_t>(ifs.gcount()) == bytes;
}

template <int HEAD_DIM>
int run_for_dim(const std::vector<float>& q,
                const std::vector<float>& k_hist,
                const std::vector<float>& v_hist,
                int seq_len,
                int padded_len,
                std::vector<float>& out) {
    hls_stream<vec_t<VEC_W>> q_stream("q_stream");
    hls_stream<vec_t<VEC_W>> k_stream("k_stream");
    hls_stream<vec_t<VEC_W>> v_stream("v_stream");
    hls_stream<vec_t<VEC_W>> o_stream("o_stream");

    for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
        vec_t<VEC_W> chunk{};
        for (int j = 0; j < VEC_W; ++j) chunk[j] = q[i * VEC_W + j];
        q_stream.write(chunk);
    }

    for (int t = 0; t < seq_len; ++t) {
        for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
            vec_t<VEC_W> k_chunk{};
            vec_t<VEC_W> v_chunk{};
            for (int j = 0; j < VEC_W; ++j) {
                const int idx = t * HEAD_DIM + i * VEC_W + j;
                k_chunk[j] = k_hist[idx];
                v_chunk[j] = v_hist[idx];
            }
            k_stream.write(k_chunk);
            v_stream.write(v_chunk);
        }
    }

    fused_online_attention_pwl<HEAD_DIM>(q_stream, k_stream, v_stream, o_stream, seq_len, padded_len);

    out.assign(HEAD_DIM, 0.0f);
    for (int i = 0; i < HEAD_DIM / VEC_W; ++i) {
        vec_t<VEC_W> chunk = o_stream.read();
        for (int j = 0; j < VEC_W; ++j) out[i * VEC_W + j] = chunk[j];
    }
    return 0;
}

int run_dispatch(int head_dim,
                 const std::vector<float>& q,
                 const std::vector<float>& k_hist,
                 const std::vector<float>& v_hist,
                 int seq_len,
                 int padded_len,
                 std::vector<float>& out) {
    switch (head_dim) {
    case 64:
        return run_for_dim<64>(q, k_hist, v_hist, seq_len, padded_len, out);
    case 128:
        return run_for_dim<128>(q, k_hist, v_hist, seq_len, padded_len, out);
    case 256:
        return run_for_dim<256>(q, k_hist, v_hist, seq_len, padded_len, out);
    default:
        std::cerr << "Unsupported head_dim=" << head_dim
                  << " (supported: 64, 128, 256)\n";
        return 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_bin> <output_bin>\n";
        return 1;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];

    std::ifstream ifs(in_path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Cannot open input: " << in_path << "\n";
        return 1;
    }

    InputHeader hdr{};
    if (!read_exact(ifs, &hdr, sizeof(hdr))) {
        std::cerr << "Failed to read input header\n";
        return 1;
    }
    if (hdr.magic != kMagic) {
        std::cerr << "Bad magic in input file\n";
        return 1;
    }
    if (hdr.head_dim <= 0 || hdr.seq_len <= 0) {
        std::cerr << "Invalid dimensions in input file\n";
        return 1;
    }
    if (hdr.head_dim % VEC_W != 0) {
        std::cerr << "head_dim must be divisible by " << VEC_W << "\n";
        return 1;
    }
    if (hdr.padded_len > 0 && hdr.padded_len < hdr.seq_len) {
        std::cerr << "padded_len must be >= seq_len when provided\n";
        return 1;
    }

    const std::size_t hd = static_cast<std::size_t>(hdr.head_dim);
    const std::size_t sl = static_cast<std::size_t>(hdr.seq_len);
    std::vector<float> q(hd);
    std::vector<float> k_hist(sl * hd);
    std::vector<float> v_hist(sl * hd);

    if (!read_exact(ifs, q.data(), q.size() * sizeof(float)) ||
        !read_exact(ifs, k_hist.data(), k_hist.size() * sizeof(float)) ||
        !read_exact(ifs, v_hist.data(), v_hist.size() * sizeof(float))) {
        std::cerr << "Failed to read full Q/K/V payload\n";
        return 1;
    }

    std::vector<float> out;
    const int rc = run_dispatch(hdr.head_dim, q, k_hist, v_hist, hdr.seq_len, hdr.padded_len, out);
    if (rc != 0) return rc;

    std::ofstream ofs(out_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Cannot open output: " << out_path << "\n";
        return 1;
    }
    ofs.write(reinterpret_cast<const char*>(out.data()),
              static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!ofs) {
        std::cerr << "Failed to write output data\n";
        return 1;
    }
    return 0;
}

