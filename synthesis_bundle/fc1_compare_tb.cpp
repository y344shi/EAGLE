#include "deep_pipeline_lutmac.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using tmac::hls::VEC_W;
using tmac::hls::dense_projection_production;
using tmac::hls::dense_projection_production_scaled;
using tmac::hls::dense_projection_production_scaled_raw;
using tmac::hls::hls_stream;
using tmac::hls::pack512;
using tmac::hls::vec_t;

// Comparison harness: CPU row-major decode vs HLS broadcast path using swizzled weights + scales.

constexpr int IN_DIM = 4096;
constexpr int OUT_DIM = 4096;
constexpr int OUT_W = 128;
constexpr int GROUP_SIZE = 128;
constexpr int SCALE_EXP = 0; // keep activations unscaled

static std::vector<uint32_t> load_qweights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("Cannot open %s\n", path.c_str());
        return {};
    }
    f.seekg(0, std::ios::end);
    const auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint32_t> buf(bytes / sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    return buf;
}

static int8_t decode_nibble(uint32_t word, int j) {
    const int val = (word >> (j * 4)) & 0xF;
    return static_cast<int8_t>(val - 8);
}

static void cpu_matvec_tile(const std::vector<uint32_t>& qw,
                            const std::vector<float>& scales,
                            const std::vector<float>& x,
                            int tile_idx,
                            std::vector<float>& out_tile) {
    const int in_packed = IN_DIM / 8; // 512
    const int tile_base = tile_idx * OUT_W;
    for (int o = 0; o < OUT_W; ++o) {
        const int global_o = tile_base + o;
        float acc = 0.0f;
        const uint32_t* row = qw.data() + static_cast<size_t>(global_o) * in_packed;
        for (int ip = 0; ip < in_packed; ++ip) {
            const uint32_t word = row[ip];
            const int base_idx = ip * 8;
            const int group = base_idx / GROUP_SIZE;
            const float scale = scales[static_cast<size_t>(group) * OUT_DIM + global_o];
            for (int j = 0; j < 8; ++j) {
                const int idx = base_idx + j;
                const float w = static_cast<float>(decode_nibble(word, j));
                acc += w * scale * x[idx];
            }
        }
        out_tile[o] = acc;
    }
}

#if defined(TMAC_ENABLE_FC1_COMPARE_TB)
int main(int argc, char** argv) {
    // Default paths expect repo layout: .../T-Mac/eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/...
    std::string qweight_path = "../../../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/fc1_qweight.bin";
    std::string swizzled_path = "weights_swizzled.bin";
    std::string scales_raw_path = "../../../eagle_verified_pipeline_4bit/hls_4bit/weights_all_4bit/fc1_scales.bin";
    int tile_idx = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--tile=", 0) == 0) {
            tile_idx = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--qweight=", 0) == 0) {
            qweight_path = arg.substr(10);
        } else if (arg.rfind("--swizzled=", 0) == 0) {
            swizzled_path = arg.substr(11);
        } else if (arg.rfind("--scales=", 0) == 0) {
            scales_raw_path = arg.substr(9);
        }
    }

    const auto qweights = load_qweights(qweight_path);
    if (qweights.empty()) {
        return 1;
    }
    const size_t expected_words = static_cast<size_t>(IN_DIM) * OUT_DIM / 8;
    if (qweights.size() != expected_words) {
        std::printf("Unexpected qweight size: %zu words (expected %zu)\n", qweights.size(),
                    expected_words);
        return 1;
    }

    std::ifstream f(swizzled_path, std::ios::binary);
    if (!f) {
        std::printf("Cannot open swizzled weights at %s\n", swizzled_path.c_str());
        return 1;
    }
    std::vector<uint8_t> w_bytes((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
    if (w_bytes.size() % sizeof(pack512) != 0) {
        std::printf("Swizzled file size not multiple of pack512\n");
        return 1;
    }
    const size_t packets = w_bytes.size() / sizeof(pack512);
    const size_t packets_per_tile = IN_DIM;
    const size_t tiles = packets / packets_per_tile;
    if (tile_idx < 0 || static_cast<size_t>(tile_idx) >= tiles) {
        std::printf("Tile %d out of range (0..%zu)\n", tile_idx, tiles - 1);
        return 1;
    }
    std::vector<uint16_t> scales_fp16;
    {
        std::ifstream sf(scales_raw_path, std::ios::binary);
        if (!sf) {
            std::printf("Cannot open raw scales at %s\n", scales_raw_path.c_str());
            return 1;
        }
        sf.seekg(0, std::ios::end);
        const auto bytes = static_cast<size_t>(sf.tellg());
        sf.seekg(0, std::ios::beg);
        if (bytes % sizeof(uint16_t) != 0) {
            std::printf("Scales file size not multiple of uint16\n");
            return 1;
        }
        scales_fp16.resize(bytes / sizeof(uint16_t));
        sf.read(reinterpret_cast<char*>(scales_fp16.data()), bytes);
    }
    std::vector<float> scales(scales_fp16.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        const uint16_t h = scales_fp16[i];
        const uint16_t sign = (h >> 15) & 0x1;
        uint16_t exp = (h >> 10) & 0x1F;
        uint16_t mant = h & 0x3FF;
        uint32_t f_bits;
        if (exp == 0) {
            if (mant == 0) {
                f_bits = static_cast<uint32_t>(sign) << 31;
            } else {
                exp = 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    --exp;
                }
                mant &= 0x3FF;
                f_bits = (static_cast<uint32_t>(sign) << 31) |
                         (static_cast<uint32_t>(exp + 127 - 15) << 23) |
                         (static_cast<uint32_t>(mant) << 13);
            }
        } else if (exp == 31) {
            f_bits = (static_cast<uint32_t>(sign) << 31) | 0x7F800000 |
                     (static_cast<uint32_t>(mant) << 13);
        } else {
            f_bits = (static_cast<uint32_t>(sign) << 31) |
                     (static_cast<uint32_t>(exp + 127 - 15) << 23) |
                     (static_cast<uint32_t>(mant) << 13);
        }
        float val;
        std::memcpy(&val, &f_bits, sizeof(float));
        scales[i] = val;
    }

    const pack512* weights_all = reinterpret_cast<const pack512*>(w_bytes.data());
    const pack512* weights_tile = weights_all + static_cast<size_t>(tile_idx) * packets_per_tile;

    // Deterministic activation vector.
    std::vector<float> x(IN_DIM);
    for (int i = 0; i < IN_DIM; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.001f);
    }

    // CPU reference (scale=1).
    std::vector<float> cpu_out(OUT_W);
    cpu_matvec_tile(qweights, scales, x, tile_idx, cpu_out);

    // HLS DUT with scales.
    hls_stream<vec_t<VEC_W>> ain, cout;
    for (int chunk = 0; chunk < IN_DIM / VEC_W; ++chunk) {
        vec_t<VEC_W> v{};
        for (int j = 0; j < VEC_W; ++j) {
            v[j] = x[chunk * VEC_W + j];
        }
        ain.write(v);
    }
    const int tile_base = tile_idx * OUT_W;
    dense_projection_production_scaled_raw<SCALE_EXP, IN_DIM, OUT_W, GROUP_SIZE, OUT_DIM>(
        ain, cout, weights_tile, scales.data(), tile_base);

    std::vector<float> dut_out(OUT_W);
    for (int oc = 0; oc < OUT_W / VEC_W; ++oc) {
        vec_t<VEC_W> v = cout.read();
        for (int j = 0; j < VEC_W; ++j) {
            dut_out[oc * VEC_W + j] = v[j];
        }
    }

    int errors = 0;
    for (int i = 0; i < OUT_W; ++i) {
        const float diff = std::fabs(cpu_out[i] - dut_out[i]);
        if (diff > 1e-3f) {
            if (errors < 8) {
                std::printf("Mismatch lane %d: got %f expected %f (diff=%f)\n",
                            i, dut_out[i], cpu_out[i], diff);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        std::printf("[compare tile %d] PASS (with scales)\n", tile_idx);
        return 0;
    }
    std::printf("[compare tile %d] FAIL with %d mismatches (with scales)\n", tile_idx, errors);
    return 1;
}
#endif
