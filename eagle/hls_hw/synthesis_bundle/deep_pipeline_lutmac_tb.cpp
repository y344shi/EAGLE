#include "deep_pipeline_lutmac.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using tmac::hls::VEC_W;
using tmac::hls::dense_projection_production;
using tmac::hls::dense_projection_production_scaled;
using tmac::hls::lut_mac_broadcast;
using tmac::hls::pack512;
using tmac::hls::vec_t;
using tmac::hls::hls_stream;

constexpr int SCALE_EXP = 0; // keep activations unscaled for easy checking

// Build a pack512 with raw nibble encoding (stored as unsigned, decoded as raw-8).
static pack512 build_weights(const std::vector<int8_t>& w4) {
    pack512 p{};
    for (int i = 0; i < static_cast<int>(w4.size()); ++i) {
        const int8_t w_real = w4[i];
        const uint8_t raw = static_cast<uint8_t>(w_real + 8) & 0xF;
#ifdef __SYNTHESIS__
        const int bit = i * 4;
        p.range(bit + 3, bit) = raw;
#else
        const int byte_idx = i >> 1;
        const bool high = i & 1;
        if (high) {
            p.bytes[byte_idx] |= (raw << 4);
        } else {
            p.bytes[byte_idx] |= raw;
        }
#endif
    }
    return p;
}

// Scaled smoke test that exercises per-group scales and both T-MAC and DSP paths.
int run_smoke_scaled() {
    constexpr int INPUT_DIM = 128;
    constexpr int OUT_W = 128;
    constexpr int GROUP_SIZE = 64;
    constexpr int NUM_GROUPS = INPUT_DIM / GROUP_SIZE;

    // Build input vector (pattern: 1, -1, ...)
    vec_t<VEC_W> a_chunks[INPUT_DIM / VEC_W];
    for (int i = 0; i < INPUT_DIM / VEC_W; ++i) {
        vec_t<VEC_W> chunk{};
        for (int j = 0; j < VEC_W; ++j) {
            const int idx = i * VEC_W + j;
            chunk[j] = (idx & 1) ? -1.0f : 1.0f;
        }
        a_chunks[i] = chunk;
    }

    // Weights: ramp -8..7 per lane, repeated per input scalar.
    std::vector<pack512> w_pkts(INPUT_DIM);
    for (int k = 0; k < INPUT_DIM; ++k) {
        std::vector<int8_t> row(OUT_W);
        for (int j = 0; j < OUT_W; ++j) row[j] = static_cast<int8_t>((j % 16) - 8);
        w_pkts[k] = build_weights(row);
    }

    // Scales: group 0 = 0.5, others = 2.0.
    std::vector<float> scales(NUM_GROUPS * OUT_W);
    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int j = 0; j < OUT_W; ++j) {
            scales[g * OUT_W + j] = (g == 0) ? 0.5f : 2.0f;
        }
    }

    // Golden reference (CPU math).
    vec_t<OUT_W> golden{};
    for (int k = 0; k < INPUT_DIM; ++k) {
        const float a = a_chunks[k / VEC_W][k % VEC_W];
        const int g = k / GROUP_SIZE;
        for (int j = 0; j < OUT_W; ++j) {
            const int8_t w = static_cast<int8_t>((j % 16) - 8);
            golden[j] += a * static_cast<float>(w) * scales[g * OUT_W + j];
        }
    }

    // DUT: T-MAC path
    hls_stream<vec_t<VEC_W>> ain_tmac, cout_tmac;
    for (auto& c : a_chunks) ain_tmac.write(c);
    dense_projection_production_scaled<SCALE_EXP, INPUT_DIM, OUT_W, GROUP_SIZE, true>(
        ain_tmac, cout_tmac, w_pkts.data(), scales.data());

    // DUT: DSP path
    hls_stream<vec_t<VEC_W>> ain_dsp, cout_dsp;
    for (auto& c : a_chunks) ain_dsp.write(c);
    dense_projection_production_scaled<SCALE_EXP, INPUT_DIM, OUT_W, GROUP_SIZE, false>(
        ain_dsp, cout_dsp, w_pkts.data(), scales.data());

    vec_t<OUT_W> dut_tmac{}, dut_dsp{};
    for (int oc = 0; oc < OUT_W / VEC_W; ++oc) {
        vec_t<VEC_W> chunk_t = cout_tmac.read();
        vec_t<VEC_W> chunk_d = cout_dsp.read();
        for (int j = 0; j < VEC_W; ++j) {
            const int lane = oc * VEC_W + j;
            dut_tmac[lane] = chunk_t[j];
            dut_dsp[lane] = chunk_d[j];
        }
    }

    int errors = 0;
    for (int i = 0; i < OUT_W; ++i) {
        const float diff_t = std::fabs(golden[i] - dut_tmac[i]);
        const float diff_d = std::fabs(golden[i] - dut_dsp[i]);
        if (diff_t > 1e-4f || diff_d > 1e-4f) {
            if (errors < 8) {
                std::printf("Lane %d mismatch: gold=%f tmac=%f dsp=%f (dt=%f dd=%f)\n",
                            i, golden[i], dut_tmac[i], dut_dsp[i], diff_t, diff_d);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        std::printf("[smoke_scaled] PASS (T-MAC vs DSP match)\n");
        return 0;
    }
    std::printf("[smoke_scaled] FAIL with %d mismatches\n", errors);
    return 1;
}

// FC1 tile test that consumes the swizzled weight dump produced by pack_fc1.py.
int run_fc1_tile(const std::string& weight_path, int tile_idx) {
    constexpr int INPUT_DIM = 4096;
    constexpr int OUT_DIM = 4096;
    constexpr int OUT_W = 128;
    const size_t expected_bytes = static_cast<size_t>(INPUT_DIM) * OUT_DIM / 2;

    std::ifstream file(weight_path, std::ios::binary);
    if (!file) {
        std::printf("Swizzled weights not found at %s\n", weight_path.c_str());
        std::printf("Run pack_fc1.py to generate them (default output is this folder).\n");
        return 1;
    }

    file.seekg(0, std::ios::end);
    const std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (file_size <= 0) {
        std::printf("Weight file is empty or unreadable: %s\n", weight_path.c_str());
        return 1;
    }

    std::vector<uint8_t> w_bytes(static_cast<size_t>(file_size));
    file.read(reinterpret_cast<char*>(w_bytes.data()), w_bytes.size());
    if (!file) {
        std::printf("Failed to read %s\n", weight_path.c_str());
        return 1;
    }
    if (w_bytes.size() != expected_bytes) {
        std::printf(
            "Warning: expected %zu bytes, file has %zu bytes (still attempting)\n",
            expected_bytes, w_bytes.size());
    }
    if (w_bytes.size() % sizeof(pack512) != 0) {
        std::printf("Weight file size (%zu) is not a multiple of pack512\n", w_bytes.size());
        return 1;
    }

    const size_t packets = w_bytes.size() / sizeof(pack512);
    const size_t packets_per_tile = INPUT_DIM;
    const size_t tiles = packets / packets_per_tile;
    if (tile_idx < 0 || static_cast<size_t>(tile_idx) >= tiles) {
        std::printf("Tile index %d out of range (available tiles: %zu)\n", tile_idx, tiles);
        return 1;
    }
    const pack512* weights_all = reinterpret_cast<const pack512*>(w_bytes.data());
    const pack512* weights_tile = weights_all + static_cast<size_t>(tile_idx) * packets_per_tile;

    hls_stream<vec_t<VEC_W>> ain, cout;
    for (int i = 0; i < INPUT_DIM / VEC_W; ++i) {
        vec_t<VEC_W> chunk{};
        for (int j = 0; j < VEC_W; ++j) {
            chunk[j] = 0.5f; // simple, reproducible input
        }
        ain.write(chunk);
    }

    vec_t<OUT_W> golden{};
    for (int k = 0; k < INPUT_DIM; ++k) {
        lut_mac_broadcast<SCALE_EXP, OUT_W>(0.5f, weights_tile[k], golden);
    }

    dense_projection_production<SCALE_EXP, INPUT_DIM, OUT_W>(ain, cout, weights_tile);
    vec_t<OUT_W> dut{};
    for (int oc = 0; oc < OUT_W / VEC_W; ++oc) {
        vec_t<VEC_W> chunk = cout.read();
        for (int j = 0; j < VEC_W; ++j) {
            dut[oc * VEC_W + j] = chunk[j];
        }
    }

    int errors = 0;
    for (int i = 0; i < OUT_W; ++i) {
        const float diff = std::fabs(golden[i] - dut[i]);
        if (diff > 1e-6f) {
            if (errors < 8) {
                std::printf("[tile %d] mismatch lane %d: got %f expected %f\n",
                            tile_idx, i, dut[i], golden[i]);
            }
            ++errors;
        }
    }

    if (errors == 0) {
        std::printf("[fc1 tile %d] PASS\n", tile_idx);
        return 0;
    }
    std::printf("[fc1 tile %d] FAIL with %d mismatches\n", tile_idx, errors);
    return 1;
}

int main(int argc, char** argv) {
    std::string weight_path = "weights_swizzled.bin";
    int tile_idx = 0;
    bool smoke_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--smoke") {
            smoke_only = true;
        } else if (arg.rfind("--tile=", 0) == 0) {
            tile_idx = std::atoi(arg.c_str() + 7);
        } else {
            weight_path = arg;
        }
    }

    if (smoke_only) {
        return run_smoke_scaled();
    }

    int status = run_fc1_tile(weight_path, tile_idx);
    if (status != 0) {
        std::printf("Use --smoke for a quick self-contained check.\n");
    }
    return status;
}
