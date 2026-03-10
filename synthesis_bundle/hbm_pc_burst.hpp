#ifndef TMAC_HBM_PC_BURST_HPP
#define TMAC_HBM_PC_BURST_HPP

#include <cstdint>

#include "tmac_utils.hpp"

namespace tmac {
namespace hls {

#ifdef __SYNTHESIS__
using hbm_word256_t = ap_uint<256>;
#else
struct hbm_word256_t {
    uint8_t bytes[32]{};
};
#endif

constexpr int kHbmWordBits = 256;
constexpr int kHbmWordBytes = kHbmWordBits / 8;

inline int64_t bytes_to_hbm_words(int64_t bytes) {
    if (bytes <= 0) {
        return 0;
    }
    return (bytes + kHbmWordBytes - 1) / kHbmWordBytes;
}

inline int64_t align_words(int64_t words, int64_t align_words) {
    if (align_words <= 1) {
        return words;
    }
    return ((words + align_words - 1) / align_words) * align_words;
}

template <typename T>
inline T* pc_word_offset_ptr(hbm_word256_t* base, int64_t word_offset) {
    return reinterpret_cast<T*>(base + word_offset);
}

template <typename T>
inline const T* pc_word_offset_ptr(const hbm_word256_t* base, int64_t word_offset) {
    return reinterpret_cast<const T*>(base + word_offset);
}

}  // namespace hls
}  // namespace tmac

#endif  // TMAC_HBM_PC_BURST_HPP
