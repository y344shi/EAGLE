#ifndef TMAC_FOLDED_MATMUL_DUMMY_B_HPP
#define TMAC_FOLDED_MATMUL_DUMMY_B_HPP

#include <cstdint>

#include "folded_matmul_dummy_dims.hpp"

namespace tmac {
namespace hls {

// Dummy weight tensor for constant-folded standalone matmul IP.
static const uint8_t kFoldedDummyB[kFoldTopN][kFoldTopK] = {};

} // namespace hls
} // namespace tmac

#endif // TMAC_FOLDED_MATMUL_DUMMY_B_HPP
