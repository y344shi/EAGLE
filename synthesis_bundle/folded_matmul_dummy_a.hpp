#ifndef TMAC_FOLDED_MATMUL_DUMMY_A_HPP
#define TMAC_FOLDED_MATMUL_DUMMY_A_HPP

#include <cstdint>

#include "folded_matmul_dummy_dims.hpp"

namespace tmac {
namespace hls {

// Dummy activation tensor for constant-folded standalone matmul IP.
static const uint8_t kFoldedDummyA[kFoldTopM][kFoldTopK] = {};

} // namespace hls
} // namespace tmac

#endif // TMAC_FOLDED_MATMUL_DUMMY_A_HPP
