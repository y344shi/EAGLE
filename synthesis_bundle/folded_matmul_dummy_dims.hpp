#ifndef TMAC_FOLDED_MATMUL_DUMMY_DIMS_HPP
#define TMAC_FOLDED_MATMUL_DUMMY_DIMS_HPP

namespace tmac {
namespace hls {

constexpr int kFoldTopM = 4;
constexpr int kFoldTopK = 4096;
constexpr int kFoldTopN = 4096;
constexpr int kFoldTopOutputElems = kFoldTopM * kFoldTopN;

} // namespace hls
} // namespace tmac

#endif // TMAC_FOLDED_MATMUL_DUMMY_DIMS_HPP
