#ifndef TMAC_UTILS_HPP
#define TMAC_UTILS_HPP

#include <cmath>
#include <cstdint>

#ifndef TMAC_WEIGHT_BITS
#define TMAC_WEIGHT_BITS 2
#endif

#if (TMAC_WEIGHT_BITS != 2) && (TMAC_WEIGHT_BITS != 4)
#error "TMAC_WEIGHT_BITS must be 2 or 4"
#endif

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
namespace tmac {
namespace hls {
using pack512 = ap_uint<512>;
template <int W>
using vec_t = ::hls::vector<float, W>;
template <typename T>
using hls_stream = ::hls::stream<T>;
} // namespace hls
} // namespace tmac
#else
#include <array>
#include <queue>

namespace hls {
inline float sqrt(float x) { return std::sqrt(x); }
inline float exp(float x) { return std::exp(x); }

template <typename T, int N>
struct vector : public std::array<T, N> {
    vector() { this->fill(T(0)); }
    using std::array<T, N>::array;
};

template <typename T>
class stream {
  public:
    stream() = default;
    explicit stream(const char*) {}
    void write(const T& v) { q_.push(v); }
    T read() {
        T v = q_.front();
        q_.pop();
        return v;
    }
    bool empty() const { return q_.empty(); }

  private:
    std::queue<T> q_;
};
} // namespace hls

namespace tmac {
namespace hls {
struct pack512 {
    uint8_t bytes[64]{};
};
template <int W>
using vec_t = ::hls::vector<float, W>;
template <typename T>
using hls_stream = ::hls::stream<T>;
} // namespace hls
} // namespace tmac
#endif

namespace tmac {
namespace hls {
constexpr int VEC_W = 16;
constexpr int TREE_WIDTH = 4;
constexpr int kQuantBits = TMAC_WEIGHT_BITS;
constexpr int kPack512WeightElems = 512 / TMAC_WEIGHT_BITS;
constexpr int kI32WeightElems = 32 / TMAC_WEIGHT_BITS;
constexpr int kQuantMask = (1 << TMAC_WEIGHT_BITS) - 1;
} // namespace hls
} // namespace tmac

#endif // TMAC_UTILS_HPP
