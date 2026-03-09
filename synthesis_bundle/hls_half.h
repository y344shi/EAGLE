#ifndef TMAC_COMPAT_HLS_HALF_H
#define TMAC_COMPAT_HLS_HALF_H

// Host-side compatibility shim for builds outside Vitis HLS.
// Vitis provides `half` in <hls_half.h>; on host we map it to IEEE fp16.
using half = _Float16;

#endif // TMAC_COMPAT_HLS_HALF_H
