// Exact copy of source/kernels/routed_storage/portable_exp.h at source parent
// 4ad416bf5e7a7bb91e3da202e4d6f4bd017ab745. Build -ffp-contract=off.
// Ciru: deterministic FP32 exponential for connected quantization semantics.
// Mathematical Taylor coefficients, no fitted tables or vendor exp primitive.
// Explicit F32 operations match portable_exp.py.
#pragma once

__device__ __forceinline__ float ornith_portable_exp(float x) {
    if (isnan(x)) return x;
    if (x <= -104.f) return 0.f;
    if (x >= 89.f) return INFINITY;
    const float n_float = nearbyintf(x * 0x1.715476p+0f);
    const int n = static_cast<int>(n_float);
    const float high = n_float * 0x1.62e400p-1f;
    const float low = n_float * 0x1.7f7d1cp-20f;
    const float reduced_high = x - high;
    const float r = reduced_high - low;
    float p = 0x1.a01a02p-16f;
    p = p * r + 0x1.a01a02p-13f;
    p = p * r + 0x1.6c16c2p-10f;
    p = p * r + 0x1.111112p-7f;
    p = p * r + 0x1.555556p-5f;
    p = p * r + 0x1.555556p-3f;
    p = p * r + 0x1p-1f;
    p = p * r + 0x1p+0f;
    p = p * r + 0x1p+0f;
    return ldexpf(p, n);
}
