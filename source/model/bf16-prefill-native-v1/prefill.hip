// Adapted from asf0, halo-box/strix-llama.cpp PR172 at62e9d99, MIT license.
// BF16 / standard softmax / varlen / causal adaptations by Ciru,2026.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>
#define GGML_USE_HIP 1
#define RDNA3_5 1
// Flash attention for RDNA3.5 WMMA (v_wmma_f32_16x16x16_f16), D = 64, 128, 256 or 512, f16 K/V. For prefill a K/V cache of
// another type (q8_0, q4_0, ...) is converted to f16 first, into the scratch space behind dst: at 32k keys and 4096
// tokens that is ~1% of the attention time, while the generic kernels such a cache fell back to are ~2x slower.
// Small batches at depth read q8_0 K/V directly (dequantized while staged to LDS): they are bound by the K/V reads.
//
// The generic MMA kernel keeps fp32 accumulators for 16 query columns x all V dims per warp (128 VGPRs at
// D = 256), so Q cannot stay in registers, every WMMA re-reads both operands from LDS and the kernel
// spills; D = 512 is not supported by it at all on RDNA and falls back to the FMA tile kernel.
// Here a block of 4*DQ warps owns 64 query columns (4 column tiles of 16) and walks the KV cache in steps of
// 64 / 32 / 32 / 16 keys at D = 64 / 128 / 256 / 512 (32 WMMAs per warp and step):
//   - S^T = K Q^T: the DQ = D/128 warps of a column tile split D (one warp at D <= 128); each keeps its share
//     of Q in registers (64 VGPRs), scores all key tiles of the step over its share (one LDS operand per WMMA) and the shares
//     are summed through LDS. Every warp of a column tile then holds all scores of its columns, so the
//     softmax needs no max exchange.
//   - P stays in the lane that owns the column; one permlanex16 turns the WMMA C layout into the B operand
//     layout.
//   - O^T += V^T P: warp w owns V dims [32w, 32w+32) for all 64 columns (64 VGPRs of accumulators). V is
//     transposed once per block while it is staged into LDS.
//   - K and V^T have separate LDS buffers: the next K and this step's V^T are both staged between the two
//     barriers of a step, from registers that were loaded while the scores were computed.
//   - a pre-pass classifies the mask per (token tile, 32 keys): all-zero steps ignore the mask, all -inf
//     steps are skipped, only mixed steps (near the diagonal) read it.
//   - the head group varies fastest across blocks, so the blocks in flight read whole KV cache rows.
//   - small batches (decode, speculative verification) have too few column tiles to fill the GPU: the key steps are
//     split over gridDim.y blocks, which write unnormalized partials and (max, row sum) for flash_attn_combine_results.
// Column c of a block is (token jt*NCOLS1 + c/NCOLS2, Q head head0 + c%NCOLS2); all NCOLS2 heads share
// one K/V head.

#if defined(GGML_USE_HIP) && defined(RDNA3_5)
#define FATTN_RDNA35_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA3_5)

namespace fattn_rdna35 {

constexpr int CLS_KEYS   = 32;     // granularity of the mask classes
constexpr int MAX_CLS    = 8192;   // mask classes kept in LDS (2 bits each): up to 262144 keys
constexpr float m_init   = -1e30f; // finite so that fully masked columns never produce inf - inf
constexpr float log2e    = 1.44269504088896340736f;
constexpr float T_LAZY   = 8.0f;   // OWN: the reference max moves when a step max is more than this above it (log2 units)
constexpr float T_HI     = 14.0f;  // OWN: above this P would leave the f16 range -> redo the step against the new max

template <int D>
struct cfg {
    static constexpr int NCOLS  = 64;                   // query columns per block (32 at D = 512: 16.3 vs 25.6 TFLOPS, K/V reuse halves)
    static constexpr int NCT    = NCOLS / 16;           // column tiles
    static constexpr int DQ     = D > 128 ? D / 128 : 1; // warps per column tile, splitting D in the score phase
    static constexpr int QCH    = D / DQ / 16;          // 16-dim chunks of Q per warp
    static constexpr int NWARPS = NCT * DQ;
    static constexpr int NTHR   = NWARPS * 32;
    static constexpr int OT     = D / NWARPS / 16;      // warp w owns V dims [16*OT*w, 16*OT*(w+1))
    static constexpr int NKEYS  = D == 512 ? 16 : D == 64 ? 64 : 32; // keys per step: 32 WMMAs per warp and step
    static constexpr int KT     = NKEYS / 16;           // key tiles per step
    static constexpr int SEGS   = D / 8;                // 16 B segments per K/V row
    static constexpr int KPT    = NKEYS * SEGS / NTHR;  // K and V rows (keys) per thread and step
    // D = 512 runs at 192 VGPRs (2 blocks per WGP). With V prefetched as well and all P operands of a key tile loaded at
    // once, the loop spilled and its scratch reloads waited for the K/V loads (vmcnt is in order): 26.7 TFLOPS. Loading V
    // after the first barrier, one score chain and P one operand at a time: 29.5. At D = 256 both cost ~1%.
    static constexpr bool TIGHT = D == 512;
    // D = 256: warp dq of a column tile finishes key tile dq of the step (half the score exchange and softmax per warp).
    static constexpr bool OWN   = false;
    static constexpr int SQK    = D + 8;                // row stride (halves) of sK:     conflict-free 16-row reads
    static constexpr int SVT    = NKEYS + 8;            // row stride (halves) of sVt/sP: conflict-free 16-row reads
    static constexpr int NUNITS = NKEYS / 8;            // 8-key units of a V^T row (swizzled)

    static constexpr int SZ_K   = NKEYS * SQK * 2;
    static constexpr int SZ_VT  = D * SVT * 2;
    static constexpr int SZ_P   = NCOLS * SVT * 2;
    // partial scores [warp][key tile][lane][8]; one warp per column tile (D <= 128) needs no exchange, only the column maxima
    static constexpr int SZ_SX  = DQ == 1 ? NCOLS * 4 : NWARPS * (OWN ? 1 : KT) * 32 * 8 * 4;
    static constexpr int SZ_RED = (OWN ? 4 : 2) * NCOLS * 4;            // per column: step maxima and row sums (OWN), or rescale factors and row sums
    static constexpr int SZ_CLS = MAX_CLS / 4;

    static constexpr int OFF_K   = 0;
    static constexpr int OFF_VT  = OFF_K   + SZ_K;
    static constexpr int OFF_P   = OFF_VT  + SZ_VT;
    static constexpr int OFF_SX  = OFF_P   + SZ_P;
    static constexpr int OFF_RED = OFF_SX  + SZ_SX;
    static constexpr int OFF_CLS = OFF_RED + SZ_RED;
    static constexpr int SZ_SMEM = OFF_CLS + SZ_CLS;

    static_assert(KPT == 2 || KPT == 4, "V transpose handles 2 or 4 keys per thread");
    static_assert(SEGS % 8 == 0 && NTHR % SEGS == 0, "thread layout");
    static_assert(QCH == 8 || (DQ == 1 && QCH == 4), "Q chunks per warp");
    static_assert(NKEYS % CLS_KEYS == 0 || CLS_KEYS % NKEYS == 0, "mask classes per step");
    static_assert(SZ_SMEM + 16 <= 65536, "LDS budget");
};

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef unsigned short ushort16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4_t  __attribute__((ext_vector_type(4))); // plain vector: stays in registers (HIP's int4 class may not)

#ifdef FATTN_RDNA35_AVAILABLE

static __device__ __forceinline__ float exp_natural(float x) {
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
}
static __device__ __forceinline__ uint16_t bf16_rne(float x) {
    uint32_t u = __float_as_uint(x);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static __device__ __forceinline__ half2 bf16_pair(float a, float b) {
    return __builtin_bit_cast(half2, uint32_t(bf16_rne(a)) | (uint32_t(bf16_rne(b)) << 16));
}
static __device__ __forceinline__ float8_t wmma(const half16_t a, const half16_t b, const float8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(ushort16_t, a), __builtin_bit_cast(ushort16_t, b), c);
}

// Barrier that only orders LDS: __syncthreads() also waits for all outstanding global loads (vmcnt(0)) and
// invalidates the vector L0 cache, which would cut the K/V prefetch short at every barrier.
static __device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// value of the lane in the other half of the wave (lane ^ 16)
static __device__ __forceinline__ int swap_halves(const int x) {
    return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false, false);
}
static __device__ __forceinline__ float swap_halves(const float x) {
    return __int_as_float(swap_halves(__float_as_int(x)));
}

// q8_0 K/V (small batches, see the launcher): a 16 B f16 segment of 8 dims is a quarter of a q8_0 block. Its 8 quants
// are loaded raw with the block scale and dequantized when staged to LDS (same f16 values as dequantize_q8_0).
// The quants of segment s start at byte 34*(s/4) + 2 + 8*(s%4) of the row: 2 mod 4 in even blocks, where a third dword
// is loaded and the two data dwords are funnel-shifted out; rows and heads start at multiples of 4 B (host check).
static __device__ __forceinline__ u32x4_t load_seg_q8(const char * row, const int seg) {
    const uint32_t  blk = seg / 4;
    const uint32_t  off = 34u*blk + 2u + 8u*(seg % 4);
    const uint32_t * p  = (const uint32_t *) (row + (off & ~3u));
    u32x4_t r;
    r[0] = p[0];
    r[1] = p[1];
    r[2] = blk % 2 == 0 ? p[2] : 0u;
    r[3] = *(const uint16_t *) (row + 34u*blk);
    return r;
}
static __device__ __forceinline__ u32x4_t dequant_seg_q8(const u32x4_t r, const int seg) {
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    const uint32_t sh = (seg / 4) % 2 == 0 ? 2u : 0u;
    // q + 128 as the low byte of an f16 with exponent 10: 1152 + q, exact
    const uint32_t lo = __builtin_amdgcn_alignbyte(r[1], r[0], sh) ^ 0x80808080u;
    const uint32_t hi = __builtin_amdgcn_alignbyte(r[2], r[1], sh) ^ 0x80808080u;
    const uint32_t b[4] = {
        __builtin_amdgcn_perm(0x64646464u, lo, 0x04010400u), __builtin_amdgcn_perm(0x64646464u, lo, 0x04030402u),
        __builtin_amdgcn_perm(0x64646464u, hi, 0x04010400u), __builtin_amdgcn_perm(0x64646464u, hi, 0x04030402u),
    };
    const h2_t d = __builtin_bit_cast(h2_t, r[3] * 0x00010001u);
    const h2_t o = {(_Float16) -1152.0f, (_Float16) -1152.0f};
    u32x4_t out;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        out[e] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2_t, b[e]) + o) * d); // q exact, one rounding of d*q
    }
    return out;
}

// K tile: KPT x 16 B per thread; the lanes of a wave read 512 contiguous bytes of a key row.
template <int D, bool Q8>
static __device__ __forceinline__ void load_K(u32x4_t * regs, const char * K, const uint32_t nb11, const int k0, const int length) {
    using C = cfg<D>;
#pragma unroll
    for (int p = 0; p < C::KPT; ++p) {
        const int idx = p*C::NTHR + threadIdx.x;
        if constexpr (Q8) {
            regs[p] = load_seg_q8(K + uint32_t(k0 + idx/C::SEGS)*nb11, idx % C::SEGS);
        } else {
            regs[p] = k0 + idx/C::SEGS < length ? *(const u32x4_t *) (K + (uint32_t(k0 + idx/C::SEGS)*nb11 + uint32_t(idx % C::SEGS)*16u)) : u32x4_t{};
        }
    }
}
template <int D, bool Q8>
static __device__ __forceinline__ void store_K(const u32x4_t * regs, _Float16 * sK) {
    using C = cfg<D>;
#pragma unroll
    for (int p = 0; p < C::KPT; ++p) {
        const int idx = p*C::NTHR + threadIdx.x;
        *(u32x4_t *) (sK + (idx/C::SEGS)*C::SQK + (idx % C::SEGS)*8) = Q8 ? dequant_seg_q8(regs[p], idx % C::SEGS) : regs[p];
    }
}

// V tile: thread = (group kg of KPT keys, 8-dim segment seg). Lanes 8j..8j+7 of a wave read 8 consecutive
// segments of one key, i.e. full 128 B cache lines.
template <int D>
static __device__ __forceinline__ void v_thread_coords(int & kg, int & seg) {
    constexpr int WPR = cfg<D>::SEGS / 8; // warps per V row
    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    seg = lane % 8 + 8*(w % WPR);
    kg  = lane / 8 + 4*(w / WPR);
}
template <int D, bool Q8>
static __device__ __forceinline__ void load_V(u32x4_t * regs, const char * V, const uint32_t nb21, const int k0, const int length) {
    using C = cfg<D>;
    int kg, seg;
    v_thread_coords<D>(kg, seg);
#pragma unroll
    for (int kk = 0; kk < C::KPT; ++kk) {
        if constexpr (Q8) {
            regs[kk] = load_seg_q8(V + uint32_t(k0 + C::KPT*kg + kk)*nb21, seg);
        } else {
            regs[kk] = k0 + C::KPT*kg + kk < length ? *(const u32x4_t *) (V + (uint32_t(k0 + C::KPT*kg + kk)*nb21 + uint32_t(seg)*16u)) : u32x4_t{};
        }
    }
}
// V^T[dim][key] in LDS: within a row the 8-key units are swizzled with the dim's segment, so that the lanes
// of a store (same key group, consecutive segments) spread over more banks.
template <int D>
static __device__ __forceinline__ int vt_unit(const int unit, const int dim) {
    return unit ^ ((dim / 8) % cfg<D>::NUNITS);
}
// registers -> LDS as V^T: the KPT keys x 8 dims block of the thread is transposed with v_perm_b32.
template <int D, bool Q8>
static __device__ __forceinline__ void store_Vt(const u32x4_t * regs_in, _Float16 * sVt) {
    using C = cfg<D>;
    int kg, seg;
    v_thread_coords<D>(kg, seg);
    u32x4_t regs[C::KPT];
#pragma unroll
    for (int kk = 0; kk < C::KPT; ++kk) {
        regs[kk] = Q8 ? dequant_seg_q8(regs_in[kk], seg) : regs_in[kk];
    }
    // regs[kk][q]: dims (2q, 2q+1) of key kk
    const int col = 8*vt_unit<D>(C::KPT*kg/8, 8*seg) + (C::KPT*kg) % 8;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        _Float16 * row_even = sVt + (8*seg + 2*q + 0)*C::SVT + col;
        _Float16 * row_odd  = sVt + (8*seg + 2*q + 1)*C::SVT + col;
        const uint32_t e01 = __builtin_amdgcn_perm(regs[1][q], regs[0][q], 0x05040100u); // dim 2q:   keys 0, 1
        const uint32_t o01 = __builtin_amdgcn_perm(regs[1][q], regs[0][q], 0x07060302u); // dim 2q+1: keys 0, 1
        if constexpr (C::KPT == 4) {
            const uint32_t e23 = __builtin_amdgcn_perm(regs[3][q], regs[2][q], 0x05040100u);
            const uint32_t o23 = __builtin_amdgcn_perm(regs[3][q], regs[2][q], 0x07060302u);
            *(uint2 *) row_even = make_uint2(e01, e23);
            *(uint2 *) row_odd  = make_uint2(o01, o23);
        } else {
            *(uint32_t *) row_even = e01;
            *(uint32_t *) row_odd  = o01;
        }
    }
}
// A operand of V^T for dims [dim0, dim0+16) (dim0 % 16 == 0) and keys [16c, 16c+16): lane row dim0 + l16.
template <int D>
static __device__ __forceinline__ half16_t load_Vt_operand(const _Float16 * sVt, const int dim, const int c) {
    typedef int int8v_t __attribute__((ext_vector_type(8)));
    const _Float16 * row = sVt + dim*cfg<D>::SVT;
    const int4 lo = *(const int4 *) (row + 8*vt_unit<D>(2*c + 0, dim));
    const int4 hi = *(const int4 *) (row + 8*vt_unit<D>(2*c + 1, dim));
    const int8v_t v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    return __builtin_bit_cast(half16_t, v);
}

#endif // FATTN_RDNA35_AVAILABLE

// D = 512: a 16-wave block at up to 256 VGPRs fits once per WGP, so the SIMDs idle while its warps wait at the barriers.
// Capped at 192 VGPRs (8 waves per SIMD) a second block fits (LDS allows two) and fills these gaps (see cfg::TIGHT).
template <int D, int NCOLS2, bool Q8>
__launch_bounds__(cfg<D>::NTHR, cfg<D>::NWARPS == 16 ? 8 : 1) // HIP: the second argument is the minimum number of waves per SIMD
static __global__ void flash_attn_rdna35(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, const uint8_t * __restrict__ mask_cls, const int * __restrict__ KV_max, uint16_t * __restrict__ dst,
        float * __restrict__ dst_parts, float2 * __restrict__ dst_meta, const float * __restrict__ sinks,
        const float scale_q, const float softcap_log2, const int max_queries, const int ne02, const int max_keys, const int gqa_ratio,
        const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23,
        const int ne31, const int ne33, const int64_t nb31, const int64_t nb33,
        const int * QStarts, const int * KStarts, const int64_t os0, const int64_t os1, float * lse, const int64_t ls0, const int64_t ls1) {
#ifdef FATTN_RDNA35_AVAILABLE
    using C = cfg<D>;
    constexpr int NCOLS  = C::NCOLS;
    constexpr int NCOLS1 = NCOLS / NCOLS2;
    constexpr int NUSED  = NCOLS1 * NCOLS2; // NCOLS2 = 6: 60 of the 64 columns
    constexpr int NCT    = C::NCT;
    constexpr int OT     = C::OT;
    constexpr int KT     = C::KT;
    constexpr int DQ     = C::DQ;
    constexpr int NKEYS  = C::NKEYS;
    constexpr int SVT    = C::SVT;

    __shared__ __attribute__((aligned(16))) char smem[C::SZ_SMEM];
    __shared__ int sRescaleStep; // last step in which some column max moved
    __shared__ int sRedo;        // OWN: last step whose P had to be redone against a new max
    _Float16 * sK     = (_Float16 *) (smem + C::OFF_K);
    _Float16 * sVt    = (_Float16 *) (smem + C::OFF_VT);
    _Float16 * sP     = (_Float16 *) (smem + C::OFF_P);
    float    * sSx    = (float    *) (smem + C::OFF_SX);
    float    * sAlpha = (float    *) (smem + C::OFF_RED);
    float    * sMax   = sAlpha;                              // OWN: [key tile][l16][column tile]
    float    * sL     = sAlpha + (C::OWN ? 2*NCOLS : NCOLS); // OWN: [key tile][column]

    const int lane = threadIdx.x % 32;
    const int w    = threadIdx.x / 32;
    const int l16  = lane % 16;
    const int hh   = lane / 16;
    const int ct   = w / DQ;      // column tile this warp scores
    const int dq   = w % DQ;      // share of D this warp scores: dims [128dq, 128dq+128)
    const int c_s  = 16*ct + l16; // column of this lane in the score phase

    const int n_hg   = ne02 / NCOLS2;
    const int ntiles = gridDim.x / n_hg;
    const int jt     = blockIdx.x / n_hg;
    const int head0  = (blockIdx.x % n_hg)*NCOLS2;
    const int seq    = blockIdx.z;
    const int qstart = QStarts[seq];
    const int kstart = KStarts[seq];
    const int ne01 = QStarts[seq+1] - qstart;
    const int ne11 = KStarts[seq+1] - kstart;
    if (jt*NCOLS1 >= ne01) return;
    const int z_KV   = head0 / gqa_ratio;

    const char * K_h = K + int64_t(kstart)*nb11 + z_KV*nb12;
    const char * V_h = V + int64_t(kstart)*nb21 + z_KV*nb22;
    const uint32_t nb11u = (uint32_t) nb11; // row offsets of one head stay below 4 GiB (checked on the host)
    const uint32_t nb21u = (uint32_t) nb21;
    const half * mask_h = mask ? (const half *) (mask + (seq % ne33)*nb33) : nullptr;
    const int64_t s31 = nb31 / sizeof(half);

    const int kv_end = min(ne11, ne11-ne01+(jt+1)*NCOLS1);
    const int n_steps = (kv_end + NKEYS - 1) / NKEYS;

    // split-KV: this block takes the key steps [st_begin, st_end)
    const int nsplit   = gridDim.y;
    const int st_begin = int(int64_t(n_steps)*blockIdx.y/nsplit);
    const int st_end   = int(int64_t(n_steps)*(blockIdx.y + 1)/nsplit);

    // mask classes per 32 keys (see mask_classify), uniform across the block; no mask = all zero
    uint8_t * cls = mask_cls ? (uint8_t *) (smem + C::OFF_CLS) : nullptr;
    if (cls) {
        const int n_cls  = (n_steps*NKEYS + CLS_KEYS - 1) / CLS_KEYS;
        const int stride = (ne11/CLS_KEYS + 3) / 4;
        const uint8_t * cls_g = mask_cls + (int64_t(seq % ne33)*ntiles + jt)*stride;
        for (int i = threadIdx.x; i < (n_cls + 3)/4; i += C::NTHR) {
            cls[i] = cls_g[i];
        }
    }
    if (threadIdx.x == 0) {
        sRescaleStep = -1;
        sRedo        = -1;
    }
    auto step_class = [&](const int st) -> int {
        return (st+1)*NKEYS <= ne11-ne01+jt*NCOLS1+1 ? 0 : 2;
    };
    auto next_active = [&](int st) -> int {
        while (st < st_end && step_class(st) == 1) {
            ++st;
        }
        return st;
    };

    // Q of column c_s, dims [128dq, 128dq+128) -> registers as B operands, pre-scaled by scale*log2(e)
    // so that the softmax can use exp2 (with a logit softcap by scale/softcap, see softcap below).
    half16_t qb[C::QCH];
    {
        const int  token = jt*NCOLS1 + c_s/NCOLS2;
        const bool valid = token < ne01 && c_s < NUSED;
        const uint16_t * qrow = (const uint16_t *) (Q + int64_t(qstart)*nb01 + int64_t(head0 + c_s % NCOLS2)*nb02 + int64_t(valid ? token : 0)*nb01) + (D/DQ)*dq;
#pragma unroll
        for (int k = 0; k < C::QCH; ++k) {
#pragma unroll
            for (int e = 0; e < 16; e += 4) {
                for (int j=0; j<4; ++j) {
                    const uint16_t bits = valid ? qrow[16*k+e+j] : uint16_t(0);
                    qb[k][e+j] = __builtin_bit_cast(_Float16, bits);
                }
            }
        }
    }
    __syncthreads(); // classes and sRescaleStep visible

    u32x4_t kregs[C::KPT];
    u32x4_t vregs[C::KPT];
    int cur = next_active(st_begin);
    if (cur < st_end) {
        load_K<D, Q8>(kregs, K_h, nb11u, cur*NKEYS, ne11);
        store_K<D, Q8>(kregs, sK);
    }
    __syncthreads();

    float8_t O[OT][NCT]; // [dim tile][column tile]
#pragma unroll
    for (int i = 0; i < OT; ++i) {
#pragma unroll
        for (int t = 0; t < NCT; ++t) {
            O[i][t] = float8_t{};
        }
    }
    float m = m_init; // running max and row sum of column c_s (identical in all warps of a column tile)
    float l = 0.0f;   // OWN: reference max of column c_s and row sum over key tile dq
    float l_prev = 0.0f;
    float mref[NCT];  // OWN: reference max of the PV columns 16t + l16
#pragma unroll
    for (int t = 0; t < NCT; ++t) {
        mref[t] = m_init;
    }

    // OWN: P = exp2(S - mr) of key tile dq -> B operand layout in sP; returns the column sum of P.
    // own[q] = (key 4q + hh, key 4q + 2 + hh), other[q] from lane ^ 16 = (key 4q + 1 - hh, key 4q + 3 - hh)
    auto store_P = [&](const float8_t & sc, const float mr) -> float {
        float p[8];
        float rowsum = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            p[r] = exp_natural(sc[r] - mr);
            rowsum += p[r];
        }
        uint32_t Pb[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const half2 h2 = bf16_pair(p[2*q], p[2*q + 1]);
            const uint32_t own   = __builtin_bit_cast(uint32_t, h2);
            const uint32_t other = (uint32_t) swap_halves((int) own);
            const uint32_t lo_src = hh ? other : own;
            const uint32_t hi_src = hh ? own   : other;
            Pb[2*q + 0] = __builtin_amdgcn_perm(hi_src, lo_src, 0x05040100u);
            Pb[2*q + 1] = __builtin_amdgcn_perm(hi_src, lo_src, 0x07060302u);
        }
        if (hh == 0) {
            int4 * dstp = (int4 *) (sP + c_s*SVT + 16*dq);
            dstp[0] = make_int4(Pb[0], Pb[1], Pb[2], Pb[3]);
            dstp[1] = make_int4(Pb[4], Pb[5], Pb[6], Pb[7]);
        }
        return rowsum + swap_halves(rowsum);
    };

    // logit softcap: Q carries scale/softcap, the full scores become softcap*tanh(s) in log2 units (before the mask)
    auto softcap = [&](float8_t & sc) {
        #pragma unroll
        for (int r=0; r<8; ++r) sc[r] *= scale_q;
    };

    while (cur < st_end) {
        const int  k0       = cur*NKEYS;
        const int  nxt      = next_active(cur + 1);
        const bool has_next = nxt < st_end;
        const bool mixed    = step_class(cur) == 2;

        // V of this step (not at D = 512) and K of the next one: in flight while the scores are computed.
        if constexpr (!C::TIGHT) {
            load_V<D, Q8>(vregs, V_h, nb21u, k0, ne11);
        }
        if (has_next) {
            load_K<D, Q8>(kregs, K_h, nb11u, nxt*NKEYS, ne11);
        }

        // ---- partial S^T = K Q^T over dims [128dq, 128dq+128), all key tiles of the step ----
        float8_t S[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            S[kt] = float8_t{};
        }
        if constexpr (!C::TIGHT) {
            // K operands double-buffered: the loads of chunk k + 1 are in flight during the WMMAs of chunk k
            const _Float16 * ka[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                ka[kt] = sK + (16*(C::OWN ? kt ^ dq : kt) + l16)*C::SQK + (D/DQ)*dq;
            }
            half16_t a[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                a[kt] = *(const half16_t *) ka[kt];
            }
#pragma unroll
            for (int k = 0; k < C::QCH; ++k) {
                half16_t an[KT];
                if (k < C::QCH - 1) {
#pragma unroll
                    for (int kt = 0; kt < KT; ++kt) {
                        an[kt] = *(const half16_t *) (ka[kt] + 16*(k + 1));
                    }
                }
                __builtin_amdgcn_sched_barrier(0); // keep the loads of chunk k + 1 ahead of the WMMAs of chunk k
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    S[kt] = wmma(a[kt], qb[k], S[kt]);
                    if (k < C::QCH - 1) {
                        a[kt] = an[kt];
                    }
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < C::QCH; ++k) {
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    S[kt] = wmma(*(const half16_t *) (sK + (16*(C::OWN ? kt ^ dq : kt) + l16)*C::SQK + (D/DQ)*dq + 16*k), qb[k], S[kt]);
                }
            }
        }
        // S[kt][r]: key k0 + 16kt + 2r + hh, column c_s (OWN: S[0] is key tile dq, S[1] key tile dq ^ 1)
#pragma unroll
        for (int kt = C::OWN ? 1 : 0; kt < KT && DQ > 1; ++kt) {
            float4 * px = (float4 *) (sSx + ((C::OWN ? w : KT*w + kt)*32 + lane)*8);
            px[0] = make_float4(S[kt][0], S[kt][1], S[kt][2], S[kt][3]);
            px[1] = make_float4(S[kt][4], S[kt][5], S[kt][6], S[kt][7]);
        }
        lds_barrier(); // B1: partial scores visible, all warps are done reading K

        if constexpr (C::OWN) {
            // Own key tile: add the share of the partner warp. The reference max m moves lazily (as in FlashAttention-4),
            // after V^T P and only when a step max is more than T_LAZY above it; P up to 2^T_HI is exact in f16.
            {
                const float4 * px = (const float4 *) (sSx + ((ct*DQ + (dq ^ 1))*32 + lane)*8);
                const float4 x0 = px[0];
                const float4 x1 = px[1];
                S[0][0] += x0.x; S[0][1] += x0.y; S[0][2] += x0.z; S[0][3] += x0.w;
                S[0][4] += x1.x; S[0][5] += x1.y; S[0][6] += x1.z; S[0][7] += x1.w;
            }
            softcap(S[0]);
            if (mixed) {
                const int row = min(jt*NCOLS1 + c_s/NCOLS2, ne31 - 1);
                const int4 * mp = (const int4 *) (mask_h + row*s31 + k0 + 16*dq);
                const int4 m0 = mp[0];
                const int4 m1 = mp[1];
                const uint32_t md[8] = {(uint32_t) m0.x, (uint32_t) m0.y, (uint32_t) m0.z, (uint32_t) m0.w,
                                        (uint32_t) m1.x, (uint32_t) m1.y, (uint32_t) m1.z, (uint32_t) m1.w};
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const uint16_t bits = (uint16_t) (hh ? md[r] >> 16 : md[r] & 0xFFFFu);
                    S[0][r] += __half2float(__ushort_as_half(bits)) * log2e;
                }
            }
            float mx = S[0][0];
#pragma unroll
            for (int r = 1; r < 8; ++r) {
                mx = fmaxf(mx, S[0][r]);
            }
            mx = fmaxf(mx, swap_halves(mx));
            if (hh == 0) {
                sMax[(dq*16 + l16)*NCT + ct] = mx;
            }
            if (mx > m + T_HI) {
                sRedo = cur;
            }
            l_prev = l;
            l += store_P(S[0], m);
        } else if (dq == 0) {
            // With one key tile (D = 512) only warp dq = 0 of a column tile uses the softmax (P, rescale factors, row sums);
            // at D <= 128 one warp owns a column tile.
            // ---- full scores: add the shares of the other warps of the column tile ----
#pragma unroll
            for (int j = 1; j < DQ; ++j) {
                const int wp = ct*DQ + (dq + j) % DQ;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    const float4 * px = (const float4 *) (sSx + ((KT*wp + kt)*32 + lane)*8);
                    const float4 x0 = px[0];
                    const float4 x1 = px[1];
                    S[kt][0] += x0.x; S[kt][1] += x0.y; S[kt][2] += x0.z; S[kt][3] += x0.w;
                    S[kt][4] += x1.x; S[kt][5] += x1.y; S[kt][6] += x1.z; S[kt][7] += x1.w;
                }
            }
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                softcap(S[kt]);
            }
            if (mixed) {
                const int token = jt*NCOLS1+c_s/NCOLS2;
                #pragma unroll
                for (int kt=0; kt<KT; ++kt) {
                    #pragma unroll
                    for (int r=0; r<8; ++r) {
                        const int key=k0+16*kt+2*r+hh;
                        if (key>=ne11 || key>=ne11-ne01+token+1) S[kt][r]=-INFINITY;
                    }
                }
            }

            // ---- softmax over the keys of the step ----
            {
                float mx = S[0][0];
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        mx = fmaxf(mx, S[kt][r]);
                    }
                }
                const float m_new = fmaxf(m, fmaxf(mx, swap_halves(mx)));
                const float alpha = exp_natural(m - m_new);
                m = m_new;

                float rowsum = 0.0f;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        S[kt][r] = exp_natural(S[kt][r] - m_new);
                        rowsum += S[kt][r];
                    }
                }
                l = l*alpha + rowsum + swap_halves(rowsum);
                if (dq == 0 && hh == 0) {
                    sAlpha[c_s] = alpha;
                    if (alpha != 1.0f) {
                        sRescaleStep = cur;
                    }
                }

                // P of the key tiles kt % DQ == dq -> B operand layout.
                // own[q] = (key 4q + hh, key 4q + 2 + hh), other[q] from lane ^ 16 = (key 4q + 1 - hh, key 4q + 3 - hh)
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) {
                    uint32_t Pb[8];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const half2 h2 = bf16_pair(S[kt][2*q], S[kt][2*q + 1]);
                        const uint32_t own   = __builtin_bit_cast(uint32_t, h2);
                        const uint32_t other = (uint32_t) swap_halves((int) own);
                        const uint32_t lo_src = hh ? other : own; // provides the even key of each pair
                        const uint32_t hi_src = hh ? own   : other;
                        Pb[2*q + 0] = __builtin_amdgcn_perm(hi_src, lo_src, 0x05040100u); // keys 4q,   4q+1
                        Pb[2*q + 1] = __builtin_amdgcn_perm(hi_src, lo_src, 0x07060302u); // keys 4q+2, 4q+3
                    }
                    if (hh == 0) {
                        int4 * dstp = (int4 *) (sP + c_s*SVT + 16*kt);
                        dstp[0] = make_int4(Pb[0], Pb[1], Pb[2], Pb[3]);
                        dstp[1] = make_int4(Pb[4], Pb[5], Pb[6], Pb[7]);
                    }
                }
            }
        }

        // ---- stage V^T of this step and K of the next one (both buffers are free after B1) ----
        if constexpr (C::TIGHT) {
            load_V<D, Q8>(vregs, V_h, nb21u, k0, ne11);
        }
        store_Vt<D, Q8>(vregs, sVt);
        if (has_next) {
            store_K<D, Q8>(kregs, sK);
        }
        lds_barrier(); // B2: P, V^T, rescale factors and the next K visible

        // ---- O^T = alpha*O^T + V^T P for dims [32w, 32w+32) ----
        float alpha[NCT];
        bool  redo  = false;
        bool  moved = false;
        if constexpr (C::OWN) {
            redo = sRedo == cur;
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const float smax = fmaxf(sMax[l16*NCT + t], sMax[(16 + l16)*NCT + t]);
                const float mn   = redo || smax > mref[t] + T_LAZY ? fmaxf(mref[t], smax) : mref[t];
                alpha[t] = exp_natural(mref[t] - mn);
                moved    = moved || mn != mref[t];
                mref[t]  = mn;
            }
            float alpha_s = alpha[0];
            float m_s     = mref[0];
#pragma unroll
            for (int t = 1; t < NCT; ++t) {
                if (ct == t) {
                    alpha_s = alpha[t];
                    m_s     = mref[t];
                }
            }
            m = m_s;
            if (redo) {
                // P of this step left the f16 range against the old max: rescale first and redo P (first steps only, in practice)
#pragma unroll
                for (int t = 0; t < NCT; ++t) {
#pragma unroll
                    for (int i = 0; i < OT; ++i) {
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            O[i][t][r] *= alpha[t];
                        }
                    }
                }
                l = l_prev*alpha_s + store_P(S[0], m);
                lds_barrier();
            } else {
                l *= alpha_s;
            }
        } else if (sRescaleStep == cur) {
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const float alpha = sAlpha[16*t + l16];
#pragma unroll
                for (int i = 0; i < OT; ++i) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        O[i][t][r] *= alpha;
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KT && !C::TIGHT; ++c) {
            half16_t b[NCT];
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                b[t] = *(const half16_t *) (sP + (16*t + l16)*SVT + 16*c);
            }
#pragma unroll
            for (int i = 0; i < OT; ++i) {
                const half16_t a = load_Vt_operand<D>(sVt, 16*OT*w + 16*i + l16, c);
#pragma unroll
                for (int t = 0; t < NCT; ++t) {
                    O[i][t] = wmma(a, b[t], O[i][t]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < KT && C::TIGHT; ++c) {
            half16_t a[OT];
#pragma unroll
            for (int i = 0; i < OT; ++i) {
                a[i] = load_Vt_operand<D>(sVt, 16*OT*w + 16*i + l16, c);
            }
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
                const half16_t b = *(const half16_t *) (sP + (16*t + l16)*SVT + 16*c);
#pragma unroll
                for (int i = 0; i < OT; ++i) {
                    O[i][t] = wmma(a[i], b, O[i][t]);
                }
            }
        }
        if (C::OWN && !redo && __any(moved)) {
#pragma unroll
            for (int t = 0; t < NCT; ++t) {
#pragma unroll
                for (int i = 0; i < OT; ++i) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        O[i][t][r] *= alpha[t];
                    }
                }
            }
        }
        cur = nxt;
    }

    // ---- normalize by the row sums and write (split-KV: partials and (max, row sum) in natural log units) ----
    float * sM = sSx; // column maxima (not OWN), nobody reads sSx after the last B2
    if (C::OWN ? hh == 0 : dq == 0 && hh == 0) {
        sL[(C::OWN ? dq*NCOLS : 0) + c_s] = l;
        if (!C::OWN) {
            sM[c_s] = m;
        }
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < NCT; ++t) {
        const int c = 16*t + l16;
        const int token = jt*NCOLS1 + c/NCOLS2;
        if (token >= ne01 || c >= NUSED) {
            continue;
        }
        const int head = head0 + c % NCOLS2;
        float l_sum = C::OWN ? sL[c] + sL[NCOLS + c] : sL[c];
        float m_c   = C::OWN ? mref[t] : sM[c];
        float o_scl = 1.0f;
        if (sinks && blockIdx.y == 0) {
            // the sink logit of the head joins the softmax denominator (once, in the first split)
            const float s = sinks[head]*log2e;
            if (s > m_c) {
                o_scl = exp_natural(m_c - s);
                l_sum = l_sum*o_scl + 1.0f;
                m_c   = s;
            } else {
                l_sum += exp_natural(s - m_c);
            }
        }
        const int64_t j_dst = int64_t(qstart+token)*ne02+head;
        if (nsplit > 1) {
            float * part = dst_parts + (j_dst*nsplit + blockIdx.y)*D;
#pragma unroll
            for (int i = 0; i < OT; ++i) {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    part[16*OT*w + 16*i + 2*r + hh] = O[i][t][r]*o_scl;
                }
            }
            if (w == 0 && hh == 0) {
                dst_meta[j_dst*nsplit + blockIdx.y] = make_float2(m_c * (1.0f/log2e), l_sum);
            }
            continue;
        }
        const float inv = l_sum > 0.0f ? o_scl/l_sum : 0.0f;
        uint16_t * dst_c = dst + int64_t(qstart+token)*os0+head*os1;
        if (w==0 && hh==0) lse[head*ls0+int64_t(qstart+token)*ls1] = m_c + logf(l_sum);
#pragma unroll
        for (int i = 0; i < OT; ++i) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                dst_c[16*OT*w + 16*i + 2*r + hh] = bf16_rne(O[i][t][r]*inv);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, mask_cls, KV_max, dst, dst_parts, dst_meta, sinks, scale_q, softcap_log2, ne01, ne02, ne11, gqa_ratio,
        nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23, ne31, ne33, nb31, nb33);
    NO_DEVICE_CODE;
#endif // FATTN_RDNA35_AVAILABLE
}


} // namespace fattn_rdna35
extern "C" int bf16_prefill_native(
    const void *q,const void *k,const void *v,void *o,float *lse,const int *qs,const int *ks,
    int maxq,int reqs,int64_t q0,int64_t q1,int64_t k0,int64_t k1,int64_t v0,int64_t v1,
    int64_t o0,int64_t o1,int64_t l0,int64_t l1,float scale,void *stream) {
    using namespace fattn_rdna35;
    flash_attn_rdna35<256,8,false><<<dim3(2*((maxq+7)/8),1,reqs),dim3(256),0,(hipStream_t)stream>>>(
       (const char*)q,(const char*)k,(const char*)v,nullptr,nullptr,nullptr,(uint16_t*)o,
       nullptr,nullptr,nullptr,scale,0.0f,maxq,16,262144,8,
       q0*2,q1*2,0,k0*2,k1*2,0,v0*2,v1*2,0,0,1,0,0,qs,ks,o0,o1,lse,l0,l1);
    return (int)hipGetLastError();
}
