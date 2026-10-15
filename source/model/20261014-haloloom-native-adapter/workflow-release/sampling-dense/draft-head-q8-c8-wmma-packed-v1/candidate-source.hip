// Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
// Direct BF16-operand dense M16/N32 consumer. Four independent wave32 jobs/block.
// Preserve quantized values and use the original N16/G32 tiled layout.
#include "q8_dense_bf16_live.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

namespace minicpm_head_q8_c8_wmma_packed_v1 {
using i8x16 = int8_t __attribute__((ext_vector_type(16)));
using u16x8 = uint16_t __attribute__((ext_vector_type(8)));
using u16x16 = uint16_t __attribute__((ext_vector_type(16)));
using u32x8 = uint32_t __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));
constexpr int WavesPerBlock=4;

bool valid_shape(int M,int N,int K) {
    return M==16 && N==248192 && K==2048;
}
__device__ __forceinline__ uint16_t bf16_rne(float value) {
    uint32_t bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u) return uint16_t((bits>>16)|0x40u);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__device__ __forceinline__ bool bf16_finite(uint16_t bits) {
    return (bits&0x7f80u)!=0x7f80u;
}

__global__ __launch_bounds__(128) void projection(
    const uint16_t* x,const int8_t* codes,const uint16_t* scale_bits,
    uint16_t* out,uint32_t* flags,const int32_t* live_rows) {
    constexpr int M=16,N=248192,K=2048;
    const int lane=threadIdx.x&31,lane16=lane&15,piece=lane>>4;
    const int ntiles=N/32,groups=K/32;
    const int job=int(blockIdx.x)*WavesPerBlock+(threadIdx.x>>5);
    const int ntile=job,m=lane16; // N is exactly divisible by four N32 wave jobs.
    // Read the current device scalar once. No row or wave exits on liveness;
    // all weight checks and operand exchanges/barriers remain active.
    const int requested_live_rows=live_rows?*live_rows:M;
    const bool valid_live_rows=requested_live_rows>=1 && requested_live_rows<=M;
    const int effective_live_rows=valid_live_rows?requested_live_rows:0;
    const bool live=m<effective_live_rows;
    const int n=ntile*32+lane;
    f32x8 acc0={},acc1={};
    uint32_t local_flags=valid_live_rows?0u:64u;
    for(int group=0;group<groups;++group) {
        // Low/high halfwaves load distinct N16 rows, then exchange packed operands.
        const size_t bank=size_t(n/16)*groups+group;
        const float scale=__half2float(__ushort_as_half(scale_bits[bank*16+lane16]));
        if(!isfinite(scale)) local_flags|=8u;
#pragma unroll
        for(int half=0;half<2;++half) {
            const size_t wi=((bank*2+half)*16+lane16)*16;
            const i8x16 q=*reinterpret_cast<const i8x16*>(codes+wi);
            u16x16 w,a={};
#pragma unroll
            for(int j=0;j<16;++j) {
                // Separate multiply then integer RNE: no scale/accumulator fusion.
                const float decoded=float(q[j])*scale;
                w[j]=bf16_rne(decoded);
                if(!bf16_finite(w[j])) local_flags|=8u;
            }
            if(live && piece==0) {
                const uint16_t* input=x+size_t(m)*K+group*32+half*16;
                const u16x8 lo=*reinterpret_cast<const u16x8*>(input);
                const u16x8 hi=*reinterpret_cast<const u16x8*>(input+8);
#pragma unroll
                for(int j=0;j<8;++j) { a[j]=lo[j];a[j+8]=hi[j]; }
#pragma unroll
                for(int j=0;j<16;++j) if(!bf16_finite(a[j])) local_flags|=8u;
            }
            const u32x8 wb=__builtin_bit_cast(u32x8,w),ab=__builtin_bit_cast(u32x8,a);
            u32x8 w0b,w1b,abroadcast;
#pragma unroll
            for(int j=0;j<8;++j) {
                // All32 lanes reach exchanges, including padded M rows.
                const uint32_t peer_w=__shfl_xor(wb[j],16,32);
                const uint32_t peer_a=__shfl_xor(ab[j],16,32);
                w0b[j]=piece?peer_w:wb[j];w1b[j]=piece?wb[j]:peer_w;
                abroadcast[j]=piece?peer_a:ab[j];
            }
            const u16x16 w0=__builtin_bit_cast(u16x16,w0b);
            const u16x16 w1=__builtin_bit_cast(u16x16,w1b);
            const u16x16 av=__builtin_bit_cast(u16x16,abroadcast);
            acc0=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(w0,av,acc0);
            acc1=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(w1,av,acc1);
        }
    }
    if(m<M) {
#pragma unroll
        for(int v=0;v<8;++v) {
            uint16_t y0=0,y1=0;
            if(live) {
                y0=bf16_rne(acc0[v]);y1=bf16_rne(acc1[v]);
                if(!isfinite(acc0[v]) || !isfinite(acc1[v])) local_flags|=16u;
                if(!bf16_finite(y0) || !bf16_finite(y1)) local_flags|=32u;
            }
            out[size_t(m)*N+ntile*32+2*v+piece]=y0;
            out[size_t(m)*N+ntile*32+16+2*v+piece]=y1;
        }
    }
    if(local_flags) atomicOr(flags,local_flags);
}
} // namespace minicpm_head_q8_c8_wmma_packed_v1

extern "C" hipError_t minicpm_draft_head_q8_c8_wmma_packed_v1_get_layout(
    int M,int N,int K,OrnithQ8DenseBF16Layout* layout) {
    if(!layout || !minicpm_head_q8_c8_wmma_packed_v1::valid_shape(M,N,K)) return hipErrorInvalidValue;
    *layout={size_t(M)*K*2,size_t(N)*K,size_t(N)*K/16,size_t(M)*N*2,0};
    return hipSuccess;
}
extern "C" hipError_t minicpm_draft_head_q8_c8_wmma_packed_v1_launch(
    const void* x,const void* codes,const void* scale_bits,void* out,void* flags,const int32_t* live_rows,
    int M,int N,int K,hipStream_t stream) {
    if(!minicpm_head_q8_c8_wmma_packed_v1::valid_shape(M,N,K) || !x || !codes || !scale_bits || !out || !flags)
        return hipErrorInvalidValue;
    if((reinterpret_cast<uintptr_t>(x)|reinterpret_cast<uintptr_t>(codes))&15u)
        return hipErrorInvalidValue;
    if((reinterpret_cast<uintptr_t>(scale_bits)|reinterpret_cast<uintptr_t>(out))&1u ||
       (reinterpret_cast<uintptr_t>(flags)|reinterpret_cast<uintptr_t>(live_rows))&3u)
        return hipErrorInvalidValue;
    const int jobs=((M+15)/16)*(N/32);
    minicpm_head_q8_c8_wmma_packed_v1::projection<<<(jobs+3)/4,128,0,stream>>>(
        static_cast<const uint16_t*>(x),static_cast<const int8_t*>(codes),
        static_cast<const uint16_t*>(scale_bits),static_cast<uint16_t*>(out),
        static_cast<uint32_t*>(flags),live_rows);
    return hipGetLastError();
}
