// Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
// Reuse each effective BF16 weight across verification rows while preserving
// the retained ROW_VEC=32 DOT2 order and per-row XOR reduction exactly.
#include "q8_dense_bf16_live.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>
#ifndef ROW_TILE
#define ROW_TILE 4
#endif
#ifndef ROW_UNROLL
#define ROW_UNROLL 2
#endif
namespace minicpm_head_q8_v1 {
using codes_vec=int8_t __attribute__((ext_vector_type(32)));
using input_vec=uint16_t __attribute__((ext_vector_type(32)));
using short2=int16_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint16_t bf16_rne(float value) {
    uint32_t bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u)return uint16_t((bits>>16)|0x40u);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__device__ __forceinline__ uint16_t finite_weight_bits(int code,float scale) {
    uint32_t bits=__float_as_uint(float(code)*scale);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
template<int N>
__global__ __launch_bounds__(128) void projection(
    const uint16_t* x,const int8_t* codes,const uint16_t* scales,
    uint16_t* out,uint32_t* flags,const int32_t* live_rows,int M) {
    constexpr int K=2048;
    const int lane=threadIdx.x&31;
    const int n=int(blockIdx.y)*4+(threadIdx.x>>5);
    const int row0=int(blockIdx.x)*ROW_TILE;
    const int live=live_rows?*live_rows:M;
    const bool valid=live>=1 && live<=M;
    uint32_t status=valid?0u:64u;
    float acc[ROW_TILE]={};
#pragma unroll ROW_UNROLL
    for(int tile=0;tile<K/(32*32);++tile) {
        const int k=tile*32*32+lane*32;
        const float scale=__half2float(__ushort_as_half(scales[n*(K/32)+k/32]));
        const codes_vec q=*reinterpret_cast<const codes_vec*>(codes+size_t(n)*K+k);
        input_vec weight={};
#pragma unroll
        for(int j=0;j<32;++j)weight[j]=finite_weight_bits(q[j],scale);
#pragma unroll
        for(int r=0;r<ROW_TILE;++r) {
            const bool active=valid && row0+r<live;
            input_vec a={};
            if(active)a=*reinterpret_cast<const input_vec*>(x+size_t(row0+r)*K+k);
#pragma unroll
            for(int j=0;j<32;j+=2) {
                const short2 w={int16_t(weight[j]),int16_t(weight[j+1])};
                const short2 input={int16_t(a[j]),int16_t(a[j+1])};
                acc[r]=__builtin_amdgcn_fdot2_f32_bf16(w,input,acc[r],false);
            }
        }
    }
#pragma unroll
    for(int r=0;r<ROW_TILE;++r) {
#pragma unroll
        for(int delta=16;delta;delta/=2)acc[r]+=__shfl_xor(acc[r],delta,32);
        if(lane==0 && row0+r<M) {
            const bool active=valid && row0+r<live;
            const uint16_t y=active?bf16_rne(acc[r]):0;
            if(active && !isfinite(acc[r]))status|=16u;
            if(active && (y&0x7f80u)==0x7f80u)status|=32u;
            out[size_t(row0+r)*N+n]=y;
        }
    }
    if(status)atomicOr(flags,status);
}
}
static bool admitted(int M,int N,int K) {
    return M==3 && N==248192 && K==2048;
}
extern "C" hipError_t minicpm_draft_head_q8_v1_get_layout(int M,int N,int K,OrnithQ8DenseBF16Layout* layout) {
    if(!layout || !admitted(M,N,K))return hipErrorInvalidValue;
    *layout={size_t(M)*K*2,size_t(N)*K,size_t(N)*(K/32)*2,size_t(M)*N*2,0};return hipSuccess;
}
extern "C" hipError_t minicpm_draft_head_q8_v1_launch(
    const void* x,const void* codes,const void* scales,void* out,void* flags,const int32_t* live_rows,
    int M,int N,int K,hipStream_t stream) {
    if(!admitted(M,N,K) || !x || !codes || !scales || !out || !flags)return hipErrorInvalidValue;
    if(reinterpret_cast<uintptr_t>(x)&63u || reinterpret_cast<uintptr_t>(codes)&31u)return hipErrorInvalidValue;
    if((reinterpret_cast<uintptr_t>(scales)|reinterpret_cast<uintptr_t>(out))&1u ||
       (reinterpret_cast<uintptr_t>(flags)|reinterpret_cast<uintptr_t>(live_rows))&3u)return hipErrorInvalidValue;
    const dim3 grid((M+ROW_TILE-1)/ROW_TILE,N/4);
    minicpm_head_q8_v1::projection<248192><<<grid,128,0,stream>>>(
        static_cast<const uint16_t*>(x),static_cast<const int8_t*>(codes),static_cast<const uint16_t*>(scales),
        static_cast<uint16_t*>(out),static_cast<uint32_t*>(flags),live_rows,M);
    return hipGetLastError();
}
