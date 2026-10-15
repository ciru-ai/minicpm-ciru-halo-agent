// Preserve GU/down finite/error checks; accumulate their bits before one final per-thread OR.
// Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
// Same retained G256 arithmetic; fuse M64 gate/up and exact portable SiLU middle.
// GU M64/N64 and down M64/N128 both stage whole K256 groups in48KiB with two barriers.
// A whole block owns descriptor64/N64 GU or descriptor64/N128 down;16 ordered K16 IU8 WMMAs
// accumulate into one INT32/G256 before the unchanged FP32 scale epilogue.
// No new quantization or bank copy. Gate/up workspace fields are intentionally unused.
#include "w8g256_m128_gu64k_producer_v3.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>
#include "q8_complete_exp.h"

namespace ultra_w8g256_m128_gu64k_producer_v3 {
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));
struct Descriptor { int32_t expert, first, rows, reserved; };
constexpr int ProjectionBlocks = 512;

size_t align256(size_t value) { return (value+255)&~size_t(255); }
template<class T> T* at(void* base,size_t offset) {
    return reinterpret_cast<T*>(static_cast<uint8_t*>(base)+offset);
}
bool valid_shape(int Tcap,int E,int H,int I,int Top) {
    return Tcap>=0 && Tcap<=32768 && E>=1 && E<=256 && Top>=1 && Top<=E && Top<=32 &&
        H>=256 && H<=8192 && H%256==0 && I>=256 && I<=8192 && I%256==0;
}

__global__ void histogram(const int32_t* ids,int32_t* counts,uint32_t* flags,int R,int E) {
    const int route=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(route>=R) return;
    const int e=ids[route];
    if(e>=0 && e<E) atomicAdd(counts+e,1);
    else if(e!=-1) atomicOr(flags,0x100u);
}
__global__ void scan(const int32_t* counts,int32_t* offsets,int32_t* cursors,int E) {
    int total=0;
    for(int e=0;e<E;++e) { offsets[e]=total;cursors[e]=0;total+=counts[e]; }
    offsets[E]=total;
}
__global__ void scatter(const int32_t* ids,const int32_t* offsets,int32_t* cursors,
    int32_t* routes,int R,int E) {
    const int route=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(route>=R) return;
    const int e=ids[route];
    if(e>=0 && e<E) routes[offsets[e]+atomicAdd(cursors+e,1)]=route;
}
__global__ void partition(const int32_t* counts,const int32_t* offsets,
    Descriptor* descriptors,uint32_t* queue,int E) {
    uint32_t count=0;
    for(int e=0;e<E;++e) {
        for(int j=0;j<counts[e];j+=64)
            descriptors[count++]={e,offsets[e]+j,counts[e]-j<64?counts[e]-j:64,0};
    }
    queue[0]=count;queue[1]=uint32_t(offsets[E]);
}

// Parent H128 arithmetic verbatim, with dynamic H/I and this candidate's flags.
// The all-disabled BF16 token branch is uniform across its complete128 block.
template<bool InputBF16> __global__ __launch_bounds__(128) void transform(
    const void* input,float* transformed,const int32_t* ids,uint32_t* flags,
    int K,int E,int Top) {
    const size_t base=size_t(blockIdx.x)*128;
    const int lane=threadIdx.x;
    if constexpr(InputBF16) {
        const size_t token=base/K;
        bool live=false;
        for(int slot=0;slot<Top;++slot) {
            const int id=ids[token*Top+slot];live|=id>=0 && id<E;
        }
        if(!live) { transformed[base+lane]=0.f;return; }
    }
    float value;
    if constexpr(InputBF16)
        value=__uint_as_float(uint32_t(static_cast<const uint16_t*>(input)[base+lane])<<16);
    else value=static_cast<const float*>(input)[base+lane];
    if(!isfinite(value)) atomicOr(flags,8u);
    __shared__ float first[128],second[128];
    first[lane]=value;__syncthreads();
    float* src=first;float* dst=second;
#pragma unroll
    for(int step=1;step<128;step*=2) {
        const int lo=lane&~step;
        dst[lane]=(lane&step)?src[lo]-src[lo+step]:src[lo]+src[lo+step];
        __syncthreads();float* swap=src;src=dst;dst=swap;
    }
    const float result=src[lane]*0x1.6a09e6p-4f;
    transformed[base+lane]=result;
    if(!isfinite(result)) atomicOr(flags,8u);
}

// One wave per G256. Same parent A8 values, maximum reduction, FP16 scale,
// correctly rounded divisions and signed RNE codes; store I8 directly instead
// of the parent's lossless low/high-nibble encoding. No activation sums needed.
__global__ __launch_bounds__(32) void quantize(
    const float* input,int8_t* x_q,uint16_t* x_scale_bits,uint32_t* flags) {
    const int lane=threadIdx.x;
    const size_t group=blockIdx.x;
    float values[8],maximum=0.f;
#pragma unroll
    for(int part=0;part<8;++part) {
        const float original=input[group*256+part*32+lane];
        const bool finite=isfinite(original);
        if(!finite) atomicOr(flags,8u);
        values[part]=finite?original:0.f;
        maximum=fmaxf(maximum,fabsf(values[part]));
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)
        maximum=fmaxf(maximum,__shfl_xor(maximum,delta,32));
    const float raw=__fdiv_rn(maximum,127.f);
    const bool valid=isfinite(raw) && raw<=65504.f;
    if(!valid) atomicOr(flags,8u);
    const __half scale=__float2half_rn(maximum==0.f || !valid?1.f:fmaxf(raw,0x1p-14f));
#pragma unroll
    for(int part=0;part<8;++part) {
        int q=maximum==0.f || !valid?0:__float2int_rn(__fdiv_rn(values[part],__half2float(scale)));
        q=q<-127?-127:(q>127?127:q);
        x_q[group*256+part*32+lane]=int8_t(q);
    }
    if(lane==0) x_scale_bits[group]=__half_as_ushort(scale);
}

// Each wave consumes two independent N16 columns with one activation operand.
// Same ordered IU8 chains and FP32 G256 epilogues; only down tile reuse changes.
// Same ascending H128 FP32 additions/subtractions. Steps1..16 stay in a
// wave; steps32/64 combine register parts. Quantize the identical normalized
// value directly; no transformed-FP32 materialization or extra model math.
// A whole G256 is held as eight values in one wave. H128 small steps
// exchange lanes; steps32/64 combine vector parts in the original order.
// The normalized FP32 values and parent A8 max/division/RNE rules are unchanged.
template<bool InputBF16> __global__ __launch_bounds__(32) void prepare(
    const void* input,int8_t* x_q,uint16_t* x_scale_bits,const int32_t* ids,
    uint32_t* flags,int K,int E,int Top) {
    const int lane=threadIdx.x;
    const size_t group=blockIdx.x,base=group*256;
    if constexpr(InputBF16) {
        const size_t token=base/K;
        bool live=false;
        for(int slot=0;slot<Top;++slot) {
            const int id=ids[token*Top+slot];live|=id>=0 && id<E;
        }
        if(!live) {
#pragma unroll
            for(int part=0;part<8;++part)x_q[base+part*32+lane]=0;
            if(lane==0)x_scale_bits[group]=0x3c00u;
            return;
        }
    }
    float values[8];
#pragma unroll
    for(int part=0;part<8;++part) {
        if constexpr(InputBF16)
            values[part]=__uint_as_float(uint32_t(static_cast<const uint16_t*>(input)[base+part*32+lane])<<16);
        else values[part]=static_cast<const float*>(input)[base+part*32+lane];
        if(!isfinite(values[part]))atomicOr(flags,8u);
    }
#pragma unroll
    for(int step=1;step<=16;step*=2) {
#pragma unroll
        for(int part=0;part<8;++part) {
            const float other=__shfl_xor(values[part],step,32);
            values[part]=(lane&step)?other-values[part]:values[part]+other;
        }
    }
#pragma unroll
    for(int part=0;part<8;part+=2) {
        const float lo=values[part],hi=values[part+1];
        values[part]=lo+hi;values[part+1]=lo-hi;
    }
#pragma unroll
    for(int chunk=0;chunk<8;chunk+=4) {
#pragma unroll
        for(int offset=0;offset<2;++offset) {
            const float lo=values[chunk+offset],hi=values[chunk+offset+2];
            values[chunk+offset]=lo+hi;values[chunk+offset+2]=lo-hi;
        }
    }
    float maximum=0.f;
#pragma unroll
    for(int part=0;part<8;++part) {
        const float normalized=values[part]*0x1.6a09e6p-4f;
        const bool finite=isfinite(normalized);
        if(!finite)atomicOr(flags,8u);
        values[part]=finite?normalized:0.f;
        maximum=fmaxf(maximum,fabsf(values[part]));
    }
#pragma unroll
    for(int delta=16;delta>0;delta/=2)
        maximum=fmaxf(maximum,__shfl_xor(maximum,delta,32));
    const float raw=__fdiv_rn(maximum,127.f);
    const bool valid=isfinite(raw) && raw<=65504.f;
    if(!valid)atomicOr(flags,8u);
    const __half scale=__float2half_rn(maximum==0.f || !valid?1.f:fmaxf(raw,0x1p-14f));
#pragma unroll
    for(int part=0;part<8;++part) {
        int q=maximum==0.f || !valid?0:__float2int_rn(__fdiv_rn(values[part],__half2float(scale)));
        q=q<-127?-127:(q>127?127:q);
        x_q[base+part*32+lane]=int8_t(q);
    }
    if(lane==0)x_scale_bits[group]=__half_as_ushort(scale);
}

template<bool Gate> __global__ __launch_bounds__(512) void projection_m64n128(
    const int8_t* codes,const uint16_t* scale_bits,const int8_t* x_q,const uint16_t* x_scale_bits,
    const Descriptor* descriptors,const int32_t* routes,const uint32_t* queue,
    float* output,uint32_t* flags,int E,int N,int K,int T,int Top) {
    static_assert(!Gate,"This wider projection is the isolated down child");
    __shared__ int32_t shared_w[16*4*128];
    __shared__ int32_t shared_a[16*4*64];
    const int thread=threadIdx.x,lane=thread&31,lane16=lane&15,piece=lane>>4;
    const int wave=thread>>5,load_part=wave,output_column=(wave&3)*16;
    const int output_row=(wave>>2)*16+lane16;
    const int load_row=lane,load_row_hi=lane+32;
    const int tiles=N/128,groups=K/256,code_groups=K/32,R=T*Top;
    const int jobs=int(queue[0])*tiles;
    for(int job=int(blockIdx.x);job<jobs;job+=int(gridDim.x)) {
        const auto desc=descriptors[job/tiles];
        const int tile=job%tiles;
        if(desc.expert<0 || desc.expert>=E || desc.rows<1 || desc.rows>64 ||
            desc.first<0 || desc.first>R-desc.rows) {
            if(thread==0) atomicOr(flags,1u);
            continue;
        }
        bool live=output_row<desc.rows;
        int route=live?routes[desc.first+output_row]:0;
        if(live && (route<0 || route>=R)) { atomicOr(flags,2u);live=false; }
        const int inputrow=live?route:0;
        bool load_live=load_row<desc.rows,load_live_hi=load_row_hi<desc.rows;
        const int load_route=load_live?routes[desc.first+load_row]:0;
        const int load_route_hi=load_live_hi?routes[desc.first+load_row_hi]:0;
        if(load_live && (load_route<0 || load_route>=R)) { atomicOr(flags,2u);load_live=false; }
        if(load_live_hi && (load_route_hi<0 || load_route_hi>=R)) { atomicOr(flags,2u);load_live_hi=false; }
        const int load_inputrow=load_live?load_route:0,load_inputrow_hi=load_live_hi?load_route_hi:0;
        const size_t output_tile0=size_t(desc.expert)*(N/16)+tile*8+(wave&3);
        const size_t scale_bank0=output_tile0*groups,scale_bank1=(output_tile0+4)*groups;
        float values0[8]={},values1[8]={};
        uint32_t local_flags=0;
        for(int group=0;group<groups;++group) {
            i32x8 accum0={},accum1={};
#pragma unroll
            for(int stage=0;stage<1;++stage) {
                const int kstart=group*256+stage*256+load_part*16;
                const int code_group=kstart/32,half=(kstart/16)&1;
#pragma unroll
                for(int half_n=0;half_n<4;++half_n) {
                    const int local_n=half_n*32+lane,n=tile*128+local_n;
                    const size_t bank=(size_t(desc.expert)*(N/16)+n/16)*code_groups+code_group;
                    const size_t wb=((bank*2+half)*16+(n&15))*16;
                    const i32x4 w=*reinterpret_cast<const i32x4*>(codes+wb);
#pragma unroll
                    for(int word=0;word<4;++word)
                        shared_w[(load_part*4+word)*128+local_n]=w[word];
                }
                i32x4 loaded_a={},loaded_a_hi={};
                if(load_live) loaded_a=*reinterpret_cast<const i32x4*>(x_q+size_t(load_inputrow)*K+kstart);
                if(load_live_hi) loaded_a_hi=*reinterpret_cast<const i32x4*>(x_q+size_t(load_inputrow_hi)*K+kstart);
#pragma unroll
                for(int word=0;word<4;++word) {
                    shared_a[(load_part*4+word)*64+load_row]=loaded_a[word];
                    shared_a[(load_part*4+word)*64+load_row_hi]=loaded_a_hi[word];
                }
                __syncthreads();
#pragma unroll 4
                for(int part=0;part<16;++part) {
                    i32x4 w0,w1,a;
#pragma unroll
                    for(int word=0;word<4;++word) {
                        w0[word]=shared_w[(part*4+word)*128+output_column+lane16];
                        w1[word]=shared_w[(part*4+word)*128+output_column+64+lane16];
                        a[word]=shared_a[(part*4+word)*64+output_row];
                    }
                    accum0=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,w0,true,a,accum0,false);
                    accum1=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,w1,true,a,accum1,false);
                }
                __syncthreads();
            }
            const uint32_t own_scale0=scale_bits[(scale_bank0+group)*16+lane16];
            const uint32_t own_scale1=scale_bits[(scale_bank1+group)*16+lane16];
            if(!isfinite(__half2float(__ushort_as_half(uint16_t(own_scale0)))) ||
               !isfinite(__half2float(__ushort_as_half(uint16_t(own_scale1))))) local_flags|=8u;
            const float activation_scale=live?
                __half2float(__ushort_as_half(x_scale_bits[size_t(inputrow)*groups+group])):0.f;
#pragma unroll
            for(int v=0;v<8;++v) {
                const uint16_t bits0=uint16_t(__shfl(own_scale0,2*v+piece,32));
                const uint16_t bits1=uint16_t(__shfl(own_scale1,2*v+piece,32));
                const float scaled0=activation_scale*float(accum0[v]);
                const float scaled1=activation_scale*float(accum1[v]);
                const float product0=__half2float(__ushort_as_half(bits0))*scaled0;
                const float product1=__half2float(__ushort_as_half(bits1))*scaled1;
                values0[v]=values0[v]+product0;
                values1[v]=values1[v]+product1;
            }
        }
        if(live) {
#pragma unroll
            for(int v=0;v<8;++v) {
                output[size_t(route)*N+tile*128+output_column+2*v+piece]=values0[v];
                output[size_t(route)*N+tile*128+output_column+64+2*v+piece]=values1[v];
                local_flags|=(isfinite(values0[v]) && isfinite(values1[v]))?0u:16u;
            }
        }
        if(local_flags)atomicOr(flags,local_flags);
    }
}


__global__ void partition_gate_hybrid(const int32_t* counts,const int32_t* offsets,
    Descriptor* descriptors,uint32_t* queue,int E) {
    uint32_t full=0;
    for(int e=0;e<E;++e)
        for(int j=0;j+128<=counts[e];j+=128)
            descriptors[full++]={e,offsets[e]+j,128,0};
    uint32_t count=full;
    for(int e=0;e<E;++e)
        for(int j=(counts[e]/128)*128;j<counts[e];j+=64)
            descriptors[count++]={e,offsets[e]+j,counts[e]-j<64?counts[e]-j:64,0};
    queue[0]=full;queue[1]=uint32_t(offsets[E]);queue[2]=count-full;queue[3]=1u;
}

// Full M128/N64 gate/up uses64KiB:32KiB weights and32KiB activations.
// Both M64 halves reuse every weight vector, preserving each K16 chain.

// Grouping is complete on this stream and routes remain immutable through GU.
// An invalid value preserves the original route-error bit and masked fallback.
__global__ void validate_grouped_routes(const int32_t* routes,uint32_t* queue,
    uint32_t* flags,int R) {
    const int index=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=int(queue[1]))return;
    const int route=routes[index];
    if(route<0 || route>=R) {
        atomicExch(queue+3,0u);
        atomicOr(flags,2u);
    }
}

__global__ __launch_bounds__(512) void gate_up_full_m128n64(
    const int8_t* gate_codes,const uint16_t* gate_scale_bits,
    const int8_t* up_codes,const uint16_t* up_scale_bits,
    const int8_t* x_q,const uint16_t* x_scale_bits,
    const Descriptor* descriptors,const int32_t* routes,const uint32_t* queue,
    float* output,uint32_t* flags,int E,int N,int K,int T,int Top) {
    __shared__ int32_t shared_g[16*4*64],shared_u[16*4*64];
    __shared__ int32_t shared_a[16*4*128];
    const int thread=threadIdx.x,lane=thread&31,lane16=lane&15,piece=lane>>4;
    const int wave=thread>>5,load_part=wave,output_column=(wave&3)*16;
    const int output_row=(wave>>2)*16+lane16;
    const int tiles=N/64,groups=K/256,code_groups=K/32,R=T*Top;
    const int jobs=int(queue[0])*tiles;
    for(int job=int(blockIdx.x);job<jobs;job+=int(gridDim.x)) {
        const auto desc=descriptors[job/tiles];
        const int tile=job%tiles;
        if(desc.expert<0 || desc.expert>=E || desc.rows!=128 ||
            desc.first<0 || desc.first>R-128) {
            if(thread==0)atomicOr(flags,1u);
            continue;
        }
        if(__builtin_amdgcn_readfirstlane(queue[3])!=0) {
        int route[2],inputrow[2],load_inputrow[4];
#pragma unroll
        for(int m=0;m<2;++m) {
            route[m]=routes[desc.first+output_row+m*64];
            inputrow[m]=route[m]/Top;
        }
#pragma unroll
        for(int quarter=0;quarter<4;++quarter)
            load_inputrow[quarter]=routes[desc.first+lane+quarter*32]/Top;
        const size_t output_tile=size_t(desc.expert)*(N/16)+tile*4+(wave&3);
        const size_t scale_bank=output_tile*groups;
        float gate_values[2][8]={},up_values[2][8]={};
        uint32_t local_flags=0;
        for(int group=0;group<groups;++group) {
            i32x8 gate_accum[2]={},up_accum[2]={};
            const int kstart=group*256+load_part*16;
            const int code_group=kstart/32,half=(kstart/16)&1;
#pragma unroll
            for(int half_n=0;half_n<2;++half_n) {
                const int local_n=half_n*32+lane,n=tile*64+local_n;
                const size_t bank=(size_t(desc.expert)*(N/16)+n/16)*code_groups+code_group;
                const size_t wb=((bank*2+half)*16+(n&15))*16;
                const i32x4 g=*reinterpret_cast<const i32x4*>(gate_codes+wb);
                const i32x4 u=*reinterpret_cast<const i32x4*>(up_codes+wb);
#pragma unroll
                for(int word=0;word<4;++word) {
                    shared_g[(load_part*4+word)*64+local_n]=g[word];
                    shared_u[(load_part*4+word)*64+local_n]=u[word];
                }
            }
#pragma unroll
            for(int quarter=0;quarter<4;++quarter) {
                i32x4 a={};
                a=*reinterpret_cast<const i32x4*>(x_q+size_t(load_inputrow[quarter])*K+kstart);
#pragma unroll
                for(int word=0;word<4;++word)
                    shared_a[(load_part*4+word)*128+lane+quarter*32]=a[word];
            }
            __syncthreads();
#pragma unroll 4
            for(int part=0;part<16;++part) {
                i32x4 g,u;
#pragma unroll
                for(int word=0;word<4;++word) {
                    g[word]=shared_g[(part*4+word)*64+output_column+lane16];
                    u[word]=shared_u[(part*4+word)*64+output_column+lane16];
                }
#pragma unroll
                for(int m=0;m<2;++m) {
                    i32x4 a;
#pragma unroll
                    for(int word=0;word<4;++word)
                        a[word]=shared_a[(part*4+word)*128+output_row+m*64];
                    gate_accum[m]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,g,true,a,gate_accum[m],false);
                    up_accum[m]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,u,true,a,up_accum[m],false);
                }
            }
            __syncthreads();
            const uint32_t own_gate_scale=gate_scale_bits[(scale_bank+group)*16+lane16];
            const uint32_t own_up_scale=up_scale_bits[(scale_bank+group)*16+lane16];
            if(!isfinite(__half2float(__ushort_as_half(uint16_t(own_gate_scale)))) ||
               !isfinite(__half2float(__ushort_as_half(uint16_t(own_up_scale)))))local_flags|=8u;
#pragma unroll
            for(int m=0;m<2;++m) {
                const float activation_scale=
                    __half2float(__ushort_as_half(x_scale_bits[size_t(inputrow[m])*groups+group]));
#pragma unroll
                for(int v=0;v<8;++v) {
                    const uint16_t gbits=uint16_t(__shfl(own_gate_scale,2*v+piece,32));
                    const uint16_t ubits=uint16_t(__shfl(own_up_scale,2*v+piece,32));
                    const float gscaled=activation_scale*float(gate_accum[m][v]);
                    const float uscaled=activation_scale*float(up_accum[m][v]);
                    const float gproduct=__half2float(__ushort_as_half(gbits))*gscaled;
                    const float uproduct=__half2float(__ushort_as_half(ubits))*uscaled;
                    gate_values[m][v]=gate_values[m][v]+gproduct;
                    up_values[m][v]=up_values[m][v]+uproduct;
                }
            }
        }
#pragma unroll
        for(int m=0;m<2;++m) {
            {
#pragma unroll
                for(int v=0;v<8;++v) {
                    const float g=gate_values[m][v],u=up_values[m][v];
                    local_flags|=(isfinite(g) && isfinite(u))?0u:16u;
                    const float denominator=1.f+ornith_portable_exp(-g);
                    const float silu=__fdiv_rn(g,denominator),value=silu*u;
                    output[size_t(route[m])*N+tile*64+output_column+2*v+piece]=value;
                    local_flags|=isfinite(value)?0u:0x200u;
                }
            }
        }
        if(local_flags)atomicOr(flags,local_flags);
        } else {
        int route[2],inputrow[2],load_inputrow[4];
        bool live[2],load_live[4];
#pragma unroll
        for(int m=0;m<2;++m) {
            route[m]=routes[desc.first+output_row+m*64];
            live[m]=route[m]>=0 && route[m]<R;
            if(!live[m])atomicOr(flags,2u);
            inputrow[m]=live[m]?route[m]/Top:0;
            if(live[m] && (inputrow[m]<0 || inputrow[m]>=T)) {
                atomicOr(flags,4u);live[m]=false;
            }
        }
#pragma unroll
        for(int quarter=0;quarter<4;++quarter) {
            const int r=routes[desc.first+lane+quarter*32];
            load_live[quarter]=r>=0 && r<R;
            if(!load_live[quarter])atomicOr(flags,2u);
            load_inputrow[quarter]=load_live[quarter]?r/Top:0;
            if(load_live[quarter] && (load_inputrow[quarter]<0 || load_inputrow[quarter]>=T)) {
                atomicOr(flags,4u);load_live[quarter]=false;
            }
        }
        const size_t output_tile=size_t(desc.expert)*(N/16)+tile*4+(wave&3);
        const size_t scale_bank=output_tile*groups;
        float gate_values[2][8]={},up_values[2][8]={};
        uint32_t local_flags=0;
        for(int group=0;group<groups;++group) {
            i32x8 gate_accum[2]={},up_accum[2]={};
            const int kstart=group*256+load_part*16;
            const int code_group=kstart/32,half=(kstart/16)&1;
#pragma unroll
            for(int half_n=0;half_n<2;++half_n) {
                const int local_n=half_n*32+lane,n=tile*64+local_n;
                const size_t bank=(size_t(desc.expert)*(N/16)+n/16)*code_groups+code_group;
                const size_t wb=((bank*2+half)*16+(n&15))*16;
                const i32x4 g=*reinterpret_cast<const i32x4*>(gate_codes+wb);
                const i32x4 u=*reinterpret_cast<const i32x4*>(up_codes+wb);
#pragma unroll
                for(int word=0;word<4;++word) {
                    shared_g[(load_part*4+word)*64+local_n]=g[word];
                    shared_u[(load_part*4+word)*64+local_n]=u[word];
                }
            }
#pragma unroll
            for(int quarter=0;quarter<4;++quarter) {
                i32x4 a={};
                if(load_live[quarter])a=*reinterpret_cast<const i32x4*>(x_q+size_t(load_inputrow[quarter])*K+kstart);
#pragma unroll
                for(int word=0;word<4;++word)
                    shared_a[(load_part*4+word)*128+lane+quarter*32]=a[word];
            }
            __syncthreads();
#pragma unroll 4
            for(int part=0;part<16;++part) {
                i32x4 g,u;
#pragma unroll
                for(int word=0;word<4;++word) {
                    g[word]=shared_g[(part*4+word)*64+output_column+lane16];
                    u[word]=shared_u[(part*4+word)*64+output_column+lane16];
                }
#pragma unroll
                for(int m=0;m<2;++m) {
                    i32x4 a;
#pragma unroll
                    for(int word=0;word<4;++word)
                        a[word]=shared_a[(part*4+word)*128+output_row+m*64];
                    gate_accum[m]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,g,true,a,gate_accum[m],false);
                    up_accum[m]=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,u,true,a,up_accum[m],false);
                }
            }
            __syncthreads();
            const uint32_t own_gate_scale=gate_scale_bits[(scale_bank+group)*16+lane16];
            const uint32_t own_up_scale=up_scale_bits[(scale_bank+group)*16+lane16];
            if(!isfinite(__half2float(__ushort_as_half(uint16_t(own_gate_scale)))) ||
               !isfinite(__half2float(__ushort_as_half(uint16_t(own_up_scale)))))local_flags|=8u;
#pragma unroll
            for(int m=0;m<2;++m) {
                const float activation_scale=live[m]?
                    __half2float(__ushort_as_half(x_scale_bits[size_t(inputrow[m])*groups+group])):0.f;
#pragma unroll
                for(int v=0;v<8;++v) {
                    const uint16_t gbits=uint16_t(__shfl(own_gate_scale,2*v+piece,32));
                    const uint16_t ubits=uint16_t(__shfl(own_up_scale,2*v+piece,32));
                    const float gscaled=activation_scale*float(gate_accum[m][v]);
                    const float uscaled=activation_scale*float(up_accum[m][v]);
                    const float gproduct=__half2float(__ushort_as_half(gbits))*gscaled;
                    const float uproduct=__half2float(__ushort_as_half(ubits))*uscaled;
                    gate_values[m][v]=gate_values[m][v]+gproduct;
                    up_values[m][v]=up_values[m][v]+uproduct;
                }
            }
        }
#pragma unroll
        for(int m=0;m<2;++m) {
            if(live[m]) {
#pragma unroll
                for(int v=0;v<8;++v) {
                    const float g=gate_values[m][v],u=up_values[m][v];
                    local_flags|=(isfinite(g) && isfinite(u))?0u:16u;
                    const float denominator=1.f+ornith_portable_exp(-g);
                    const float silu=__fdiv_rn(g,denominator),value=silu*u;
                    output[size_t(route[m])*N+tile*64+output_column+2*v+piece]=value;
                    local_flags|=isfinite(value)?0u:0x200u;
                }
            }
        }
        if(local_flags)atomicOr(flags,local_flags);
        }
    }
}

__global__ __launch_bounds__(512) void gate_up_tail_m64n64(
    const int8_t* gate_codes,const uint16_t* gate_scale_bits,
    const int8_t* up_codes,const uint16_t* up_scale_bits,
    const int8_t* x_q,const uint16_t* x_scale_bits,
    const Descriptor* descriptors,const int32_t* routes,const uint32_t* queue,
    float* output,uint32_t* flags,int E,int N,int K,int T,int Top) {
    // Each word holds4 consecutive signed-I8 K values. Distinct rows are
    // contiguous LDS addresses; duplicate operand lanes read the same word.
    __shared__ int32_t shared_g[16*4*64],shared_u[16*4*64];
    __shared__ int32_t shared_a[16*4*64];
    const int thread=threadIdx.x,lane=thread&31,lane16=lane&15,piece=lane>>4;
    const int wave=thread>>5,load_part=wave,output_column=(wave&3)*16;
    const int output_row=(wave>>2)*16+lane16;
    const int load_row=lane,load_row_hi=lane+32;
    const int tiles=N/64,groups=K/256,code_groups=K/32,R=T*Top;
    const int jobs=int(queue[2])*tiles;
    // All descriptor guards and the job loop are whole-block uniform.
    for(int job=int(blockIdx.x);job<jobs;job+=int(gridDim.x)) {
        const auto desc=descriptors[int(queue[0])+job/tiles];
        const int tile=job%tiles;
        if(desc.expert<0 || desc.expert>=E || desc.rows<1 || desc.rows>64 ||
            desc.first<0 || desc.first>R-desc.rows) {
            if(thread==0) atomicOr(flags,1u);
            continue;
        }
        bool live=output_row<desc.rows;
        int route=live?routes[desc.first+output_row]:0;
        if(live && (route<0 || route>=R)) { atomicOr(flags,2u);live=false; }
        const int inputrow=live?route/Top:0;
        if(live && (inputrow<0 || inputrow>=T)) {
            atomicOr(flags,4u);live=false;
        }
        // K256 has sixteen K16 producer parts; every lane produces both row halves.
        bool load_live=load_row<desc.rows;
        const int load_route=load_live?routes[desc.first+load_row]:0;
        if(load_live && (load_route<0 || load_route>=R)) { atomicOr(flags,2u);load_live=false; }
        const int load_inputrow=load_live?load_route/Top:0;
        if(load_live && (load_inputrow<0 || load_inputrow>=T)) { atomicOr(flags,4u);load_live=false; }
        bool load_live_hi=load_row_hi<desc.rows;
        const int load_route_hi=load_live_hi?routes[desc.first+load_row_hi]:0;
        if(load_live_hi && (load_route_hi<0 || load_route_hi>=R)) { atomicOr(flags,2u);load_live_hi=false; }
        const int load_inputrow_hi=load_live_hi?load_route_hi/Top:0;
        if(load_live_hi && (load_inputrow_hi<0 || load_inputrow_hi>=T)) { atomicOr(flags,4u);load_live_hi=false; }
        const size_t output_tile=size_t(desc.expert)*(N/16)+tile*4+(wave&3);
        const size_t scale_bank=output_tile*groups;
        float gate_values[8]={},up_values[8]={};
        uint32_t local_flags=0;
        for(int group=0;group<groups;++group) {
            i32x8 gate_accum={},up_accum={};
#pragma unroll
            for(int stage=0;stage<1;++stage) {
                const int kstart=group*256+stage*256+load_part*16;
                const int code_group=kstart/32,half=(kstart/16)&1;
                // Producer wave owns one K16 part; two N32 rounds uniquely
                // cover all64 weight rows without duplicate global reads.
#pragma unroll
                for(int half_n=0;half_n<2;++half_n) {
                    if(wave<16) {
                    const int local_n=half_n*32+lane,n=tile*64+local_n;
                    const size_t bank=(size_t(desc.expert)*(N/16)+n/16)*code_groups+code_group;
                    const size_t wb=((bank*2+half)*16+(n&15))*16;
                    const i32x4 g=*reinterpret_cast<const i32x4*>(gate_codes+wb);
                    const i32x4 u=*reinterpret_cast<const i32x4*>(up_codes+wb);
#pragma unroll
                    for(int word=0;word<4;++word)
                        { shared_g[(load_part*4+word)*64+local_n]=g[word];
                          shared_u[(load_part*4+word)*64+local_n]=u[word]; }
                    }
                }
                // Every producer writes its distinct K16/M row; inactive rows
                // still write zero before all16 consumer waves reach the barrier.
                i32x4 loaded_a={},loaded_a_hi={};
                if(load_live) loaded_a=*reinterpret_cast<const i32x4*>(x_q+size_t(load_inputrow)*K+kstart);
                if(load_live_hi) loaded_a_hi=*reinterpret_cast<const i32x4*>(x_q+size_t(load_inputrow_hi)*K+kstart);
#pragma unroll
                for(int word=0;word<4;++word) {
                    shared_a[(load_part*4+word)*64+load_row]=loaded_a[word];
                    shared_a[(load_part*4+word)*64+load_row_hi]=loaded_a_hi[word];
                }
                __syncthreads();
#pragma unroll 4
                for(int part=0;part<16;++part) {
                    i32x4 g,u,a;
#pragma unroll
                    for(int word=0;word<4;++word) {
                        g[word]=shared_g[(part*4+word)*64+output_column+lane16];
                        u[word]=shared_u[(part*4+word)*64+output_column+lane16];
                        a[word]=shared_a[(part*4+word)*64+output_row];
                    }
                    gate_accum=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,g,true,a,gate_accum,false);
                    up_accum=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,u,true,a,up_accum,false);
                }
                // Retain integer accum across all16 ordered K16 parts of this K256 stage. The next stage
                // must not overwrite operands before all16 M16/N16 waves consume.
                __syncthreads();
            }
            const uint32_t own_gate_scale=gate_scale_bits[(scale_bank+group)*16+lane16];
            const uint32_t own_up_scale=up_scale_bits[(scale_bank+group)*16+lane16];
            if(!isfinite(__half2float(__ushort_as_half(uint16_t(own_gate_scale)))) ||
               !isfinite(__half2float(__ushort_as_half(uint16_t(own_up_scale))))) local_flags|=8u;
            const float activation_scale=live?
                __half2float(__ushort_as_half(x_scale_bits[size_t(inputrow)*groups+group])):0.f;
#pragma unroll
            for(int v=0;v<8;++v) {
                const uint16_t gbits=uint16_t(__shfl(own_gate_scale,2*v+piece,32));
                const uint16_t ubits=uint16_t(__shfl(own_up_scale,2*v+piece,32));
                const float gscaled=activation_scale*float(gate_accum[v]);
                const float uscaled=activation_scale*float(up_accum[v]);
                const float gproduct=__half2float(__ushort_as_half(gbits))*gscaled;
                const float uproduct=__half2float(__ushort_as_half(ubits))*uscaled;
                gate_values[v]=gate_values[v]+gproduct;
                up_values[v]=up_values[v]+uproduct;
            }
        }
        if(live) {
#pragma unroll
            for(int v=0;v<8;++v) {
                const float g=gate_values[v],u=up_values[v];
                local_flags|=(isfinite(g) && isfinite(u))?0u:16u;
                const float denominator=1.f+ornith_portable_exp(-g);
                const float silu=__fdiv_rn(g,denominator),result=silu*u;
                output[size_t(route)*N+tile*64+output_column+2*v+piece]=result;
                local_flags|=isfinite(result)?0u:0x200u;
            }
        }
        if(local_flags)atomicOr(flags,local_flags);
    }
}


__global__ void middle_kernel(const float* gate,const float* up,float* middle,
    const int32_t* ids,uint32_t* flags,size_t values,int E,int I) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=values) return;
    const size_t route=index/I;
    if(ids[route]<0 || ids[route]>=E) { middle[index]=0.f;return; }
    const float g=gate[index],u=up[index];
    const float denominator=1.f+ornith_portable_exp(-g);
    const float silu=__fdiv_rn(g,denominator),result=silu*u;
    middle[index]=result;
    if(!isfinite(result)) atomicOr(flags,0x200u);
}
__device__ __forceinline__ uint16_t bf16_rne(float value) {
    unsigned bits=__float_as_uint(value);
    if((bits&0x7fffffffu)>0x7f800000u) return uint16_t((bits>>16)|0x40u);
    return uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__global__ void reduce_slots(const float* routes,const float* weights,const int32_t* ids,
    float* diagnostic,uint16_t* output,uint32_t* flags,size_t values,int E,int H,int Top) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=values) return;
    const size_t token=index/H,column=index%H;
    float sum=0.f;
    for(int slot=0;slot<Top;++slot) {
        const size_t route=token*Top+slot;
        if(ids[route]>=0 && ids[route]<E) {
            const float product=routes[route*H+column]*weights[route];
            sum=sum+product;
        }
    }
    diagnostic[index]=sum;
    const uint16_t result=bf16_rne(sum);
    output[index]=result;
    if(!isfinite(sum) || (result&0x7f80u)==0x7f80u) atomicOr(flags,0x400u);
}
// Fixed Ornith H2048/Top8 view: one block owns1024 contiguous columns.
// Four columns/thread share uniform route IDs/weights, with ordered FP32
// multiply then add for slots0..7. No FMA or changed BF16 rounding boundary.
using f32x4 = float __attribute__((ext_vector_type(4)));
using u16x4 = uint16_t __attribute__((ext_vector_type(4)));
__global__ __launch_bounds__(256) void reduce_slots_h2048_top8_v4(
    const float* routes,const float* weights,const int32_t* ids,
    float* diagnostic,uint16_t* output,uint32_t* flags,int E) {
    const size_t token=blockIdx.x>>1;
    const int column=int(blockIdx.x&1)*1024+int(threadIdx.x)*4;
    f32x4 sum={};
#pragma unroll
    for(int slot=0;slot<8;++slot) {
        const size_t route=token*8+slot;
        const int32_t expert=ids[route];
        if(expert>=0 && expert<E) {
            const f32x4 value=*reinterpret_cast<const f32x4*>(routes+route*2048+column);
            const f32x4 product=value*weights[route];
            sum=sum+product;
        }
    }
    const size_t index=token*2048+column;
    *reinterpret_cast<f32x4*>(diagnostic+index)=sum;
    u16x4 result={};
    bool bad=false;
#pragma unroll
    for(int c=0;c<4;++c) {
        result[c]=bf16_rne(sum[c]);
        bad=bad || !isfinite(sum[c]) || (result[c]&0x7f80u)==0x7f80u;
    }
    *reinterpret_cast<u16x4*>(output+index)=result;
    if(bad)atomicOr(flags,0x400u);
}
OrnithW8G256H128QuantLayout quant_layout(const OrnithQ8CompleteLayout& base,
    int Tcap,int H,int I,int Top) {
    OrnithW8G256H128QuantLayout q{};
    size_t cursor=align256(base.final_f32+size_t(Tcap)*H*4),R=size_t(Tcap)*Top;
    auto add=[&](size_t& field,size_t bytes){field=cursor;cursor=align256(cursor+bytes);};
    q.gate_transformed=0;
    add(q.gate_q,size_t(Tcap)*H);add(q.gate_scales,size_t(Tcap)*(H/256)*2);
    q.down_transformed=0;
    add(q.down_q,R*I);add(q.down_scales,R*(I/256)*2);
    return q;
}

}

extern "C" int ornith_w8g256_m128_gu64k_producer_v3_get_layout(int Tcap,int E,int H,int I,int Top,
    OrnithQ8CompleteLayout* out) {
    using namespace ultra_w8g256_m128_gu64k_producer_v3;
    if(!out || !valid_shape(Tcap,E,H,I,Top)) return int(hipErrorInvalidValue);
    OrnithQ8CompleteLayout layout{};
    size_t cursor=0,R=size_t(Tcap)*Top;
    layout.descriptor_capacity=(R+15)/16+E;
    auto add=[&](size_t& field,size_t bytes){field=cursor;cursor=align256(cursor+bytes);};
    add(layout.counts,size_t(E)*4);add(layout.offsets,size_t(E+1)*4);
    add(layout.cursors,size_t(E)*4);add(layout.routes,R*4);
    add(layout.descriptors,layout.descriptor_capacity*sizeof(Descriptor));add(layout.queue,16);
    layout.gate=layout.up=0;add(layout.middle,R*I*4);
    add(layout.route_out,R*H*4);add(layout.final_f32,size_t(Tcap)*H*4);
    const auto q=quant_layout(layout,Tcap,H,I,Top);
    layout.workspace_bytes=align256(q.down_scales+R*(I/256)*2);*out=layout;
    return int(hipSuccess);
}

extern "C" int ornith_w8g256_m128_gu64k_producer_v3_get_quant_layout(int Tcap,int E,int H,int I,int Top,
    OrnithW8G256H128QuantLayout* out) {
    OrnithQ8CompleteLayout base{};
    if(!out || ornith_w8g256_m128_gu64k_producer_v3_get_layout(Tcap,E,H,I,Top,&base)!=0)
        return int(hipErrorInvalidValue);
    *out=ultra_w8g256_m128_gu64k_producer_v3::quant_layout(base,Tcap,H,I,Top);
    return int(hipSuccess);
}

#define ULTRA_CHECK_LAUNCH() do {const auto err=hipGetLastError();if(err!=hipSuccess)return int(err);} while(0)

extern "C" int ornith_w8g256_m128_gu64k_producer_v3_group_routes(const int32_t* ids,void* workspace,
    size_t bytes,uint32_t* flags,int T,int Tcap,int E,int H,int I,int Top,void* raw_stream) {
    using namespace ultra_w8g256_m128_gu64k_producer_v3;
    OrnithQ8CompleteLayout layout{};
    if(ornith_w8g256_m128_gu64k_producer_v3_get_layout(Tcap,E,H,I,Top,&layout)!=0 || T<0 || T>Tcap ||
        bytes<layout.workspace_bytes) return int(hipErrorInvalidValue);
    if(T==0) return int(hipSuccess);
    if(!ids || !workspace || !flags) return int(hipErrorInvalidValue);
    if(reinterpret_cast<uintptr_t>(workspace)&255u ||
        (reinterpret_cast<uintptr_t>(ids)|reinterpret_cast<uintptr_t>(flags))&3u)
        return int(hipErrorInvalidValue);
    const auto stream=hipStream_t(raw_stream);
    const int R=T*Top;
    const auto err=hipMemsetAsync(at<int32_t>(workspace,layout.counts),0,size_t(E)*4,stream);
    if(err!=hipSuccess) return int(err);
    histogram<<<(R+255)/256,256,0,stream>>>(ids,at<int32_t>(workspace,layout.counts),flags,R,E);ULTRA_CHECK_LAUNCH();
    scan<<<1,1,0,stream>>>(at<int32_t>(workspace,layout.counts),at<int32_t>(workspace,layout.offsets),
        at<int32_t>(workspace,layout.cursors),E);ULTRA_CHECK_LAUNCH();
    scatter<<<(R+255)/256,256,0,stream>>>(ids,at<int32_t>(workspace,layout.offsets),
        at<int32_t>(workspace,layout.cursors),at<int32_t>(workspace,layout.routes),R,E);ULTRA_CHECK_LAUNCH();
    partition_gate_hybrid<<<1,1,0,stream>>>(at<int32_t>(workspace,layout.counts),at<int32_t>(workspace,layout.offsets),
        at<Descriptor>(workspace,layout.descriptors),at<uint32_t>(workspace,layout.queue),E);ULTRA_CHECK_LAUNCH();
    return int(hipSuccess);
}

extern "C" int ornith_w8g256_m128_gu64k_producer_v3_launch(
    const uint16_t* input_bf16,const float* route_weights_f32,const int32_t* ids,
    const int8_t* gate_codes,const uint16_t* gate_scale_bits,
    const int8_t* up_codes,const uint16_t* up_scale_bits,
    const int8_t* down_codes,const uint16_t* down_scale_bits,
    void* workspace,size_t bytes,uint16_t* output_bf16,uint32_t* flags,
    int T,int Tcap,int E,int H,int I,int Top,void* raw_stream) {
    using namespace ultra_w8g256_m128_gu64k_producer_v3;
    OrnithQ8CompleteLayout layout{};
    if(ornith_w8g256_m128_gu64k_producer_v3_get_layout(Tcap,E,H,I,Top,&layout)!=0 || T<0 || T>Tcap ||
        bytes<layout.workspace_bytes) return int(hipErrorInvalidValue);
    if(T==0) return int(hipSuccess);
    if(!input_bf16 || !route_weights_f32 || !ids || !gate_codes || !gate_scale_bits ||
        !up_codes || !up_scale_bits || !down_codes || !down_scale_bits ||
        !workspace || !output_bf16 || !flags) return int(hipErrorInvalidValue);
    if((reinterpret_cast<uintptr_t>(gate_codes)|reinterpret_cast<uintptr_t>(up_codes)|
        reinterpret_cast<uintptr_t>(down_codes))&15u ||
        (reinterpret_cast<uintptr_t>(input_bf16)|reinterpret_cast<uintptr_t>(output_bf16)|
        reinterpret_cast<uintptr_t>(gate_scale_bits)|reinterpret_cast<uintptr_t>(up_scale_bits)|
        reinterpret_cast<uintptr_t>(down_scale_bits))&1u ||
        (reinterpret_cast<uintptr_t>(route_weights_f32)|reinterpret_cast<uintptr_t>(ids)|
        reinterpret_cast<uintptr_t>(flags))&3u || reinterpret_cast<uintptr_t>(workspace)&255u)
        return int(hipErrorInvalidValue);
    const int group_error=ornith_w8g256_m128_gu64k_producer_v3_group_routes(ids,workspace,bytes,flags,T,Tcap,E,H,I,Top,raw_stream);
    if(group_error) return group_error;
    const auto stream=hipStream_t(raw_stream);
    const int R=T*Top;
    const auto descriptors=at<Descriptor>(workspace,layout.descriptors);
    const auto routes=at<int32_t>(workspace,layout.routes);
    const auto queue=at<uint32_t>(workspace,layout.queue);
    const auto q=quant_layout(layout,Tcap,H,I,Top);
    const size_t gate_groups=size_t(T)*(H/256),down_groups=size_t(R)*(I/256);
    prepare<true><<<unsigned(gate_groups),32,0,stream>>>(input_bf16,
        at<int8_t>(workspace,q.gate_q),at<uint16_t>(workspace,q.gate_scales),ids,flags,H,E,Top);ULTRA_CHECK_LAUNCH();
    // Disabled routes must remain zero before the unchanged down H128/quantizer.
    const auto middle_clear=hipMemsetAsync(at<float>(workspace,layout.middle),0,size_t(R)*I*4,stream);
    if(middle_clear!=hipSuccess) return int(middle_clear);
    validate_grouped_routes<<<(R+255)/256,256,0,stream>>>(routes,queue,flags,R);ULTRA_CHECK_LAUNCH();
    gate_up_full_m128n64<<<ProjectionBlocks,512,0,stream>>>(gate_codes,gate_scale_bits,
        up_codes,up_scale_bits,at<int8_t>(workspace,q.gate_q),at<uint16_t>(workspace,q.gate_scales),
        descriptors,routes,queue,at<float>(workspace,layout.middle),flags,E,I,H,T,Top);ULTRA_CHECK_LAUNCH();
    gate_up_tail_m64n64<<<ProjectionBlocks,512,0,stream>>>(gate_codes,gate_scale_bits,
        up_codes,up_scale_bits,at<int8_t>(workspace,q.gate_q),at<uint16_t>(workspace,q.gate_scales),
        descriptors,routes,queue,at<float>(workspace,layout.middle),flags,E,I,H,T,Top);ULTRA_CHECK_LAUNCH();
    prepare<false><<<unsigned(down_groups),32,0,stream>>>(at<float>(workspace,layout.middle),
        at<int8_t>(workspace,q.down_q),at<uint16_t>(workspace,q.down_scales),ids,flags,I,E,Top);ULTRA_CHECK_LAUNCH();
    partition<<<1,1,0,stream>>>(at<int32_t>(workspace,layout.counts),at<int32_t>(workspace,layout.offsets),
        descriptors,queue,E);ULTRA_CHECK_LAUNCH();
    projection_m64n128<false><<<ProjectionBlocks,512,0,stream>>>(down_codes,down_scale_bits,at<int8_t>(workspace,q.down_q),at<uint16_t>(workspace,q.down_scales),
        descriptors,routes,queue,at<float>(workspace,layout.route_out),flags,E,H,I,T,Top);ULTRA_CHECK_LAUNCH();
    const size_t output_values=size_t(T)*H;
    if(H==2048 && Top==8) {
        reduce_slots_h2048_top8_v4<<<unsigned(T*2),256,0,stream>>>(at<float>(workspace,layout.route_out),
            route_weights_f32,ids,at<float>(workspace,layout.final_f32),output_bf16,flags,E);ULTRA_CHECK_LAUNCH();
    } else {
        reduce_slots<<<unsigned((output_values+255)/256),256,0,stream>>>(at<float>(workspace,layout.route_out),
            route_weights_f32,ids,at<float>(workspace,layout.final_f32),output_bf16,flags,output_values,E,H,Top);ULTRA_CHECK_LAUNCH();
    }
    return int(hipSuccess);
}
#undef ULTRA_CHECK_LAUNCH
