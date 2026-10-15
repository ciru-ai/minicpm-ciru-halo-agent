// Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "q8_complete.h"

// NEW quantized model arithmetic, not original-G32 compatibility. Weight codes
// are signed I8 [E,N/16,K/32,2,16,16]; one FP16 scale [E,N/16,K/256,16]
// covers eight successive K32 code tiles. Scales may be FP16 subnormal.
// Original BF16 weights are H128-transformed and quantized separately; this
// consumer never requantizes or reconstructs existing G32 banks.
//
// Layout keeps the retained complete-op prefix; private diagnostic tail stores
// FP32 H128 activations, contiguous I8 activations, and raw FP16 G256 scales.
struct OrnithW8G256H128QuantLayout {
    size_t gate_transformed, gate_q, gate_scales;
    size_t down_transformed, down_q, down_scales;
};

extern "C" int ornith_w8g256_h128_get_layout(int Tcap,int E,int H,int I,int Top,
    OrnithQ8CompleteLayout* out);
extern "C" int ornith_w8g256_h128_get_quant_layout(int Tcap,int E,int H,int I,int Top,
    OrnithW8G256H128QuantLayout* out);
extern "C" int ornith_w8g256_h128_group_routes(const int32_t* ids,void* workspace,
    size_t bytes,uint32_t* flags,int T,int Tcap,int E,int H,int I,int Top,void* stream);

// Supports 0<=T<=Tcap<=32768; 1<=E<=256; 1<=Top<=min(E,32); H/I multiples
// of256 in[256,8192]. E2/Top2 is the bounded pilot; E256/Top8 is the real shape.
// x BF16[T,H]; route weights FP32[T,Top]; ids I32[T,Top], -1 disables a slot.
// Gate/up output dimensions I/H; down H/I. Arena aligned256 and all operands
// nonoverlapping. Codes/activation rows require aligned16; FP16/BF16 aligned2;
// routing/flags aligned4. Buffers must live until the supplied stream finishes.
//
// Gate expands BF16 to FP32; down consumes FP32 portable-SiLU*up directly.
// Both use ascending H128 steps1,2,4,8,16,32,64 and multiply0x3db504f3.
// Parent A8/G256: sanitize nonfinite to0 and flag; maximum/127 uses FP32 RN;
// nonzero valid scale=max(raw,2^-14) rounded FP16 RN, zero/invalid scale1;
// q=clip(RNE(value/stored_scale),-127,127), zero/invalid raw gives q0.
// Each G256 dot uses16 ascending K16 signed IU8 WMMAs, then FP32 sa*float(dot),
// sw*scaled, ascending FP32 group addition. Build with contraction disabled.
// Grouping, portable FP32 middle and slot-order reduction follow q8_complete.h.
// Flags:1 descriptor,2 route,4 activation row,8 nonfinite transform/A8/weight
// scale,16 projection,0x100 invalid ID,0x200 middle,0x400 reduction/BF16 output.
// No flag reset, allocation, host copy or synchronization. Success is launch
// admission only. No original-G32/F16/BF16 bitwise equality is promised.
extern "C" int ornith_w8g256_h128_a8_launch(
    const uint16_t* input_bf16,const float* route_weights_f32,const int32_t* ids,
    const int8_t* gate_codes,const uint16_t* gate_scale_bits,
    const int8_t* up_codes,const uint16_t* up_scale_bits,
    const int8_t* down_codes,const uint16_t* down_scale_bits,
    void* workspace,size_t bytes,uint16_t* output_bf16,uint32_t* flags,
    int T,int Tcap,int E,int H,int I,int Top,void* stream);
