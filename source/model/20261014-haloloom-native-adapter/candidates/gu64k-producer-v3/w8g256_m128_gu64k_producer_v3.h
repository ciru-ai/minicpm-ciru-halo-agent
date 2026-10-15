// Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "w8g256_h128_a8.h"

// Exact v1 H128/W8-G256/A8-G256 arithmetic and bank/complete-layout contract.
// GU M64/N64/K256 and down M64/N128/K256 projection tiles; fused gate/up emits the exact middle directly:
// sixteen wave32 consumers;48KiB GU/down LDS. Descriptor64 unchanged.
// H/I multiples256 guarantee whole N64 and K256 tiles. All16 increasingK16
// IU8 instructions accumulate into INT32 before the original FP32 scale
// epilogue sa*float(dot),sw*scaled,ascendingG256 sum. No extra model storage.
// Expected bitwise v1 output for finite valid inputs, not new model arithmetic.
// Same shapes, alignments, sticky flags and asynchronous lifetime as v1.
extern "C" int ornith_w8g256_m128_gu64k_producer_v3_get_layout(
    int Tcap,int E,int H,int I,int Top,OrnithQ8CompleteLayout* out);
extern "C" int ornith_w8g256_m128_gu64k_producer_v3_get_quant_layout(
    int Tcap,int E,int H,int I,int Top,OrnithW8G256H128QuantLayout* out);
extern "C" int ornith_w8g256_m128_gu64k_producer_v3_group_routes(
    const int32_t* ids,void* workspace,size_t bytes,uint32_t* flags,
    int T,int Tcap,int E,int H,int I,int Top,void* stream);
extern "C" int ornith_w8g256_m128_gu64k_producer_v3_launch(
    const uint16_t* input_bf16,const float* route_weights_f32,const int32_t* ids,
    const int8_t* gate_codes,const uint16_t* gate_scale_bits,
    const int8_t* up_codes,const uint16_t* up_scale_bits,
    const int8_t* down_codes,const uint16_t* down_scale_bits,
    void* workspace,size_t bytes,uint16_t* output_bf16,uint32_t* flags,
    int T,int Tcap,int E,int H,int I,int Top,void* stream);

// H128 steps1..16 use wave exchanges; steps32/64 use vector parts, in original order.
// Fused A8 preparation emits the identical codes/scales; transformed tails unused.

// Compact offsets require this library query. Unused gate/up and transformed
// offsets are zero sentinels; no valid diagnostic buffers are promised there.

// Distinct child: ordered float4 routed reduction for H2048/Top8; parent fallback retained.
