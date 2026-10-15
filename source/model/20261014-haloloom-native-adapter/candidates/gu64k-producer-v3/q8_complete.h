// Copyright 2026 Ciru. SPDX-License-Identifier: Apache-2.0
// Isolated complete routed-Q8 operation; no serving hooks or implicit GPU work.
#pragma once
#include <stddef.h>
#include <stdint.h>

// All offsets are bytes in a caller-owned, at-least-256B-aligned device arena.
// gate/up/middle are FP32 [Tcap*Top,I]; route_out FP32 [Tcap*Top,H].
// final_f32 is the pre-BF16 reduction, retained for numerical diagnostics.
// queue is uint32[2]: descriptor count, valid route count, both remain on GPU.
struct OrnithQ8CompleteLayout {
    size_t workspace_bytes;
    size_t counts, offsets, cursors, routes, descriptors, queue;
    size_t gate, up, middle, route_out, final_f32;
    size_t descriptor_capacity;
};

#ifdef __cplusplus
extern "C" {
#endif
// CPU-only layout query. Return 0 or hipErrorInvalidValue as int.
// Supports 0<=Tcap<=32768, 1<=E<=256, 1<=Top<=min(E,32),
// H/I multiples of 32 in [32,8192]. Ornith uses E256,H2048,I512,Top8.
int ornith_q8_complete_get_layout(int Tcap, int E, int H, int I, int Top,
                                  struct OrnithQ8CompleteLayout* out);

// Reusable GPU-only grouping into this layout; covers EVERY nonempty expert,
// including 1..4 rows. IDs [T,Top] use -1 for disabled slots. Other invalid IDs
// are skipped and set flag 0x100. No device-to-host count or synchronization.
int ornith_q8_complete_group_routes(const int32_t* ids, void* workspace,
    size_t bytes, uint32_t* flags, int T, int Tcap, int E, int H, int I, int Top,
    void* stream);

// Full operation: grouping -> gate + up -> FP32 SiLU*up -> down -> FP32
// weighted reduction -> BF16 output. Original Q8 basis; NO H128 transforms.
//
// input_bf16 [T,H], route_weights_f32 [T,Top], ids [T,Top], output_bf16 [T,H].
// Separate gate/up packed Q8: codes [E,I/16,H/32,2,16,16], scale_bits
// [E,I/16,H/32,16]. Down swaps I/H. Codes and raw FP16 scale bits unchanged.
// Gate input rounds BF16->FP16, down rounds FP32->FP16 directly. No BF16
// intermediate. Every input/weight/output buffer must be distinct from arena.
//
// flags is caller-initialized device uint32, not cleared by this entry:
// 1 bad descriptor, 2 bad route, 4 bad activation row, 8 nonfinite F16 operand,
// 16 nonfinite projection, 0x100 invalid ID, 0x200 nonfinite SiLU*up,
// 0x400 nonfinite final reduction/BF16. Inspect after stream completion.
// Success means launch admission only; no allocation, copies to host, or sync.
int ornith_q8_complete_f16_launch(
    const uint16_t* input_bf16, const float* route_weights_f32, const int32_t* ids,
    const int8_t* gate_codes, const uint16_t* gate_scale_bits,
    const int8_t* up_codes, const uint16_t* up_scale_bits,
    const int8_t* down_codes, const uint16_t* down_scale_bits,
    void* workspace, size_t bytes, uint16_t* output_bf16, uint32_t* flags,
    int T, int Tcap, int E, int H, int I, int Top, void* stream);
#ifdef __cplusplus
}
#endif
