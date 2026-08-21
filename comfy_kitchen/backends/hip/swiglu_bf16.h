// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

union Bf16Bits {
    __bf16 value;
    uint16_t bits;
};

__forceinline__ __device__ __bf16 bf16_from_bits(uint16_t bits) {
    Bf16Bits result{};
    result.bits = bits;
    return result.value;
}

// Match torch.nn.functional.silu(gate) * up for BF16 operands: SiLU is rounded
// to BF16 before the multiply and the product is rounded to BF16 on return.
__forceinline__ __device__ __bf16 swiglu_bf16_value(
    __bf16 gate, __bf16 up) {
    Bf16Bits gate_storage{};
    gate_storage.value = gate;
    const float gate_f = static_cast<float>(gate);
    __bf16 silu;
    // PyTorch's current ROCm BF16 SiLU differs from the native float expf
    // path at four finite BF16 inputs. Preserve those exact rounded values;
    // every other finite BF16 input maps identically on gfx12.
    switch (gate_storage.bits) {
        case 0x40be: silu = bf16_from_bits(0x40bd); break;  //  5.9375
        case 0xc2af: silu = bf16_from_bits(0x8395); break;  // -87.5
        case 0xc2b0: silu = bf16_from_bits(0x8335); break;  // -88.0
        case 0xc2b1: silu = bf16_from_bits(0x82dd); break;  // -88.5
        default:
            silu = static_cast<__bf16>(
                gate_f / (1.0f + expf(-gate_f)));
            break;
    }
    return static_cast<__bf16>(
        static_cast<float>(silu) * static_cast<float>(up));
}

}  // namespace comfy::hip_backend
