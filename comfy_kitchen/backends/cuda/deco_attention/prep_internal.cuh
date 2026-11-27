// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "prep.h"
#include "layout.h"

namespace comfy_deco {
struct PrepGeometry {
    int sequence, heads, tiles;
};

struct InputView {
    const char* data;
    int64_t token_bytes, head_bytes;

    __device__ __forceinline__ const char* row(int token, int head) const {
        return data + token * token_bytes + head * head_bytes;
    }
};

struct PrepBuffers {
    unsigned char *q4, *q_sf, *image;
    uint16_t *ut, *side;
    float *score_scale, *v4_scale, *chosen_scales;
    uint32_t* maxima;
};

cudaError_t basis_and_v(PrepGeometry shape, InputView q, InputView k, InputView v,
                        const int32_t* rows, const void* rotation, PrepBuffers out,
                        int sms, cudaStream_t stream);
} // namespace comfy_deco
