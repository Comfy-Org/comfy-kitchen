// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "launch.h"

namespace comfy_deco::detail {

struct CoreArguments {
    const void* q4;
    const void* q_sf;
    const void* score_scale;
    const void* v4_scale;
    const void* image;
    void* out;
    void* scratch;
};

template <int Sequence, int Heads>
cudaError_t launch_shape(const CoreArguments& args, int sms, bool single_writer, cudaStream_t stream);

} // namespace comfy_deco::detail
