// SPDX-License-Identifier: Apache-2.0
#include "launch_detail.h"

namespace comfy_deco {
namespace {

using Launcher = cudaError_t (*)(const detail::CoreArguments&, int, bool, cudaStream_t);
#ifdef COMFY_HAVE_DECO_FP4
struct Shape {
    int sequence, heads;
    Launcher launch;
};

constexpr Shape shapes[] = {
#define DECO_SHAPE(sequence, heads) {sequence, heads, detail::launch_shape<sequence, heads>},
#include "core_shapes.inc"
#undef DECO_SHAPE
};

Launcher lookup(int sequence, int heads) {
    for (const auto& shape : shapes) {
        if (shape.sequence == sequence && shape.heads == heads) return shape.launch;
    }
    return nullptr;
}
#else
Launcher lookup(int, int) { return nullptr; }
#endif

} // namespace

bool core_supported(int sequence, int heads, int cc) {
    switch (cc) {
#ifdef COMFY_DECO_SM120
    case 120:
        return lookup(sequence, heads) != nullptr;
#endif
#ifdef COMFY_DECO_SM121
    case 121:
        return lookup(sequence, heads) != nullptr;
#endif
    default:
        return false;
    }
}

cudaError_t launch_core(int sequence, int heads, int sms, bool single_writer, const void* q4,
                        const void* q_sf, const void* score_scale, const void* v4_scale,
                        const void* image, void* out, void* scratch, cudaStream_t stream) {
    const auto launch = lookup(sequence, heads);
    if (!launch) return cudaErrorNotSupported;
    if (sms <= 0) return cudaErrorInvalidValue;
    return launch({q4, q_sf, score_scale, v4_scale, image, out, scratch}, sms, single_writer, stream);
}

} // namespace comfy_deco
