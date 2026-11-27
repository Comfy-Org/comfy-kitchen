// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace comfy_deco::detail {

// Q4 is token-major packed FP4. The output is token-major BF16.
// Each split pair owns layout::split_pair_bytes and one uint32 counter.
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

// Only the host binding needs the dispatch table.
#ifdef COMFY_DECO_CCS
namespace comfy_deco {

using Launcher = cudaError_t (*)(const detail::CoreArguments&, int, bool, cudaStream_t);
struct Shape {
    std::size_t sequence, heads;
    Launcher launch;
};
inline constexpr Shape shapes[] = {
#define DECO_SHAPE(seq, heads) {seq, heads, detail::launch_shape<seq, heads>},
#include "core_shapes.inc"
#undef DECO_SHAPE
};
inline constexpr int compiled_architectures[] = {COMFY_DECO_CCS};

inline bool arch_supported(int cc) {
    for (int image : compiled_architectures)
        if (image == cc) return true;
    return false;
}

inline Launcher lookup_kernel(std::size_t sequence, std::size_t heads, int cc) {
    if (arch_supported(cc)) {
        for (const auto& shape : shapes)
            if (shape.sequence == sequence && shape.heads == heads) return shape.launch;
    }
    return nullptr;
}

} // namespace comfy_deco
#endif
