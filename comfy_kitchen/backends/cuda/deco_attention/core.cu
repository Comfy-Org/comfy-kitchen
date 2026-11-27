// SPDX-License-Identifier: Apache-2.0
#include "core.cuh"

#define DECO_SHAPE(sequence, heads) \
    template cudaError_t comfy_deco::detail::launch_shape<sequence, heads>( \
        const comfy_deco::detail::CoreArguments&, int, bool, cudaStream_t);
#include "core_shapes.inc"
#undef DECO_SHAPE
