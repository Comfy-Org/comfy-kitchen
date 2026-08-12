#pragma once

#include <tuple>

namespace at::cuda::philox {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState state) {
    return {state.seed, state.offset};
}

} // namespace at::cuda::philox
