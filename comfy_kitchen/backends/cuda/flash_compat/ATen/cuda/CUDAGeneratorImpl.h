#pragma once

#include <cstdint>

namespace at {

struct PhiloxCudaState {
    uint64_t seed = 0;
    uint64_t offset = 0;
};

} // namespace at
