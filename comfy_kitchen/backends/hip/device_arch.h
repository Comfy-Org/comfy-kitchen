// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Per-device hardware queries shared by the launchers that pick a schedule.
#pragma once

#include <atomic>
#include <cstring>

#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

// Evaluates Select(device) once per device ordinal and caches the result.
template <bool (*Select)(int)>
inline bool cached_device_query() {
    constexpr int kMaxDevices = 16;
    static std::atomic<int> cache[kMaxDevices] = {};
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return false;
    if (device < 0 || device >= kMaxDevices) return Select(device);

    int selected = cache[device].load(std::memory_order_relaxed);
    if (selected == 0) {
        selected = Select(device) ? 2 : 1;
        cache[device].store(selected, std::memory_order_relaxed);
    }
    return selected == 2;
}

inline bool select_nonduplicated_wmma(int device) {
    hipDeviceProp_t properties{};
    return hipGetDeviceProperties(&properties, device) == hipSuccess &&
        (std::strncmp(properties.gcnArchName, "gfx117", 6) == 0 ||
         std::strncmp(properties.gcnArchName, "gfx12", 5) == 0);
}

// gfx12 and gfx117 use the non-duplicated WMMA operand layout. The tuned
// INT8 GEMM and 512-thread ConvRot schedules were tuned on these targets;
// other WMMA targets keep the generic schedules.
inline bool use_nonduplicated_wmma_schedule() {
    return cached_device_query<select_nonduplicated_wmma>();
}

inline bool select_has_wmma(int device) {
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, device) != hipSuccess) return false;
    return std::strncmp(properties.gcnArchName, "gfx11", 5) == 0 ||
        std::strncmp(properties.gcnArchName, "gfx12", 5) == 0;
}

// Matrix cores: the line the WMMA policies in mma.h are built on, and what the
// INT8 GEMM dispatches on. RDNA2 and older take the SIMT v_dot4 path instead.
inline bool device_has_wmma() {
    return cached_device_query<select_has_wmma>();
}

}  // namespace comfy::hip_backend
