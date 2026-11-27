// SPDX-License-Identifier: Apache-2.0
// Argument checks shared by the HIP binding translation units.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

inline void check_hip_launch() {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel launch failed: ") + hipGetErrorString(err));
    }
}

// Every kernel takes raw pointers plus caller-supplied extents and indexes up to
// those extents with no bounds of its own, so a tensor smaller than the extents it
// is launched with is an out-of-bounds device access. _C is importable, so these
// are checked here rather than trusted from the Python layer.
inline void require_len(const nb::ndarray<>& t, int64_t need, const char* fn, const char* name) {
    if (need < 0 || t.size() < static_cast<size_t>(need)) {
        throw std::runtime_error(std::string(fn) + ": " + name + " has " +
                                 std::to_string(t.size()) + " elements, needs at least " +
                                 std::to_string(need));
    }
}

// Kernels that synthesize strides from extents instead of reading tensor strides
// require packed row-major operands, including views passed directly to _C.
inline void require_packed_contiguous(const nb::ndarray<>& t, const char* fn, const char* name) {
    int64_t expected = 1;
    for (int axis = static_cast<int>(t.ndim()) - 1; axis >= 0; --axis) {
        if (t.shape(axis) != 1 && t.stride(axis) != expected) {
            throw std::runtime_error(std::string(fn) + ": " + name +
                                     " must be contiguous in the packed layout");
        }
        expected *= static_cast<int64_t>(t.shape(axis));
    }
}

inline void require_positive(int v, const char* fn, const char* name) {
    if (v <= 0) {
        throw std::runtime_error(std::string(fn) + ": " + name + " must be positive, got " +
                                 std::to_string(v));
    }
}
