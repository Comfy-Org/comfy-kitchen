// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <nanobind/ndarray.h>

#include "tensor.h"

namespace comfy::tensor {

inline DType dtype_from_dlpack(const nanobind::dlpack::dtype& dtype) {
    using DTypeCode = nanobind::dlpack::dtype_code;
    const auto code = static_cast<DTypeCode>(dtype.code);
    if (code == DTypeCode::Float) {
        if (dtype.bits == 32) return DType::Float32;
        if (dtype.bits == 16) return DType::Float16;
        if (dtype.bits == 8) return DType::Float8E4M3;
    } else if (code == DTypeCode::Bfloat && dtype.bits == 16) {
        return DType::BFloat16;
    } else if (code == DTypeCode::UInt && dtype.bits == 8) {
        return DType::UInt8;
    } else if (code == DTypeCode::Int && dtype.bits == 8) {
        return DType::Int8;
    }
    return DType::Unknown;
}

template <std::size_t Rank, typename... Args>
TensorArg<Rank> make_tensor_arg(const nanobind::ndarray<Args...>& array) {
    if (array.ndim() != Rank) {
        throw std::runtime_error("unexpected tensor rank");
    }
    const DType dtype = dtype_from_dlpack(array.dtype());
    if (dtype == DType::Unknown) {
        throw std::runtime_error("unsupported tensor dtype");
    }
    TensorArg<Rank> arg{};
    arg.data = const_cast<void*>(static_cast<const void*>(array.data()));
    arg.meta.dtype = dtype;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
        arg.meta.sizes[axis] = static_cast<std::int64_t>(array.shape(axis));
        arg.meta.strides[axis] = array.stride(axis);
    }
    return arg;
}

}  // namespace comfy::tensor
