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
    constexpr std::uint8_t kDLFloat8E4M3FN = 10;
    constexpr std::uint8_t kDLFloat8E5M2 = 12;

    if (dtype.lanes != 1) return DType::Unknown;

    const auto code = static_cast<DTypeCode>(dtype.code);
    if (code == DTypeCode::Float) {
        if (dtype.bits == 32) return DType::Float32;
        if (dtype.bits == 16) return DType::Float16;
    } else if (code == DTypeCode::Bfloat && dtype.bits == 16) {
        return DType::BFloat16;
    } else if (code == DTypeCode::UInt && dtype.bits == 8) {
        return DType::UInt8;
    } else if (code == DTypeCode::Int) {
        if (dtype.bits == 8) return DType::Int8;
        if (dtype.bits == 32) return DType::Int32;
    } else if (code == DTypeCode::Bool) {
        return DType::Bool;
    } else if (dtype.code == kDLFloat8E4M3FN) {
        return DType::Float8E4M3;
    } else if (dtype.code == kDLFloat8E5M2) {
        return DType::Float8E5M2;
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

template <typename... Args>
bool is_contiguous(const nanobind::ndarray<Args...>& array) {
    if (array.size() == 0) return true;
    std::int64_t expected_stride = 1;
    for (std::size_t axis = array.ndim(); axis-- > 0;) {
        if (array.shape(axis) > 1 && array.stride(axis) != expected_stride) {
            return false;
        }
        expected_stride *= array.shape(axis);
    }
    return true;
}

template <std::size_t Rank, typename... Args>
TensorArg<Rank> make_contiguous_tensor_arg(const nanobind::ndarray<Args...>& array) {
    TensorArg<Rank> arg = make_tensor_arg<Rank>(array);
    if (!is_contiguous(array)) {
        throw std::runtime_error("expected a contiguous tensor");
    }
    return arg;
}

template <typename... Args>
TensorArg<1> make_flat_tensor_arg(const nanobind::ndarray<Args...>& array) {
    if (!is_contiguous(array)) {
        throw std::runtime_error("expected a contiguous tensor");
    }
    const DType dtype = dtype_from_dlpack(array.dtype());
    if (dtype == DType::Unknown) {
        throw std::runtime_error("unsupported tensor dtype");
    }
    TensorArg<1> arg{};
    arg.data = const_cast<void*>(static_cast<const void*>(array.data()));
    arg.meta.sizes[0] = static_cast<std::int64_t>(array.size());
    arg.meta.strides[0] = 1;
    arg.meta.dtype = dtype;
    return arg;
}
}  // namespace comfy::tensor
