// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace comfy::tensor {

enum class DType : std::int32_t {
    Unknown = -1,
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    UInt8 = 3,
    Int8 = 4,
    Float8E4M3 = 5,
    Float8E5M2 = 6,
};

template <std::size_t Rank>
struct TensorMeta {
    static_assert(Rank > 0, "rank-zero tensors are not needed by the initial native ABI");

    std::int64_t sizes[Rank]{};
    std::int64_t strides[Rank]{};
    DType dtype = DType::Unknown;
};

template <std::size_t Rank>
struct TensorArg {
    void* data = nullptr;
    TensorMeta<Rank> meta{};
};

static_assert(std::is_standard_layout_v<TensorMeta<1>>);
static_assert(std::is_trivially_copyable_v<TensorMeta<1>>);
static_assert(std::is_standard_layout_v<TensorArg<1>>);
static_assert(std::is_trivially_copyable_v<TensorArg<1>>);
static_assert(sizeof(TensorArg<1>) == 32);
static_assert(sizeof(TensorArg<4>) == 80);
static_assert(sizeof(TensorArg<6>) == 112);

}  // namespace comfy::tensor
