// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "deco_attention/launch.h"
#ifdef COMFY_HAVE_DECO_FP4
#include "deco_attention/prep.h"
#endif

namespace nb = nanobind;
namespace {
using Tensor = nb::ndarray<nb::device::cuda>;
using Dense = nb::ndarray<nb::device::cuda, nb::c_contig>;
using Rows = nb::ndarray<const int32_t, nb::device::cuda, nb::c_contig>;
using Workspace = nb::ndarray<uint8_t, nb::device::cuda, nb::c_contig>;
constexpr nb::dlpack::dtype bf16{static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

size_t workspace_size(int sequence, int heads, int device) {
#ifdef COMFY_HAVE_DECO_FP4
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device));
    if (comfy_deco::core_supported(sequence, heads, props.major * 10 + props.minor))
        return comfy_deco::prep_layout(sequence, heads, props.multiProcessorCount).bytes;
#endif
    return 0;
}

#ifdef COMFY_HAVE_DECO_FP4
void sample_rows(Tensor tags, Rows order, nb::ndarray<int32_t, nb::device::cuda, nb::c_contig> rows,
                 Workspace workspace, uintptr_t stream_handle) {
    int device;
    check(cudaGetDevice(&device));
    const auto dtype = tags.dtype();
    const bool is_unsigned = dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
    const bool is_signed = dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    if (tags.ndim() != 1 || tags.shape(0) < 64 || tags.shape(0) > 131072 ||
        tags.stride(0) < 0 || tags.device_id() != device || dtype.lanes != 1 ||
        !((is_signed && (dtype.bits == 8 || dtype.bits == 16 || dtype.bits == 32 || dtype.bits == 64)) ||
          (is_unsigned && dtype.bits == 8)))
        throw std::invalid_argument("Invalid DeCo token tags");
    const int sequence = static_cast<int>(tags.shape(0));
    const size_t count_bytes = sizeof(uint32_t) * ((tags.shape(0) + 255) / 256);
    if (order.ndim() != 1 || order.shape(0) != tags.shape(0) || order.device_id() != device ||
        rows.ndim() != 1 || rows.size() != static_cast<size_t>((std::max)(64, sequence / 256)) ||
        rows.device_id() != device || workspace.ndim() != 1 || workspace.device_id() != device ||
        workspace.size() < count_bytes || reinterpret_cast<uintptr_t>(workspace.data()) % alignof(uint32_t) ||
        reinterpret_cast<uintptr_t>(order.data()) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(rows.data()) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(tags.data()) % (dtype.bits / 8))
        throw std::invalid_argument("Invalid DeCo sampling buffers");
    check(comfy_deco::sample_tags(tags.data(), dtype.bits, is_unsigned, tags.stride(0),
                                  order.data(), rows.data(), workspace.data(), sequence,
                                  reinterpret_cast<cudaStream_t>(stream_handle)));
}

void forward(Tensor q, Tensor k, Tensor v, Rows rows, Dense rotation, Dense out,
             Workspace workspace, bool single_writer, uintptr_t stream_handle) {
    int device;
    check(cudaGetDevice(&device));
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device));
    if (q.ndim() != 4 || q.shape(1) > INT32_MAX || q.shape(2) > INT32_MAX)
        throw std::invalid_argument("DeCo requires four-dimensional BHND inputs");
    const int heads = static_cast<int>(q.shape(1));
    const int sequence = static_cast<int>(q.shape(2));
    const int sms = props.multiProcessorCount;
    if (!comfy_deco::core_supported(sequence, heads, props.major * 10 + props.minor))
        throw std::invalid_argument("No DeCo kernel for this device and shape");
    for (const auto& input : {q, k, v}) {
        if (input.device_id() != device || input.dtype() != bf16 || input.ndim() != 4 ||
            input.shape(0) != 1 || input.shape(1) != q.shape(1) || input.shape(2) != q.shape(2) ||
            input.shape(3) != 128 || input.stride(3) != 1 || input.stride(2) <= 0 ||
            input.stride(2) % 8 || (heads > 1 && (input.stride(1) <= 0 || input.stride(1) % 8)) ||
            reinterpret_cast<uintptr_t>(input.data()) % 16)
            throw std::invalid_argument("Expected aligned BF16 DeCo inputs on the current CUDA device");
    }
    const auto w = comfy_deco::prep_layout(sequence, heads, sms);
    if (rows.device_id() != device || rows.ndim() != 1 ||
        rows.shape(0) != static_cast<size_t>((std::max)(64, sequence / 256)) ||
        rotation.device_id() != device || rotation.dtype() != bf16 || rotation.ndim() != 2 ||
        rotation.shape(0) != 128 || rotation.shape(1) != 128 ||
        reinterpret_cast<uintptr_t>(rotation.data()) % 16 ||
        out.device_id() != device || out.dtype() != bf16 || out.ndim() != 4 ||
        out.shape(0) != 1 || out.shape(1) != q.shape(2) || out.shape(2) != q.shape(1) ||
        out.shape(3) != 128 || reinterpret_cast<uintptr_t>(out.data()) % 16 ||
        workspace.device_id() != device || workspace.ndim() != 1 || workspace.size() < w.bytes ||
        reinterpret_cast<uintptr_t>(workspace.data()) % 256)
        throw std::invalid_argument("Invalid DeCo sample rows, rotation, output or workspace");
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    auto* p = workspace.data();
    check(comfy_deco::prepare(q.data(), k.data(), v.data(), rows.data(), rotation.data(),
                              p, sequence, heads, sms, q.stride(2)*2, q.stride(1)*2,
                              k.stride(2)*2, k.stride(1)*2, v.stride(2)*2, v.stride(1)*2, stream));
    check(comfy_deco::launch_core(sequence, heads, sms, single_writer, p + w.q4, p + w.q_sf,
                                  p + w.score_scale, p + w.v4_scale, p + w.img, out.data(),
                                  p + w.split, stream));
}
#endif
}  // namespace

void register_deco_attention(nb::module_& m) {
    nb::list shapes, architectures;
#ifdef COMFY_HAVE_DECO_FP4
#define DECO_SHAPE(sequence, heads) shapes.append(nb::make_tuple(sequence, heads));
#include "deco_attention/core_shapes.inc"
#undef DECO_SHAPE
#ifdef COMFY_DECO_SM120
    architectures.append(nb::make_tuple(12, 0));
#endif
#ifdef COMFY_DECO_SM121
    architectures.append(nb::make_tuple(12, 1));
#endif
#endif
    m.attr("deco_attention_shapes") = shapes;
    m.attr("deco_attention_architectures") = architectures;
    m.def("deco_attention_workspace_size", &workspace_size);
    m.def("deco_attention_supported", [](int sequence, int heads, int device) {
        return workspace_size(sequence, heads, device) != 0;
    });
#ifdef COMFY_HAVE_DECO_FP4
    m.def("deco_attention_sample_rows", &sample_rows);
    m.def("deco_attention_forward", &forward);
#endif
}
