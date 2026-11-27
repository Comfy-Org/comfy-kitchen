// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <stdexcept>

#include "deco_attention/launch.h"
#include "deco_attention/prep.h"

namespace nb = nanobind;
namespace {
template<typename... Args> using CudaArray = nb::ndarray<nb::device::cuda, Args...>;
using AttentionShape = nb::shape<1, -1, -1, 128>;
using Table = CudaArray<const int32_t, nb::ndim<1>, nb::c_contig>;
using Workspace = CudaArray<uint8_t, nb::ndim<1>, nb::c_contig>;
using Code = nb::dlpack::dtype_code;
constexpr nb::dlpack::dtype bf16{static_cast<uint8_t>(Code::Bfloat), 16, 1};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

cudaDeviceProp properties(int device) {
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device));
    return props;
}

bool aligned(const void* data, size_t bytes) {
    return data != nullptr && reinterpret_cast<uintptr_t>(data) % bytes == 0;
}

size_t workspace_size(int sequence, int heads, int device) {
    const auto props = properties(device);
    if (!comfy_deco::lookup_kernel(sequence, heads, 10 * props.major + props.minor)) return 0;
    return comfy_deco::prep_layout(sequence, heads, props.multiProcessorCount).bytes;
}

void forward(CudaArray<AttentionShape> q, CudaArray<AttentionShape> k, CudaArray<AttentionShape> v,
             Table table, CudaArray<nb::shape<128, 128>, nb::c_contig> rotation,
             CudaArray<AttentionShape, nb::c_contig> out, Workspace workspace,
             bool single_writer, uintptr_t stream_handle, std::optional<CudaArray<nb::ndim<1>>> tags) {
    int device;
    check(cudaGetDevice(&device));
    const auto props = properties(device);
    const auto kernel = comfy_deco::lookup_kernel(q.shape(2), q.shape(1), 10 * props.major + props.minor);
    if (!kernel) throw std::invalid_argument("No DeCo kernel for this device and shape");
    const int heads = static_cast<int>(q.shape(1)), sequence = static_cast<int>(q.shape(2));
    for (const auto& input : {q, k, v}) {
        if (input.device_id() != device || input.dtype() != bf16 || input.shape(1) != q.shape(1) ||
            input.shape(2) != q.shape(2) || input.stride(3) != 1 || input.stride(2) <= 0 ||
            input.stride(2) % 8 || (heads > 1 && (input.stride(1) <= 0 || input.stride(1) % 8)) ||
            !aligned(input.data(), 16))
            throw std::invalid_argument("Expected aligned BF16 DeCo inputs on the current CUDA device");
    }
    const int sms = props.multiProcessorCount;
    const auto w = comfy_deco::prep_layout(sequence, heads, sms);
    const auto table_size = tags ? sequence : (std::max)(64, sequence / 256);
    if (table.device_id() != device || table.size() != static_cast<size_t>(table_size) ||
        !aligned(table.data(), alignof(int32_t)) || rotation.device_id() != device ||
        rotation.dtype() != bf16 || !aligned(rotation.data(), 16) || out.device_id() != device ||
        out.dtype() != bf16 || out.shape(1) != q.shape(2) || out.shape(2) != q.shape(1) ||
        !aligned(out.data(), 16) || workspace.device_id() != device || workspace.size() < w.bytes ||
        !aligned(workspace.data(), 256))
        throw std::invalid_argument("Invalid DeCo sampling table, rotation, output or workspace");

    auto* p = workspace.data();
    auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
    const int32_t* rows = table.data();
    if (tags) {
        const auto dtype = tags->dtype();
        const bool is_unsigned = dtype.code == static_cast<uint8_t>(Code::UInt);
        if (tags->device_id() != device || tags->size() != q.shape(2) || tags->stride(0) < 0 ||
            dtype.lanes != 1 || (is_unsigned ? dtype.bits != 8 : dtype.code != static_cast<uint8_t>(Code::Int)) ||
            !tags->itemsize() || !aligned(tags->data(), tags->itemsize()))
            throw std::invalid_argument("Invalid DeCo token tags");
        // Prep consumes the sampled rows before the core writes any output on this stream.
        auto* drawn = static_cast<int32_t*>(out.data());
        check(comfy_deco::sample_tags(tags->data(), dtype.bits, is_unsigned, tags->stride(0),
                                      table.data(), drawn, p, sequence, stream));
        rows = drawn;
    }
    check(comfy_deco::prepare(w, q.data(), k.data(), v.data(), rows, rotation.data(), p, sequence, heads, sms,
                              q.stride(2) * 2, q.stride(1) * 2, k.stride(2) * 2, k.stride(1) * 2,
                              v.stride(2) * 2, v.stride(1) * 2, stream));
    check(kernel({p + w.q4, p + w.q_sf, p + w.score_scale, p + w.v4_scale, p + w.img,
                  out.data(), p + w.split}, sms, single_writer, stream));
}
} // namespace

void register_deco_attention(nb::module_& m) {
    nb::list shapes, architectures;
    for (const auto& shape : comfy_deco::shapes) shapes.append(nb::make_tuple(shape.sequence, shape.heads));
    for (int cc : comfy_deco::compiled_architectures) architectures.append(nb::make_tuple(cc / 10, cc % 10));
    m.attr("deco_attention_kernels") = nb::make_tuple(nb::tuple(architectures), nb::tuple(shapes));
    m.def("deco_attention_workspace_size", &workspace_size);
    m.def("deco_attention_forward", &forward, nb::arg("q"), nb::arg("k"), nb::arg("v"),
          nb::arg("table"), nb::arg("rotation"), nb::arg("out"), nb::arg("workspace"),
          nb::arg("single_writer"), nb::arg("stream"), nb::arg("tags").none());
}
