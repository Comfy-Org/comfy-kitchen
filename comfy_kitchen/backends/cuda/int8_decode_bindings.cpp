// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>

namespace nb = nanobind;
using Tensor = nb::ndarray<nb::device::cuda, nb::c_contig>;
using Bytes = nb::ndarray<int8_t, nb::device::cuda, nb::c_contig>;
using Scales = nb::ndarray<float, nb::device::cuda, nb::c_contig>;
using Length = nb::ndarray<int64_t, nb::device::cuda, nb::c_contig>;

extern "C" void launch_int8_decode_quant_q(const void*, int8_t*, float*, int, int, int, cudaStream_t);
extern "C" void launch_int8_decode_quant_k(const void*, int8_t*, float*, const int64_t*, int, int, int, int, int, bool, cudaStream_t);
extern "C" void launch_int8_decode_quant_v(const void*, int8_t*, float*, const int64_t*, int, int, int, int, int, bool, cudaStream_t);
extern "C" void launch_int8_decode(int8_t*, int8_t*, int8_t*, float*, float*, float*, const int64_t*,
    void*, float*, void*, float*, int, int, int, int, int, int, cudaStream_t);

void register_int8_decode(nb::module_& m) {
    m.def("_int8_decode_update", [](Tensor k, Tensor v, Bytes ki, Bytes vi, Scales ks, Scales vs,
                                  Length length, bool initialize, uintptr_t stream) {
        if (k.ndim() != 4 || v.ndim() != 4 || ki.ndim() != 5 || vi.ndim() != 5 ||
            k.dtype() != nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1} || v.dtype() != k.dtype() ||
            k.shape(3) != 256 || v.size() != k.size() || ki.shape(0) != k.shape(0) || ki.shape(2) != k.shape(1) ||
            ki.shape(4) != 256 || ki.shape(3) % 64 || ki.shape(1) * ki.shape(3) < k.shape(2) || vi.size() != ki.size() ||
            ks.size() != ki.size() / (64 * 256) * 4 || vs.size() != ki.size() / ki.shape(3) || length.size() != 1)
            throw std::runtime_error("Invalid int8 decode cache tensors");
        auto s = reinterpret_cast<cudaStream_t>(stream);
        launch_int8_decode_quant_k(k.data(), ki.data(), ks.data(), length.data(), k.shape(0), k.shape(1), k.shape(2), ki.shape(1), ki.shape(3), initialize, s);
        launch_int8_decode_quant_v(v.data(), vi.data(), vs.data(), length.data(), k.shape(0), k.shape(1), k.shape(2), ki.shape(1), ki.shape(3), initialize, s);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    });
    m.def("_int8_decode", [](Tensor q, Bytes qi, Scales qs, Bytes k, Bytes v, Scales ks, Scales vs,
                            Length length, Tensor partial, Scales lse, Tensor out, Scales out_lse, int seq, uintptr_t stream) {
        if (q.ndim() != 4 || k.ndim() != 5 || q.shape(3) != 256 || q.shape(2) > 64 || seq < 1 || q.shape(2) % seq ||
            q.dtype() != nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1} ||
            partial.dtype() != q.dtype() || out.dtype() != q.dtype() || qi.size() != q.size() ||
            qs.size() != q.shape(0) * q.shape(1) * 64 || q.shape(0) != k.shape(0) || q.shape(1) != k.shape(2) ||
            partial.size() != q.size() * k.shape(1) || lse.size() != partial.size() / 256 ||
            out.size() != q.size() || out_lse.size() != q.size() / 256 || length.size() != 1)
            throw std::runtime_error("Invalid int8 decode query or workspace");
        auto s = reinterpret_cast<cudaStream_t>(stream);
        launch_int8_decode_quant_q(q.data(), qi.data(), qs.data(), q.shape(0), q.shape(1), q.shape(2), s);
        launch_int8_decode(qi.data(), k.data(), v.data(), qs.data(), ks.data(), vs.data(), length.data(),
            partial.data(), lse.data(), out.data(), out_lse.data(), q.shape(0), q.shape(1), q.shape(2), k.shape(1), k.shape(3), seq, s);
    });
}
