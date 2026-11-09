// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>

namespace nb = nanobind;
using Tensor = nb::ndarray<nb::device::cuda, nb::c_contig>;
using Bytes = nb::ndarray<int8_t, nb::device::cuda, nb::c_contig>;
using Scales = nb::ndarray<float, nb::device::cuda, nb::c_contig>;
using Length = nb::ndarray<int64_t, nb::device::cuda, nb::c_contig>;
using Mask = nb::ndarray<nb::device::cuda>;

extern "C" void launch_int8_decode_quant_q(const void*, int8_t*, float*, int, int, int, cudaStream_t);
extern "C" void launch_int8_prefill_quant_q(const void*, int8_t*, float*, int, int, int, cudaStream_t);
extern "C" void launch_int8_decode_quant_k(const void*, int8_t*, float*, const int64_t*, int, int, int, int, int, bool, cudaStream_t);
extern "C" void launch_int8_decode_quant_v(const void*, int8_t*, float*, const int64_t*, int, int, int, int, int, bool, cudaStream_t);
extern "C" void launch_int8_decode(int8_t*, int8_t*, int8_t*, float*, float*, float*, const int64_t*,
    void*, float*, void*, float*, int, int, int, int, int, int, cudaStream_t);
extern "C" void launch_int8_prefill(int8_t*, int8_t*, int8_t*, float*, float*, float*, void*,
    const void*, int, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int, cudaStream_t);

void register_int8_decode(nb::module_& m) {
    m.def("_int8_decode_update", [](Tensor k, Tensor v, Bytes ki, Bytes vi, Scales ks, Scales vs,
                                  Length length, bool initialize, uintptr_t stream) {
        if (k.ndim() != 4 || v.ndim() != 4 || ki.ndim() != 4 || vi.ndim() != 2 || ks.ndim() != 3 || vs.ndim() != 4 ||
            k.dtype() != nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1} || v.dtype() != k.dtype() ||
            k.shape(3) != 256 || v.size() != k.size() || ki.shape(0) != k.shape(0) || ki.shape(1) != k.shape(1) ||
            ki.shape(3) != 256 || vs.shape(1) == 0 || ki.shape(2) % (vs.shape(1) * 64) || ki.shape(2) < k.shape(2) ||
            vi.shape(0) != k.shape(0) * k.shape(1) * 256 || vi.shape(1) != ki.shape(2) ||
            ks.size() != ki.size() / (64 * 256) * 4 || vs.shape(0) != ki.shape(0) || vs.shape(2) != ki.shape(1) || vs.shape(3) != 256 || length.size() != 1)
            throw std::runtime_error("Invalid int8 decode cache tensors");
        auto s = reinterpret_cast<cudaStream_t>(stream);
        launch_int8_decode_quant_k(k.data(), ki.data(), ks.data(), length.data(), k.shape(0), k.shape(1), k.shape(2), vs.shape(1), ki.shape(2) / vs.shape(1), initialize, s);
        launch_int8_decode_quant_v(v.data(), vi.data(), vs.data(), length.data(), k.shape(0), k.shape(1), k.shape(2), vs.shape(1), ki.shape(2) / vs.shape(1), initialize, s);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    });
    m.def("_int8_decode", [](Tensor q, Bytes qi, Scales qs, Bytes k, Bytes v, Scales ks, Scales vs,
                            Length length, Tensor partial, Scales lse, Tensor out, Scales out_lse, int seq, uintptr_t stream) {
        if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 2 || ks.ndim() != 3 || vs.ndim() != 4 || q.shape(3) != 256 || q.shape(2) > 64 || seq < 1 || q.shape(2) % seq ||
            q.dtype() != nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1} ||
            partial.dtype() != q.dtype() || out.dtype() != q.dtype() || qi.size() != q.size() ||
            qs.size() != q.shape(0) * q.shape(1) * 64 || q.shape(0) != k.shape(0) || q.shape(1) != k.shape(1) ||
            k.shape(3) != 256 || vs.shape(1) == 0 || k.shape(2) % (vs.shape(1) * 64) ||
            v.size() != k.size() || v.shape(1) != k.shape(2) || ks.size() != k.size() / (64 * 256) * 4 ||
            vs.shape(0) != k.shape(0) || vs.shape(2) != k.shape(1) || vs.shape(3) != 256 ||
            partial.size() != q.size() * vs.shape(1) || lse.size() != partial.size() / 256 ||
            out.size() != q.size() || out_lse.size() != q.size() / 256 || length.size() != 1)
            throw std::runtime_error("Invalid int8 decode query or workspace");
        auto s = reinterpret_cast<cudaStream_t>(stream);
        launch_int8_decode_quant_q(q.data(), qi.data(), qs.data(), q.shape(0), q.shape(1), q.shape(2), s);
        launch_int8_decode(qi.data(), k.data(), v.data(), qs.data(), ks.data(), vs.data(), length.data(),
            partial.data(), lse.data(), out.data(), out_lse.data(), q.shape(0), q.shape(1), q.shape(2), vs.shape(1), k.shape(2) / vs.shape(1), seq, s);
    });
    m.def("_int8_prefill", [](Tensor q, Bytes qi, Scales qs, Bytes k, Bytes v, Scales ks, Scales vs,
                              Tensor out, nb::object mask, int length, uintptr_t stream) {
        if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 2 || ks.ndim() != 3 || vs.ndim() != 4 ||
            q.dtype() != nb::dlpack::dtype{static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1} || out.dtype() != q.dtype() ||
            q.shape(3) != 256 || k.shape(3) != 256 || q.shape(2) == 0 || k.shape(1) == 0 || q.shape(0) != k.shape(0) || q.shape(1) % k.shape(1) || qi.size() != q.size() ||
            qs.size() != q.shape(0) * q.shape(1) * ((q.shape(2) + 127) / 128) * 64 || out.size() != q.size() ||
            v.size() != k.size() || v.shape(1) != k.shape(2) || ks.size() != k.size() / (64 * 256) * 4 ||
            vs.shape(0) != k.shape(0) || vs.shape(1) == 0 || k.shape(2) % (vs.shape(1) * 64) || vs.shape(2) != k.shape(1) || vs.shape(3) != 256 ||
            length < 1 || length > k.shape(2))
            throw std::runtime_error("Invalid int8 prefill tensors");
        Mask bias;
        int code = 0;
        int64_t mb = 0, mh = 0, mq = 0, mk = 0;
        if (!mask.is_none()) {
            bias = nb::cast<Mask>(mask);
            const int dims = bias.ndim();
            if (dims < 2 || dims > 4 || bias.shape(dims - 1) != length ||
                (bias.shape(dims - 2) != 1 && bias.shape(dims - 2) != q.shape(2)) ||
                (dims > 2 && bias.shape(0) != 1 && bias.shape(0) != q.shape(0)) ||
                (dims == 4 && bias.shape(1) != 1 && bias.shape(1) != q.shape(1)))
                throw std::runtime_error("Invalid int8 prefill mask");
            mk = bias.stride(dims - 1);
            mq = bias.shape(dims - 2) == 1 ? 0 : bias.stride(dims - 2);
            if (dims > 2) mb = bias.shape(0) == 1 ? 0 : bias.stride(0);
            if (dims == 4) mh = bias.shape(1) == 1 ? 0 : bias.stride(1);
            auto dtype = bias.dtype();
            if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool)) code = 3;
            else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat) && dtype.bits == 16) code = 2;
            else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float) && dtype.bits == 16) code = 1;
            else if (dtype.code != static_cast<uint8_t>(nb::dlpack::dtype_code::Float) || dtype.bits != 32)
                throw std::runtime_error("Invalid int8 prefill mask dtype");
        }
        auto s = reinterpret_cast<cudaStream_t>(stream);
        launch_int8_prefill_quant_q(q.data(), qi.data(), qs.data(), q.shape(0), q.shape(1), q.shape(2), s);
        launch_int8_prefill(qi.data(), k.data(), v.data(), qs.data(), ks.data(), vs.data(), out.data(),
            bias.is_valid() ? bias.data() : nullptr, code, mb, mh, mq, mk,
            q.shape(0), q.shape(1), k.shape(1), q.shape(2), length, k.shape(2), k.shape(2) / vs.shape(1), s);
    }, nb::arg("q"), nb::arg("qi"), nb::arg("qs"), nb::arg("k"), nb::arg("v"),
       nb::arg("ks"), nb::arg("vs"), nb::arg("out"), nb::arg("mask").none(), nb::arg("length"), nb::arg("stream"));
}
