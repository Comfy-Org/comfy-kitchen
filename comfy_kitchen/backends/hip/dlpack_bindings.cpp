// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <optional>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;

// Matches comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE:
// 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8, 5=float8_e4m3fn, 6=float8_e5m2
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
    if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) {
        if (dtype.bits == 32) return 0;
        if (dtype.bits == 16) return 1;
        if (dtype.bits == 8) return 5;
    } else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat) && dtype.bits == 16) {
        return 2;
    } else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) && dtype.bits == 8) {
        return 3;
    } else if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int) && dtype.bits == 8) {
        return 4;
    }
    return -1;
}

extern "C" {
void launch_quantize_per_tensor_fp8_kernel(const void*, const void*, void*, int64_t, int, int,
                                           hipStream_t);
void launch_dequantize_per_tensor_fp8_kernel(const void*, const void*, void*, int64_t, int, int,
                                             hipStream_t);
void launch_stochastic_round_fp8_kernel(void*, const void*, int64_t, int, int, int, hipStream_t);

void launch_scaled_mm_fp8_kernel(const void*, const void*, void*, const void*, const void*,
                                 const void*, int, int, int, int, int, hipStream_t);
void launch_int8_gemm_kernel(const void*, const void*, void*, const void*, const void*, int,
                             const void*, int, int, int, int, int, hipStream_t);
void launch_convrot_w4a4_gemm_kernel(const void*, const void*, void*, const void*, const void*,
                                     const void*, int, int, int, int, int, hipStream_t);

void launch_quantize_int8_rowwise_kernel(const void*, int, void*, void*, int, int, hipStream_t);
void launch_quantize_int8_convrot_kernel(const void*, int, void*, void*, int, int, int,
                                         hipStream_t);
void launch_quantize_int8_tensorwise_kernel(const void*, int, void*, void*, void*, int64_t,
                                            hipStream_t);
void launch_convrot_quant_int4_kernel(const void*, int, void*, void*, int, int, int, hipStream_t);
void launch_unpack_int4_kernel(const void*, void*, int64_t, hipStream_t);
int convrot_max_k_host();

void launch_adaln_kernel(const void*, const void*, const void*, void*, int, int, int, int, float,
                         int, int, int, hipStream_t);
void launch_gemv_awq_kernel(const void*, const void*, const void*, const void*, const void*, void*,
                            int, int, int, int, int, int, int, int, hipStream_t);
void launch_svdquant_lora_down_kernel(const void*, const void*, void*, int, int, int, int, int,
                                      hipStream_t);
void launch_svdquant_quant_kernel(const void*, const void*, void*, void*, int, int, int, int, int,
                                  bool, hipStream_t);
void launch_svdquant_gemm_kernel(const void*, const void*, void*, const void*, const void*,
                                 const void*, const void*, const void*, int, int, int, int, int,
                                 int, int, int, int, int, bool, hipStream_t);
void launch_apply_rope_kernel(const void*, const void*, const void*, void*, void*, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int,
                              bool, hipStream_t);
}

static void check_hip_launch() {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel launch failed: ") + hipGetErrorString(err));
    }
}

using OptArray = std::optional<nb::ndarray<>>;

static const void* opt_data(const OptArray& t) {
    return t.has_value() ? t->data() : nullptr;
}

static int opt_code(const OptArray& t) {
    return t.has_value() ? map_dtype_to_code(t->dtype()) : 0;
}

void quantize_per_tensor_fp8(nb::ndarray<> input, nb::ndarray<> scale, nb::ndarray<> output,
                             int input_dtype_code, int output_dtype_code, int64_t numel,
                             uintptr_t stream_ptr) {
    launch_quantize_per_tensor_fp8_kernel(input.data(), scale.data(), output.data(), numel,
                                          input_dtype_code, output_dtype_code,
                                          reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void dequantize_per_tensor_fp8(nb::ndarray<> input, nb::ndarray<> scale, nb::ndarray<> output,
                               int input_dtype_code, int output_dtype_code, int64_t numel,
                               uintptr_t stream_ptr) {
    launch_dequantize_per_tensor_fp8_kernel(input.data(), scale.data(), output.data(), numel,
                                            input_dtype_code, output_dtype_code,
                                            reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void stochastic_round_fp8(nb::ndarray<> rng_and_output, nb::ndarray<> input, int output_dtype_code,
                          int64_t numel, uintptr_t stream_ptr) {
    int rng_dtype_code = map_dtype_to_code(rng_and_output.dtype());
    if (rng_dtype_code != 3) {
        throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
    }
    int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for stochastic_round_fp8");
    }
    if (output_dtype_code < 5 || output_dtype_code > 6) {
        throw std::runtime_error("Unsupported output dtype for stochastic_round_fp8");
    }
    launch_stochastic_round_fp8_kernel(rng_and_output.data(), input.data(), numel, rng_dtype_code,
                                       input_dtype_code, output_dtype_code,
                                       reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void scaled_mm_fp8(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> scale_a,
                   nb::ndarray<> scale_b, OptArray bias, int M, int N, int K, int out_code,
                   uintptr_t stream_ptr) {
    launch_scaled_mm_fp8_kernel(a.data(), b.data(), c.data(), scale_a.data(), scale_b.data(),
                                opt_data(bias), opt_code(bias), M, N, K, out_code,
                                reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void int8_gemm(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> scale_a,
               nb::ndarray<> scale_b, int scale_b_stride, OptArray bias, int M, int N, int K,
               int out_code, uintptr_t stream_ptr) {
    launch_int8_gemm_kernel(a.data(), b.data(), c.data(), scale_a.data(), scale_b.data(),
                            scale_b_stride, opt_data(bias), opt_code(bias), M, N, K, out_code,
                            reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void convrot_w4a4_gemm(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> x_scale,
                       nb::ndarray<> w_scale, OptArray bias, int M, int N, int K, int out_code,
                       uintptr_t stream_ptr) {
    launch_convrot_w4a4_gemm_kernel(a.data(), b.data(), c.data(), x_scale.data(), w_scale.data(),
                                    opt_data(bias), opt_code(bias), M, N, K, out_code,
                                    reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void quantize_int8_rowwise(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales, int M, int K,
                           uintptr_t stream_ptr) {
    launch_quantize_int8_rowwise_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                        scales.data(), M, K,
                                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void quantize_int8_convrot(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales, int M, int K,
                           int group_size, uintptr_t stream_ptr) {
    launch_quantize_int8_convrot_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                        scales.data(), M, K, group_size,
                                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void quantize_int8_tensorwise(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scale,
                              nb::ndarray<> scratch, int64_t numel, uintptr_t stream_ptr) {
    launch_quantize_int8_tensorwise_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                           scale.data(), scratch.data(), numel,
                                           reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void convrot_quant_int4(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales, int M, int K,
                        int group_size, uintptr_t stream_ptr) {
    launch_convrot_quant_int4_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                     scales.data(), M, K, group_size,
                                     reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void unpack_int4(nb::ndarray<> q, nb::ndarray<> out, int64_t nbytes, uintptr_t stream_ptr) {
    launch_unpack_int4_kernel(q.data(), out.data(), nbytes,
                              reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void adaln(nb::ndarray<> x, nb::ndarray<> scale, nb::ndarray<> shift, nb::ndarray<> out, int N,
           int D, int scale_group, int shift_group, float eps, uintptr_t stream_ptr) {
    launch_adaln_kernel(x.data(), scale.data(), shift.data(), out.data(), N, D, scale_group,
                        shift_group, eps, map_dtype_to_code(x.dtype()),
                        map_dtype_to_code(scale.dtype()), map_dtype_to_code(shift.dtype()),
                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// xq is (batch, dim1, dim2, head_dim); freqs is (fb, fd1, fd2, head_dim/2, 2, 2).
// Shapes and strides are read off the arrays so the broadcast rules stay in one
// place rather than being recomputed on the Python side.
void apply_rope(nb::ndarray<> xq, OptArray xk, nb::ndarray<> freqs, nb::ndarray<> xq_out,
                OptArray xk_out, bool split_half, uintptr_t stream_ptr) {
    if (xq.ndim() != 4) throw std::runtime_error("apply_rope expects a 4D input");
    if (freqs.ndim() != 6) throw std::runtime_error("apply_rope expects a 6D freqs_cis");
    if (xq.shape(3) % 2 != 0) throw std::runtime_error("apply_rope expects an even head_dim");

    // The kernel indexes freqs as (fb, fd1, fd2, head_dim/2, 2, 2), broadcasting a
    // leading dim only when it is 1. Anything else walks off the end of the array.
    if (freqs.shape(3) != xq.shape(3) / 2 || freqs.shape(4) != 2 || freqs.shape(5) != 2) {
        throw std::runtime_error("apply_rope expects freqs_cis trailing dims (head_dim/2, 2, 2)");
    }
    for (size_t i = 0; i < 3; ++i) {
        if (freqs.shape(i) != 1 && freqs.shape(i) != xq.shape(i)) {
            throw std::runtime_error(
                "apply_rope expects each leading freqs_cis dim to be 1 or match the input");
        }
    }

    // The kernel addresses xk and both outputs with xq's strides and dtype code, and
    // writes xk_out whenever xk is present. Anything that does not share xq's layout
    // has to be launched on its own.
    auto same_layout_as_xq = [&xq](const nb::ndarray<>& t, const char* name) {
        if (t.ndim() != 4 || t.dtype() != xq.dtype()) {
            throw std::runtime_error(std::string("apply_rope expects ") + name +
                                     " to be 4D with xq's dtype");
        }
        for (size_t i = 0; i < 4; ++i) {
            if (t.shape(i) != xq.shape(i) || t.stride(i) != xq.stride(i)) {
                throw std::runtime_error(std::string("apply_rope expects ") + name +
                                         " to have xq's shape and strides");
            }
        }
    };

    if (xk.has_value() != xk_out.has_value()) {
        throw std::runtime_error("apply_rope expects xk and xk_out together or not at all");
    }
    same_layout_as_xq(xq_out, "xq_out");
    if (xk.has_value()) {
        same_layout_as_xq(*xk, "xk");
        same_layout_as_xq(*xk_out, "xk_out");
    }

    launch_apply_rope_kernel(
        xq.data(), xk.has_value() ? xk->data() : nullptr, freqs.data(), xq_out.data(),
        xk_out.has_value() ? xk_out->data() : nullptr,
        static_cast<int64_t>(xq.shape(0)), static_cast<int64_t>(xq.shape(1)),
        static_cast<int64_t>(xq.shape(2)), static_cast<int64_t>(xq.shape(3)),
        static_cast<int64_t>(freqs.shape(0)), static_cast<int64_t>(freqs.shape(1)),
        static_cast<int64_t>(freqs.shape(2)),
        static_cast<int64_t>(xq.stride(0)), static_cast<int64_t>(xq.stride(1)),
        static_cast<int64_t>(xq.stride(2)), static_cast<int64_t>(xq.stride(3)),
        static_cast<int64_t>(freqs.stride(0)), static_cast<int64_t>(freqs.stride(1)),
        static_cast<int64_t>(freqs.stride(2)), static_cast<int64_t>(freqs.stride(3)),
        static_cast<int64_t>(freqs.stride(4)), static_cast<int64_t>(freqs.stride(5)),
        map_dtype_to_code(xq.dtype()), map_dtype_to_code(freqs.dtype()), split_half,
        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void gemv_awq_w4a16(nb::ndarray<> x, nb::ndarray<> qweight, nb::ndarray<> wscales,
                    nb::ndarray<> wzeros, OptArray bias, nb::ndarray<> out, int M, int N, int K,
                    int group_size, uintptr_t stream_ptr) {
    launch_gemv_awq_kernel(x.data(), qweight.data(), wscales.data(), wzeros.data(), opt_data(bias),
                           out.data(), M, N, K, group_size, map_dtype_to_code(x.dtype()),
                           map_dtype_to_code(wscales.dtype()), opt_code(bias),
                           map_dtype_to_code(out.dtype()),
                           reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void svdquant_lora_down(nb::ndarray<> x, nb::ndarray<> lora_down, nb::ndarray<> lora_act, int M,
                        int K, int R, uintptr_t stream_ptr) {
    launch_svdquant_lora_down_kernel(x.data(), lora_down.data(), lora_act.data(), M, K, R,
                                     map_dtype_to_code(x.dtype()),
                                     map_dtype_to_code(lora_down.dtype()),
                                     reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void svdquant_quantize(nb::ndarray<> x, nb::ndarray<> smooth, nb::ndarray<> q,
                       nb::ndarray<> ascales, int M, int M_pad, int K, bool act_unsigned,
                       uintptr_t stream_ptr) {
    launch_svdquant_quant_kernel(x.data(), smooth.data(), q.data(), ascales.data(), M, M_pad, K,
                                 map_dtype_to_code(x.dtype()), map_dtype_to_code(ascales.dtype()),
                                 act_unsigned, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void svdquant_gemm(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> ascales,
                   nb::ndarray<> wscales, nb::ndarray<> lora_act, nb::ndarray<> lora_up,
                   OptArray bias, int M, int N, int K, int R, bool act_unsigned,
                   uintptr_t stream_ptr) {
    launch_svdquant_gemm_kernel(
        a.data(), b.data(), c.data(), ascales.data(), wscales.data(), lora_act.data(),
        lora_up.data(), opt_data(bias), M, N, K, R, map_dtype_to_code(ascales.dtype()),
        map_dtype_to_code(wscales.dtype()), map_dtype_to_code(lora_act.dtype()),
        map_dtype_to_code(lora_up.dtype()), opt_code(bias), map_dtype_to_code(c.dtype()),
        act_unsigned, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

NB_MODULE(_C, m) {
    m.doc() = "ComfyKitchen HIP backend native operations (gfx12 WMMA)";
    m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8);
    m.def("dequantize_per_tensor_fp8", &dequantize_per_tensor_fp8);
    m.def("stochastic_round_fp8", &stochastic_round_fp8);
    m.def("scaled_mm_fp8", &scaled_mm_fp8);
    m.def("int8_gemm", &int8_gemm);
    m.def("convrot_w4a4_gemm", &convrot_w4a4_gemm);
    m.def("quantize_int8_rowwise", &quantize_int8_rowwise);
    m.def("quantize_int8_convrot", &quantize_int8_convrot);
    m.def("quantize_int8_tensorwise", &quantize_int8_tensorwise);
    m.def("convrot_quant_int4", &convrot_quant_int4);
    m.def("convrot_max_k", &convrot_max_k_host);
    m.def("unpack_int4", &unpack_int4);
    m.def("adaln", &adaln);
    m.def("apply_rope", &apply_rope);
    m.def("gemv_awq_w4a16", &gemv_awq_w4a16);
    m.def("svdquant_lora_down", &svdquant_lora_down);
    m.def("svdquant_quantize", &svdquant_quantize);
    m.def("svdquant_gemm", &svdquant_gemm);
}
