// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include "launchers.h"

namespace nb = nanobind;

// Maps a DLPack dtype onto comfy_kitchen.backends.eager.quantization.DTYPE_TO_CODE:
// 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8.
//
// The fp8 codes (5=e4m3, 6=e5m2) are never produced here. DLPack gives e4m3 and
// e5m2 their own dtype codes (10 and 12), which nanobind's dtype_code does not
// name, so an fp8 tensor would land in the -1 branch. The Python layer therefore
// hands fp8 across as uint8 and passes the fp8 code alongside it as an int.
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
    if (dtype.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) {
        if (dtype.bits == 32) return 0;
        if (dtype.bits == 16) return 1;
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
void launch_convrot_w4a4_gemm_kernel(const void*, const void*, void*, const void*, const void*,
                                     const void*, int, int, int, int, int, hipStream_t);

void launch_quantize_int8_rowwise_kernel(const void*, int, void*, void*, int, int, hipStream_t);
void launch_quantize_int8_convrot_kernel(const void*, int, void*, void*, void*, void*, int, int,
                                         int, int, hipStream_t);
void launch_quantize_int8_tensorwise_kernel(const void*, int, void*, void*, void*, int64_t,
                                            hipStream_t);
void launch_dequantize_int8_simple_kernel(const void*, const void*, void*, int64_t, int64_t, int,
                                          int, hipStream_t);
void launch_dequantize_int8_convrot_weight_kernel(const void*, const void*, void*, int, int, int,
                                                  int, int, hipStream_t);
void launch_convrot_quant_int4_kernel(const void*, int, void*, void*, int, int, int, hipStream_t);
void launch_unpack_int4_kernel(const void*, void*, int64_t, hipStream_t);
int convrot_max_k_host(int);
int convrot_int8_needs_spill_host(int, int, int);

void launch_quantize_w4a8_convrot_kernel(const void*, const void*, void*, void*, void*, int64_t,
                                         int64_t, int, bool, uint64_t, hipStream_t);
int w4a8_requant_max_k_kernel();
void launch_na3d_kernel(const void*, const void*, const void*, void*, int, int, int, int, int, int,
                        int, int, int, int, int, int, float, int, hipStream_t);

void launch_sage_quant_qk_int8(const void*, void*, void*, const void*, void*, void*, void*, int,
                               int, int, int, int, int, int, int, int64_t, int64_t, int64_t,
                               int64_t, int64_t, int64_t, int, int, hipStream_t);
void launch_sage_quant_v_int8(const void*, void*, void*, int, int, int, int, int, int64_t, int64_t,
                              int64_t, int, hipStream_t);
void launch_sage_int8_attn(const void*, const void*, const void*, void*, const void*, const void*,
                           const void*, const void*, int64_t, int64_t, int64_t, int64_t, int, int,
                           int, int, int, int, int, int, int, int, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                           float, int, hipStream_t);

void launch_adaln_kernel(const void*, const void*, const void*, void*, int, int, int, int, float,
                         int, int, int, bool, hipStream_t);
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
                              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int, int, bool, hipStream_t);
void launch_rms_rope_kernel(const void*, const void*, const void*, const void*, const void*, void*,
                            void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int,
                            int, float, bool, hipStream_t);
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

// _C is importable, so these entry points cannot assume the Python layer put them
// together. The kernels dereference scale[0] and index up to numel off raw
// pointers, so a caller-supplied count larger than the tensor, or a scale of the
// wrong dtype, is an out-of-bounds device access rather than an exception.
// The epilogues cast the scales to raw float32 and index scale_a[row] up to M and
// scale_b[col * stride] up to N, so a short or wrongly-typed scale is an
// out-of-bounds device read rather than an exception.
static void require_scale_len(const nb::ndarray<>& s, size_t need, const char* fn,
                              const char* name) {
    if (map_dtype_to_code(s.dtype()) != 0) {
        throw std::runtime_error(std::string(fn) + ": " + name + " must be float32");
    }
    if (s.size() < need) {
        throw std::runtime_error(std::string(fn) + ": " + name + " has " +
                                 std::to_string(s.size()) + " elements, needs at least " +
                                 std::to_string(need));
    }
}

static void require_scale(const nb::ndarray<>& scale, const char* fn) {
    require_scale_len(scale, 1, fn, "scale");
}

// Every kernel takes raw pointers plus caller-supplied extents and indexes up to
// those extents with no bounds of its own, so a tensor smaller than the extents it
// is launched with is an out-of-bounds device access. _C is importable, so these
// are checked here rather than trusted from the Python layer.
static void require_len(const nb::ndarray<>& t, int64_t need, const char* fn, const char* name) {
    if (need < 0 || t.size() < static_cast<size_t>(need)) {
        throw std::runtime_error(std::string(fn) + ": " + name + " has " +
                                 std::to_string(t.size()) + " elements, needs at least " +
                                 std::to_string(need));
    }
}

// lo..hi are DTYPE_TO_CODE values: 0..2 float32/16/bfloat16, 3 uint8, 4 int8.
static void require_dtype(const nb::ndarray<>& t, int lo, int hi, const char* fn,
                          const char* name) {
    const int code = map_dtype_to_code(t.dtype());
    if (code < lo || code > hi) {
        throw std::runtime_error(std::string(fn) + ": " + name + " has an unsupported dtype");
    }
}

// Mirrors kSvdGroup in ops/svdquant_w4a4.hip: one scale per 64-element group.
constexpr int kSvdGroup = 64;

static void require_positive(int v, const char* fn, const char* name) {
    if (v <= 0) {
        throw std::runtime_error(std::string(fn) + ": " + name + " must be positive, got " +
                                 std::to_string(v));
    }
}

// A negative extent survives require_len whenever it is multiplied by another
// negative one (M*K stays positive), then reaches a launcher where it becomes an
// enormous unsigned grid. Reject each extent on its own before any product.
static void require_nonneg(int v, const char* fn, const char* name) {
    if (v < 0) {
        throw std::runtime_error(std::string(fn) + ": " + name + " must be non-negative, got " +
                                 std::to_string(v));
    }
}

// The GEMM launchers pick the output element width from out_code, independently
// of c's own dtype, and write c at that width. A wider out_code than c's dtype
// overruns the allocation, so the two have to name the same type.
static void require_out_matches(const nb::ndarray<>& c, int out_code, const char* fn) {
    if (out_code != map_dtype_to_code(c.dtype())) {
        throw std::runtime_error(std::string(fn) + ": out_code does not match the output dtype");
    }
}

// The epilogues also index bias[col] up to N, decoded with bias_code.
static void require_bias(const OptArray& bias, int n, const char* fn) {
    if (!bias.has_value()) return;
    const int code = map_dtype_to_code(bias->dtype());
    if (code < 0 || code > 2) {
        throw std::runtime_error(std::string(fn) + ": bias must be float32/float16/bfloat16");
    }
    if (bias->size() < static_cast<size_t>(n)) {
        throw std::runtime_error(std::string(fn) + ": bias has " + std::to_string(bias->size()) +
                                 " elements, needs at least " + std::to_string(n));
    }
}

static void require_numel(int64_t numel, const nb::ndarray<>& t, const char* fn, const char* name) {
    if (numel < 0 || static_cast<size_t>(numel) > t.size()) {
        throw std::runtime_error(std::string(fn) + ": numel=" + std::to_string(numel) +
                                 " exceeds " + name + " (" + std::to_string(t.size()) +
                                 " elements)");
    }
}

// fp8 crosses as uint8 (see map_dtype_to_code), so the fp8 side is checked as a
// code and the float side against the tensor's own dtype.
static void require_code(int code, int lo, int hi, const char* fn, const char* name) {
    if (code < lo || code > hi) {
        throw std::runtime_error(std::string(fn) + ": unsupported " + name + " code " +
                                 std::to_string(code));
    }
}

void quantize_per_tensor_fp8(nb::ndarray<> input, nb::ndarray<> scale, nb::ndarray<> output,
                             int input_dtype_code, int output_dtype_code, int64_t numel,
                             uintptr_t stream_ptr) {
    constexpr const char* kFn = "quantize_per_tensor_fp8";
    require_code(input_dtype_code, 0, 2, kFn, "input dtype");
    require_code(output_dtype_code, 5, 6, kFn, "output dtype");
    if (map_dtype_to_code(input.dtype()) != input_dtype_code) {
        throw std::runtime_error(std::string(kFn) + ": input dtype does not match its code");
    }
    // fp8 crosses as uint8; the output buffer must be that storage (as in scaled_mm_fp8).
    require_dtype(output, 3, 3, kFn, "output");
    require_scale(scale, kFn);
    require_numel(numel, input, kFn, "input");
    require_numel(numel, output, kFn, "output");

    launch_quantize_per_tensor_fp8_kernel(input.data(), scale.data(), output.data(), numel,
                                          input_dtype_code, output_dtype_code,
                                          reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void dequantize_per_tensor_fp8(nb::ndarray<> input, nb::ndarray<> scale, nb::ndarray<> output,
                               int input_dtype_code, int output_dtype_code, int64_t numel,
                               uintptr_t stream_ptr) {
    constexpr const char* kFn = "dequantize_per_tensor_fp8";
    require_code(input_dtype_code, 5, 6, kFn, "input dtype");
    require_code(output_dtype_code, 0, 2, kFn, "output dtype");
    if (map_dtype_to_code(output.dtype()) != output_dtype_code) {
        throw std::runtime_error(std::string(kFn) + ": output dtype does not match its code");
    }
    // fp8 crosses as uint8; the input buffer must be that storage (as in scaled_mm_fp8).
    require_dtype(input, 3, 3, kFn, "input");
    require_scale(scale, kFn);
    require_numel(numel, input, kFn, "input");
    require_numel(numel, output, kFn, "output");

    launch_dequantize_per_tensor_fp8_kernel(input.data(), scale.data(), output.data(), numel,
                                            input_dtype_code, output_dtype_code,
                                            reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void stochastic_round_fp8(nb::ndarray<> rng_and_output, nb::ndarray<> input, int output_dtype_code,
                          int64_t numel, uintptr_t stream_ptr) {
    constexpr const char* kFn = "stochastic_round_fp8";
    int rng_dtype_code = map_dtype_to_code(rng_and_output.dtype());
    if (rng_dtype_code != 3) {
        throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
    }
    int input_dtype_code = map_dtype_to_code(input.dtype());
    require_code(input_dtype_code, 0, 2, kFn, "input dtype");
    require_code(output_dtype_code, 5, 6, kFn, "output dtype");
    require_numel(numel, input, kFn, "input");
    require_numel(numel, rng_and_output, kFn, "rng");

    launch_stochastic_round_fp8_kernel(rng_and_output.data(), input.data(), numel, rng_dtype_code,
                                       input_dtype_code, output_dtype_code,
                                       reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void scaled_mm_fp8(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> scale_a,
                   nb::ndarray<> scale_b, OptArray bias, int M, int N, int K, int out_code,
                   uintptr_t stream_ptr) {
    constexpr const char* kFn = "scaled_mm_fp8";
    require_nonneg(M, kFn, "M");
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    // fp8 crosses the boundary as uint8: a is (M, K), b is (N, K), c is (M, N).
    require_dtype(a, 3, 3, kFn, "a");
    require_dtype(b, 3, 3, kFn, "b");
    require_dtype(c, 0, 2, kFn, "c");
    require_out_matches(c, out_code, kFn);
    require_len(a, static_cast<int64_t>(M) * K, kFn, "a");
    require_len(b, static_cast<int64_t>(N) * K, kFn, "b");
    require_len(c, static_cast<int64_t>(M) * N, kFn, "c");
    // EpiTensorwise reads scale_a[0] * scale_b[0].
    require_scale_len(scale_a, 1, kFn, "scale_a");
    require_scale_len(scale_b, 1, kFn, "scale_b");
    require_bias(bias, N, kFn);

    launch_scaled_mm_fp8_kernel(a.data(), b.data(), c.data(), scale_a.data(), scale_b.data(),
                                opt_data(bias), opt_code(bias), M, N, K, out_code,
                                reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void int8_gemm(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> scale_a,
               nb::ndarray<> scale_b, int scale_b_stride, OptArray bias, int M, int N, int K,
               int out_code, uintptr_t stream_ptr) {
    constexpr const char* kFn = "int8_gemm";
    // EpiRowwise reads scale_a[row] over M rows and scale_b[col * stride] over N
    // columns; a stride of 0 collapses the weight scale to a single scalar.
    if (scale_b_stride != 0 && scale_b_stride != 1) {
        throw std::runtime_error(std::string(kFn) + ": scale_b_stride must be 0 or 1, got " +
                                 std::to_string(scale_b_stride));
    }
    require_nonneg(M, kFn, "M");
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    // a is (M, K) int8, b is (N, K) int8, c is (M, N).
    require_dtype(a, 4, 4, kFn, "a");
    require_dtype(b, 4, 4, kFn, "b");
    require_dtype(c, 0, 2, kFn, "c");
    require_out_matches(c, out_code, kFn);
    require_len(a, static_cast<int64_t>(M) * K, kFn, "a");
    require_len(b, static_cast<int64_t>(N) * K, kFn, "b");
    require_len(c, static_cast<int64_t>(M) * N, kFn, "c");
    require_scale_len(scale_a, static_cast<size_t>(M), kFn, "scale_a");
    require_scale_len(scale_b, scale_b_stride == 1 ? static_cast<size_t>(N) : 1, kFn, "scale_b");
    require_bias(bias, N, kFn);

    launch_int8_gemm_kernel(a.data(), b.data(), c.data(), scale_a.data(), scale_b.data(),
                            scale_b_stride, opt_data(bias), opt_code(bias), M, N, K, N /*ldc*/,
                            out_code, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void convrot_w4a4_gemm(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> x_scale,
                       nb::ndarray<> w_scale, OptArray bias, int M, int N, int K, int out_code,
                       uintptr_t stream_ptr) {
    constexpr const char* kFn = "convrot_w4a4_gemm";
    require_nonneg(M, kFn, "M");
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    // int4 packs two nibbles per byte, so the operand rows are K / 2 bytes wide.
    if (K % 2 != 0) {
        throw std::runtime_error(std::string(kFn) + ": K must be even, got " + std::to_string(K));
    }
    require_dtype(a, 4, 4, kFn, "a");
    require_dtype(b, 4, 4, kFn, "b");
    require_dtype(c, 0, 2, kFn, "c");
    require_out_matches(c, out_code, kFn);
    require_len(a, static_cast<int64_t>(M) * (K / 2), kFn, "a");
    require_len(b, static_cast<int64_t>(N) * (K / 2), kFn, "b");
    require_len(c, static_cast<int64_t>(M) * N, kFn, "c");
    // EpiRowwise with a stride of 1: per-row activation scale, per-column weight scale.
    require_scale_len(x_scale, static_cast<size_t>(M), kFn, "x_scale");
    require_scale_len(w_scale, static_cast<size_t>(N), kFn, "w_scale");
    require_bias(bias, N, kFn);

    launch_convrot_w4a4_gemm_kernel(a.data(), b.data(), c.data(), x_scale.data(), w_scale.data(),
                                    opt_data(bias), opt_code(bias), M, N, K, out_code,
                                    reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// The int4 quantizers pack two nibbles per byte, so the packed row is K / 2 bytes
// and an odd K would round it down and drop the tail.
static void require_convrot_group(int k, int group_size, const char* fn) {
    // A negative K divisible by group_size would clear the divisibility check below,
    // and with M=0 the length products collapse to zero, so it would otherwise reach
    // the launcher as a negative extent.
    require_nonneg(k, fn, "K");
    if (group_size != 16 && group_size != 64 && group_size != 256) {
        throw std::runtime_error(std::string(fn) + ": group_size must be 16, 64 or 256, got " +
                                 std::to_string(group_size));
    }
    if (k % group_size != 0) {
        throw std::runtime_error(std::string(fn) + ": K=" + std::to_string(k) +
                                 " is not divisible by group_size=" + std::to_string(group_size));
    }
}

void quantize_int8_rowwise(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales, int M, int K,
                           uintptr_t stream_ptr) {
    constexpr const char* kFn = "quantize_int8_rowwise";
    require_nonneg(M, kFn, "M");
    require_nonneg(K, kFn, "K");
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(q, 4, 4, kFn, "q");
    require_len(x, static_cast<int64_t>(M) * K, kFn, "x");
    require_len(q, static_cast<int64_t>(M) * K, kFn, "q");
    require_scale_len(scales, static_cast<size_t>(M), kFn, "scales");

    launch_quantize_int8_rowwise_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                        scales.data(), M, K,
                                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// act_code folds an elementwise activation into the rotation's load.
void quantize_int8_convrot(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales,
                           OptArray spill_rotated, OptArray spill_partials, int M, int K,
                           int group_size, int act_code, uintptr_t stream_ptr) {
    constexpr const char* kFn = "quantize_int8_convrot";
    require_nonneg(M, kFn, "M");
    require_convrot_group(K, group_size, kFn);
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(q, 4, 4, kFn, "q");
    // K is the activated (written) width; swiglu (code 2, see INPUT_ACT_TO_CODE)
    // reads a [gate | up] row twice as wide. The launcher rejects unknown codes.
    const int64_t in_width = act_code == 2 ? 2 : 1;
    require_len(x, static_cast<int64_t>(M) * K * in_width, kFn, "x");
    require_len(q, static_cast<int64_t>(M) * K, kFn, "q");
    require_scale_len(scales, static_cast<size_t>(M), kFn, "scales");

    void* spill_rotated_ptr = nullptr;
    void* spill_partials_ptr = nullptr;
    if (spill_rotated.has_value()) {
        require_dtype(*spill_rotated, 0, 2, kFn, "spill_rotated");
        require_len(*spill_rotated, static_cast<int64_t>(M) * K, kFn, "spill_rotated");
        if (map_dtype_to_code(spill_rotated->dtype()) != map_dtype_to_code(x.dtype())) {
            throw std::runtime_error(std::string(kFn) + ": spill_rotated dtype must match x");
        }
        spill_rotated_ptr = spill_rotated->data();
    }
    if (spill_partials.has_value()) {
        require_dtype(*spill_partials, 0, 0, kFn, "spill_partials");
        require_len(*spill_partials, static_cast<int64_t>(M) * (K / 256), kFn, "spill_partials");
        spill_partials_ptr = spill_partials->data();
    }

    launch_quantize_int8_convrot_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                        scales.data(), spill_rotated_ptr, spill_partials_ptr, M, K,
                                        group_size, act_code,
                                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void quantize_int8_tensorwise(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scale,
                              nb::ndarray<> scratch, int64_t numel, uintptr_t stream_ptr) {
    constexpr const char* kFn = "quantize_int8_tensorwise";
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(q, 4, 4, kFn, "q");
    require_len(x, numel, kFn, "x");
    require_len(q, numel, kFn, "q");
    require_scale_len(scale, 1, kFn, "scale");
    // scratch is the kernel's int32 atomic accumulator, not a float scale: it has
    // to be a single 4-byte element, and the kernel atomicMaxes into it, so it must
    // start at zero rather than carry a stale absmax from the caller's buffer.
    require_len(scratch, 1, kFn, "scratch");
    if (scratch.dtype().bits != 32) {
        throw std::runtime_error(std::string(kFn) + ": scratch must be a 32-bit element");
    }
    auto stream = reinterpret_cast<hipStream_t>(stream_ptr);
    if (hipMemsetAsync(scratch.data(), 0, sizeof(unsigned int), stream) != hipSuccess) {
        throw std::runtime_error(std::string(kFn) + ": failed to clear scratch");
    }

    launch_quantize_int8_tensorwise_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                           scale.data(), scratch.data(), numel, stream);
    check_hip_launch();
}

void dequantize_int8_simple(nb::ndarray<> q, nb::ndarray<> scale, nb::ndarray<> out,
                            int64_t inner_dim, int scale_mode, uintptr_t stream_ptr) {
    constexpr const char* kFn = "dequantize_int8_simple";
    require_dtype(q, 4, 4, kFn, "q");
    require_scale_len(scale, 0, kFn, "scale");
    require_dtype(out, 0, 2, kFn, "out");
    if (out.size() != q.size()) {
        throw std::runtime_error(std::string(kFn) + ": output shape mismatch");
    }
    if (scale_mode < 0 || scale_mode > 2) {
        throw std::runtime_error(std::string(kFn) + ": invalid scale mode");
    }

    const int64_t numel = static_cast<int64_t>(q.size());
    if (numel > 0) {
        if (inner_dim <= 0) {
            throw std::runtime_error(std::string(kFn) + ": inner_dim must be positive");
        }
        size_t expected_scale = 1;
        if (scale_mode == 1) {
            expected_scale = q.size();
        } else if (scale_mode == 2) {
            if (numel % inner_dim != 0) {
                throw std::runtime_error(
                    std::string(kFn) + ": numel must be divisible by inner_dim");
            }
            expected_scale = static_cast<size_t>(numel / inner_dim);
        }
        if (scale.size() != expected_scale) {
            throw std::runtime_error(
                std::string(kFn) + ": scale has " + std::to_string(scale.size()) +
                " elements, expected " + std::to_string(expected_scale));
        }
    }

    launch_dequantize_int8_simple_kernel(
        q.data(), scale.data(), out.data(), numel, inner_dim, scale_mode,
        map_dtype_to_code(out.dtype()), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void dequantize_int8_convrot_weight(nb::ndarray<> q, nb::ndarray<> scale, nb::ndarray<> out,
                                    int M, int K, int group_size, uintptr_t stream_ptr) {
    constexpr const char* kFn = "dequantize_int8_convrot_weight";
    require_nonneg(M, kFn, "M");
    require_convrot_group(K, group_size, kFn);
    require_dtype(q, 4, 4, kFn, "q");
    require_scale_len(scale, 0, kFn, "scale");
    require_dtype(out, 0, 2, kFn, "out");
    require_len(q, static_cast<int64_t>(M) * K, kFn, "q");
    require_len(out, static_cast<int64_t>(M) * K, kFn, "out");
    if (q.size() != out.size() ||
        q.size() != static_cast<size_t>(static_cast<int64_t>(M) * K)) {
        throw std::runtime_error(std::string(kFn) + ": input/output shape mismatch");
    }
    if (scale.size() != 1 && scale.size() != static_cast<size_t>(M)) {
        throw std::runtime_error(
            std::string(kFn) + ": scale must contain one value or one value per row");
    }

    launch_dequantize_int8_convrot_weight_kernel(
        q.data(), scale.data(), out.data(), M, K, static_cast<int>(scale.size()),
        group_size, map_dtype_to_code(out.dtype()),
        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void convrot_quant_int4(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales, int M, int K,
                        int group_size, uintptr_t stream_ptr) {
    constexpr const char* kFn = "convrot_quant_int4";
    require_nonneg(M, kFn, "M");
    require_convrot_group(K, group_size, kFn);
    if (K % 2 != 0) {
        throw std::runtime_error(std::string(kFn) + ": K must be even, got " + std::to_string(K));
    }
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(q, 4, 4, kFn, "q");
    require_len(x, static_cast<int64_t>(M) * K, kFn, "x");
    require_len(q, static_cast<int64_t>(M) * (K / 2), kFn, "q");
    require_scale_len(scales, static_cast<size_t>(M), kFn, "scales");

    launch_convrot_quant_int4_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                     scales.data(), M, K, group_size,
                                     reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void unpack_int4(nb::ndarray<> q, nb::ndarray<> out, int64_t nbytes, uintptr_t stream_ptr) {
    constexpr const char* kFn = "unpack_int4";
    require_dtype(q, 4, 4, kFn, "q");
    require_dtype(out, 4, 4, kFn, "out");
    require_len(q, nbytes, kFn, "q");
    require_len(out, nbytes * 2, kFn, "out");  // one byte unpacks to two nibbles

    launch_unpack_int4_kernel(q.data(), out.data(), nbytes,
                              reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// scale_code names the s_rel storage: 0 float32, 5 e4m3 (crossing as uint8, as
// elsewhere). codebook is optional; without it the levels are uniform.
// Fused W4A8 requantize (group_size 16): a rotated weight [N, K] becomes packed
// int4 [N, K/2], raw e4m3 s_rel [N, K/16] and f32 s_channel [N] in one launch.
void quantize_w4a8_convrot(nb::ndarray<> rotated, nb::ndarray<> codebook, nb::ndarray<> packed,
                           nb::ndarray<> s_rel, nb::ndarray<> s_channel, int N, int K,
                           bool stochastic, uint64_t seed, uintptr_t stream_ptr) {
    constexpr const char* kFn = "quantize_w4a8_convrot";
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    if (K % 16 != 0) {
        throw std::runtime_error(std::string(kFn) + ": K must be a multiple of 16");
    }
    require_dtype(rotated, 0, 2, kFn, "rotated");
    require_dtype(packed, 4, 4, kFn, "packed");
    require_dtype(s_rel, 3, 3, kFn, "s_rel");
    require_scale_len(s_channel, static_cast<size_t>(N), kFn, "s_channel");
    require_scale_len(codebook, 16, kFn, "codebook");
    require_len(rotated, static_cast<int64_t>(N) * K, kFn, "rotated");
    require_len(packed, static_cast<int64_t>(N) * (K / 2), kFn, "packed");
    require_len(s_rel, static_cast<int64_t>(N) * (K / 16), kFn, "s_rel");

    launch_quantize_w4a8_convrot_kernel(rotated.data(), codebook.data(), packed.data(),
                                        s_rel.data(), s_channel.data(), N, K,
                                        map_dtype_to_code(rotated.dtype()), stochastic, seed,
                                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void dequant_int4_grouped_to_int8(nb::ndarray<> qdata, nb::ndarray<> s_rel, int scale_code,
                                  OptArray codebook, nb::ndarray<> out, int N, int K,
                                  int group_size, uintptr_t stream_ptr) {
    constexpr const char* kFn = "dequant_int4_grouped_to_int8";
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    require_positive(group_size, kFn, "group_size");
    if (scale_code != 0 && scale_code != 5) {
        throw std::runtime_error(std::string(kFn) + ": s_rel must be float32 or float8_e4m3fn");
    }
    if (K % group_size != 0) {
        throw std::runtime_error(std::string(kFn) + ": group_size must divide K");
    }
    require_dtype(qdata, 4, 4, kFn, "qdata");
    require_dtype(out, 4, 4, kFn, "out");
    require_dtype(s_rel, scale_code == 0 ? 0 : 3, scale_code == 0 ? 0 : 3, kFn, "s_rel");
    require_len(qdata, static_cast<int64_t>(N) * (K / 2), kFn, "qdata");
    require_len(out, static_cast<int64_t>(N) * K, kFn, "out");
    require_len(s_rel, static_cast<int64_t>(N) * (K / group_size), kFn, "s_rel");
    if (codebook.has_value()) {
        require_scale_len(*codebook, 16, kFn, "codebook");
    }

    launch_dequant_int4_grouped_to_int8_kernel(
        qdata.data(), s_rel.data(), scale_code, opt_data(codebook), out.data(), N, K, group_size,
        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// Chunked W4A8: decode chunk_cols weight columns at a time and run the INT8 GEMM
// on each chunk, writing an N-wide slice of out. xq/xs are the already rotated and
// quantized activation, so the loop only touches the weight.
void w4a8_int8_gemm_chunked(nb::ndarray<> xq, nb::ndarray<> qdata, nb::ndarray<> s_rel,
                            int scale_code, OptArray codebook, nb::ndarray<> s_channel,
                            nb::ndarray<> xs, OptArray bias, nb::ndarray<> workspace,
                            nb::ndarray<> out, int M, int N, int K, int group_size, int chunk_cols,
                            int out_code, uintptr_t stream_ptr) {
    constexpr const char* kFn = "w4a8_int8_gemm_chunked";
    require_nonneg(M, kFn, "M");
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    require_positive(group_size, kFn, "group_size");
    require_positive(chunk_cols, kFn, "chunk_cols");
    if (scale_code != 0 && scale_code != 5) {
        throw std::runtime_error(std::string(kFn) + ": s_rel must be float32 or float8_e4m3fn");
    }
    if (K % group_size != 0) {
        throw std::runtime_error(std::string(kFn) + ": group_size must divide K");
    }
    require_dtype(xq, 4, 4, kFn, "xq");
    require_dtype(qdata, 4, 4, kFn, "qdata");
    require_dtype(workspace, 4, 4, kFn, "workspace");
    require_dtype(out, 0, 2, kFn, "out");
    require_out_matches(out, out_code, kFn);
    require_dtype(s_rel, scale_code == 0 ? 0 : 3, scale_code == 0 ? 0 : 3, kFn, "s_rel");
    require_len(xq, static_cast<int64_t>(M) * K, kFn, "xq");
    require_len(qdata, static_cast<int64_t>(N) * (K / 2), kFn, "qdata");
    require_len(s_rel, static_cast<int64_t>(N) * (K / group_size), kFn, "s_rel");
    require_len(out, static_cast<int64_t>(M) * N, kFn, "out");
    // Every chunk decodes into the same scratch, so it has to hold the widest one.
    require_len(workspace, static_cast<int64_t>(chunk_cols < N ? chunk_cols : N) * K, kFn,
                "workspace");
    require_scale_len(s_channel, static_cast<size_t>(N), kFn, "s_channel");
    require_scale_len(xs, static_cast<size_t>(M), kFn, "xs");
    require_bias(bias, N, kFn);
    if (codebook.has_value()) {
        require_scale_len(*codebook, 16, kFn, "codebook");
    }

    launch_w4a8_int8_gemm_chunked_kernel(
        xq.data(), qdata.data(), s_rel.data(), scale_code, opt_data(codebook), s_channel.data(),
        xs.data(), opt_data(bias), opt_code(bias), workspace.data(), out.data(), M, N, K,
        group_size, chunk_cols, out_code, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// subtract_mean selects LayerNorm (adaln) or RMSNorm (rms_adaln) statistics.
static void adaln_impl(const char* kFn, nb::ndarray<>& x, nb::ndarray<>& scale,
                       nb::ndarray<>& shift, nb::ndarray<>& out, int N, int D, int scale_group,
                       int shift_group, float eps, bool subtract_mean, uintptr_t stream_ptr) {
    require_nonneg(N, kFn, "N");
    require_nonneg(D, kFn, "D");
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(out, 0, 2, kFn, "out");
    require_dtype(scale, 0, 2, kFn, "scale");
    require_dtype(shift, 0, 2, kFn, "shift");
    // The launcher gets one dtype code (x's) and writes out at that width, so a
    // wider x than out would overrun the output buffer.
    if (map_dtype_to_code(out.dtype()) != map_dtype_to_code(x.dtype())) {
        throw std::runtime_error(std::string(kFn) + ": out must have the same dtype as x");
    }
    require_len(x, static_cast<int64_t>(N) * D, kFn, "x");
    require_len(out, static_cast<int64_t>(N) * D, kFn, "out");
    // The kernel reads scale[(row / scale_group) * D + i] for row < N, i < D. An
    // empty input divides by nothing and launches no blocks, and the group sizes
    // the caller derives from it are 0, so only constrain them when there are rows.
    if (N > 0 && D > 0) {
        require_positive(scale_group, kFn, "scale_group");
        require_positive(shift_group, kFn, "shift_group");
        require_len(scale, (static_cast<int64_t>(N - 1) / scale_group + 1) * D, kFn, "scale");
        require_len(shift, (static_cast<int64_t>(N - 1) / shift_group + 1) * D, kFn, "shift");
    }

    launch_adaln_kernel(x.data(), scale.data(), shift.data(), out.data(), N, D, scale_group,
                        shift_group, eps, map_dtype_to_code(x.dtype()),
                        map_dtype_to_code(scale.dtype()), map_dtype_to_code(shift.dtype()),
                        subtract_mean, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// Fused 3D neighborhood attention. Every extent is caller-supplied and the kernel
// indexes up to it, so the operands are sized here rather than trusted.
void na3d(nb::ndarray<> q, nb::ndarray<> k, nb::ndarray<> v, nb::ndarray<> out, int batch,
          int t_size, int h_size, int w_size, int num_heads, int head_dim, int kt, int kh, int kw,
          int causal_t, int causal_h, int causal_w, float scale, int dtype_code,
          uintptr_t stream_ptr) {
    constexpr const char* kFn = "na3d";
    require_positive(batch, kFn, "batch");
    require_positive(t_size, kFn, "t_size");
    require_positive(h_size, kFn, "h_size");
    require_positive(w_size, kFn, "w_size");
    require_positive(num_heads, kFn, "num_heads");
    require_positive(head_dim, kFn, "head_dim");
    require_positive(kt, kFn, "kt");
    require_positive(kh, kFn, "kh");
    require_positive(kw, kFn, "kw");
    if (dtype_code != 1 && dtype_code != 2) {
        throw std::runtime_error(std::string(kFn) + ": q must be float16 or bfloat16");
    }
    // The kernel reads k and v off bare pointers with q's extents, so a differing
    // dtype or a shorter operand is an out-of-bounds device read.
    const int64_t need = static_cast<int64_t>(batch) * t_size * h_size * w_size * num_heads *
                         head_dim;
    require_dtype(q, dtype_code, dtype_code, kFn, "q");
    require_dtype(k, dtype_code, dtype_code, kFn, "k");
    require_dtype(v, dtype_code, dtype_code, kFn, "v");
    require_dtype(out, dtype_code, dtype_code, kFn, "out");
    require_len(q, need, kFn, "q");
    require_len(k, need, kFn, "k");
    require_len(v, need, kFn, "v");
    require_len(out, need, kFn, "out");

    launch_na3d_kernel(q.data(), k.data(), v.data(), out.data(), batch, t_size, h_size, w_size,
                       num_heads, head_dim, kt, kh, kw, causal_t, causal_h, causal_w, scale,
                       dtype_code, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void adaln(nb::ndarray<> x, nb::ndarray<> scale, nb::ndarray<> shift, nb::ndarray<> out, int N,
           int D, int scale_group, int shift_group, float eps, uintptr_t stream_ptr) {
    adaln_impl("adaln", x, scale, shift, out, N, D, scale_group, shift_group, eps,
               /*subtract_mean=*/true, stream_ptr);
}

void rms_adaln(nb::ndarray<> x, nb::ndarray<> scale, nb::ndarray<> shift, nb::ndarray<> out, int N,
               int D, int scale_group, int shift_group, float eps, uintptr_t stream_ptr) {
    adaln_impl("rms_adaln", x, scale, shift, out, N, D, scale_group, shift_group, eps,
               /*subtract_mean=*/false, stream_ptr);
}

// x is (batch, dim1, dim2, head_dim); freqs is (fb, fd1, fd2, rot_dim/2, 2, 2).
// Shapes and strides are read off the arrays so the broadcast rules stay in one
// place rather than being recomputed on the Python side. Shared by apply_rope and
// rms_rope, which index both tensors the same way. rot_dim is the rotated head-dim
// prefix; 0 rotates everything.
static int64_t require_rope_shapes(const nb::ndarray<>& x, const nb::ndarray<>& freqs,
                                   const char* fn, int64_t rot_dim = 0) {
    // map_dtype_to_code returns -1 for anything else, which the device decoder
    // would misread; the other operands are checked against x's dtype separately.
    require_dtype(x, 0, 2, fn, "x");
    require_dtype(freqs, 0, 2, fn, "freqs_cis");
    if (x.ndim() != 4) throw std::runtime_error(std::string(fn) + " expects a 4D input");
    if (freqs.ndim() != 6) throw std::runtime_error(std::string(fn) + " expects a 6D freqs_cis");

    // Only the rotated prefix is paired, so an odd head_dim is fine as long as the
    // resolved rot is even: the leftover tail is norm-only. apply_rope stays
    // covered because its rot resolves to head_dim.
    const int64_t head_dim = static_cast<int64_t>(x.shape(3));
    const int64_t rot = rot_dim > 0 ? rot_dim : head_dim;
    if (rot % 2 != 0 || rot > head_dim) {
        throw std::runtime_error(std::string(fn) + " expects an even rot_dim <= head_dim");
    }

    // The kernel indexes freqs as (fb, fd1, fd2, rot_dim/2, 2, 2), broadcasting a
    // leading dim only when it is 1. Anything else walks off the end of the array.
    if (freqs.shape(3) != static_cast<size_t>(rot / 2) || freqs.shape(4) != 2 ||
        freqs.shape(5) != 2) {
        throw std::runtime_error(std::string(fn) +
                                 " expects freqs_cis trailing dims (rot_dim/2, 2, 2)");
    }
    for (size_t i = 0; i < 3; ++i) {
        if (freqs.shape(i) != 1 && freqs.shape(i) != x.shape(i)) {
            throw std::runtime_error(
                std::string(fn) +
                " expects each leading freqs_cis dim to be 1 or match the input");
        }
    }
    return rot;
}

// An output is walked with its own strides, so it only has to match the extents.
static void require_rope_extents(const nb::ndarray<>& x, const nb::ndarray<>& t, const char* name,
                                 const char* fn) {
    if (t.ndim() != 4 || t.dtype() != x.dtype()) {
        throw std::runtime_error(std::string(fn) + " expects " + name +
                                 " to be 4D with the input's dtype");
    }
    for (size_t i = 0; i < 4; ++i) {
        if (t.shape(i) != x.shape(i)) {
            throw std::runtime_error(std::string(fn) + " expects " + name +
                                     " to have the input's shape");
        }
    }
}

// A q/k pair is read off one set of strides and one dtype code, so the two have
// to agree on both. The stride of a length-1 axis is skipped: it is only ever
// multiplied by index zero.
static void require_rope_layout(const nb::ndarray<>& x, const nb::ndarray<>& t, const char* name,
                                const char* fn) {
    require_rope_extents(x, t, name, fn);
    for (size_t i = 0; i < 4; ++i) {
        if (x.shape(i) > 1 && t.stride(i) != x.stride(i)) {
            throw std::runtime_error(std::string(fn) + " expects " + name +
                                     " to have the input's strides");
        }
    }
}

void apply_rope(nb::ndarray<> xq, OptArray xk, nb::ndarray<> freqs, nb::ndarray<> xq_out,
                OptArray xk_out, bool split_half, uintptr_t stream_ptr) {
    constexpr const char* kFn = "apply_rope";
    require_rope_shapes(xq, freqs, kFn);

    if (xk.has_value() != xk_out.has_value()) {
        throw std::runtime_error("apply_rope expects xk and xk_out together or not at all");
    }
    // Inputs share one stride set and outputs another, so the two sets are
    // independent: xq may be a strided view while xq_out is dense.
    require_rope_extents(xq, xq_out, "xq_out", kFn);
    if (xk.has_value()) {
        require_rope_layout(xq, *xk, "xk", kFn);
        require_rope_layout(xq_out, *xk_out, "xk_out", kFn);
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
        static_cast<int64_t>(xq_out.stride(0)), static_cast<int64_t>(xq_out.stride(1)),
        static_cast<int64_t>(xq_out.stride(2)), static_cast<int64_t>(xq_out.stride(3)),
        static_cast<int64_t>(freqs.stride(0)), static_cast<int64_t>(freqs.stride(1)),
        static_cast<int64_t>(freqs.stride(2)), static_cast<int64_t>(freqs.stride(3)),
        static_cast<int64_t>(freqs.stride(4)), static_cast<int64_t>(freqs.stride(5)),
        map_dtype_to_code(xq.dtype()), map_dtype_to_code(freqs.dtype()), split_half,
        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// Fused RMSNorm + RoPE. Same operand layout as apply_rope, plus a per-head_dim
// weight for each input.
void rms_rope(nb::ndarray<> q, OptArray k, nb::ndarray<> freqs, nb::ndarray<> q_scale,
              OptArray k_scale, nb::ndarray<> q_out, OptArray k_out, float epsilon,
              bool split_half, uintptr_t stream_ptr, int64_t rot_dim) {
    constexpr const char* kFn = "rms_rope";
    const int64_t rot = require_rope_shapes(q, freqs, kFn, rot_dim);

    // k, its scale and its output are one operand set.
    if (k.has_value() != k_out.has_value() || k.has_value() != k_scale.has_value()) {
        throw std::runtime_error("rms_rope expects k, k_scale and k_out together or not at all");
    }
    require_rope_extents(q, q_out, "q_out", kFn);
    if (k.has_value()) {
        require_rope_layout(q, *k, "k", kFn);
        require_rope_layout(q_out, *k_out, "k_out", kFn);
    }

    // The weight is indexed by element within the row, so it needs one entry per
    // head_dim, decoded with a single code shared by both weights.
    const int64_t head_dim = static_cast<int64_t>(q.shape(3));
    require_dtype(q_scale, 0, 2, kFn, "q_scale");
    require_len(q_scale, head_dim, kFn, "q_scale");
    if (k_scale.has_value()) {
        if (k_scale->dtype() != q_scale.dtype()) {
            throw std::runtime_error("rms_rope expects k_scale to have q_scale's dtype");
        }
        require_len(*k_scale, head_dim, kFn, "k_scale");
    }

    launch_rms_rope_kernel(
        q.data(), k.has_value() ? k->data() : nullptr, freqs.data(), q_scale.data(),
        k_scale.has_value() ? k_scale->data() : nullptr, q_out.data(),
        k_out.has_value() ? k_out->data() : nullptr,
        static_cast<int64_t>(q.shape(0)), static_cast<int64_t>(q.shape(1)),
        static_cast<int64_t>(q.shape(2)), head_dim, rot,
        static_cast<int64_t>(freqs.shape(0)), static_cast<int64_t>(freqs.shape(1)),
        static_cast<int64_t>(freqs.shape(2)),
        static_cast<int64_t>(q.stride(0)), static_cast<int64_t>(q.stride(1)),
        static_cast<int64_t>(q.stride(2)), static_cast<int64_t>(q.stride(3)),
        static_cast<int64_t>(q_out.stride(0)), static_cast<int64_t>(q_out.stride(1)),
        static_cast<int64_t>(q_out.stride(2)), static_cast<int64_t>(q_out.stride(3)),
        static_cast<int64_t>(freqs.stride(0)), static_cast<int64_t>(freqs.stride(1)),
        static_cast<int64_t>(freqs.stride(2)), static_cast<int64_t>(freqs.stride(3)),
        static_cast<int64_t>(freqs.stride(4)), static_cast<int64_t>(freqs.stride(5)),
        map_dtype_to_code(q.dtype()), map_dtype_to_code(freqs.dtype()),
        map_dtype_to_code(q_scale.dtype()), epsilon, split_half,
        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void gemv_awq_w4a16(nb::ndarray<> x, nb::ndarray<> qweight, nb::ndarray<> wscales,
                    nb::ndarray<> wzeros, OptArray bias, nb::ndarray<> out, int M, int N, int K,
                    int group_size, uintptr_t stream_ptr) {
    constexpr const char* kFn = "gemv_awq_w4a16";
    require_nonneg(M, kFn, "M");
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    // The launcher enforces the packing invariants (group_size a positive multiple
    // of 8, K a multiple of both 8 and group_size); these are the operand extents.
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(qweight, 4, 4, kFn, "qweight");
    require_dtype(out, 0, 2, kFn, "out");
    require_dtype(wscales, 0, 2, kFn, "wscales");
    require_dtype(wzeros, 0, 2, kFn, "wzeros");
    // Only wscales' dtype code reaches the kernel; it decodes wzeros with the same
    // code, so a differing wzeros dtype would be misread.
    if (map_dtype_to_code(wzeros.dtype()) != map_dtype_to_code(wscales.dtype())) {
        throw std::runtime_error(std::string(kFn) + ": wzeros must have the same dtype as wscales");
    }
    require_len(x, static_cast<int64_t>(M) * K, kFn, "x");
    require_len(out, static_cast<int64_t>(M) * N, kFn, "out");
    if (group_size > 0 && K % 2 == 0) {
        require_len(qweight, static_cast<int64_t>(N) * (K / 2), kFn, "qweight");
        // scale and zero are read at (k / group_size) * N + n.
        const int64_t groups = static_cast<int64_t>(K) / group_size;
        require_len(wscales, groups * N, kFn, "wscales");
        require_len(wzeros, groups * N, kFn, "wzeros");
    }
    require_bias(bias, N, kFn);

    launch_gemv_awq_kernel(x.data(), qweight.data(), wscales.data(), wzeros.data(), opt_data(bias),
                           out.data(), M, N, K, group_size, map_dtype_to_code(x.dtype()),
                           map_dtype_to_code(wscales.dtype()), opt_code(bias),
                           map_dtype_to_code(out.dtype()),
                           reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void svdquant_lora_down(nb::ndarray<> x, nb::ndarray<> lora_down, nb::ndarray<> lora_act, int M,
                        int K, int R, uintptr_t stream_ptr) {
    constexpr const char* kFn = "svdquant_lora_down";
    require_nonneg(M, kFn, "M");
    require_nonneg(K, kFn, "K");
    require_nonneg(R, kFn, "R");
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(lora_down, 0, 2, kFn, "lora_down");
    // The launcher writes lora_act through a float*, so it must be float32 storage.
    require_dtype(lora_act, 0, 0, kFn, "lora_act");
    require_len(x, static_cast<int64_t>(M) * K, kFn, "x");
    require_len(lora_down, static_cast<int64_t>(K) * R, kFn, "lora_down");
    require_len(lora_act, static_cast<int64_t>(M) * R, kFn, "lora_act");

    launch_svdquant_lora_down_kernel(x.data(), lora_down.data(), lora_act.data(), M, K, R,
                                     map_dtype_to_code(x.dtype()),
                                     map_dtype_to_code(lora_down.dtype()),
                                     reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void svdquant_quantize(nb::ndarray<> x, nb::ndarray<> smooth, nb::ndarray<> q,
                       nb::ndarray<> ascales, int M, int M_pad, int K, bool act_unsigned,
                       uintptr_t stream_ptr) {
    constexpr const char* kFn = "svdquant_quantize";
    if (K % kSvdGroup != 0) {
        throw std::runtime_error(std::string(kFn) + ": K=" + std::to_string(K) +
                                 " must be a multiple of " + std::to_string(kSvdGroup));
    }
    if (M_pad < M) {
        throw std::runtime_error(std::string(kFn) + ": M_pad must be at least M");
    }
    require_nonneg(M, kFn, "M");
    require_nonneg(K, kFn, "K");
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(smooth, 0, 2, kFn, "smooth");
    require_dtype(q, 4, 4, kFn, "q");
    require_dtype(ascales, 0, 2, kFn, "ascales");
    // The launcher passes only ascales' dtype code; the kernel decodes smooth with
    // it too, so a differing smooth dtype would be misread.
    if (map_dtype_to_code(smooth.dtype()) != map_dtype_to_code(ascales.dtype())) {
        throw std::runtime_error(std::string(kFn) + ": smooth must have the same dtype as ascales");
    }
    require_len(x, static_cast<int64_t>(M) * K, kFn, "x");
    require_len(smooth, K, kFn, "smooth");
    require_len(q, static_cast<int64_t>(M_pad) * (K / 2), kFn, "q");
    // ascales is (K / 64, M_pad): the group stride is M_pad.
    require_len(ascales, (static_cast<int64_t>(K) / kSvdGroup) * M_pad, kFn, "ascales");

    launch_svdquant_quant_kernel(x.data(), smooth.data(), q.data(), ascales.data(), M, M_pad, K,
                                 map_dtype_to_code(x.dtype()), map_dtype_to_code(ascales.dtype()),
                                 act_unsigned, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void svdquant_gemm(nb::ndarray<> a, nb::ndarray<> b, nb::ndarray<> c, nb::ndarray<> ascales,
                   nb::ndarray<> wscales, nb::ndarray<> lora_act, nb::ndarray<> lora_up,
                   OptArray bias, int M, int N, int K, int R, bool act_unsigned,
                   uintptr_t stream_ptr) {
    constexpr const char* kFn = "svdquant_gemm";
    require_nonneg(M, kFn, "M");
    require_nonneg(N, kFn, "N");
    require_nonneg(K, kFn, "K");
    require_nonneg(R, kFn, "R");
    if (K % kSvdGroup != 0) {
        throw std::runtime_error(std::string(kFn) + ": K=" + std::to_string(K) +
                                 " must be a multiple of " + std::to_string(kSvdGroup));
    }
    require_dtype(a, 4, 4, kFn, "act");
    require_dtype(b, 4, 4, kFn, "wgt");
    require_dtype(c, 0, 2, kFn, "out");
    require_dtype(ascales, 0, 2, kFn, "ascales");
    require_dtype(wscales, 0, 2, kFn, "wscales");
    require_dtype(lora_act, 0, 2, kFn, "lora_act");
    require_dtype(lora_up, 0, 2, kFn, "lora_up");

    const int64_t groups = static_cast<int64_t>(K) / kSvdGroup;
    require_len(a, static_cast<int64_t>(M) * (K / 2), kFn, "act");
    require_len(b, static_cast<int64_t>(N) * (K / 2), kFn, "wgt");
    require_len(c, static_cast<int64_t>(M) * N, kFn, "out");
    // The kernel reads ascales at g * M + row, so M is also its row stride.
    require_len(ascales, groups * M, kFn, "ascales");
    require_len(wscales, groups * N, kFn, "wscales");
    require_len(lora_act, static_cast<int64_t>(M) * R, kFn, "lora_act");
    require_len(lora_up, static_cast<int64_t>(N) * R, kFn, "lora_up");
    require_bias(bias, N, kFn);

    launch_svdquant_gemm_kernel(
        a.data(), b.data(), c.data(), ascales.data(), wscales.data(), lora_act.data(),
        lora_up.data(), opt_data(bias), M, N, K, R, map_dtype_to_code(ascales.dtype()),
        map_dtype_to_code(wscales.dtype()), map_dtype_to_code(lora_act.dtype()),
        map_dtype_to_code(lora_up.dtype()), opt_code(bias), map_dtype_to_code(c.dtype()),
        act_unsigned, reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// ---------------------------------------------------------------------------
// INT8 attention
//
// Buffer shapes differ from the CUDA backend because the scale granularity does;
// see sage_attention/quant_qk_int8.hip. _C is importable on its own, so every
// extent the kernels index off a raw pointer is checked here rather than trusted
// from the Python layer.
// ---------------------------------------------------------------------------

// Must match kCtaQ and the key tile in sage_attention/int8_attn.hip.
constexpr int kSageCtaQ = 128;
constexpr int kSageCtaK = 64;
constexpr int kSageKeyGroup = 16;

static int sage_padded_q(int qo_len) { return ((qo_len + kSageCtaQ - 1) / kSageCtaQ) * kSageCtaQ; }

static int sage_padded_k(int kv_len, int cta_k) { return ((kv_len + cta_k - 1) / cta_k) * cta_k; }

static void sage_check_cta_k(int cta_k, const char* fn) {
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error(std::string(fn) + ": cta_k must be 64 or 128");
    }
}

// Same selection the CUDA backend makes: short keys take the cheap H4 blocks,
// longer ones the widest Hadamard that fits the head dimension. Q and K must
// agree on it or their dot products change. 128 is the signed H128; padded D256
// keeps the plain one, which is 129.
static int sage_rotation(int kv_len, int head_dim) {
    if (kv_len <= 256) return 4;
    if (head_dim >= 256) return 129;
    return head_dim >= 128 ? 128 : 64;
}

static void sage_check_shapes(const nb::ndarray<>& q, const nb::ndarray<>& k,
                              const nb::ndarray<>& v, const char* fn) {
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
        throw std::runtime_error(std::string(fn) + ": q, k and v must be 4D [B, H, L, D]");
    }
    const int head_dim = static_cast<int>(q.shape(3));
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
        throw std::runtime_error(std::string(fn) + ": head_dim must be 64, 128 or 256, got " +
                                 std::to_string(head_dim));
    }
    if (k.shape(0) != q.shape(0) || v.shape(0) != q.shape(0) || v.shape(1) != k.shape(1) ||
        v.shape(2) != k.shape(2) || k.shape(3) != q.shape(3) || v.shape(3) != q.shape(3)) {
        throw std::runtime_error(std::string(fn) + ": incompatible q, k and v shapes");
    }
    // The modulo below divides by the k/v head count, and _C is importable, so a
    // zero here would be a SIGFPE rather than an exception.
    if (q.shape(0) == 0 || q.shape(1) == 0 || k.shape(1) == 0 || q.shape(2) == 0 ||
        k.shape(2) == 0) {
        throw std::runtime_error(
            std::string(fn) + ": batch, head counts and sequence lengths must be positive");
    }
    if (q.shape(1) % k.shape(1) != 0) {
        throw std::runtime_error(std::string(fn) +
                                 ": q head count must be a multiple of the k/v head count");
    }
}

// sage_attend synthesizes Q/K/V/O strides from the extents rather than reading
// them, so anything but the packed row-major layout is read as though it were
// packed. The Python layer only ever allocates fresh contiguous buffers; a caller
// reaching _C directly can pass a view.
static void require_packed_contiguous(const nb::ndarray<>& t, const char* fn, const char* name) {
    int64_t expected = 1;
    for (int axis = static_cast<int>(t.ndim()) - 1; axis >= 0; --axis) {
        if (t.shape(axis) != 1 && t.stride(axis) != expected) {
            throw std::runtime_error(std::string(fn) + ": " + name +
                                     " must be contiguous in the packed layout");
        }
        expected *= static_cast<int64_t>(t.shape(axis));
    }
}

static void sage_check_quantized(const nb::ndarray<>& q_int8, const nb::ndarray<>& q_scale,
                                 const nb::ndarray<>& k_int8, const nb::ndarray<>& k_scale,
                                 const nb::ndarray<>& v_int8, const nb::ndarray<>& v_scale,
                                 int batch, int q_heads, int kv_heads, int qo_len, int kv_len,
                                 int head_dim, int cta_k, const char* fn) {
    const int64_t padded_q = sage_padded_q(qo_len);
    const int64_t padded_k = sage_padded_k(kv_len, cta_k);
    require_dtype(q_int8, 4, 4, fn, "q_int8");
    require_dtype(k_int8, 4, 4, fn, "k_int8");
    require_dtype(v_int8, 4, 4, fn, "v_int8");
    require_len(q_int8, static_cast<int64_t>(batch) * q_heads * qo_len * head_dim, fn, "q_int8");
    require_len(k_int8, static_cast<int64_t>(batch) * kv_heads * kv_len * head_dim, fn, "k_int8");
    require_len(v_int8, static_cast<int64_t>(batch) * kv_heads * head_dim * padded_k, fn,
                "v_int8");
    require_scale_len(q_scale, static_cast<size_t>(batch) * q_heads * padded_q, fn, "q_scale");
    require_scale_len(k_scale, static_cast<size_t>(batch) * kv_heads * (padded_k / kSageKeyGroup),
                      fn, "k_scale");
    require_scale_len(v_scale, static_cast<size_t>(batch) * kv_heads * head_dim, fn, "v_scale");
    // The quantizers and sage_attend both synthesize every stride from the
    // extents, so a strided view would be written, and later read, as though it
    // were packed. Both entry points reach this helper.
    require_packed_contiguous(q_int8, fn, "q_int8");
    require_packed_contiguous(k_int8, fn, "k_int8");
    require_packed_contiguous(v_int8, fn, "v_int8");
    require_packed_contiguous(q_scale, fn, "q_scale");
    require_packed_contiguous(k_scale, fn, "k_scale");
    require_packed_contiguous(v_scale, fn, "v_scale");
}

static void sage_quantize(const nb::ndarray<>& q, const nb::ndarray<>& k, const nb::ndarray<>& v,
                          const nb::ndarray<>& q_int8, const nb::ndarray<>& q_scale,
                          const nb::ndarray<>& k_int8, const nb::ndarray<>& k_scale,
                          const nb::ndarray<>& v_int8, const nb::ndarray<>& v_scale,
                          const nb::ndarray<>& anchor_indices, int input_dtype_code, int cta_k,
                          hipStream_t stream, const char* fn) {
    const int batch = static_cast<int>(q.shape(0));
    const int q_heads = static_cast<int>(q.shape(1));
    const int qo_len = static_cast<int>(q.shape(2));
    const int head_dim = static_cast<int>(q.shape(3));
    const int kv_heads = static_cast<int>(k.shape(1));
    const int kv_len = static_cast<int>(k.shape(2));
    const int padded_k = sage_padded_k(kv_len, cta_k);

    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error(std::string(fn) +
                                 ": input dtype must be float32, float16 or bfloat16");
    }
    require_dtype(q, input_dtype_code, input_dtype_code, fn, "q");
    require_dtype(k, input_dtype_code, input_dtype_code, fn, "k");
    require_dtype(v, input_dtype_code, input_dtype_code, fn, "v");
    if (q.stride(3) != 1 || k.stride(3) != 1 || v.stride(3) != 1) {
        throw std::runtime_error(std::string(fn) +
                                 ": the last dimension of q, k and v must be contiguous");
    }
    sage_check_quantized(q_int8, q_scale, k_int8, k_scale, v_int8, v_scale, batch, q_heads,
                         kv_heads, qo_len, kv_len, head_dim, cta_k, fn);
    // The detector writes one index per (batch, kv head); there is no int32 code
    // in map_dtype_to_code, so the width is what gets checked.
    if (anchor_indices.dtype().bits != 32) {
        throw std::runtime_error(std::string(fn) + ": anchor_indices must be a 32-bit element");
    }
    require_len(anchor_indices, static_cast<int64_t>(batch) * kv_heads, fn, "anchor_indices");

    launch_sage_quant_qk_int8(q.data(), q_int8.data(), q_scale.data(), k.data(), k_int8.data(),
                              k_scale.data(), anchor_indices.data(), batch, q_heads, qo_len,
                              sage_padded_q(qo_len), kv_heads, kv_len, padded_k / kSageKeyGroup,
                              head_dim, q.stride(0), q.stride(1), q.stride(2), k.stride(0),
                              k.stride(1), k.stride(2), input_dtype_code,
                              sage_rotation(kv_len, head_dim), stream);

    launch_sage_quant_v_int8(v.data(), v_int8.data(), v_scale.data(), batch, kv_heads, kv_len,
                             head_dim, padded_k, v.stride(0), v.stride(1), v.stride(2),
                             input_dtype_code, stream);
}

// An expanded mask carries zero strides, so a per-key mask arrives with
// mask_stride_q == 0 and its query term drops out of the kernel's addressing on
// its own. That needs no separate mask mode.
static void sage_mask_info(const OptArray& attn_mask, int batch, int q_heads, int qo_len,
                           int kv_len, const void*& ptr, int64_t& stride_b, int64_t& stride_h,
                           int64_t& stride_q, int64_t& stride_k, int& dtype_code,
                           const char* fn) {
    ptr = nullptr;
    stride_b = stride_h = stride_q = stride_k = 0;
    dtype_code = -1;
    if (!attn_mask.has_value()) return;

    const auto& mask = attn_mask.value();
    if (mask.ndim() != 4 || static_cast<int>(mask.shape(0)) != batch ||
        static_cast<int>(mask.shape(1)) != q_heads || static_cast<int>(mask.shape(2)) != qo_len ||
        static_cast<int>(mask.shape(3)) != kv_len) {
        throw std::runtime_error(std::string(fn) +
                                 ": attention mask must be expanded to [B, H_q, Lq, Lk]");
    }
    if (mask.dtype().code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool)) {
        dtype_code = 3;
    } else {
        // map_dtype_to_code gives uint8 the same code 3 that marks a bool mask
        // here, and mask_keep would then read it as one. Only the float codes may
        // come through this branch.
        dtype_code = map_dtype_to_code(mask.dtype());
        if (dtype_code > 2) dtype_code = -1;
    }
    if (dtype_code < 0 || dtype_code > 3) {
        throw std::runtime_error(std::string(fn) +
                                 ": attention mask must be bool, float16, bfloat16 or float32");
    }
    ptr = mask.data();
    stride_b = mask.stride(0);
    stride_h = mask.stride(1);
    stride_q = mask.stride(2);
    stride_k = mask.stride(3);
}

static void sage_attend(const nb::ndarray<>& q_int8, const nb::ndarray<>& k_int8,
                        const nb::ndarray<>& v_int8, const nb::ndarray<>& o,
                        const nb::ndarray<>& q_scale, const nb::ndarray<>& k_scale,
                        const nb::ndarray<>& v_scale, const OptArray& attn_mask, int batch,
                        int q_heads, int kv_heads, int qo_len, int kv_len, int head_dim,
                        int cta_k, float sm_scale, int output_dtype_code, hipStream_t stream,
                        const char* fn) {
    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error(std::string(fn) + ": output dtype must be float16 or bfloat16");
    }
    require_dtype(o, output_dtype_code, output_dtype_code, fn, "o");
    require_len(o, static_cast<int64_t>(batch) * q_heads * qo_len * head_dim, fn, "o");
    // The output strides below are synthesized from the extents rather than read
    // from o, so anything but the packed layout would be written as though it
    // were packed. Both entry points reach this, so the check belongs here.
    if (o.ndim() != 4 || static_cast<int>(o.shape(0)) != batch ||
        static_cast<int>(o.shape(1)) != q_heads || static_cast<int>(o.shape(2)) != qo_len ||
        static_cast<int>(o.shape(3)) != head_dim) {
        throw std::runtime_error(std::string(fn) + ": o must be [B, H_q, Lq, D]");
    }
    require_packed_contiguous(o, fn, "o");

    const void* mask_ptr = nullptr;
    int64_t mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k;
    int mask_dtype_code;
    sage_mask_info(attn_mask, batch, q_heads, qo_len, kv_len, mask_ptr, mask_stride_b,
                   mask_stride_h, mask_stride_q, mask_stride_k, mask_dtype_code, fn);

    const int padded_k = sage_padded_k(kv_len, cta_k);
    launch_sage_int8_attn(
        q_int8.data(), k_int8.data(), v_int8.data(), o.data(), q_scale.data(), k_scale.data(),
        v_scale.data(), mask_ptr, mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
        mask_dtype_code, cta_k, batch, qo_len, kv_len, sage_padded_q(qo_len), q_heads,
        kv_heads, head_dim, padded_k / kSageKeyGroup,
        static_cast<int64_t>(q_heads) * qo_len * head_dim, static_cast<int64_t>(qo_len) * head_dim,
        static_cast<int64_t>(kv_heads) * kv_len * head_dim,
        static_cast<int64_t>(kv_len) * head_dim,
        static_cast<int64_t>(kv_heads) * head_dim * padded_k,
        static_cast<int64_t>(head_dim) * padded_k, padded_k,
        static_cast<int64_t>(q_heads) * qo_len * head_dim, static_cast<int64_t>(qo_len) * head_dim,
        head_dim, sm_scale, output_dtype_code, stream);
    check_hip_launch();
}

// Quantize plus attention in one call. Every scratch buffer is allocated by the
// Python layer.
void sage_sdpa(nb::ndarray<> q, nb::ndarray<> k, nb::ndarray<> v, nb::ndarray<> o,
               nb::ndarray<> q_int8, nb::ndarray<> q_scale, nb::ndarray<> k_int8,
               nb::ndarray<> k_scale, nb::ndarray<> v_int8, nb::ndarray<> v_scale,
               nb::ndarray<> anchor_indices, float sm_scale, int cta_k, int input_dtype_code,
               int output_dtype_code, uintptr_t stream_ptr, OptArray attn_mask = std::nullopt) {
    constexpr const char* kFn = "sage_sdpa";
    sage_check_shapes(q, k, v, kFn);
    sage_check_cta_k(cta_k, kFn);
    const auto stream = reinterpret_cast<hipStream_t>(stream_ptr);
    const int batch = static_cast<int>(q.shape(0));
    const int q_heads = static_cast<int>(q.shape(1));
    const int qo_len = static_cast<int>(q.shape(2));
    const int head_dim = static_cast<int>(q.shape(3));
    const int kv_heads = static_cast<int>(k.shape(1));
    const int kv_len = static_cast<int>(k.shape(2));

    sage_quantize(q, k, v, q_int8, q_scale, k_int8, k_scale, v_int8, v_scale, anchor_indices,
                  input_dtype_code, cta_k, stream, kFn);
    check_hip_launch();
    sage_attend(q_int8, k_int8, v_int8, o, q_scale, k_scale, v_scale, attn_mask, batch, q_heads,
                kv_heads, qo_len, kv_len, head_dim, cta_k, sm_scale, output_dtype_code, stream,
                kFn);
}

// Quantization half of the split API. Deliberately the same launches with the
// same tiling as sage_sdpa, so deferring the attention until after the caller
// releases q, k and v changes no numbers.
void sage_sdpa_quantize(nb::ndarray<> q, nb::ndarray<> k, nb::ndarray<> v, nb::ndarray<> q_int8,
                        nb::ndarray<> q_scale, nb::ndarray<> k_int8, nb::ndarray<> k_scale,
                        nb::ndarray<> v_int8, nb::ndarray<> v_scale,
                        nb::ndarray<> anchor_indices, int cta_k, int input_dtype_code,
                        uintptr_t stream_ptr) {
    constexpr const char* kFn = "sage_sdpa_quantize";
    sage_check_shapes(q, k, v, kFn);
    sage_check_cta_k(cta_k, kFn);
    sage_quantize(q, k, v, q_int8, q_scale, k_int8, k_scale, v_int8, v_scale, anchor_indices,
                  input_dtype_code, cta_k, reinterpret_cast<hipStream_t>(stream_ptr), kFn);
    check_hip_launch();
}

// Attention half of the split API, over the packed layouts sage_sdpa_quantize
// produced. No floating-point q, k or v is retained or reconstructed.
void sage_sdpa_prequantized(nb::ndarray<> q_int8, nb::ndarray<> k_int8, nb::ndarray<> v_int8,
                            nb::ndarray<> o, nb::ndarray<> q_scale, nb::ndarray<> k_scale,
                            nb::ndarray<> v_scale, int cta_k, float sm_scale,
                            int output_dtype_code, uintptr_t stream_ptr,
                            OptArray attn_mask = std::nullopt) {
    constexpr const char* kFn = "sage_sdpa_prequantized";
    if (q_int8.ndim() != 4 || k_int8.ndim() != 4 || o.ndim() != 4 || v_int8.ndim() != 2) {
        throw std::runtime_error(std::string(kFn) +
                                 ": q, k and o must be 4D and packed v must be 2D");
    }
    sage_check_cta_k(cta_k, kFn);
    const int batch = static_cast<int>(q_int8.shape(0));
    const int q_heads = static_cast<int>(q_int8.shape(1));
    const int qo_len = static_cast<int>(q_int8.shape(2));
    const int head_dim = static_cast<int>(q_int8.shape(3));
    const int kv_heads = static_cast<int>(k_int8.shape(1));
    const int kv_len = static_cast<int>(k_int8.shape(2));
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
        throw std::runtime_error(std::string(kFn) + ": head_dim must be 64, 128 or 256, got " +
                                 std::to_string(head_dim));
    }
    if (batch == 0 || q_heads == 0 || kv_heads == 0 || qo_len == 0 || kv_len == 0) {
        throw std::runtime_error(
            std::string(kFn) + ": batch, head counts and sequence lengths must be positive");
    }
    if (static_cast<int>(k_int8.shape(0)) != batch ||
        static_cast<int>(k_int8.shape(3)) != head_dim || q_heads % kv_heads != 0) {
        throw std::runtime_error(std::string(kFn) + ": incompatible quantized tensor shapes");
    }
    sage_check_quantized(q_int8, q_scale, k_int8, k_scale, v_int8, v_scale, batch, q_heads,
                         kv_heads, qo_len, kv_len, head_dim, cta_k, kFn);
    // V is packed as [B * H_kv * D, padded_k], and padded_k follows cta_k. Element
    // count alone cannot tell a buffer packed against a different cta_k from a
    // correct one, and the kernel would read shifted rows rather than fail.
    if (v_int8.shape(1) != static_cast<size_t>(sage_padded_k(kv_len, cta_k))) {
        throw std::runtime_error(std::string(kFn) + ": packed v row width " +
                                 std::to_string(v_int8.shape(1)) + " does not match cta_k " +
                                 std::to_string(cta_k));
    }

    sage_attend(q_int8, k_int8, v_int8, o, q_scale, k_scale, v_scale, attn_mask, batch, q_heads,
                kv_heads, qo_len, kv_len, head_dim, cta_k, sm_scale, output_dtype_code,
                reinterpret_cast<hipStream_t>(stream_ptr), kFn);
}


// BF16 decode attention. Every extent below is derived from the operands rather
// than taken from the caller, and the kernel indexes with the strides passed
// here, so a mismatch is an out-of-bounds device access. Mirrors the checks in
// the CUDA binding and adds the two the HIP kernel needs: it reads four
// elements at a time, and it addresses the output with q's strides.
void flash_attention_decode(nb::ndarray<> q, nb::ndarray<> k, nb::ndarray<> v,
                            nb::ndarray<> kv_lengths, nb::ndarray<> output,
                            nb::ndarray<> softmax_lse, nb::ndarray<> softmax_lse_accum,
                            nb::ndarray<> output_accum, int num_splits, uintptr_t stream_ptr) {
    constexpr const char* kFn = "flash_attention_decode";
    constexpr int kHeadDim = 128;
    constexpr int kElemsPerLoad = 4;
    if (q.ndim() != 3 || k.ndim() != 4 || v.ndim() != 4 || output.ndim() != 3 ||
        kv_lengths.ndim() != 1) {
        throw std::runtime_error(std::string(kFn) + ": operand rank mismatch");
    }
    const int batch = static_cast<int>(k.shape(0));
    const int kv_capacity = static_cast<int>(k.shape(1));
    const int heads = static_cast<int>(k.shape(2));
    require_positive(batch, kFn, "batch");
    require_positive(kv_capacity, kFn, "kv_capacity");
    require_positive(heads, kFn, "heads");
    const int query_length = static_cast<int>(q.shape(0)) / batch;
    require_positive(query_length, kFn, "query_length");
    if (static_cast<int64_t>(q.shape(0)) != static_cast<int64_t>(batch) * query_length ||
        static_cast<int>(q.shape(1)) != heads || static_cast<int>(q.shape(2)) != kHeadDim ||
        static_cast<int>(k.shape(3)) != kHeadDim) {
        throw std::runtime_error(std::string(kFn) + ": invalid q/k dimensions");
    }
    if (static_cast<int>(v.shape(0)) != batch || static_cast<int>(v.shape(1)) != kv_capacity ||
        static_cast<int>(v.shape(2)) != heads || static_cast<int>(v.shape(3)) != kHeadDim) {
        throw std::runtime_error(std::string(kFn) + ": k/v shape mismatch");
    }
    if (output.shape(0) != q.shape(0) || static_cast<int>(output.shape(1)) != heads ||
        static_cast<int>(output.shape(2)) != kHeadDim ||
        kv_lengths.size() != static_cast<size_t>(batch)) {
        throw std::runtime_error(std::string(kFn) + ": output or length shape mismatch");
    }
    require_dtype(q, 2, 2, kFn, "q");
    require_dtype(k, 2, 2, kFn, "k");
    require_dtype(v, 2, 2, kFn, "v");
    require_dtype(output, 2, 2, kFn, "output");
    if (map_dtype_to_code(kv_lengths.dtype()) != -1 ||
        kv_lengths.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) ||
        kv_lengths.dtype().bits != 32) {
        throw std::runtime_error(std::string(kFn) + ": kv_lengths must be int32");
    }

    const size_t lse_size = static_cast<size_t>(batch) * heads * query_length;
    if (softmax_lse.size() != lse_size || num_splits < 1 || num_splits > 32 ||
        (num_splits > 1 &&
         (softmax_lse_accum.size() != lse_size * num_splits ||
          output_accum.size() != lse_size * kHeadDim * num_splits))) {
        throw std::runtime_error(std::string(kFn) + ": invalid split workspace");
    }
    require_dtype(softmax_lse, 0, 0, kFn, "softmax_lse");
    // The lengths and the split workspace are indexed linearly off their base
    // pointers, so a strided view of the right size is read as though packed.
    require_packed_contiguous(kv_lengths, kFn, "kv_lengths");
    require_packed_contiguous(softmax_lse, kFn, "softmax_lse");
    if (num_splits > 1) {
        require_dtype(softmax_lse_accum, 0, 0, kFn, "softmax_lse_accum");
        require_dtype(output_accum, 0, 0, kFn, "output_accum");
        require_packed_contiguous(softmax_lse_accum, kFn, "softmax_lse_accum");
        require_packed_contiguous(output_accum, kFn, "output_accum");
    }

    if (k.stride(0) != v.stride(0) || k.stride(1) != v.stride(1) || k.stride(2) != v.stride(2) ||
        k.stride(3) != 1 || v.stride(3) != 1 || q.stride(2) != 1 || output.stride(2) != 1) {
        throw std::runtime_error(std::string(kFn) + ": unsupported tensor strides");
    }
    // The kernel writes the output off q's strides, the way the CUDA launcher
    // does. Here that is checked rather than assumed.
    if (output.stride(0) != q.stride(0) || output.stride(1) != q.stride(1)) {
        throw std::runtime_error(std::string(kFn) + ": output strides must match q");
    }
    // Rows are read four elements at a time, so every row start has to stay on
    // that boundary.
    const int64_t vectored[] = {q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2)};
    for (int64_t stride : vectored) {
        if (stride % kElemsPerLoad != 0) {
            throw std::runtime_error(std::string(kFn) + ": strides must be a multiple of 4");
        }
    }
    // A view carries a byte offset, so operands with aligned strides can still
    // begin off the boundary the four-element load needs.
    constexpr uintptr_t kLoadBytes = kElemsPerLoad * sizeof(uint16_t);
    const void* row_starts[] = {q.data(), k.data(), v.data(), output.data()};
    for (const void* base : row_starts) {
        if (reinterpret_cast<uintptr_t>(base) % kLoadBytes != 0) {
            throw std::runtime_error(std::string(kFn) + ": q, k, v and output must be 8-byte aligned");
        }
    }
    // The launcher takes raw pointers, so anything but ROCm device memory would
    // be dereferenced on the device as though it were. The CUDA binding gets
    // this from its nb::device::cuda operand types; this one takes plain
    // ndarrays and has to ask. Testing for "not kDLCPU" is not enough: pinned
    // host memory reports kDLCUDAHost and is still a host pointer.
    constexpr int kDeviceRocm = nb::device::rocm::value;
    const nb::ndarray<>* operands[] = {&q,      &k,          &v,
                                       &output, &kv_lengths, &softmax_lse};
    for (const nb::ndarray<>* t : operands) {
        if (t->device_type() != kDeviceRocm || t->device_id() != q.device_id()) {
            throw std::runtime_error(std::string(kFn) +
                                     ": every operand must be ROCm device memory on q's device");
        }
    }
    if (num_splits > 1 &&
        (softmax_lse_accum.device_type() != kDeviceRocm ||
         output_accum.device_type() != kDeviceRocm ||
         softmax_lse_accum.device_id() != q.device_id() ||
         output_accum.device_id() != q.device_id())) {
        throw std::runtime_error(std::string(kFn) +
                                 ": split workspace must be ROCm device memory on q's device");
    }

    launch_flash_decode(
        q.data(), k.data(), v.data(), static_cast<const int*>(kv_lengths.data()), output.data(),
        static_cast<float*>(softmax_lse.data()),
        num_splits > 1 ? static_cast<float*>(output_accum.data()) : nullptr,
        num_splits > 1 ? static_cast<float*>(softmax_lse_accum.data()) : nullptr, batch,
        query_length, heads, kv_capacity, num_splits, q.stride(0) * query_length, q.stride(0),
        q.stride(1), k.stride(0), k.stride(1), k.stride(2),
        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}


// ---------------------------------------------------------------------------
// Sol-Attn sparse attention
// ---------------------------------------------------------------------------
//
// _C is importable, so none of this can assume the Python layer put it together.
// The kernels reach q/k/v and the workspace as bare pointers sized from (B, T, H),
// so a short buffer, or a layout the 16-byte staging loads cannot use, is an
// out-of-bounds device access rather than an exception.

// The kernels read these as packed arrays of one dtype, so an element count on
// its own is not enough: a narrower dtype makes the buffer shorter in bytes than
// the kernel reads, and a strided view of the right count is accessed as though
// it were packed. `code` is a DTYPE_TO_CODE value.
static void sol_need_elems(const nb::ndarray<>& a, int64_t n, int code, const char* fn,
                           const char* what) {
    if (static_cast<int64_t>(a.size()) != n) {
        throw std::runtime_error(std::string(fn) + ": " + what + " must have " +
                                 std::to_string(n) + " elements, got " + std::to_string(a.size()));
    }
    if (map_dtype_to_code(a.dtype()) != code) {
        throw std::runtime_error(std::string(fn) + ": " + what + " has an unsupported dtype");
    }
    int64_t expect = 1;
    for (int i = static_cast<int>(a.ndim()) - 1; i >= 0; --i) {
        if (a.shape(i) > 1 && a.stride(i) != expect) {
            throw std::runtime_error(std::string(fn) + ": " + what + " must be contiguous");
        }
        expect *= static_cast<int64_t>(a.shape(i));
    }
}

// Extents size every workspace slot and every grid, and they reach the kernels as
// plain ints. A non-positive one plans a workspace nothing checks again.
static void sol_need_extents(int64_t batch, int64_t seq_len, int64_t num_heads, const char* fn) {
    if (batch <= 0 || seq_len <= 0 || num_heads <= 0) {
        throw std::runtime_error(std::string(fn) +
                                 ": batch, seq_len and num_heads must all be positive");
    }
}

// Both sinks are half-open [start, end) block ranges, mirroring
// sol_attn_common_call_rule on the Python side. A negative start would make the
// routing kernel emit that many extra list entries, wrapped to huge uint16 block
// ids, and overrun the per-query row.
static void sol_need_sinks(int64_t start, int64_t end, const char* fn, const char* what) {
    if (start < 0 || end < 0 || end < start) {
        throw std::runtime_error(std::string(fn) + ": " + what +
                                 " must be a [start, end) range with 0 <= start <= end");
    }
}

static void sol_check_block_len(const nb::ndarray<>& b, int64_t seq_len, const char* fn) {
    if (b.dtype().code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) ||
        b.dtype().bits != 32 || b.ndim() != 1 || b.stride(0) != 1 ||
        static_cast<int64_t>(b.size()) != (seq_len + 63) / 64) {
        throw std::runtime_error(std::string(fn) +
                                 ": block_len must be a contiguous 1-D int32 array of "
                                 "ceil(T/64) elements");
    }
}

static void sol_need_workspace(const nb::ndarray<>& ws, int64_t batch, int64_t seq_len,
                               int64_t num_heads, const char* fn) {
    int64_t v[32];
    const int n = sol_attn_plan(static_cast<int>(batch), static_cast<int>(seq_len),
                                static_cast<int>(num_heads), v, 32);
    if (n > 32 || static_cast<int64_t>(ws.size()) < v[n - 1]) {  // last slot is "total"
        throw std::runtime_error(std::string(fn) + ": workspace too small for this shape");
    }
    // sol_attn_plan reports byte offsets and the Python layer slices the workspace
    // with them to read the block means back out, so a workspace of anything but
    // bytes reads the wrong slots. The size check above already bounds the
    // allocation, since no element is narrower than a byte; this pins the units.
    if (map_dtype_to_code(ws.dtype()) != 3) {
        throw std::runtime_error(std::string(fn) + ": workspace must be a uint8 array");
    }
}

// q/k/v/out element code for launch_sol_attn: 0 = bfloat16, 1 = float16, -1 = neither
static int sol_elem_code(const nb::ndarray<>& a) {
    const int code = map_dtype_to_code(a.dtype());
    return code == 2 ? 0 : code == 1 ? 1 : -1;
}
static void sol_need_bthd(const nb::ndarray<>& a, int64_t b, int64_t t, int64_t h, int64_t d,
                          int elem, const char* fn, const char* what) {
    if (a.ndim() != 4 || sol_elem_code(a) != elem ||
        a.shape(0) != static_cast<size_t>(b) || a.shape(1) != static_cast<size_t>(t) ||
        a.shape(2) != static_cast<size_t>(h) || a.shape(3) != static_cast<size_t>(d)) {
        throw std::runtime_error(std::string(fn) + ": " + what +
                                 " must be a (B, T, H, D) array of q's dtype (bfloat16 or float16)");
    }
}

// The kernels stage rows with 16-byte loads: unit last stride, 16-byte base, and
// leading strides (of non-singleton dims) that keep every row 16-byte aligned.
static void sol_need_staging_layout(const nb::ndarray<>& a, const char* fn, const char* what) {
    bool ok = a.stride(3) == 1 && reinterpret_cast<uintptr_t>(a.data()) % 16 == 0;
    for (int i = 0; i < 3; ++i) ok = ok && (a.shape(i) <= 1 || a.stride(i) % 8 == 0);
    if (!ok) {
        throw std::runtime_error(
            std::string(fn) + ": " + what +
            " must have a contiguous last dim, a 16-byte aligned base and leading strides that "
            "are multiples of 8");
    }
}

static void sol_need_contiguous(const nb::ndarray<>& a, const char* fn, const char* what) {
    int64_t expect = 1;
    for (int i = static_cast<int>(a.ndim()) - 1; i >= 0; --i) {
        if (a.shape(i) > 1 && a.stride(i) != expect) {
            throw std::runtime_error(std::string(fn) + ": " + what + " must be contiguous");
        }
        expect *= static_cast<int64_t>(a.shape(i));
    }
}

// Workspace dims and slot byte offsets, from the C++ Plan (the one definition).
nb::dict sol_attn_plan_py(int64_t batch, int64_t seq_len, int64_t num_heads) {
    int64_t v[32];
    const int n = sol_attn_plan(static_cast<int>(batch), static_cast<int>(seq_len),
                                static_cast<int>(num_heads), v, 32);
    if (n > 32) throw std::runtime_error("sol_attn_plan: Plan grew past the binding's buffer");
    nb::dict d;
    for (int i = 0; i < n && sol_attn_plan_names[i]; ++i) d[sol_attn_plan_names[i]] = v[i];
    return d;
}

void sol_attn(nb::ndarray<> q, nb::ndarray<> k, nb::ndarray<> v, nb::ndarray<> out,
              nb::ndarray<> workspace, int64_t batch, int64_t seq_len, int64_t num_heads,
              int64_t head_dim, float tau, float scale, int64_t sink_start, int64_t sink_end,
              int64_t sink_q_start, int64_t sink_q_end, uintptr_t stream_ptr,
              OptArray key_bias = std::nullopt, OptArray threshold = std::nullopt,
              OptArray block_len = std::nullopt, bool tail = true) {
    constexpr const char* kFn = "sol_attn";
    auto stream = reinterpret_cast<hipStream_t>(stream_ptr);
    sol_need_extents(batch, seq_len, num_heads, kFn);
    sol_need_sinks(sink_start, sink_end, kFn, "sink_blocks");
    sol_need_sinks(sink_q_start, sink_q_end, kFn, "sink_q");
    if (threshold) {
        sol_need_elems(*threshold, batch * num_heads * ((seq_len + 63) / 64), 0, kFn, "threshold");
    }
    if (block_len) sol_check_block_len(*block_len, seq_len, kFn);
    const int elem = sol_elem_code(q);
    if (elem < 0) throw std::runtime_error(std::string(kFn) + ": q must be bfloat16 or float16");
    sol_need_bthd(q, batch, seq_len, num_heads, head_dim, elem, kFn, "q");
    sol_need_bthd(k, batch, seq_len, num_heads, head_dim, elem, kFn, "k");
    sol_need_bthd(v, batch, seq_len, num_heads, head_dim, elem, kFn, "v");
    sol_need_bthd(out, batch, seq_len, num_heads, head_dim, elem, kFn, "out");
    sol_need_staging_layout(q, kFn, "q");
    sol_need_staging_layout(k, kFn, "k");
    sol_need_staging_layout(v, kFn, "v");
    sol_need_contiguous(out, kFn, "out");
    sol_need_workspace(workspace, batch, seq_len, num_heads, kFn);
    if (key_bias) sol_need_elems(*key_bias, batch * seq_len, 0, kFn, "key_bias");
    // Explicit strides: only the last dim must be contiguous (BHND views go in as-is).
    launch_sol_attn(q.data(), k.data(), v.data(), out.data(), workspace.data(),
                    static_cast<int>(batch), static_cast<int>(seq_len),
                    static_cast<int>(num_heads), static_cast<int>(head_dim), elem, tau, scale,
                    opt_data(key_bias), opt_data(threshold), opt_data(block_len), tail ? 1 : 0,
                    static_cast<int>(sink_start), static_cast<int>(sink_end),
                    static_cast<int>(sink_q_start), static_cast<int>(sink_q_end), q.stride(0),
                    q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0),
                    v.stride(1), v.stride(2), stream);
    check_hip_launch();
}

void sol_producer_begin_py(nb::ndarray<> workspace, int64_t batch, int64_t seq_len,
                           int64_t num_heads, uintptr_t stream_ptr) {
    sol_need_extents(batch, seq_len, num_heads, "sol_producer_begin");
    sol_need_workspace(workspace, batch, seq_len, num_heads, "sol_producer_begin");
    sol_producer_begin(workspace.data(), static_cast<int>(batch), static_cast<int>(seq_len),
                       static_cast<int>(num_heads), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void sol_producer_chunk_py(nb::ndarray<> workspace, nb::ndarray<> qkv, nb::ndarray<> fab,
                           nb::ndarray<> qw, nb::ndarray<> kw, nb::ndarray<> kmean,
                           nb::ndarray<> vscale, float rope_eps, int64_t rot_dim, int64_t t0,
                           int64_t m, int64_t batch, int64_t seq_len, int64_t num_heads,
                           uintptr_t stream_ptr, OptArray block_len = std::nullopt) {
    constexpr const char* kFn = "sol_producer_chunk";
    sol_need_extents(batch, seq_len, num_heads, kFn);
    if (batch != 1) throw std::runtime_error(std::string(kFn) + ": the producer path is B=1 only");
    if (rot_dim <= 0 || rot_dim > 128 || rot_dim % 8) {
        throw std::runtime_error(std::string(kFn) +
                                 ": rot_dim must be a multiple of 8 in (0, 128]");
    }
    if (t0 < 0 || m < 0 || t0 + m > seq_len || (m && t0 % 64)) {
        throw std::runtime_error(
            std::string(kFn) +
            ": chunk [t0, t0 + m) must lie in [0, seq_len] with a 64-aligned start");
    }
    if (block_len) sol_check_block_len(*block_len, seq_len, kFn);
    sol_need_workspace(workspace, batch, seq_len, num_heads, kFn);
    sol_need_elems(qkv, m * 3 * num_heads * 128, 2, kFn, "qkv");
    sol_need_elems(fab, seq_len * rot_dim * 2, 0, kFn, "fab");
    sol_need_elems(qw, 128, 2, kFn, "qw");
    sol_need_elems(kw, 128, 2, kFn, "kw");
    sol_need_elems(kmean, batch * num_heads * 128, 0, kFn, "kmean");
    sol_need_elems(vscale, batch * num_heads * 128, 0, kFn, "vscale");
    sol_producer_chunk(workspace.data(), qkv.data(), fab.data(), qw.data(), kw.data(),
                       kmean.data(), vscale.data(), opt_data(block_len), rope_eps,
                       static_cast<int>(rot_dim), static_cast<int>(t0), static_cast<int>(m),
                       static_cast<int>(batch), static_cast<int>(seq_len),
                       static_cast<int>(num_heads), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void sol_attn_core_py(nb::ndarray<> workspace, nb::ndarray<> out, nb::ndarray<> vscale,
                      nb::ndarray<> kmean_next, nb::ndarray<> vamax_out, int64_t batch,
                      int64_t seq_len, int64_t num_heads, float tau, float scale,
                      int64_t sink_start, int64_t sink_end, int64_t sink_q_start,
                      int64_t sink_q_end, uintptr_t stream_ptr, OptArray threshold = std::nullopt,
                      OptArray block_len = std::nullopt, bool tail = true) {
    constexpr const char* kFn = "sol_attn_core";
    sol_need_extents(batch, seq_len, num_heads, kFn);
    sol_need_sinks(sink_start, sink_end, kFn, "sink_blocks");
    sol_need_sinks(sink_q_start, sink_q_end, kFn, "sink_q");
    const int64_t stats = batch * num_heads * 128;
    sol_need_elems(vscale, stats, 0, kFn, "vscale");
    sol_need_elems(kmean_next, stats, 0, kFn, "kmean_next");
    sol_need_elems(vamax_out, stats, 0, kFn, "vamax_out");
    if (threshold) {
        sol_need_elems(*threshold, batch * num_heads * ((seq_len + 63) / 64), 0, kFn, "threshold");
    }
    if (block_len) sol_check_block_len(*block_len, seq_len, kFn);
    sol_need_workspace(workspace, batch, seq_len, num_heads, kFn);
    sol_need_elems(out, batch * seq_len * num_heads * 128, 2, kFn, "out");
    launch_sol_attn_core(workspace.data(), out.data(), vscale.data(), kmean_next.data(),
                         vamax_out.data(), opt_data(block_len), tail ? 1 : 0,
                         static_cast<int>(batch), static_cast<int>(seq_len),
                         static_cast<int>(num_heads), tau, scale, opt_data(threshold),
                         static_cast<int>(sink_start), static_cast<int>(sink_end),
                         static_cast<int>(sink_q_start), static_cast<int>(sink_q_end),
                         reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

NB_MODULE(_C, m) {
    m.doc() = "ComfyKitchen HIP backend native operations (RDNA2-RDNA4, WMMA on gfx11/gfx12)";
    m.def("sol_attn_plan", &sol_attn_plan_py,
          "Workspace dims, slot byte offsets and total bytes for this shape",
          nb::arg("batch"), nb::arg("seq_len"), nb::arg("num_heads"));
    m.def("sol_attn", &sol_attn,
          "Sol-Attn training-free sparse attention (BF16 or FP16 in/out, head_dim 128)",
          nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("out"), nb::arg("workspace"),
          nb::arg("batch"), nb::arg("seq_len"), nb::arg("num_heads"), nb::arg("head_dim"),
          nb::arg("tau"), nb::arg("scale"), nb::arg("sink_start"), nb::arg("sink_end"),
          nb::arg("sink_q_start"), nb::arg("sink_q_end"), nb::arg("stream_ptr"),
          nb::arg("key_bias") = nb::none(), nb::arg("threshold") = nb::none(),
          nb::arg("block_len") = nb::none(), nb::arg("tail") = true);
    m.def("sol_producer_begin", &sol_producer_begin_py,
          nb::arg("workspace"), nb::arg("batch"), nb::arg("seq_len"), nb::arg("num_heads"),
          nb::arg("stream_ptr"));
    m.def("sol_producer_chunk", &sol_producer_chunk_py,
          nb::arg("workspace"), nb::arg("qkv"), nb::arg("fab"), nb::arg("qw"), nb::arg("kw"),
          nb::arg("kmean"), nb::arg("vscale"), nb::arg("rope_eps"), nb::arg("rot_dim"),
          nb::arg("t0"), nb::arg("m"), nb::arg("batch"), nb::arg("seq_len"), nb::arg("num_heads"),
          nb::arg("stream_ptr"), nb::arg("block_len") = nb::none());
    m.def("sol_attn_core", &sol_attn_core_py,
          nb::arg("workspace"), nb::arg("out"), nb::arg("vscale"), nb::arg("kmean_next"),
          nb::arg("vamax_out"), nb::arg("batch"), nb::arg("seq_len"), nb::arg("num_heads"),
          nb::arg("tau"), nb::arg("scale"), nb::arg("sink_start"), nb::arg("sink_end"),
          nb::arg("sink_q_start"), nb::arg("sink_q_end"), nb::arg("stream_ptr"),
          nb::arg("threshold") = nb::none(), nb::arg("block_len") = nb::none(),
          nb::arg("tail") = true);
    m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8);
    m.def("dequantize_per_tensor_fp8", &dequantize_per_tensor_fp8);
    m.def("stochastic_round_fp8", &stochastic_round_fp8);
    m.def("scaled_mm_fp8", &scaled_mm_fp8);
    m.def("int8_gemm", &int8_gemm);
    m.def("convrot_w4a4_gemm", &convrot_w4a4_gemm);
    m.def("quantize_int8_rowwise", &quantize_int8_rowwise);
    m.def("quantize_int8_convrot", &quantize_int8_convrot);
    m.def("quantize_int8_tensorwise", &quantize_int8_tensorwise);
    m.def("dequantize_int8_simple", &dequantize_int8_simple);
    m.def("dequantize_int8_convrot_weight", &dequantize_int8_convrot_weight);
    m.def("convrot_quant_int4", &convrot_quant_int4);
    m.def("convrot_max_k", &convrot_max_k_host);
    m.def("convrot_int8_needs_spill", &convrot_int8_needs_spill_host, nb::arg("m"),
          nb::arg("k"), nb::arg("in_code"));
    m.def("unpack_int4", &unpack_int4);
    m.def("dequant_int4_grouped_to_int8", &dequant_int4_grouped_to_int8);
    m.def("quantize_w4a8_convrot", &quantize_w4a8_convrot);
    m.def("w4a8_requant_max_k", &w4a8_requant_max_k_kernel);
    m.def("w4a8_int8_gemm_chunked", &w4a8_int8_gemm_chunked);
    m.def("na3d", &na3d);
    m.def("flash_attention_decode", &flash_attention_decode, nb::arg("q"), nb::arg("k"),
          nb::arg("v"), nb::arg("kv_lengths"), nb::arg("output"), nb::arg("softmax_lse"),
          nb::arg("softmax_lse_accum"), nb::arg("output_accum"), nb::arg("num_splits"),
          nb::arg("stream_ptr"));
    m.def("sage_sdpa", &sage_sdpa, nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("o"),
          nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"), nb::arg("k_scale"),
          nb::arg("v_int8"), nb::arg("v_scale"), nb::arg("anchor_indices"), nb::arg("sm_scale"),
          nb::arg("cta_k"), nb::arg("input_dtype_code"), nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"), nb::arg("attn_mask") = nb::none());
    m.def("sage_sdpa_quantize", &sage_sdpa_quantize, nb::arg("q"), nb::arg("k"), nb::arg("v"),
          nb::arg("q_int8"), nb::arg("q_scale"), nb::arg("k_int8"), nb::arg("k_scale"),
          nb::arg("v_int8"), nb::arg("v_scale"), nb::arg("anchor_indices"), nb::arg("cta_k"),
          nb::arg("input_dtype_code"), nb::arg("stream_ptr"));
    m.def("sage_sdpa_prequantized", &sage_sdpa_prequantized, nb::arg("q_int8"), nb::arg("k_int8"),
          nb::arg("v_int8"), nb::arg("o"), nb::arg("q_scale"), nb::arg("k_scale"),
          nb::arg("v_scale"), nb::arg("cta_k"), nb::arg("sm_scale"),
          nb::arg("output_dtype_code"), nb::arg("stream_ptr"), nb::arg("attn_mask") = nb::none());
    m.def("adaln", &adaln);
    m.def("rms_adaln", &rms_adaln);
    m.def("apply_rope", &apply_rope);
    m.def("rms_rope", &rms_rope);
    m.def("gemv_awq_w4a16", &gemv_awq_w4a16);
    m.def("svdquant_lora_down", &svdquant_lora_down);
    m.def("svdquant_quantize", &svdquant_quantize);
    m.def("svdquant_gemm", &svdquant_gemm);
}
