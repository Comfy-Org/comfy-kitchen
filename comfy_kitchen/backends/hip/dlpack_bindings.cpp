// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

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
                            scale_b_stride, opt_data(bias), opt_code(bias), M, N, K, out_code,
                            reinterpret_cast<hipStream_t>(stream_ptr));
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

void quantize_int8_convrot(nb::ndarray<> x, nb::ndarray<> q, nb::ndarray<> scales, int M, int K,
                           int group_size, uintptr_t stream_ptr) {
    constexpr const char* kFn = "quantize_int8_convrot";
    require_nonneg(M, kFn, "M");
    require_convrot_group(K, group_size, kFn);
    require_dtype(x, 0, 2, kFn, "x");
    require_dtype(q, 4, 4, kFn, "q");
    require_len(x, static_cast<int64_t>(M) * K, kFn, "x");
    require_len(q, static_cast<int64_t>(M) * K, kFn, "q");
    require_scale_len(scales, static_cast<size_t>(M), kFn, "scales");

    launch_quantize_int8_convrot_kernel(x.data(), map_dtype_to_code(x.dtype()), q.data(),
                                        scales.data(), M, K, group_size,
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

void adaln(nb::ndarray<> x, nb::ndarray<> scale, nb::ndarray<> shift, nb::ndarray<> out, int N,
           int D, int scale_group, int shift_group, float eps, uintptr_t stream_ptr) {
    constexpr const char* kFn = "adaln";
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
                        reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

// xq is (batch, dim1, dim2, head_dim); freqs is (fb, fd1, fd2, head_dim/2, 2, 2).
// Shapes and strides are read off the arrays so the broadcast rules stay in one
// place rather than being recomputed on the Python side.
void apply_rope(nb::ndarray<> xq, OptArray xk, nb::ndarray<> freqs, nb::ndarray<> xq_out,
                OptArray xk_out, bool split_half, uintptr_t stream_ptr) {
    constexpr const char* kFn = "apply_rope";
    // map_dtype_to_code returns -1 for anything else, which the device decoder
    // would misread; xk and the outputs are then checked against xq's dtype below.
    require_dtype(xq, 0, 2, kFn, "xq");
    require_dtype(freqs, 0, 2, kFn, "freqs_cis");
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
