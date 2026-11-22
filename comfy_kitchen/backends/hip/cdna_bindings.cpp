// SPDX-License-Identifier: Apache-2.0
// INT8 linear for wave64 CDNA, see ops/gemm_int8_mfma.hip. Linked into _cdna only.
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include "binding_checks.h"

extern "C" {
void launch_quantize_int8_rowwise_bf16_exact_kernel(const void* x, void* q, void* scales, int M, int K,
                                                    hipStream_t stream);
void launch_int8_gemm_mfma_kernel(const void* a, const void* b, void* c, const void* scale_a, const void* scale_b,
                                  int scale_b_stride, const void* bias, int M, int N, int K, hipStream_t stream);
}

using rocm_array = nb::ndarray<nb::device::rocm>;

static constexpr auto kInt = nb::dlpack::dtype_code::Int;
static constexpr auto kFloat = nb::dlpack::dtype_code::Float;
static constexpr auto kBfloat = nb::dlpack::dtype_code::Bfloat;

static void require_buf(const rocm_array& a, nb::dlpack::dtype_code code, int bits, int64_t need, const char* fn,
                        const char* name) {
    const nb::ndarray<> t(a);
    if (t.dtype().code != static_cast<uint8_t>(code) || t.dtype().bits != bits) {
        throw std::runtime_error(std::string(fn) + ": " + name + " has the wrong dtype, expected " +
                                 std::to_string(bits) + "-bit");
    }
    require_len(t, need, fn, name);
    if (t.size() != 0) require_packed_contiguous(t, fn, name);
}

// The GEMM reads A and B rows 16 bytes at a time.
static void require_16b_aligned(const rocm_array& a, const char* fn, const char* name) {
    if (reinterpret_cast<uintptr_t>(a.data()) % 16 != 0) {
        throw std::runtime_error(std::string(fn) + ": " + name + " must be 16-byte aligned");
    }
}

// x [M, K] bf16 is quantized row-wise into the workspace (M * K int8 bytes, padded to 16, then M float32
// row scales), then multiplied by w [N, K] int8 into out [M, N] bf16. One binding call keeps the host
// cost per linear down; the denoisers call this thousands of times per step.
void int8_linear_mfma(rocm_array x, rocm_array w, rocm_array scale_b, int scale_b_stride,
                      std::optional<rocm_array> bias, rocm_array workspace, rocm_array out, int M, int N, int K,
                      uintptr_t stream_ptr) {
    constexpr const char* kFn = "int8_linear_mfma";
    if (M < 0 || N < 0 || K < 0) throw std::runtime_error(std::string(kFn) + ": M, N and K must be non-negative");
    if (scale_b_stride != 0 && scale_b_stride != 1) {
        throw std::runtime_error(std::string(kFn) + ": scale_b_stride must be 0 or 1, got " +
                                 std::to_string(scale_b_stride));
    }
    const int64_t mk = static_cast<int64_t>(M) * K, q_bytes = (mk + 15) / 16 * 16;
    require_buf(x, kBfloat, 16, mk, kFn, "x");
    require_buf(w, kInt, 8, static_cast<int64_t>(N) * K, kFn, "w");
    require_buf(scale_b, kFloat, 32, scale_b_stride ? N : 1, kFn, "scale_b");
    if (bias) require_buf(*bias, kBfloat, 16, N, kFn, "bias");
    require_buf(workspace, kInt, 8, q_bytes + 4 * static_cast<int64_t>(M), kFn, "workspace");
    require_buf(out, kBfloat, 16, static_cast<int64_t>(M) * N, kFn, "out");
    require_16b_aligned(w, kFn, "w");
    require_16b_aligned(workspace, kFn, "workspace");

    auto stream = reinterpret_cast<hipStream_t>(stream_ptr);
    auto* q = static_cast<int8_t*>(workspace.data());
    void* x_scale = q + q_bytes;
    launch_quantize_int8_rowwise_bf16_exact_kernel(x.data(), q, x_scale, M, K, stream);
    check_hip_launch();
    launch_int8_gemm_mfma_kernel(q, w.data(), out.data(), x_scale, scale_b.data(), scale_b_stride,
                                 bias ? bias->data() : nullptr, M, N, K, stream);
    check_hip_launch();
}

void register_cdna_int8(nb::module_& m) {
    m.def("int8_linear_mfma", &int8_linear_mfma,
          "Row-wise INT8 quantization and INT8 GEMM with the eager int8_linear rounding, bf16 in and out",
          nb::arg("x"), nb::arg("w"), nb::arg("scale_b"), nb::arg("scale_b_stride"), nb::arg("bias").none(),
          nb::arg("workspace"), nb::arg("out"), nb::arg("M"), nb::arg("N"), nb::arg("K"), nb::arg("stream_ptr"));
}
