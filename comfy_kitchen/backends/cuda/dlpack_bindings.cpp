/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace nb = nanobind;

// Helper: Map nanobind dtype to internal dtype code
// Returns: 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8, 5=float8_e4m3fn, 6=float8_e5m2
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
    if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Float) {
        if (dtype.bits == 32) return 0;  // float32
        if (dtype.bits == 16) return 1;  // float16
        if (dtype.bits == 8) return 5;   // float8_e4m3fn (default)
    } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Bfloat && dtype.bits == 16) {
        return 2;  // bfloat16
    } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::UInt && dtype.bits == 8) {
        return 3;  // uint8
    } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Int && dtype.bits == 8) {
        return 4;  // int8
    }
    return -1;  // unsupported
}

// Forward declarations of CUDA kernel wrappers
extern "C" {
    void launch_quantize_fp8_kernel(const void* input, void* output, 
                                    const void* scale, int64_t numel,
                                    int input_dtype_code, int output_dtype_code,
                                    cudaStream_t stream);
    
    void launch_dequantize_fp8_kernel(const void* input, void* output,
                                      const void* scale, int64_t numel,
                                      int input_dtype_code, int output_dtype_code,
                                      cudaStream_t stream);

    void launch_cublas_gemm_blockwise_fp4_kernel(
        const void* B_ptr,
        const void* B_decode_scale_ptr,
        const void* A_ptr,
        const void* A_decode_scale_ptr,
        void* D_ptr,
        const void* bias_ptr,
        int64_t M,
        int64_t N,
        int64_t K,
        const float* alpha_device_ptr,
        int out_dtype_code,
        void* workspace_ptr,
        bool accumulate,
        cudaStream_t stream);

    void launch_apply_rope_kernel(
        const void* x,
        const void* freqs,
        void* x_out,
        int64_t batch,
        int64_t n_heads,
        int64_t seq_len,
        int64_t head_dim,
        int64_t freqs_batch,
        int64_t freqs_heads,
        int64_t stride_x_batch,
        int64_t stride_x_heads,
        int64_t stride_x_seq,
        int64_t stride_x_dim,
        int64_t stride_freqs_batch,
        int64_t stride_freqs_heads,
        int64_t stride_freqs_seq,
        int64_t stride_freqs_dim,
        int64_t stride_freqs_rot,
        int64_t stride_freqs_pair,
        int input_dtype_code,
        cudaStream_t stream);

    void launch_quantize_nvfp4_kernel(
        const void* input,
        const void* global_scale,
        void* output,
        void* block_scales,
        int64_t num_rows,
        int64_t num_cols,
        int64_t orig_rows,
        int64_t orig_cols,
        float epsilon,
        int input_dtype_code,
        cudaStream_t stream);

    void launch_dequantize_nvfp4_kernel(
        const void* input,
        const void* global_scale,
        const void* block_scales,
        void* output,
        int64_t num_rows,
        int64_t num_cols,
        int output_dtype_code,
        cudaStream_t stream);
}

// Nanobind wrapper for quantize_per_tensor_fp8
void quantize_per_tensor_fp8(
    nb::ndarray<nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> scale,
    nb::ndarray<nb::device::cuda> output,
    int input_dtype_code,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {
    
    // Validate input dtype code (0=float32, 1=float16, 2=bfloat16)
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for quantize_per_tensor_fp8");
    }
    
    // Validate output dtype code (5=e4m3fn, 6=e5m2)
    if (output_dtype_code < 5 || output_dtype_code > 6) {
        throw std::runtime_error("Unsupported output dtype for quantize_per_tensor_fp8");
    }
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_fp8_kernel(input.data(), output.data(), scale.data(), 
                              numel, input_dtype_code, output_dtype_code, stream);
}

// Nanobind wrapper for dequantize_per_tensor_fp8
void dequantize_per_tensor_fp8(
    nb::ndarray<nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> scale,
    nb::ndarray<nb::device::cuda> output,
    int input_dtype_code,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {
    
    // Validate input dtype code (5=float8_e4m3fn, 6=float8_e5m2)
    if (input_dtype_code != 5 && input_dtype_code != 6) {
        throw std::runtime_error("Unsupported input dtype code for dequantize_per_tensor_fp8 (must be 5 or 6)");
    }
    
    // Validate output dtype code (0=float32, 1=float16, 2=bfloat16)
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for dequantize_per_tensor_fp8 (must be float32, float16, or bfloat16)");
    }
    
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_fp8_kernel(input.data(), output.data(), scale.data(),
                                 numel, input_dtype_code, output_dtype_code, stream);
}

// Nanobind wrapper for cublas_gemm_blockwise_fp4
void cublas_gemm_blockwise_fp4(
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> b,
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> block_scale_b,
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> a,
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> block_scale_a,
    nb::ndarray<nb::device::cuda> out,
    int out_dtype_code,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::device::cuda> workspace,
    bool accumulate,
    nb::ndarray<float, nb::device::cuda> alpha,
    uintptr_t stream_ptr) {

    // Get dimensions: B is (N, K_b), A is (M, K_a) in packed format
    int64_t N = b.shape(0);
    int64_t K_b = b.shape(1);
    int64_t M = a.shape(0);
    int64_t K_a = a.shape(1);

    if (K_a != K_b) {
        throw std::runtime_error("Matrix dimensions do not match");
    }

    // K is the number of FP4 elements (2 per uint8)
    int64_t K = 2 * K_a;

    // Validate output dtype code (0=float32, 1=float16, 2=bfloat16)
    if (out_dtype_code < 0 || out_dtype_code > 2) {
        throw std::runtime_error("Invalid output dtype code");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Handle optional bias (check if pointer is null or size is 0)
    const void* bias_ptr = (bias.data() && bias.size() > 0) ? bias.data() : nullptr;

    // Call the kernel
    launch_cublas_gemm_blockwise_fp4_kernel(
        b.data(),
        block_scale_b.data(),
        a.data(),
        block_scale_a.data(),
        out.data(),
        bias_ptr,
        M,
        N,
        K,
        static_cast<const float*>(alpha.data()),
        out_dtype_code,
        workspace.data(),
        accumulate,
        stream);
}

// Nanobind wrapper for quantize_nvfp4
void quantize_nvfp4(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> global_scale,
    nb::ndarray<nb::device::cuda> output,
    nb::ndarray<nb::device::cuda> block_scales,
    float epsilon,
    bool pad_16x,
    uintptr_t stream_ptr) {

    // Get input dimensions (orig_rows, orig_cols)
    int64_t orig_rows = input.shape(0);
    int64_t orig_cols = input.shape(1);

    // Calculate effective padded dimensions
    int64_t num_rows = orig_rows;
    int64_t num_cols = orig_cols;
    
    if (pad_16x) {
        // Round up to nearest multiple of 16
        num_rows = (orig_rows + 15) / 16 * 16;
        num_cols = (orig_cols + 15) / 16 * 16;
    }

    // Get input dtype code
    int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for FP4 quantization (must be float32, float16, or bfloat16)");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_nvfp4_kernel(
        input.data(),
        global_scale.data(),
        output.data(),
        block_scales.data(),
        num_rows,
        num_cols,
        orig_rows,
        orig_cols,
        epsilon,
        input_dtype_code,
        stream);
}

// Nanobind wrapper for dequantize_nvfp4
void dequantize_nvfp4(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> global_scale,
    nb::ndarray<nb::device::cuda> block_scales,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int output_dtype_code,
    uintptr_t stream_ptr) {

    // Get output dimensions (should match input logical dimensions)
    int64_t num_rows = output.shape(0);
    int64_t num_cols = output.shape(1);

    // Validate output dtype code (0=float32, 1=float16, 2=bfloat16)
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for FP4 dequantization (must be float32, float16, or bfloat16)");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_nvfp4_kernel(
        input.data(),
        global_scale.data(),
        block_scales.data(),
        output.data(),
        num_rows,
        num_cols,
        output_dtype_code,
        stream);
}

// Nanobind wrapper for apply_rope
void apply_rope(
    nb::ndarray<nb::device::cuda> x,
    nb::ndarray<nb::device::cuda> freqs,
    nb::ndarray<nb::device::cuda> x_out,
    uintptr_t stream_ptr) {

    // Get x dimensions: (batch, n_heads, seq_len, head_dim)
    int64_t batch = x.shape(0);
    int64_t n_heads = x.shape(1);
    int64_t seq_len = x.shape(2);
    int64_t head_dim = x.shape(3);

    // Get freqs dimensions
    int64_t freqs_batch = freqs.shape(0);
    int64_t freqs_heads = freqs.shape(1);

    // Validate dimensions match
    if (freqs.shape(2) != seq_len) {
        throw std::runtime_error("Freqs seq_len dimension must match input seq_len");
    }
    if (freqs.shape(3) != head_dim / 2) {
        throw std::runtime_error("Freqs dimension 3 must be head_dim//2");
    }

    // Validate output shape matches input
    if (x_out.ndim() != 4 ||
        x_out.shape(0) != batch || x_out.shape(1) != n_heads ||
        x_out.shape(2) != seq_len || x_out.shape(3) != head_dim) {
        throw std::runtime_error("Output shape must match input shape");
    }

    // Get input dtype code
    int input_dtype_code = map_dtype_to_code(x.dtype());
    if (input_dtype_code < 0) {
        throw std::runtime_error("Unsupported input dtype for apply_rope");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Get strides (nanobind provides strides in elements, not bytes)
    int64_t stride_x_batch = x.stride(0);
    int64_t stride_x_heads = x.stride(1);
    int64_t stride_x_seq = x.stride(2);
    int64_t stride_x_dim = x.stride(3);

    int64_t stride_freqs_batch = freqs.stride(0);
    int64_t stride_freqs_heads = freqs.stride(1);
    int64_t stride_freqs_seq = freqs.stride(2);
    int64_t stride_freqs_dim = freqs.stride(3);
    int64_t stride_freqs_rot = freqs.stride(4);
    int64_t stride_freqs_pair = freqs.stride(5);

    // Launch kernel
    launch_apply_rope_kernel(
        x.data(),
        freqs.data(),
        x_out.data(),
        batch,
        n_heads,
        seq_len,
        head_dim,
        freqs_batch,
        freqs_heads,
        stride_x_batch,
        stride_x_heads,
        stride_x_seq,
        stride_x_dim,
        stride_freqs_batch,
        stride_freqs_heads,
        stride_freqs_seq,
        stride_freqs_dim,
        stride_freqs_rot,
        stride_freqs_pair,
        input_dtype_code,
        stream
    );
}

// Python module definition
NB_MODULE(_C, m) {
    m.doc() = "comfy_kitchen CUDA kernels - nanobind + DLPack interface (NO PyTorch C++ dependencies)";
    
    m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8,
          "Quantize to FP8 using nanobind ndarrays",
          nb::arg("input"),
          nb::arg("scale"),
          nb::arg("output"),
          nb::arg("input_dtype_code"),
          nb::arg("output_dtype_code"),
          nb::arg("numel"),
          nb::arg("stream_ptr"));
    
    m.def("dequantize_per_tensor_fp8", &dequantize_per_tensor_fp8,
          "Dequantize from FP8 using nanobind ndarrays",
          nb::arg("input"),
          nb::arg("scale"),
          nb::arg("output"),
          nb::arg("input_dtype_code"),
          nb::arg("output_dtype_code"),
          nb::arg("numel"),
          nb::arg("stream_ptr"));
    
    m.def("cublas_gemm_blockwise_fp4", &cublas_gemm_blockwise_fp4,
          "cuBLAS FP4 GEMM with block-wise scaling",
          nb::arg("b"),
          nb::arg("block_scale_b"),
          nb::arg("a"),
          nb::arg("block_scale_a"),
          nb::arg("out"),
          nb::arg("out_dtype_code"),
          nb::arg("bias"),
          nb::arg("workspace"),
          nb::arg("accumulate"),
          nb::arg("alpha"),
          nb::arg("stream_ptr"));

    m.def("apply_rope", &apply_rope,
          "Apply Rotary Position Embedding (RoPE) using nanobind ndarrays",
          nb::arg("x"),
          nb::arg("freqs"),
          nb::arg("x_out"),
          nb::arg("stream_ptr"));

    m.def("quantize_nvfp4", &quantize_nvfp4,
          "Quantize to FP4 E2M1 with E4M3 block scales using cuBLAS tiled layout",
          nb::arg("input"),
          nb::arg("global_scale"),
          nb::arg("output"),
          nb::arg("block_scales"),
          nb::arg("epsilon"),
          nb::arg("pad_16x") = false,
          nb::arg("stream_ptr"));

    m.def("dequantize_nvfp4", &dequantize_nvfp4,
          "Dequantize from FP4 E2M1 with E4M3 block scales using cuBLAS tiled layout",
          nb::arg("input"),
          nb::arg("global_scale"),
          nb::arg("block_scales"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    // Add version info
    m.attr("__version__") = "0.1.0";
    m.attr("__nanobind__") = true;
    m.attr("__stable_abi__") = true;
}
