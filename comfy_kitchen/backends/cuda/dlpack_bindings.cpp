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
#include <cstring>

#include "cublaslt_runtime.h"

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

    void launch_stochastic_round_fp8_kernel(void* rng_and_output,
                                            const void* input,
                                            int64_t numel,
                                            int rng_dtype_code,
                                            int input_dtype_code,
                                            int output_dtype_code,
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
        const void* xq,
        const void* xk,
        const void* freqs,
        void* xq_out,
        void* xk_out,
        int64_t batch,
        int64_t dim1,
        int64_t dim2,
        int64_t head_dim,
        int64_t freqs_batch,
        int64_t freqs_dim1,
        int64_t freqs_dim2,
        int64_t stride_x_batch,
        int64_t stride_x_dim1,
        int64_t stride_x_dim2,
        int64_t stride_x_dim,
        int64_t stride_freqs_batch,
        int64_t stride_freqs_dim1,
        int64_t stride_freqs_dim2,
        int64_t stride_freqs_dim,
        int64_t stride_freqs_rot,
        int64_t stride_freqs_pair,
        int input_dtype_code,
        int freqs_dtype_code,
        bool split_half,
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
        bool hi_first,
        cudaStream_t stream);

    void launch_dequantize_nvfp4_kernel(
        const void* input,
        const void* global_scale,
        const void* block_scales,
        void* output,
        int64_t num_rows,
        int64_t num_cols,
        int output_dtype_code,
        bool hi_first,
        cudaStream_t stream);

    void launch_quantize_mxfp8_kernel(
        const void* input,
        void* output,
        void* block_scales,
        int64_t num_rows,
        int64_t num_cols,
        int64_t orig_rows,
        int64_t orig_cols,
        int input_dtype_code,
        cudaStream_t stream);

    // SVDQuant W4A4 — see ops/quantize_svdquant_w4a4.cu
    void launch_svdquant_quantize_w4a4_kernel(
        const void* x,
        const void* smooth,
        const void* lora_down,
        void* q_x,
        void* ascales,
        void* lora_act,
        int M,
        int M_pad,
        int K,
        int R,
        int input_dtype_code,
        int act_unsigned,
        cudaStream_t stream);

    // SVDQuant W4A4 — see ops/scaled_mm_svdquant_w4a4.cu
    void launch_svdquant_scaled_mm_w4a4_kernel(
        const void* act,
        const void* wgt,
        const void* ascales,
        const void* wscales,
        const void* lora_act_in,
        const void* lora_up,
        const void* bias,
        void* out,
        int M,
        int N,
        int K,
        int R,
        int act_unsigned,
        int out_dtype_code,
        int tile_packed,
        int fast_accum,
        int shared_scale,
        int fuse_lora,
        cudaStream_t stream);

    // AWQ W4A16 — see ops/awq_w4a16.cu. Internal M-routing picks
    // gemv (M ≤ 8) vs gemm path; bias / LoRA-up are applied externally.
    void launch_awq_w4a16_kernel(
        const void* x,
        const void* qweight,
        const void* wscales,
        const void* wzeros,
        void* out,
        int M,
        int N,
        int K,
        int G,
        int dtype_code,
        cudaStream_t stream);

    // Fused AdaLN — see ops/adaln.cu.
    void launch_adaln_kernel(
        const void* x,
        const void* scale,
        const void* shift,
        void*       out,
        int64_t     N,
        int64_t     D,
        int64_t     scale_group,
        int64_t     shift_group,
        float       eps,
        int         dtype_code,
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

void stochastic_round_fp8(
    nb::ndarray<nb::device::cuda> rng_and_output,
    nb::ndarray<nb::device::cuda> input,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {

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

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_stochastic_round_fp8_kernel(
        rng_and_output.data(),
        input.data(),
        numel,
        rng_dtype_code,
        input_dtype_code,
        output_dtype_code,
        stream);
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

    auto& runtime = comfy::CublasLtRuntime::instance();
    if (!runtime.is_available()) {
        throw std::runtime_error("cuBLASLt not available: " + runtime.error_message());
    }

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
    bool hi_first,
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
        hi_first,
        stream);
}

// Nanobind wrapper for dequantize_nvfp4
void dequantize_nvfp4(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> global_scale,
    nb::ndarray<nb::device::cuda> block_scales,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int output_dtype_code,
    bool hi_first,
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
        hi_first,
        stream);
}

// Nanobind wrapper for quantize_mxfp8
void quantize_mxfp8(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> output,
    nb::ndarray<nb::device::cuda> block_scales,
    bool pad_32x,
    uintptr_t stream_ptr) {

    // Get input dimensions (orig_rows, orig_cols)
    int64_t orig_rows = input.shape(0);
    int64_t orig_cols = input.shape(1);

    // Calculate effective padded dimensions
    int64_t num_rows = orig_rows;
    int64_t num_cols = orig_cols;

    if (pad_32x) {
        // Round up to nearest multiple of 32
        num_rows = (orig_rows + 31) / 32 * 32;
        num_cols = (orig_cols + 31) / 32 * 32;
    }

    // Get input dtype code
    int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for MXFP8 quantization (must be float32, float16, or bfloat16)");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_mxfp8_kernel(
        input.data(),
        output.data(),
        block_scales.data(),
        num_rows,
        num_cols,
        orig_rows,
        orig_cols,
        input_dtype_code,
        stream);
}

// Nanobind wrapper for apply_rope (handles both single tensor and q/k pair)
void apply_rope(
    nb::ndarray<nb::device::cuda> xq,
    nb::ndarray<nb::device::cuda> freqs,
    nb::ndarray<nb::device::cuda> xq_out,
    nb::object xk_obj,
    nb::object xk_out_obj,
    uintptr_t stream_ptr,
    bool split_half = false) {

    // Get xq dimensions: (batch, dim1, dim2, head_dim) - layout agnostic
    int64_t batch = xq.shape(0);
    int64_t dim1 = xq.shape(1);
    int64_t dim2 = xq.shape(2);
    int64_t head_dim = xq.shape(3);

    // Get freqs dimensions (for broadcasting)
    int64_t freqs_batch = freqs.shape(0);
    int64_t freqs_dim1 = freqs.shape(1);
    int64_t freqs_dim2 = freqs.shape(2);

    // Validate freqs last dimensions
    if (freqs.shape(3) != head_dim / 2) {
        throw std::runtime_error("Freqs dimension 3 must be head_dim//2");
    }

    // Validate xq_out shape matches xq
    if (xq_out.ndim() != 4 ||
        xq_out.shape(0) != batch || xq_out.shape(1) != dim1 ||
        xq_out.shape(2) != dim2 || xq_out.shape(3) != head_dim) {
        throw std::runtime_error("Output shape must match input shape");
    }

    // Handle optional xk and xk_out
    bool has_xk = !xk_obj.is_none();
    bool has_xk_out = !xk_out_obj.is_none();
    
    if (has_xk != has_xk_out) {
        throw std::runtime_error("xk and xk_out must both be provided or both be None");
    }
    
    void* xk_data = nullptr;
    void* xk_out_data = nullptr;
    
    if (has_xk) {
        auto xk = nb::cast<nb::ndarray<nb::device::cuda>>(xk_obj);
        auto xk_out = nb::cast<nb::ndarray<nb::device::cuda>>(xk_out_obj);
        
        if (xk.ndim() != 4 ||
            xk.shape(0) != batch || xk.shape(1) != dim1 ||
            xk.shape(2) != dim2 || xk.shape(3) != head_dim) {
            throw std::runtime_error("xk shape must match xq shape");
        }
        
        if (xk_out.ndim() != 4 ||
            xk_out.shape(0) != batch || xk_out.shape(1) != dim1 ||
            xk_out.shape(2) != dim2 || xk_out.shape(3) != head_dim) {
            throw std::runtime_error("xk_out shape must match xq shape");
        }
        
        xk_data = xk.data();
        xk_out_data = xk_out.data();
    }

    // Get input dtype code
    int input_dtype_code = map_dtype_to_code(xq.dtype());
    if (input_dtype_code < 0) {
        throw std::runtime_error("Unsupported input dtype for apply_rope");
    }

    // Get freqs dtype code
    int freqs_dtype_code = map_dtype_to_code(freqs.dtype());
    if (freqs_dtype_code < 0) {
        throw std::runtime_error("Unsupported freqs dtype for apply_rope");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Get strides (nanobind provides strides in elements, not bytes)
    int64_t stride_x_batch = xq.stride(0);
    int64_t stride_x_dim1 = xq.stride(1);
    int64_t stride_x_dim2 = xq.stride(2);
    int64_t stride_x_dim = xq.stride(3);

    int64_t stride_freqs_batch = freqs.stride(0);
    int64_t stride_freqs_dim1 = freqs.stride(1);
    int64_t stride_freqs_dim2 = freqs.stride(2);
    int64_t stride_freqs_dim = freqs.stride(3);
    int64_t stride_freqs_rot = freqs.stride(4);
    int64_t stride_freqs_pair = freqs.stride(5);

    // Launch kernel
    launch_apply_rope_kernel(
        xq.data(),
        xk_data,
        freqs.data(),
        xq_out.data(),
        xk_out_data,
        batch,
        dim1,
        dim2,
        head_dim,
        freqs_batch,
        freqs_dim1,
        freqs_dim2,
        stride_x_batch,
        stride_x_dim1,
        stride_x_dim2,
        stride_x_dim,
        stride_freqs_batch,
        stride_freqs_dim1,
        stride_freqs_dim2,
        stride_freqs_dim,
        stride_freqs_rot,
        stride_freqs_pair,
        input_dtype_code,
        freqs_dtype_code,
        split_half,
        stream
    );
}

// ---------------------------------------------------------------------------
// SVDQuant W4A4 — nanobind/DLPack bindings for the native kitchen int4 kernels
// (see ops/quantize_svdquant_w4a4.cu and ops/scaled_mm_svdquant_w4a4.cu).
// ---------------------------------------------------------------------------

static int svdquant_dtype_code(const nb::dlpack::dtype& dt) {
    int c = map_dtype_to_code(dt);
    if (c < 0) throw std::runtime_error("svdquant: unsupported dtype");
    return c;
}

void svdquant_quantize_w4a4(
    nb::ndarray<nb::device::cuda> x,           // (M, K) bf16/fp16 — pre-shifted if unsigned path
    nb::ndarray<nb::device::cuda> smooth,      // (K,)
    nb::ndarray<nb::device::cuda> lora_down,   // (K, R)
    nb::ndarray<nb::device::cuda> q_x,         // (M_pad, K/2) int8
    nb::ndarray<nb::device::cuda> ascales,     // (K/G, M_pad)
    nb::ndarray<nb::device::cuda> lora_act,    // (M_pad, R) fp32
    bool act_unsigned,
    uintptr_t stream_ptr)
{
    int M = static_cast<int>(x.shape(0));
    int K = static_cast<int>(x.shape(1));
    int M_pad = static_cast<int>(q_x.shape(0));
    int R = static_cast<int>(lora_down.shape(1));
    int input_code = svdquant_dtype_code(x.dtype());

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_svdquant_quantize_w4a4_kernel(
        x.data(), smooth.data(), lora_down.data(),
        q_x.data(), ascales.data(), lora_act.data(),
        M, M_pad, K, R, input_code,
        static_cast<int>(act_unsigned), stream);
}

void svdquant_scaled_mm_w4a4(
    nb::ndarray<nb::device::cuda> act,           // (M, K/2) int8
    nb::ndarray<nb::device::cuda> wgt,           // (N, K/2) int8
    nb::ndarray<nb::device::cuda> ascales,       // (K/G, M)
    nb::ndarray<nb::device::cuda> wscales,       // (K/G, N)
    nb::ndarray<nb::device::cuda> lora_act_in,   // (M, R) fp32
    nb::ndarray<nb::device::cuda> lora_up,       // (N, R)
    nb::ndarray<nb::device::cuda> bias,          // (N,) or empty
    nb::ndarray<nb::device::cuda> out,           // (M, N)
    bool act_unsigned,
    bool fast_accum,
    bool shared_scale,
    bool fuse_lora,
    uintptr_t stream_ptr)
{
    int M = static_cast<int>(act.shape(0));
    int K = static_cast<int>(act.shape(1)) * 2;
    const bool tile_packed = (wgt.ndim() == 4);
    int N = tile_packed ? static_cast<int>(wgt.shape(0)) * 128 : static_cast<int>(wgt.shape(0));
    int R = static_cast<int>(lora_act_in.shape(1));
    int out_code = svdquant_dtype_code(out.dtype());
    if (fuse_lora && svdquant_dtype_code(lora_act_in.dtype()) != out_code) {
        throw std::runtime_error(
            "svdquant_scaled_mm_w4a4: fused LoRA-up requires lora_act_in dtype "
            "to match output/lora_up dtype");
    }

    if (tile_packed) {
        if (wgt.shape(1) != K / 64 || wgt.shape(2) != 32 || wgt.shape(3) != 128) {
            throw std::runtime_error(
                "svdquant_scaled_mm_w4a4: tile-packed weight must have shape "
                "(N/128, K/64, 32, 128)");
        }
        if (wscales.ndim() != 3 || wscales.shape(0) != wgt.shape(0) ||
            wscales.shape(1) != K / 64 || wscales.shape(2) != 128) {
            throw std::runtime_error(
                "svdquant_scaled_mm_w4a4: tile-packed wscales must have shape "
                "(N/128, K/64, 128)");
        }
    }

    const void* bias_ptr = (bias.data() != nullptr && bias.size() > 0) ? bias.data() : nullptr;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_svdquant_scaled_mm_w4a4_kernel(
        act.data(), wgt.data(),
        ascales.data(), wscales.data(),
        lora_act_in.data(), lora_up.data(), bias_ptr,
        out.data(),
        M, N, K, R,
        static_cast<int>(act_unsigned), out_code,
        static_cast<int>(tile_packed), static_cast<int>(fast_accum),
        static_cast<int>(shared_scale), static_cast<int>(fuse_lora), stream);
}

// ---------------------------------------------------------------------------
// AWQ W4A16 — int4 weight, fp16/bf16 activation matmul. See ops/awq_w4a16.cu.
// ---------------------------------------------------------------------------
void awq_w4a16(
    nb::ndarray<nb::device::cuda> x,         // (M, K) bf16/fp16
    nb::ndarray<nb::device::cuda> qweight,   // (N, K/2) int8 packed uint4
    nb::ndarray<nb::device::cuda> wscales,   // (K/G, N)
    nb::ndarray<nb::device::cuda> wzeros,    // (K/G, N)
    nb::ndarray<nb::device::cuda> out,       // (M, N)
    int group_size,
    uintptr_t stream_ptr)
{
    const int M = static_cast<int>(x.shape(0));
    const int K = static_cast<int>(x.shape(1));
    const int N = static_cast<int>(qweight.shape(0));
    const int dtype_code = svdquant_dtype_code(x.dtype());
    if (dtype_code != 1 && dtype_code != 2) {
        throw std::runtime_error("awq_w4a16: only fp16 (1) and bf16 (2) activations supported");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_awq_w4a16_kernel(
        x.data(), qweight.data(), wscales.data(), wzeros.data(), out.data(),
        M, N, K, group_size, dtype_code, stream);
}

// Nanobind wrapper for fused AdaLN
void adaln(
    nb::ndarray<nb::device::cuda> x,
    nb::ndarray<nb::device::cuda> scale,
    nb::ndarray<nb::device::cuda> shift,
    nb::ndarray<nb::device::cuda> out,
    int64_t N,
    int64_t D,
    int64_t scale_group,
    int64_t shift_group,
    float   eps,
    int     dtype_code,
    uintptr_t stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_adaln_kernel(
        x.data(), scale.data(), shift.data(), out.data(),
        N, D, scale_group, shift_group, eps, dtype_code, stream);
}

// Python module definition
extern "C" {
    void launch_cublas_gemm_int8_kernel(
        const void* A_ptr,
        const void* B_ptr,
        void* C_ptr,
        int64_t M,
        int64_t N,
        int64_t K,
        void* workspace_ptr,
        int64_t workspace_size,
        cudaStream_t stream);

    void launch_quantize_int8_rowwise_kernel(
        const void* input,
        void* output,
        void* scales,
        const void* rng,
        int64_t num_rows,
        int64_t num_cols,
        int input_dtype_code,
        bool stochastic,
        cudaStream_t stream);

    bool launch_cutlass_int8_dequant(
        const void* A,
        const void* B,
        const void* xs,
        const void* ws,
        const void* bias,
        void* D,
        int64_t M,
        int64_t N,
        int64_t K,
        int out_dtype_code,
        cudaStream_t stream);

    void launch_quantize_int8_rowwise_convrot_kernel(
        const void* input,
        void* output,
        void* scales,
        const void* rng,
        int64_t num_rows,
        int64_t num_cols,
        int group_size,
        int input_dtype_code,
        bool stochastic,
        cudaStream_t stream);

    void launch_rotate_int8_convrot_weight_kernel(
        const void* input,
        void* output,
        int64_t num_rows,
        int64_t num_cols,
        int group_size,
        int input_dtype_code,
        int output_dtype_code,
        cudaStream_t stream);

    void launch_quantize_int8_convrot_staged_kernel(
        const void* input,
        void* rotated,
        void* partial_absmax,
        void* output,
        void* scales,
        const void* rng,
        int64_t num_rows,
        int64_t num_cols,
        int group_size,
        int input_dtype_code,
        int rotated_dtype_code,
        bool stochastic,
        cudaStream_t stream);

    void launch_quantize_int8_rowwise_convrot64_kernel(
        const void* input,
        void* output,
        void* scales,
        const void* rng,
        int64_t num_rows,
        int64_t num_cols,
        int group_size,
        int input_dtype_code,
        bool stochastic,
        cudaStream_t stream);

    void launch_dequantize_int8_linear_kernel(
        const void* input,
        const void* x_scales,
        const void* weight_scales,
        const void* bias,
        void* output,
        int64_t num_rows,
        int64_t num_cols,
        int64_t weight_scale_size,
        bool has_bias,
        int output_dtype_code,
        int bias_dtype_code,
        cudaStream_t stream);

    void launch_int8_gemv_dequant_kernel(
        const void* input,
        const void* weight,
        const void* x_scales,
        const void* weight_scales,
        const void* bias,
        void* output,
        int64_t num_cols,
        int64_t K,
        int64_t weight_scale_size,
        bool has_bias,
        int output_dtype_code,
        int bias_dtype_code,
        cudaStream_t stream);

    void launch_dequantize_int8_simple_kernel(
        const void* input,
        const void* scales,
        void* output,
        int64_t total,
        int64_t inner_dim,
        int scale_mode,
        int output_dtype_code,
        cudaStream_t stream);

    void launch_dequantize_int8_convrot_kernel(
        const void* input,
        const void* scales,
        void* output,
        int64_t num_rows,
        int64_t num_cols,
        int64_t scale_size,
        int group_size,
        int output_dtype_code,
        cudaStream_t stream);

}

// Nanobind wrapper for cublas_gemm_int8
void cublas_gemm_int8(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,
    nb::ndarray<int32_t, nb::ndim<2>, nb::device::cuda> c,
    nb::ndarray<nb::device::cuda> workspace,
    uintptr_t stream_ptr) {

    auto& runtime = comfy::CublasLtRuntime::instance();
    if (!runtime.is_available()) {
        throw std::runtime_error("cuBLASLt not available: " + runtime.error_message());
    }

    // a is [M, K], b is [N, K], c is [M, N]
    int64_t M = a.shape(0);
    int64_t K = a.shape(1);
    int64_t N = b.shape(0);
    int64_t K_b = b.shape(1);

    if (K != K_b) {
        throw std::runtime_error("Matrix K dimensions do not match");
    }

    if (c.shape(0) != M || c.shape(1) != N) {
        throw std::runtime_error("Output matrix C shape does not match");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_cublas_gemm_int8_kernel(
        a.data(),
        b.data(),
        c.data(),
        M, N, K,
        workspace.data(),
        workspace.size() > 0 ? (int64_t)workspace.size() : 0,
        stream);
}

void quantize_int8_rowwise(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    nb::ndarray<nb::device::cuda> rng,
    bool stochastic,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);

    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("INT8 rowwise quantization output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT8 rowwise quantization scale shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT8 rowwise quantization");
    }
    if (stochastic) {
        if (map_dtype_to_code(rng.dtype()) != input_dtype_code || rng.size() != output.size()) {
            throw std::runtime_error("INT8 rowwise quantization RNG shape or dtype mismatch");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_kernel(
        input.data(),
        output.data(),
        scales.data(),
        rng.data(),
        M,
        K,
        input_dtype_code,
        stochastic,
        stream);
}

// INT8 GEMM + fused dequant (D = acc * xs[m] * ws[n] + bias[n]) via CUTLASS.
// Returns true on success; false means caller falls back to cuBLAS + dequant.
bool cutlass_int8_dequant(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,   // [M, K]
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,   // [N, K]
    nb::ndarray<float, nb::device::cuda> xs,                // [M] per-row act scale
    nb::ndarray<float, nb::device::cuda> ws,                // [N] per-col weight scale
    nb::ndarray<nb::device::cuda> bias,                     // [N] float or empty
    nb::ndarray<nb::ndim<2>, nb::device::cuda> d,           // [M, N] output
    int out_dtype_code,
    uintptr_t stream_ptr) {
    const int64_t M = a.shape(0);
    const int64_t K = a.shape(1);
    const int64_t N = b.shape(0);
    if (b.shape(1) != K) throw std::runtime_error("cutlass_int8_dequant: K mismatch");
    if (d.shape(0) != M || d.shape(1) != N) throw std::runtime_error("cutlass_int8_dequant: D shape mismatch");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const void* bias_ptr = bias.size() > 0 ? bias.data() : nullptr;
    return launch_cutlass_int8_dequant(a.data(), b.data(), xs.data(), ws.data(),
                                       bias_ptr, d.data(), M, N, K, out_dtype_code, stream);
}

void quantize_int8_rowwise_convrot(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    nb::ndarray<nb::device::cuda> rng,
    int64_t group_size,
    bool stochastic,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);

    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("INT8 rowwise convrot output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT8 rowwise convrot scale shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT8 rowwise convrot quantization");
    }
    if (stochastic) {
        if (map_dtype_to_code(rng.dtype()) != input_dtype_code || rng.size() != output.size()) {
            throw std::runtime_error("INT8 rowwise convrot RNG shape or dtype mismatch");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_convrot_kernel(
        input.data(),
        output.data(),
        scales.data(),
        rng.data(),
        M,
        K,
        static_cast<int>(group_size),
        input_dtype_code,
        stochastic,
        stream);
}

void rotate_int8_convrot_weight(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int64_t group_size,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("ConvRot rotate output shape mismatch");
    }

    const int input_dtype_code = map_dtype_to_code(input.dtype());
    const int output_dtype_code = map_dtype_to_code(output.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2 || output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported dtype for ConvRot rotate");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_rotate_int8_convrot_weight_kernel(
        input.data(),
        output.data(),
        M,
        K,
        static_cast<int>(group_size),
        input_dtype_code,
        output_dtype_code,
        stream);
}

void quantize_int8_convrot_staged(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> rotated,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> partial_absmax,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    nb::ndarray<nb::device::cuda> rng,
    int64_t group_size,
    bool stochastic,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    if (rotated.shape(0) != M || rotated.shape(1) != K) {
        throw std::runtime_error("ConvRot staged rotated shape mismatch");
    }
    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("ConvRot staged output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("ConvRot staged scale shape mismatch");
    }
    const int64_t n_groups = group_size > 0 ? K / group_size : 0;
    if (partial_absmax.shape(0) != M || partial_absmax.shape(1) != n_groups) {
        throw std::runtime_error("ConvRot staged partial absmax shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    const int rotated_dtype_code = map_dtype_to_code(rotated.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2 || rotated_dtype_code < 0 || rotated_dtype_code > 2) {
        throw std::runtime_error("Unsupported dtype for ConvRot staged quantization");
    }
    if (stochastic) {
        if (map_dtype_to_code(rng.dtype()) != rotated_dtype_code || rng.size() != output.size()) {
            throw std::runtime_error("ConvRot staged RNG shape or dtype mismatch");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_convrot_staged_kernel(
        input.data(),
        rotated.data(),
        partial_absmax.data(),
        output.data(),
        scales.data(),
        rng.data(),
        M,
        K,
        static_cast<int>(group_size),
        input_dtype_code,
        rotated_dtype_code,
        stochastic,
        stream);
}

void quantize_int8_rowwise_convrot64(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    nb::ndarray<nb::device::cuda> rng,
    int64_t group_size,
    bool stochastic,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);

    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("INT8 rowwise convrot64 output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT8 rowwise convrot64 scale shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT8 rowwise convrot64 quantization");
    }
    if (stochastic) {
        if (map_dtype_to_code(rng.dtype()) != input_dtype_code || rng.size() != output.size()) {
            throw std::runtime_error("INT8 rowwise convrot64 RNG shape or dtype mismatch");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_convrot64_kernel(
        input.data(),
        output.data(),
        scales.data(),
        rng.data(),
        M,
        K,
        static_cast<int>(group_size),
        input_dtype_code,
        stochastic,
        stream);
}

void dequantize_int8_linear(
    nb::ndarray<int32_t, nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> x_scales,
    nb::ndarray<float, nb::device::cuda> weight_scales,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int output_dtype_code,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t N = input.shape(1);

    if (x_scales.shape(0) != M || x_scales.shape(1) != 1) {
        throw std::runtime_error("INT8 linear activation scale shape mismatch");
    }
    if (output.shape(0) != M || output.shape(1) != N) {
        throw std::runtime_error("INT8 linear output shape mismatch");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Invalid INT8 linear output dtype code");
    }

    const bool has_bias = bias.data() && bias.size() > 0;
    int bias_dtype_code = output_dtype_code;
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("INT8 linear bias shape mismatch");
        }
        bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT8 linear");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_int8_linear_kernel(
        input.data(),
        x_scales.data(),
        weight_scales.data(),
        has_bias ? bias.data() : nullptr,
        output.data(),
        M,
        N,
        static_cast<int64_t>(weight_scales.size()),
        has_bias,
        output_dtype_code,
        bias_dtype_code,
        stream);
}

void int8_gemv_dequant(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> x_scales,
    nb::ndarray<float, nb::device::cuda> weight_scales,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int output_dtype_code,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    const int64_t N = weight.shape(0);
    if (M != 1) {
        throw std::runtime_error("INT8 GEMV dequant expects M == 1");
    }
    if (weight.shape(1) != K) {
        throw std::runtime_error("INT8 GEMV weight K mismatch");
    }
    if (x_scales.shape(0) != 1 || x_scales.shape(1) != 1) {
        throw std::runtime_error("INT8 GEMV activation scale shape mismatch");
    }
    if (output.shape(0) != 1 || output.shape(1) != N) {
        throw std::runtime_error("INT8 GEMV output shape mismatch");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Invalid INT8 GEMV output dtype code");
    }

    const bool has_bias = bias.data() && bias.size() > 0;
    int bias_dtype_code = output_dtype_code;
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("INT8 GEMV bias shape mismatch");
        }
        bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT8 GEMV");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_int8_gemv_dequant_kernel(
        input.data(),
        weight.data(),
        x_scales.data(),
        weight_scales.data(),
        has_bias ? bias.data() : nullptr,
        output.data(),
        N,
        K,
        static_cast<int64_t>(weight_scales.size()),
        has_bias,
        output_dtype_code,
        bias_dtype_code,
        stream);
}

void int8_linear_m1(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> q_scratch,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> x_scales,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight,
    nb::ndarray<float, nb::device::cuda> weight_scales,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int output_dtype_code,
    bool convrot,
    int group_size,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    const int64_t N = weight.shape(0);
    if (M != 1) {
        throw std::runtime_error("INT8 M=1 linear expects input M == 1");
    }
    if (weight.shape(1) != K) {
        throw std::runtime_error("INT8 M=1 linear weight K mismatch");
    }
    if (q_scratch.shape(0) != 1 || q_scratch.shape(1) != K) {
        throw std::runtime_error("INT8 M=1 linear q scratch shape mismatch");
    }
    if (x_scales.shape(0) != 1 || x_scales.shape(1) != 1) {
        throw std::runtime_error("INT8 M=1 linear activation scale shape mismatch");
    }
    if (output.shape(0) != 1 || output.shape(1) != N) {
        throw std::runtime_error("INT8 M=1 linear output shape mismatch");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Invalid INT8 M=1 linear output dtype code");
    }
    if (convrot && (group_size != 256 || K % 256 != 0)) {
        throw std::runtime_error("INT8 M=1 ConvRot linear requires group_size 256 and K divisible by 256");
    }

    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT8 M=1 linear");
    }

    const bool has_bias = bias.data() && bias.size() > 0;
    int bias_dtype_code = output_dtype_code;
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("INT8 M=1 linear bias shape mismatch");
        }
        bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT8 M=1 linear");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (convrot) {
        launch_quantize_int8_rowwise_convrot64_kernel(
            input.data(),
            q_scratch.data(),
            x_scales.data(),
            nullptr,
            M,
            K,
            group_size,
            input_dtype_code,
            false,
            stream);
    } else {
        launch_quantize_int8_rowwise_kernel(
            input.data(),
            q_scratch.data(),
            x_scales.data(),
            nullptr,
            M,
            K,
            input_dtype_code,
            false,
            stream);
    }
    launch_int8_gemv_dequant_kernel(
        q_scratch.data(),
        weight.data(),
        x_scales.data(),
        weight_scales.data(),
        has_bias ? bias.data() : nullptr,
        output.data(),
        N,
        K,
        static_cast<int64_t>(weight_scales.size()),
        has_bias,
        output_dtype_code,
        bias_dtype_code,
        stream);
}

void dequantize_int8_simple(
    nb::ndarray<int8_t, nb::device::cuda> input,
    nb::ndarray<float, nb::device::cuda> scale,
    nb::ndarray<nb::device::cuda> output,
    int64_t inner_dim,
    int scale_mode,
    uintptr_t stream_ptr) {

    if (output.size() != input.size()) {
        throw std::runtime_error("INT8 simple dequantization output shape mismatch");
    }
    if (scale_mode == 0 && scale.size() != 1) {
        throw std::runtime_error("INT8 simple dequantization scalar scale shape mismatch");
    }
    if (scale_mode == 1 && scale.size() != input.size()) {
        throw std::runtime_error("INT8 simple dequantization elementwise scale shape mismatch");
    }
    if (scale_mode == 2 && (inner_dim <= 0 || input.size() % inner_dim != 0 || scale.size() != input.size() / inner_dim)) {
        throw std::runtime_error("INT8 simple dequantization rowwise scale shape mismatch");
    }
    const int output_dtype_code = map_dtype_to_code(output.dtype());
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for INT8 simple dequantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_int8_simple_kernel(
        input.data(),
        scale.data(),
        output.data(),
        static_cast<int64_t>(input.size()),
        inner_dim,
        scale_mode,
        output_dtype_code,
        stream);
}

void dequantize_int8_convrot_weight(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<float, nb::device::cuda> scale,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int64_t group_size,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("INT8 convrot dequant output shape mismatch");
    }
    if (scale.size() != 1 && scale.size() != static_cast<size_t>(M)) {
        throw std::runtime_error("INT8 convrot dequant scale must be scalar or per-row");
    }
    const int output_dtype_code = map_dtype_to_code(output.dtype());
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for INT8 convrot dequantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_int8_convrot_kernel(
        input.data(),
        scale.data(),
        output.data(),
        M,
        K,
        static_cast<int64_t>(scale.size()),
        static_cast<int>(group_size),
        output_dtype_code,
        stream);
}

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

    m.def("stochastic_round_fp8", &stochastic_round_fp8,
          "Stochastically round to FP8, overwriting RNG storage with FP8 output",
          nb::arg("rng_and_output"),
          nb::arg("input"),
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

    m.def("cublas_gemm_int8", &cublas_gemm_int8,
          "INT8 GEMM using cuBLASLt IMMA tensor cores (SM >= 7.5)",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("c"),
          nb::arg("workspace"),
          nb::arg("stream_ptr"));

    m.def("quantize_int8_rowwise", &quantize_int8_rowwise,
          "Rowwise INT8 quantization for CUDA activations",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("rng"),
          nb::arg("stochastic"),
          nb::arg("stream_ptr"));

    m.def("cutlass_int8_dequant", &cutlass_int8_dequant,
          "INT8 GEMM + fused rowwise x colwise dequant + bias via CUTLASS; false -> fall back to cuBLAS",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("xs"),
          nb::arg("ws"),
          nb::arg("bias"),
          nb::arg("d"),
          nb::arg("out_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("quantize_int8_rowwise_convrot", &quantize_int8_rowwise_convrot,
          "Fused ConvRot Hadamard rotation + rowwise INT8 quantization",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("rng"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("stream_ptr"));

    m.def("rotate_int8_convrot_weight", &rotate_int8_convrot_weight,
          "ConvRot Hadamard weight rotation",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("group_size"),
          nb::arg("stream_ptr"));

    m.def("quantize_int8_convrot_staged", &quantize_int8_convrot_staged,
          "ConvRot rotation with partial absmax followed by INT8 rowwise quantization",
          nb::arg("input"),
          nb::arg("rotated"),
          nb::arg("partial_absmax"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("rng"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("stream_ptr"));

    m.def("quantize_int8_rowwise_convrot64", &quantize_int8_rowwise_convrot64,
          "Fused ConvRot rowwise INT8 quantization using 64-lane FHT groups",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("rng"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("stream_ptr"));

    m.def("dequantize_int8_linear", &dequantize_int8_linear,
          "Fused INT8 linear dequantization, bias, and output cast",
          nb::arg("input"),
          nb::arg("x_scales"),
          nb::arg("weight_scales"),
          nb::arg("bias"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("int8_gemv_dequant", &int8_gemv_dequant,
          "INT8 GEMV with fused rowwise x colwise dequantization, bias, and output cast",
          nb::arg("input"),
          nb::arg("weight"),
          nb::arg("x_scales"),
          nb::arg("weight_scales"),
          nb::arg("bias"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("int8_linear_m1", &int8_linear_m1,
          "M=1 INT8 linear: activation quantization followed by GEMV/dequant",
          nb::arg("input"),
          nb::arg("q_scratch"),
          nb::arg("x_scales"),
          nb::arg("weight"),
          nb::arg("weight_scales"),
          nb::arg("bias"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("convrot"),
          nb::arg("group_size"),
          nb::arg("stream_ptr"));

    m.def("dequantize_int8_simple", &dequantize_int8_simple,
          "INT8 dequantization to float32",
          nb::arg("input"),
          nb::arg("scale"),
          nb::arg("output"),
          nb::arg("inner_dim"),
          nb::arg("scale_mode"),
          nb::arg("stream_ptr"));

    m.def("dequantize_int8_convrot_weight", &dequantize_int8_convrot_weight,
          "INT8 ConvRot weight dequantization to float32",
          nb::arg("input"),
          nb::arg("scale"),
          nb::arg("output"),
          nb::arg("group_size"),
          nb::arg("stream_ptr"));

    m.def("apply_rope", &apply_rope,
          "Apply Rotary Position Embedding (RoPE) using nanobind ndarrays",
          nb::arg("xq"),
          nb::arg("freqs"),
          nb::arg("xq_out"),
          nb::arg("xk") = nullptr,
          nb::arg("xk_out") = nullptr,
          nb::arg("stream_ptr"),
          nb::arg("split_half") = false);

    m.def("quantize_nvfp4", &quantize_nvfp4,
          "Quantize to FP4 E2M1 with E4M3 block scales using cuBLAS tiled layout",
          nb::arg("input"),
          nb::arg("global_scale"),
          nb::arg("output"),
          nb::arg("block_scales"),
          nb::arg("epsilon"),
          nb::arg("pad_16x") = false,
          nb::arg("hi_first") = true,
          nb::arg("stream_ptr"));

    m.def("dequantize_nvfp4", &dequantize_nvfp4,
          "Dequantize from FP4 E2M1 with E4M3 block scales using cuBLAS tiled layout",
          nb::arg("input"),
          nb::arg("global_scale"),
          nb::arg("block_scales"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("hi_first") = true,
          nb::arg("stream_ptr"));

    m.def("quantize_mxfp8", &quantize_mxfp8,
          "Quantize to FP8 E4M3 with E8M0 block scales using cuBLAS tiled layout",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("block_scales"),
          nb::arg("pad_32x") = false,
          nb::arg("stream_ptr"));

    m.def("svdquant_quantize_w4a4", &svdquant_quantize_w4a4,
          "SVDQuant W4A4: smooth + int4 quantize (LoRA-down is external). "
          "act_unsigned selects scale=max/15 + clamp [0,15] for u4 MMA downstream; "
          "caller must pre-shift x to be non-negative before calling (model-level concern).",
          nb::arg("x"),
          nb::arg("smooth"),
          nb::arg("lora_down"),
          nb::arg("q_x"),
          nb::arg("ascales"),
          nb::arg("lora_act"),
          nb::arg("act_unsigned"),
          nb::arg("stream_ptr"));

    m.def("svdquant_scaled_mm_w4a4", &svdquant_scaled_mm_w4a4,
          "SVDQuant W4A4: int4 GEMM with per-group dequant",
          nb::arg("act"),
          nb::arg("wgt"),
          nb::arg("ascales"),
          nb::arg("wscales"),
          nb::arg("lora_act_in"),
          nb::arg("lora_up"),
          nb::arg("bias"),
          nb::arg("out"),
          nb::arg("act_unsigned"),
          nb::arg("fast_accum"),
          nb::arg("shared_scale"),
          nb::arg("fuse_lora"),
          nb::arg("stream_ptr"));

    m.def("awq_w4a16", &awq_w4a16,
          "AWQ W4A16: int4 weight @ fp activation (kitchen-native row-major). "
          "Internal M-routing picks gemv (M ≤ 8) vs gemm. bias / LoRA-up are "
          "applied externally; this kernel only does the dequant + matmul.",
          nb::arg("x"),
          nb::arg("qweight"),
          nb::arg("wscales"),
          nb::arg("wzeros"),
          nb::arg("out"),
          nb::arg("group_size"),
          nb::arg("stream_ptr"));

    m.def("adaln", &adaln,
          "Fused AdaLN: layernorm(x) * (1 + scale) + shift",
          nb::arg("x"),
          nb::arg("scale"),
          nb::arg("shift"),
          nb::arg("out"),
          nb::arg("N"),
          nb::arg("D"),
          nb::arg("scale_group"),
          nb::arg("shift_group"),
          nb::arg("eps"),
          nb::arg("dtype_code"),
          nb::arg("stream_ptr"));

    // Feature availability flag (computed at module load time)
    m.attr("HAS_CUBLASLT") = comfy::CublasLtRuntime::instance().is_available();

    // Add version info
    m.attr("__version__") = "0.1.0";
    m.attr("__nanobind__") = true;
    m.attr("__stable_abi__") = true;
}
