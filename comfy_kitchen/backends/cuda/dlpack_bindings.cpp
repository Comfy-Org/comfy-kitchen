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
#include <climits>
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

    // SageAttention kernel launchers
    void launch_quant_qk_per_thread_int8(
        const void* q, void* q_int8, void* q_scale,
        const void* k, void* k_int8, void* k_scale,
        int smooth_k, void* km_scratch, void* km_done,
        int B, int H_q, int Lq, int H_kv, int Lk, int C,
        int BLKQ, int WARPQ, int BLKK, int WARPK,
        int input_dtype_code, cudaStream_t stream);

    void launch_quant_v_fp8_kernel(
        const void* v, void* out, void* scale,
        int B, int H, int N, int D, int padded_N,
        int64_t sb, int64_t sh, int64_t sn,
        int input_dtype_code, cudaStream_t stream);

    void launch_sage_attn_kernel(
        const void* q, const void* k, const void* v, void* o,
        const void* q_scale, const void* k_scale, const void* v_scale,
        int B, int Lq, int Lk, int H_q, int H_kv, int D,
        int q_st_bz, int q_st_n, int q_st_h,
        int k_st_bz, int k_st_n, int k_st_h,
        int v_st_bz, int v_st_h, int v_st_d,
        int o_st_bz, int o_st_n, int o_st_h,
        int is_causal, float sm_scale, int output_dtype_code,
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
    uintptr_t stream_ptr) {

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
        stream
    );
}

// Nanobind wrapper: FP8 V quantization
void quant_v_fp8(
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> out,
    nb::ndarray<nb::device::cuda> scale,
    int padded_n,
    int input_dtype_code,
    uintptr_t stream_ptr)
{
    if (v.ndim() != 4) {
        throw std::runtime_error("quant_v_fp8: v must be 4D [B,H,N,D]");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quant_v_fp8_kernel(
        v.data(), out.data(), scale.data(),
        static_cast<int>(v.shape(0)),
        static_cast<int>(v.shape(1)),
        static_cast<int>(v.shape(2)),
        static_cast<int>(v.shape(3)),
        padded_n,
        v.stride(0), v.stride(1), v.stride(2),
        input_dtype_code, stream);
}

// Nanobind wrapper: INT8 Q/K per-thread quant (contiguous HND layout)
// smooth_k: 1 = fuse K-mean computation + subtraction into the kernel.
// km_scratch_ptr / km_done_ptr: device pointers for scratch buffers
// (pre-zeroed by the caller).  Ignored when smooth_k == 0.
void quant_qk_per_thread_int8(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    int BLKQ, int WARPQ, int BLKK, int WARPK,
    int input_dtype_code,
    uintptr_t stream_ptr,
    int smooth_k = 0,
    uintptr_t km_scratch_ptr = 0,
    uintptr_t km_done_ptr = 0)
{
    if (q.ndim() != 4 || k.ndim() != 4) {
        throw std::runtime_error("quant_qk_per_thread_int8: q and k must be 4D [B,H,L,D]");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quant_qk_per_thread_int8(
        q.data(), q_int8.data(), q_scale.data(),
        k.data(), k_int8.data(), k_scale.data(),
        smooth_k,
        reinterpret_cast<void *>(km_scratch_ptr),
        reinterpret_cast<void *>(km_done_ptr),
        static_cast<int>(q.shape(0)),
        static_cast<int>(q.shape(1)),
        static_cast<int>(q.shape(2)),
        static_cast<int>(k.shape(1)),
        static_cast<int>(k.shape(2)),
        static_cast<int>(q.shape(3)),
        BLKQ, WARPQ, BLKK, WARPK,
        input_dtype_code, stream);
}

// Nanobind wrapper: SageAttention INT8 QK / FP8 V attention kernel
void sage_attn(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_scale,
    int is_causal,
    float sm_scale,
    int output_dtype_code,
    uintptr_t stream_ptr)
{
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || o.ndim() != 4) {
        throw std::runtime_error("sage_attn: q, k, v, o must be 4D");
    }

    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error("sage_attn: output_dtype_code must be 1 (fp16) or 2 (bf16)");
    }

    const int64_t st_q_bz = static_cast<int64_t>(q.stride(0));
    const int64_t st_k_bz = static_cast<int64_t>(k.stride(0));
    const int64_t st_v_bz = static_cast<int64_t>(v.stride(0));
    const int64_t st_o_bz = static_cast<int64_t>(o.stride(0));
    if (st_q_bz > INT_MAX || st_k_bz > INT_MAX ||
        st_v_bz > INT_MAX || st_o_bz > INT_MAX) {
        throw std::overflow_error(
            "sage_attn: tensor strides exceed int32 range");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_sage_attn_kernel(
        q.data(), k.data(), v.data(), o.data(),
        q_scale.data(), k_scale.data(), v_scale.data(),
        static_cast<int>(q.shape(0)),
        static_cast<int>(q.shape(2)),
        static_cast<int>(k.shape(2)),
        static_cast<int>(q.shape(1)),
        static_cast<int>(k.shape(1)),
        static_cast<int>(q.shape(3)),
        q.stride(0), q.stride(2), q.stride(1),
        k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(2), o.stride(1),
        is_causal, sm_scale, output_dtype_code,
        stream);
}

// Fused SageAttention SDPA: quant_qk + quant_v + sage_attn in one C++ call.
// All scratch buffers are pre-allocated by the caller (Python frontend).
// smooth_k: 1 = fuse K-mean reduction + subtraction into the quant kernel.
// km_scratch_ptr / km_done_ptr: device memory for the fused mean kernel.
void sage_sdpa(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_quant,
    nb::ndarray<nb::device::cuda> v_scale,
    int is_causal,
    float sm_scale,
    int input_dtype_code,
    int output_dtype_code,
    uintptr_t stream_ptr,
    int smooth_k = 0,
    uintptr_t km_scratch_ptr = 0,
    uintptr_t km_done_ptr = 0)
{
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 || o.ndim() != 4) {
        throw std::runtime_error("sage_sdpa: q, k, v, o must be 4D [B,H,L,D]");
    }

    const int B = static_cast<int>(q.shape(0));
    const int H_q = static_cast<int>(q.shape(1));
    const int Lq = static_cast<int>(q.shape(2));
    const int D = static_cast<int>(q.shape(3));
    const int H_kv = static_cast<int>(k.shape(1));
    const int Lk = static_cast<int>(k.shape(2));

    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("sage_sdpa: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    }

    constexpr int BLKQ = 128, WARPQ = 32, BLKK = 64, WARPK = 64;
    constexpr int CTA_K = 64;
    const int padded_Lk = ((Lk + CTA_K - 1) / CTA_K) * CTA_K;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_quant_qk_per_thread_int8(
        q.data(), q_int8.data(), q_scale.data(),
        k.data(), k_int8.data(), k_scale.data(),
        smooth_k,
        reinterpret_cast<void *>(km_scratch_ptr),
        reinterpret_cast<void *>(km_done_ptr),
        B, H_q, Lq, H_kv, Lk, D,
        BLKQ, WARPQ, BLKK, WARPK,
        input_dtype_code, stream);

    launch_quant_v_fp8_kernel(
        v.data(), v_quant.data(), v_scale.data(),
        B, H_kv, Lk, D, padded_Lk,
        v.stride(0), v.stride(1), v.stride(2),
        input_dtype_code, stream);

    // int64_t arithmetic to detect overflow before narrowing to int.
    const int64_t qi_st_bz64 = static_cast<int64_t>(H_q)  * Lq * D;
    const int64_t ki_st_bz64 = static_cast<int64_t>(H_kv) * Lk * D;
    const int64_t v_st_bz64  = static_cast<int64_t>(H_kv) * D * padded_Lk;

    if (qi_st_bz64 > INT_MAX || ki_st_bz64 > INT_MAX || v_st_bz64 > INT_MAX) {
        throw std::overflow_error(
            "sage_sdpa: tensor strides exceed int32 range; reduce batch/seq/head dimensions");
    }

    const int qi_st_h = Lq * D, qi_st_n = D, qi_st_bz = static_cast<int>(qi_st_bz64);
    const int ki_st_h = Lk * D, ki_st_n = D, ki_st_bz = static_cast<int>(ki_st_bz64);
    const int o_st_h  = Lq * D, o_st_n  = D, o_st_bz  = static_cast<int>(qi_st_bz64);
    // v_quant is [B*H_kv*D, padded_Lk] (2D from quant kernel).
    // Attention expects V as [B, H, D, padded_N].
    const int v_st_d  = padded_Lk;
    const int v_st_h  = D * padded_Lk;
    const int v_st_bz = static_cast<int>(v_st_bz64);

    launch_sage_attn_kernel(
        q_int8.data(), k_int8.data(), v_quant.data(), o.data(),
        q_scale.data(), k_scale.data(), v_scale.data(),
        B, Lq, Lk, H_q, H_kv, D,
        qi_st_bz, qi_st_n, qi_st_h,
        ki_st_bz, ki_st_n, ki_st_h,
        v_st_bz, v_st_h, v_st_d,
        o_st_bz, o_st_n, o_st_h,
        is_causal, sm_scale, output_dtype_code,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sage_sdpa kernel launch failed: ") + cudaGetErrorString(err));
    }
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

    m.def("apply_rope", &apply_rope,
          "Apply Rotary Position Embedding (RoPE) using nanobind ndarrays",
          nb::arg("xq"),
          nb::arg("freqs"),
          nb::arg("xq_out"),
          nb::arg("xk") = nullptr,
          nb::arg("xk_out") = nullptr,
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

    m.def("quantize_mxfp8", &quantize_mxfp8,
          "Quantize to FP8 E4M3 with E8M0 block scales using cuBLAS tiled layout",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("block_scales"),
          nb::arg("pad_32x") = false,
          nb::arg("stream_ptr"));

    m.def("_quant_v_fp8", &quant_v_fp8,
          "Quantize V [B,H,N,D] fp16/bf16 to FP8 E4M3 rows [B*H*D,padded_N] with per-row scale",
          nb::arg("v"),
          nb::arg("out"),
          nb::arg("scale"),
          nb::arg("padded_n"),
          nb::arg("input_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("_quant_qk_per_thread_int8", &quant_qk_per_thread_int8,
          "INT8 per-thread quant for Q and K (HND), same tiling as Triton quant_per_thread",
          nb::arg("q"),
          nb::arg("q_int8"),
          nb::arg("q_scale"),
          nb::arg("k"),
          nb::arg("k_int8"),
          nb::arg("k_scale"),
          nb::arg("blk_q"),
          nb::arg("warp_q"),
          nb::arg("blk_k"),
          nb::arg("warp_k"),
          nb::arg("input_dtype_code"),
          nb::arg("stream_ptr"),
          nb::arg("smooth_k") = 0,
          nb::arg("km_scratch_ptr") = 0,
          nb::arg("km_done_ptr") = 0);

    m.def("_sage_attn", &sage_attn,
          "SageAttention INT8 QK / FP8 V attention kernel",
          nb::arg("q"),
          nb::arg("k"),
          nb::arg("v"),
          nb::arg("o"),
          nb::arg("q_scale"),
          nb::arg("k_scale"),
          nb::arg("v_scale"),
          nb::arg("is_causal"),
          nb::arg("sm_scale"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("sage_sdpa", &sage_sdpa,
          "Fused SageAttention SDPA: quant_qk + quant_v + sage_attn in one call",
          nb::arg("q"),
          nb::arg("k"),
          nb::arg("v"),
          nb::arg("o"),
          nb::arg("q_int8"),
          nb::arg("q_scale"),
          nb::arg("k_int8"),
          nb::arg("k_scale"),
          nb::arg("v_quant"),
          nb::arg("v_scale"),
          nb::arg("is_causal"),
          nb::arg("sm_scale"),
          nb::arg("input_dtype_code"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"),
          nb::arg("smooth_k") = 0,
          nb::arg("km_scratch_ptr") = 0,
          nb::arg("km_done_ptr") = 0);

    // Feature availability flag (computed at module load time)
    m.attr("HAS_CUBLASLT") = comfy::CublasLtRuntime::instance().is_available();

    // Add version info
    m.attr("__version__") = "0.1.0";
    m.attr("__nanobind__") = true;
    m.attr("__stable_abi__") = true;
}
