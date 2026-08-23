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
#include <nanobind/stl/optional.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstring>
#include <optional>
#include <string>

#include "cublaslt_runtime.h"
#include "input_act_codes.h"
#include "tensor_bindings.h"

namespace nb = nanobind;

// Helper: Map nanobind dtype to internal dtype code
// Returns: 0=float32, 1=float16, 2=bfloat16, 3=uint8, 4=int8, 5=float8_e4m3fn, 6=float8_e5m2
int map_dtype_to_code(const nb::dlpack::dtype& dtype) {
    return static_cast<int>(comfy::tensor::dtype_from_dlpack(dtype));
}

template <typename... Args>
void validate_fp_dtype_code(
    const nb::ndarray<Args...>& array, int dtype_code, const char* operation) {
    if (dtype_code < 0 || dtype_code > 2 || map_dtype_to_code(array.dtype()) != dtype_code) {
        throw std::runtime_error(
            std::string(operation) + ": dtype code must match an FP32, FP16, or BF16 tensor");
    }
}

comfy::tensor::TensorArg<4> make_packed_sage_v_arg(
    const nb::ndarray<nb::device::cuda>& array, int64_t batch, int64_t heads,
    int64_t head_dim, int64_t padded_length) {
    auto flat = comfy::tensor::make_flat_tensor_arg(array);
    if (flat.meta.sizes[0] != batch * heads * head_dim * padded_length) {
        throw std::runtime_error("packed SageAttention V has an invalid size");
    }
    comfy::tensor::TensorArg<4> arg{};
    arg.data = flat.data;
    arg.meta.dtype = flat.meta.dtype;
    arg.meta.sizes[0] = batch;
    arg.meta.sizes[1] = heads;
    arg.meta.sizes[2] = head_dim;
    arg.meta.sizes[3] = padded_length;
    arg.meta.strides[0] = heads * head_dim * padded_length;
    arg.meta.strides[1] = head_dim * padded_length;
    arg.meta.strides[2] = padded_length;
    arg.meta.strides[3] = 1;
    return arg;
}

comfy::tensor::TensorArg<1> make_int32_scratch_arg(uintptr_t pointer, int64_t size) {
    comfy::tensor::TensorArg<1> arg{};
    arg.data = reinterpret_cast<void*>(pointer);
    arg.meta.sizes[0] = size;
    arg.meta.strides[0] = 1;
    arg.meta.dtype = comfy::tensor::DType::Int32;
    return arg;
}

using comfy::tensor::DType;
using comfy::tensor::TensorArg;
using comfy::tensor::make_contiguous_tensor_arg;
using comfy::tensor::make_flat_tensor_arg;
using comfy::tensor::make_optional_flat_tensor_arg;
using comfy::tensor::make_tensor_arg;

// Forward declarations of CUDA kernel wrappers
extern "C" {
    void launch_quantize_fp8_kernel(
        TensorArg<1> input, TensorArg<1> scale, TensorArg<1> output,
        cudaStream_t stream);
    
    void launch_dequantize_fp8_kernel(
        TensorArg<1> input, TensorArg<1> scale, TensorArg<1> output,
        cudaStream_t stream);

    void launch_stochastic_round_fp8_kernel(
        TensorArg<1> rng_and_output, TensorArg<1> input, cudaStream_t stream);

    void launch_cublas_gemm_blockwise_fp4_kernel(
        TensorArg<2> b, TensorArg<2> b_scale, TensorArg<2> a,
        TensorArg<2> a_scale, TensorArg<2> out, TensorArg<1> bias,
        TensorArg<1> alpha, TensorArg<1> workspace,
        bool accumulate,
        cudaStream_t stream);

    void launch_apply_rope_kernel(
        TensorArg<4> q, TensorArg<4> k, TensorArg<6> freqs,
        TensorArg<4> q_out, TensorArg<4> k_out, bool has_k, bool split_half,
        cudaStream_t stream);

    void launch_quantize_nvfp4_kernel(
        TensorArg<2> input, TensorArg<1> global_scale, TensorArg<2> output,
        TensorArg<2> block_scales, float epsilon, bool hi_first,
        cudaStream_t stream);

    void launch_rms_rope_kernel(
        TensorArg<4> q, TensorArg<4> k, TensorArg<6> freqs,
        TensorArg<1> q_scale, TensorArg<1> k_scale, TensorArg<4> q_out,
        TensorArg<4> k_out, int64_t rot_dim, float epsilon, bool has_k,
        bool split_half, cudaStream_t stream);

    void launch_dequantize_nvfp4_kernel(
        TensorArg<2> input, TensorArg<1> global_scale,
        TensorArg<2> block_scales, TensorArg<2> output, bool hi_first,
        cudaStream_t stream);

    void launch_quantize_mxfp8_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> block_scales,
        cudaStream_t stream);

    // SageAttention kernel launchers
    void launch_quant_qk_per_thread_int8(
        TensorArg<4> q, TensorArg<4> q_int8, TensorArg<3> q_scale,
        TensorArg<4> k, TensorArg<4> k_int8, TensorArg<3> k_scale,
        int BLKQ, int WARPQ, int BLKK, int WARPK, TensorArg<1> anchor_indices,
        cudaStream_t stream);

    void launch_quant_v_int8_kernel(
        TensorArg<4> v, TensorArg<2> out, TensorArg<1> scale, int padded_N, cudaStream_t stream);

    void launch_sage_attn_kernel(
        TensorArg<4> q, TensorArg<4> k, TensorArg<4> v, TensorArg<4> o,
        TensorArg<1> q_scale, TensorArg<1> k_scale, TensorArg<1> v_scale,
        TensorArg<4> mask, int cta_k, float sm_scale, cudaStream_t stream);

    // SVDQuant W4A4 — see ops/quantize_svdquant_w4a4.cu
    void launch_svdquant_quantize_w4a4_kernel(
        TensorArg<2> x, TensorArg<1> smooth, TensorArg<2> lora_down,
        TensorArg<2> q_x, TensorArg<2> ascales, TensorArg<2> lora_act,
        int act_unsigned,
        cudaStream_t stream);

    // SVDQuant W4A4 — see ops/scaled_mm_svdquant_w4a4.cu
    void launch_svdquant_scaled_mm_w4a4_kernel(
        TensorArg<2> act, TensorArg<1> wgt, TensorArg<2> ascales,
        TensorArg<1> wscales, TensorArg<2> lora_act_in,
        TensorArg<1> lora_up, TensorArg<1> bias, TensorArg<2> out,
        int act_unsigned,
        int tile_packed,
        int fast_accum,
        int shared_scale,
        int fuse_lora,
        cudaStream_t stream);

    // AWQ W4A16 — see ops/awq_w4a16.cu. Internal M-routing picks
    // gemv (M ≤ 8) vs gemm path; bias / LoRA-up are applied externally.
    void launch_awq_w4a16_kernel(
        TensorArg<2> x, TensorArg<2> qweight, TensorArg<2> wscales,
        TensorArg<2> wzeros, TensorArg<2> out, int G,
        cudaStream_t stream);

    // Fused 3D neighborhood attention — see ops/na3d.cu.
    void launch_na3d_kernel(
        TensorArg<6> q, TensorArg<6> k, TensorArg<6> v, TensorArg<6> out,
        int kt, int kh, int kw, int causal_t, int causal_h, int causal_w,
        float scale, cudaStream_t stream);

    // Fused AdaLN — see ops/adaln.cu. subtract_mean selects LayerNorm (true)
    // or RMSNorm (false) statistics.
    void launch_adaln_kernel(
        TensorArg<2> x, TensorArg<2> scale, TensorArg<2> shift,
        TensorArg<2> out, int64_t scale_group, int64_t shift_group, float eps,
        bool subtract_mean, cudaStream_t stream);
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
    auto input_arg = make_flat_tensor_arg(input);
    auto scale_arg = make_flat_tensor_arg(scale);
    auto output_arg = make_flat_tensor_arg(output);
    if (input_dtype_code < 0 || input_dtype_code > 2 ||
        input_dtype_code != static_cast<int>(input_arg.meta.dtype)) {
        throw std::runtime_error("Unsupported input dtype for quantize_per_tensor_fp8");
    }
    if (output_dtype_code < 5 || output_dtype_code > 6) {
        throw std::runtime_error("Unsupported output dtype for quantize_per_tensor_fp8");
    }
    if (numel < 0 || input_arg.meta.sizes[0] < numel ||
        output_arg.meta.sizes[0] < numel || scale_arg.meta.sizes[0] < 1 ||
        scale_arg.meta.dtype != DType::Float32 ||
        output_arg.meta.dtype != DType::UInt8) {
        throw std::runtime_error("Invalid tensor layout for quantize_per_tensor_fp8");
    }
    input_arg.meta.sizes[0] = numel;
    output_arg.meta.dtype = static_cast<DType>(output_dtype_code);
    launch_quantize_fp8_kernel(
        input_arg, scale_arg, output_arg,
        reinterpret_cast<cudaStream_t>(stream_ptr));
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
    auto input_arg = make_flat_tensor_arg(input);
    auto scale_arg = make_flat_tensor_arg(scale);
    auto output_arg = make_flat_tensor_arg(output);
    if (input_dtype_code != 5 && input_dtype_code != 6) {
        throw std::runtime_error("Unsupported input dtype code for dequantize_per_tensor_fp8 (must be 5 or 6)");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2 ||
        output_dtype_code != static_cast<int>(output_arg.meta.dtype)) {
        throw std::runtime_error("Unsupported output dtype for dequantize_per_tensor_fp8 (must be float32, float16, or bfloat16)");
    }
    if (numel < 0 || input_arg.meta.sizes[0] < numel ||
        output_arg.meta.sizes[0] < numel || scale_arg.meta.sizes[0] < 1 ||
        scale_arg.meta.dtype != DType::Float32 ||
        input_arg.meta.dtype != DType::UInt8) {
        throw std::runtime_error("Invalid tensor layout for dequantize_per_tensor_fp8");
    }
    input_arg.meta.sizes[0] = numel;
    input_arg.meta.dtype = static_cast<DType>(input_dtype_code);
    launch_dequantize_fp8_kernel(
        input_arg, scale_arg, output_arg,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

void stochastic_round_fp8(
    nb::ndarray<nb::device::cuda> rng_and_output,
    nb::ndarray<nb::device::cuda> input,
    int output_dtype_code,
    int64_t numel,
    uintptr_t stream_ptr) {
    auto rng_arg = make_flat_tensor_arg(rng_and_output);
    auto input_arg = make_flat_tensor_arg(input);
    if (rng_arg.meta.dtype != DType::UInt8) {
        throw std::runtime_error("stochastic_round_fp8 requires uint8 RNG storage");
    }
    const int input_dtype_code = static_cast<int>(input_arg.meta.dtype);
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for stochastic_round_fp8");
    }
    if (output_dtype_code < 5 || output_dtype_code > 6) {
        throw std::runtime_error("Unsupported output dtype for stochastic_round_fp8");
    }
    if (numel < 0 || input_arg.meta.sizes[0] < numel ||
        rng_arg.meta.sizes[0] < numel) {
        throw std::runtime_error("Invalid tensor layout for stochastic_round_fp8");
    }
    input_arg.meta.sizes[0] = numel;
    rng_arg.meta.dtype = static_cast<DType>(output_dtype_code);
    launch_stochastic_round_fp8_kernel(
        rng_arg, input_arg,
        reinterpret_cast<cudaStream_t>(stream_ptr));
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

    auto b_arg = make_contiguous_tensor_arg<2>(b);
    auto b_scale_arg = make_contiguous_tensor_arg<2>(block_scale_b);
    auto a_arg = make_contiguous_tensor_arg<2>(a);
    auto a_scale_arg = make_contiguous_tensor_arg<2>(block_scale_a);
    auto out_arg = make_contiguous_tensor_arg<2>(out);
    auto bias_arg = make_flat_tensor_arg(bias);
    auto alpha_arg = make_flat_tensor_arg(alpha);
    auto workspace_arg = make_flat_tensor_arg(workspace);

    if (a_arg.meta.sizes[1] != b_arg.meta.sizes[1]) {
        throw std::runtime_error("Matrix dimensions do not match");
    }


    // Validate output dtype code (0=float32, 1=float16, 2=bfloat16)
    if (out_dtype_code < 0 || out_dtype_code > 2 ||
        out_dtype_code != static_cast<int>(out_arg.meta.dtype)) {
        throw std::runtime_error("Invalid output dtype code");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_cublas_gemm_blockwise_fp4_kernel(
        b_arg, b_scale_arg, a_arg, a_scale_arg, out_arg, bias_arg,
        alpha_arg, workspace_arg, accumulate, stream);
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
    auto input_arg = make_contiguous_tensor_arg<2>(input);
    auto global_scale_arg = make_flat_tensor_arg(global_scale);
    auto output_arg = make_contiguous_tensor_arg<2>(output);
    auto block_scales_arg = make_contiguous_tensor_arg<2>(block_scales);
    const int64_t num_rows =
        pad_16x ? (input_arg.meta.sizes[0] + 15) / 16 * 16
                : input_arg.meta.sizes[0];
    const int64_t num_cols =
        pad_16x ? (input_arg.meta.sizes[1] + 15) / 16 * 16
                : input_arg.meta.sizes[1];
    const int input_dtype_code = static_cast<int>(input_arg.meta.dtype);
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for FP4 quantization (must be float32, float16, or bfloat16)");
    }
    if (output_arg.meta.sizes[0] != num_rows ||
        output_arg.meta.sizes[1] * 2 != num_cols ||
        output_arg.meta.dtype != DType::UInt8 ||
        block_scales_arg.meta.dtype != DType::UInt8 ||
        global_scale_arg.meta.sizes[0] < 1 ||
        global_scale_arg.meta.dtype != DType::Float32) {
        throw std::runtime_error("Invalid tensor layout for FP4 quantization");
    }
    launch_quantize_nvfp4_kernel(
        input_arg, global_scale_arg, output_arg, block_scales_arg, epsilon,
        hi_first, reinterpret_cast<cudaStream_t>(stream_ptr));
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
    auto input_arg = make_contiguous_tensor_arg<2>(input);
    auto global_scale_arg = make_flat_tensor_arg(global_scale);
    auto block_scales_arg = make_contiguous_tensor_arg<2>(block_scales);
    auto output_arg = make_contiguous_tensor_arg<2>(output);
    if (output_dtype_code < 0 || output_dtype_code > 2 ||
        output_dtype_code != static_cast<int>(output_arg.meta.dtype)) {
        throw std::runtime_error("Unsupported output dtype for FP4 dequantization (must be float32, float16, or bfloat16)");
    }
    if (input_arg.meta.sizes[0] != output_arg.meta.sizes[0] ||
        input_arg.meta.sizes[1] * 2 != output_arg.meta.sizes[1] ||
        input_arg.meta.dtype != DType::UInt8 ||
        block_scales_arg.meta.dtype != DType::UInt8 ||
        global_scale_arg.meta.sizes[0] < 1 ||
        global_scale_arg.meta.dtype != DType::Float32) {
        throw std::runtime_error("Invalid tensor layout for FP4 dequantization");
    }
    launch_dequantize_nvfp4_kernel(
        input_arg, global_scale_arg, block_scales_arg, output_arg, hi_first,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Nanobind wrapper for quantize_mxfp8
void quantize_mxfp8(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::device::cuda> output,
    nb::ndarray<nb::device::cuda> block_scales,
    bool pad_32x,
    uintptr_t stream_ptr) {
    auto input_arg = make_contiguous_tensor_arg<2>(input);
    auto output_arg = make_contiguous_tensor_arg<2>(output);
    auto block_scales_arg = make_contiguous_tensor_arg<2>(block_scales);
    const int64_t num_rows =
        pad_32x ? (input_arg.meta.sizes[0] + 31) / 32 * 32
                : input_arg.meta.sizes[0];
    const int64_t num_cols =
        pad_32x ? (input_arg.meta.sizes[1] + 31) / 32 * 32
                : input_arg.meta.sizes[1];
    const int input_dtype_code = static_cast<int>(input_arg.meta.dtype);
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for MXFP8 quantization (must be float32, float16, or bfloat16)");
    }
    if (output_arg.meta.sizes[0] != num_rows ||
        output_arg.meta.sizes[1] != num_cols ||
        output_arg.meta.dtype != DType::Float8E4M3 ||
        block_scales_arg.meta.dtype != DType::UInt8) {
        throw std::runtime_error("Invalid tensor layout for MXFP8 quantization");
    }
    launch_quantize_mxfp8_kernel(
        input_arg, output_arg, block_scales_arg,
        reinterpret_cast<cudaStream_t>(stream_ptr));
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

    if (xq.ndim() != 4 || freqs.ndim() != 6) {
        throw std::runtime_error("apply_rope requires a 4D input and 6D freqs");
    }
    // Get xq dimensions: (batch, dim1, dim2, head_dim) - layout agnostic
    int64_t batch = xq.shape(0);
    int64_t dim1 = xq.shape(1);
    int64_t dim2 = xq.shape(2);
    int64_t head_dim = xq.shape(3);
    if (head_dim == 0 || head_dim % 2 != 0) {
        throw std::runtime_error(
            "apply_rope requires a positive, even head dimension");
    }

    // Get freqs dimensions (for broadcasting)
    int64_t freqs_batch = freqs.shape(0);
    int64_t freqs_dim1 = freqs.shape(1);
    int64_t freqs_dim2 = freqs.shape(2);

    // Validate broadcast and trailing rotation dimensions.
    if ((freqs_batch != 1 && freqs_batch != batch) ||
        (freqs_dim1 != 1 && freqs_dim1 != dim1) ||
        (freqs_dim2 != 1 && freqs_dim2 != dim2) ||
        freqs.shape(3) != head_dim / 2 ||
        freqs.shape(4) != 2 || freqs.shape(5) != 2) {
        throw std::runtime_error("apply_rope freqs shape is not broadcastable to input");
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
    
    TensorArg<4> xk_arg{};
    TensorArg<4> xk_out_arg{};
    
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
        
        xk_arg = make_tensor_arg<4>(xk);
        xk_out_arg = make_tensor_arg<4>(xk_out);
        if (map_dtype_to_code(xk.dtype()) != map_dtype_to_code(xq.dtype()) ||
            map_dtype_to_code(xk_out.dtype()) != map_dtype_to_code(xq.dtype())) {
            throw std::runtime_error("apply_rope inputs and outputs must share dtype");
        }
    }

    // Get input dtype code
    int input_dtype_code = map_dtype_to_code(xq.dtype());
    int output_dtype_code = map_dtype_to_code(xq_out.dtype());
    if ((input_dtype_code != 1 && input_dtype_code != 2) ||
        output_dtype_code != input_dtype_code) {
        throw std::runtime_error(
            "apply_rope inputs and outputs must share an FP16/BF16 dtype");
    }

    // Get freqs dtype code
    int freqs_dtype_code = map_dtype_to_code(freqs.dtype());
    if (freqs_dtype_code < 0 || freqs_dtype_code > 2) {
        throw std::runtime_error(
            "apply_rope frequencies must be FP32, FP16, or BF16");
    }

    launch_apply_rope_kernel(
        make_tensor_arg<4>(xq), xk_arg, make_tensor_arg<6>(freqs),
        make_tensor_arg<4>(xq_out), xk_out_arg, has_xk, split_half,
        reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Nanobind wrapper for paired fused RMSNorm + RoPE.
void rms_rope(nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> k,
              nb::ndarray<nb::device::cuda> freqs,
              nb::ndarray<nb::device::cuda> q_scale,
              nb::ndarray<nb::device::cuda> k_scale,
              nb::ndarray<nb::device::cuda> q_out,
              nb::ndarray<nb::device::cuda> k_out, float epsilon,
              uintptr_t stream_ptr, bool split_half = false,
              int64_t rot_dim = 0) {

  if (q.ndim() != 4 || k.ndim() != 4 || q_out.ndim() != 4 ||
      k_out.ndim() != 4) {
    throw std::runtime_error(
        "rms_rope Q/K inputs and outputs must be 4D BHND or BNHD tensors");
  }
  for (int axis = 0; axis < 4; ++axis) {
    if (k.shape(axis) != q.shape(axis) || q_out.shape(axis) != q.shape(axis) ||
        k_out.shape(axis) != q.shape(axis)) {
      throw std::runtime_error(
          "rms_rope Q/K input and output shapes must match");
    }
  }

  const int64_t batch = q.shape(0);
  const int64_t dim1 = q.shape(1);
  const int64_t dim2 = q.shape(2);
  const int64_t head_dim = q.shape(3);
  if (head_dim < 32 || head_dim % 32 != 0) {
    throw std::runtime_error(
        "native rms_rope requires head_dim to be a positive multiple of 32");
  }
  // rot_dim restricts the rotation to a head-dim prefix (partial rotary); the
  // norm always spans the full head_dim. 0 means rotate everything.
  const int64_t rot = rot_dim > 0 ? rot_dim : head_dim;
  if (rot % 2 != 0 || rot > head_dim) {
    throw std::runtime_error(
        "rms_rope rot_dim must be an even value <= head_dim");
  }
  if (freqs.ndim() != 6 || (freqs.shape(0) != 1 && freqs.shape(0) != batch) ||
      (freqs.shape(1) != 1 && freqs.shape(1) != dim1) ||
      (freqs.shape(2) != 1 && freqs.shape(2) != dim2) ||
      freqs.shape(3) != rot / 2 || freqs.shape(4) != 2 ||
      freqs.shape(5) != 2) {
    throw std::runtime_error(
        "rms_rope freqs shape must broadcast to Q/K");
  }
  if (q_scale.ndim() != 1 || k_scale.ndim() != 1 ||
      q_scale.shape(0) != head_dim || k_scale.shape(0) != head_dim) {
    throw std::runtime_error(
        "rms_rope scales must be 1D tensors of length head_dim");
  }

  const int input_dtype_code = map_dtype_to_code(q.dtype());
  const int k_dtype_code = map_dtype_to_code(k.dtype());
  const int q_out_dtype_code = map_dtype_to_code(q_out.dtype());
  const int k_out_dtype_code = map_dtype_to_code(k_out.dtype());
  const int freqs_dtype_code = map_dtype_to_code(freqs.dtype());
  const int scale_dtype_code = map_dtype_to_code(q_scale.dtype());
  const int k_scale_dtype_code = map_dtype_to_code(k_scale.dtype());
  if ((input_dtype_code != 1 && input_dtype_code != 2) ||
      input_dtype_code != k_dtype_code ||
      input_dtype_code != q_out_dtype_code ||
      input_dtype_code != k_out_dtype_code) {
    throw std::runtime_error(
        "rms_rope Q/K inputs and outputs must share an FP16/BF16 dtype");
  }
  if (freqs_dtype_code < 0 || scale_dtype_code < 0 ||
      scale_dtype_code != k_scale_dtype_code) {
    throw std::runtime_error("rms_rope frequencies/scales must be FP32, FP16, "
                             "or BF16; scale dtypes must match");
  }

  launch_rms_rope_kernel(
      make_tensor_arg<4>(q), make_tensor_arg<4>(k), make_tensor_arg<6>(freqs),
      make_tensor_arg<1>(q_scale), make_tensor_arg<1>(k_scale),
      make_tensor_arg<4>(q_out), make_tensor_arg<4>(k_out), rot, epsilon, true,
      split_half, reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Nanobind wrapper for single-tensor fused RMSNorm + RoPE.
void rms_rope1(nb::ndarray<nb::device::cuda> q,
               nb::ndarray<nb::device::cuda> freqs,
               nb::ndarray<nb::device::cuda> q_scale,
               nb::ndarray<nb::device::cuda> q_out, float epsilon,
               uintptr_t stream_ptr, bool split_half = false) {

  if (q.ndim() != 4 || q_out.ndim() != 4) {
    throw std::runtime_error(
        "rms_rope1 input and output must be 4D BHND or BNHD tensors");
  }
  for (int axis = 0; axis < 4; ++axis) {
    if (q_out.shape(axis) != q.shape(axis)) {
      throw std::runtime_error("rms_rope1 output shape must match input shape");
    }
  }

  const int64_t batch = q.shape(0);
  const int64_t dim1 = q.shape(1);
  const int64_t dim2 = q.shape(2);
  const int64_t head_dim = q.shape(3);
  if (head_dim < 32 || head_dim % 32 != 0) {
    throw std::runtime_error(
        "native rms_rope1 requires head_dim to be a positive multiple of 32");
  }
  if (freqs.ndim() != 6 || (freqs.shape(0) != 1 && freqs.shape(0) != batch) ||
      (freqs.shape(1) != 1 && freqs.shape(1) != dim1) ||
      (freqs.shape(2) != 1 && freqs.shape(2) != dim2) ||
      freqs.shape(3) != head_dim / 2 || freqs.shape(4) != 2 ||
      freqs.shape(5) != 2) {
    throw std::runtime_error(
        "rms_rope1 freqs shape must broadcast to input");
  }
  if (q_scale.ndim() != 1 || q_scale.shape(0) != head_dim) {
    throw std::runtime_error(
        "rms_rope1 scale must be a 1D tensor of length head_dim");
  }

  const int input_dtype_code = map_dtype_to_code(q.dtype());
  const int out_dtype_code = map_dtype_to_code(q_out.dtype());
  const int freqs_dtype_code = map_dtype_to_code(freqs.dtype());
  const int scale_dtype_code = map_dtype_to_code(q_scale.dtype());
  if ((input_dtype_code != 1 && input_dtype_code != 2) ||
      input_dtype_code != out_dtype_code) {
    throw std::runtime_error(
        "rms_rope1 input/output must share an FP16/BF16 dtype");
  }
  if (freqs_dtype_code < 0 || scale_dtype_code < 0) {
    throw std::runtime_error(
        "rms_rope1 frequencies and scale must be FP32, FP16, or BF16");
  }

  launch_rms_rope_kernel(
      make_tensor_arg<4>(q), {}, make_tensor_arg<6>(freqs),
      make_tensor_arg<1>(q_scale), {}, make_tensor_arg<4>(q_out), {}, head_dim,
      epsilon, false, split_half, reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Nanobind wrapper: signed INT8 V quantization
void quant_v_int8(
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> out,
    nb::ndarray<nb::device::cuda> scale,
    int padded_n,
    int input_dtype_code,
    uintptr_t stream_ptr)
{
    if (v.ndim() != 4) {
        throw std::runtime_error("quant_v_int8: v must be 4D [B,H,N,D]");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto v_arg = make_tensor_arg<4>(v);
    if (input_dtype_code != static_cast<int>(v_arg.meta.dtype)) {
        throw std::runtime_error("quant_v_int8: input dtype code does not match v");
    }
    launch_quant_v_int8_kernel(
        v_arg, make_contiguous_tensor_arg<2>(out),
        make_contiguous_tensor_arg<1>(scale), padded_n, stream);
}

// Nanobind wrapper: stabilized INT8 Q/K per-thread quant (contiguous HND layout)
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
    uintptr_t anchor_indices_ptr)
{
    if (q.ndim() != 4 || k.ndim() != 4) {
        throw std::runtime_error("quant_qk_per_thread_int8: q and k must be 4D [B,H,L,D]");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto q_arg = make_tensor_arg<4>(q);
    auto k_arg = make_tensor_arg<4>(k);
    if (input_dtype_code != static_cast<int>(q_arg.meta.dtype) ||
        k_arg.meta.dtype != q_arg.meta.dtype) {
        throw std::runtime_error("quant_qk_per_thread_int8: input dtype mismatch");
    }
    launch_quant_qk_per_thread_int8(
        q_arg, make_contiguous_tensor_arg<4>(q_int8),
        make_contiguous_tensor_arg<3>(q_scale), k_arg,
        make_contiguous_tensor_arg<4>(k_int8),
        make_contiguous_tensor_arg<3>(k_scale), BLKQ, WARPQ, BLKK, WARPK,
        make_int32_scratch_arg(anchor_indices_ptr,
            q_arg.meta.sizes[0] * k_arg.meta.sizes[1]), stream);
}

// Quantization half of the split INT8 SDPA API.  This deliberately launches
// the same Q/K and V kernels with the same tiling as sage_sdpa below, so moving
// the attention launch after the caller releases its input tensors does not
// change any numerical results.
void sage_sdpa_quantize(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_int8,
    nb::ndarray<nb::device::cuda> v_scale,
    int cta_k,
    int input_dtype_code,
    uintptr_t stream_ptr,
    uintptr_t anchor_indices_ptr)
{
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
        throw std::runtime_error(
            "sage_sdpa_quantize: q, k, and v must be 4D [B,H,L,D]");
    }
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error("sage_sdpa_quantize: cta_k must be 64 or 128");
    }
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error(
            "sage_sdpa_quantize: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    }
    if (!anchor_indices_ptr) {
        throw std::runtime_error(
            "sage_sdpa_quantize: anchor_indices scratch is required");
    }

    const int B = static_cast<int>(q.shape(0));
    const int H_q = static_cast<int>(q.shape(1));
    const int Lq = static_cast<int>(q.shape(2));
    const int D = static_cast<int>(q.shape(3));
    const int H_kv = static_cast<int>(k.shape(1));
    const int Lk = static_cast<int>(k.shape(2));
    const int padded_Lk = ((Lk + cta_k - 1) / cta_k) * cta_k;

    if (cta_k == 128 && D == 64) {
        throw std::runtime_error(
            "sage_sdpa_quantize: cta_k 128 is unsupported for head_dim 64");
    }

    if (k.shape(0) != B || v.shape(0) != B || v.shape(1) != H_kv ||
        v.shape(2) != Lk || k.shape(3) != D || v.shape(3) != D) {
        throw std::runtime_error("sage_sdpa_quantize: incompatible q, k, and v shapes");
    }
    if (q_int8.ndim() != 4 || k_int8.ndim() != 4 || v_int8.ndim() != 2 ||
        q_int8.shape(0) != B || q_int8.shape(1) != H_q ||
        q_int8.shape(2) != Lq || q_int8.shape(3) != D ||
        k_int8.shape(0) != B || k_int8.shape(1) != H_kv ||
        k_int8.shape(2) != Lk || k_int8.shape(3) != D ||
        v_int8.shape(0) != static_cast<size_t>(B) * H_kv * D ||
        v_int8.shape(1) != padded_Lk) {
        throw std::runtime_error("sage_sdpa_quantize: incompatible INT8 output shapes");
    }

    constexpr int BLKQ = 128;
    const int WARPQ = D == 256 ? 16 : 32;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_quant_qk_per_thread_int8(
        make_tensor_arg<4>(q), make_contiguous_tensor_arg<4>(q_int8),
        make_contiguous_tensor_arg<3>(q_scale), make_tensor_arg<4>(k),
        make_contiguous_tensor_arg<4>(k_int8),
        make_contiguous_tensor_arg<3>(k_scale), BLKQ, WARPQ, cta_k, cta_k,
        make_int32_scratch_arg(anchor_indices_ptr, B * H_kv), stream);

    launch_quant_v_int8_kernel(
        make_tensor_arg<4>(v), make_contiguous_tensor_arg<2>(v_int8),
        make_contiguous_tensor_arg<1>(v_scale), padded_Lk, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sage_sdpa_quantize kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

// Attention half of the split INT8 SDPA API.  The input tensors use the exact
// packed layouts produced by sage_sdpa_quantize; no floating-point Q/K/V
// tensor is retained or reconstructed.
void sage_sdpa_prequantized(
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> v_int8,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_scale,
    int cta_k,
    float sm_scale,
    int output_dtype_code,
    uintptr_t stream_ptr,
    std::optional<nb::ndarray<nb::device::cuda>> attn_mask = std::nullopt)
{
    if (q_int8.ndim() != 4 || k_int8.ndim() != 4 ||
        v_int8.ndim() != 2 || o.ndim() != 4) {
        throw std::runtime_error(
            "sage_sdpa_prequantized: q/k/o must be 4D and packed v must be 2D");
    }
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error("sage_sdpa_prequantized: cta_k must be 64 or 128");
    }
    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error(
            "sage_sdpa_prequantized: output_dtype_code must be 1 (fp16) or 2 (bf16)");
    }

    const int B = static_cast<int>(q_int8.shape(0));
    const int H_q = static_cast<int>(q_int8.shape(1));
    const int Lq = static_cast<int>(q_int8.shape(2));
    const int D = static_cast<int>(q_int8.shape(3));
    const int H_kv = static_cast<int>(k_int8.shape(1));
    const int Lk = static_cast<int>(k_int8.shape(2));
    const int padded_Lk = ((Lk + cta_k - 1) / cta_k) * cta_k;

    if (cta_k == 128 && (D == 64 || attn_mask.has_value())) {
        throw std::runtime_error(
            "sage_sdpa_prequantized: cta_k 128 requires unmasked head_dim 128 or 256");
    }

    if (k_int8.shape(0) != B || k_int8.shape(3) != D ||
        o.shape(0) != B || o.shape(1) != H_q || o.shape(2) != Lq ||
        o.shape(3) != D ||
        v_int8.shape(0) != static_cast<size_t>(B) * H_kv * D ||
        v_int8.shape(1) != padded_Lk) {
        throw std::runtime_error(
            "sage_sdpa_prequantized: incompatible quantized tensor shapes");
    }
    if (q_int8.stride(3) != 1 || q_int8.stride(2) != D ||
        q_int8.stride(1) != static_cast<int64_t>(Lq) * D ||
        k_int8.stride(3) != 1 || k_int8.stride(2) != D ||
        k_int8.stride(1) != static_cast<int64_t>(Lk) * D ||
        v_int8.stride(1) != 1 || v_int8.stride(0) != padded_Lk ||
        o.stride(3) != 1 || o.stride(2) != D ||
        o.stride(1) != static_cast<int64_t>(Lq) * D) {
        throw std::runtime_error(
            "sage_sdpa_prequantized: quantized tensors and output must be contiguous");
    }

    TensorArg<4> mask_arg{};
    if (attn_mask.has_value()) {
        const auto &mask = attn_mask.value();
        if (mask.ndim() != 4 || mask.shape(0) != B || mask.shape(1) != H_q ||
            mask.shape(2) != Lq || mask.shape(3) != Lk) {
            throw std::runtime_error(
                "sage_sdpa_prequantized: attention mask must be expanded to [B,H_q,Lq,Lk]");
        }
        mask_arg = make_tensor_arg<4>(mask);
        if (mask_arg.meta.dtype != DType::Bool &&
            mask_arg.meta.dtype != DType::Float16 &&
            mask_arg.meta.dtype != DType::BFloat16 &&
            mask_arg.meta.dtype != DType::Float32) {
            throw std::runtime_error(
                "sage_sdpa_prequantized: attention mask must be bool, float16, bfloat16, or float32");
        }
    }

    auto o_arg = make_contiguous_tensor_arg<4>(o);
    if (output_dtype_code != static_cast<int>(o_arg.meta.dtype)) {
        throw std::runtime_error("sage_sdpa_prequantized: output dtype mismatch");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_sage_attn_kernel(
        make_contiguous_tensor_arg<4>(q_int8),
        make_contiguous_tensor_arg<4>(k_int8),
        make_packed_sage_v_arg(v_int8, B, H_kv, D, padded_Lk), o_arg,
        make_flat_tensor_arg(q_scale), make_flat_tensor_arg(k_scale),
        make_flat_tensor_arg(v_scale), mask_arg, cta_k, sm_scale, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sage_sdpa_prequantized kernel launch failed: ") +
            cudaGetErrorString(err));
    }
}

// Nanobind wrapper: pure INT8 QK / U8-softmax / INT8-V attention kernel
void sage_attn(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_scale,
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

    constexpr int CTA_K = 64;
    const int64_t padded_k_length =
        ((static_cast<int64_t>(k.shape(2)) + CTA_K - 1) / CTA_K) * CTA_K;
    if (v.shape(3) < padded_k_length || v.shape(3) % CTA_K != 0) {
        throw std::runtime_error(
            "sage_attn: packed V sequence extent must cover K and be a multiple of 64");
    }

    auto o_arg = make_tensor_arg<4>(o);
    if (output_dtype_code != static_cast<int>(o_arg.meta.dtype)) {
        throw std::runtime_error("sage_attn: output dtype mismatch");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_sage_attn_kernel(
        make_tensor_arg<4>(q), make_tensor_arg<4>(k), make_tensor_arg<4>(v),
        o_arg, make_flat_tensor_arg(q_scale), make_flat_tensor_arg(k_scale),
        make_flat_tensor_arg(v_scale), TensorArg<4>{}, CTA_K, sm_scale, stream);
}

// Fused SageAttention SDPA: quant_qk + quant_v + sage_attn in one C++ call.
// All scratch buffers are pre-allocated by the caller (Python frontend).
void sage_sdpa(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> o,
    nb::ndarray<nb::device::cuda> q_int8,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_int8,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> v_int8,
    nb::ndarray<nb::device::cuda> v_scale,
    float sm_scale,
    int input_dtype_code,
    int output_dtype_code,
    uintptr_t stream_ptr,
    uintptr_t anchor_indices_ptr,
    std::optional<nb::ndarray<nb::device::cuda>> attn_mask = std::nullopt,
    int cta_k = 0)
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

    TensorArg<4> mask_arg{};
    if (attn_mask.has_value()) {
        const auto &mask = attn_mask.value();
        if (mask.ndim() != 4 || mask.shape(0) != B || mask.shape(1) != H_q ||
            mask.shape(2) != Lq || mask.shape(3) != Lk) {
            throw std::runtime_error(
                "sage_sdpa: attention mask must be expanded to [B,H_q,Lq,Lk]");
        }
        mask_arg = make_tensor_arg<4>(mask);
        if (mask_arg.meta.dtype != DType::Bool &&
            mask_arg.meta.dtype != DType::Float16 &&
            mask_arg.meta.dtype != DType::BFloat16 &&
            mask_arg.meta.dtype != DType::Float32) {
            throw std::runtime_error(
                "sage_sdpa: attention mask must be bool, float16, bfloat16, or float32");
        }
    }

    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("sage_sdpa: input_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    }
    if (output_dtype_code != 1 && output_dtype_code != 2) {
        throw std::runtime_error(
            "sage_sdpa: output_dtype_code must be 1 (fp16) or 2 (bf16)");
    }
    if (cta_k == 0) {
        cta_k = !attn_mask.has_value() && D >= 128 && Lk > 1024
            ? 128
            : 64;
    }
    if (cta_k != 64 && cta_k != 128) {
        throw std::runtime_error("sage_sdpa: cta_k must be 64 or 128");
    }
    if (cta_k == 128 && (D == 64 || attn_mask.has_value())) {
        throw std::runtime_error(
            "sage_sdpa: cta_k 128 requires unmasked head_dim 128 or 256");
    }
    if (!anchor_indices_ptr) {
        throw std::runtime_error(
            "sage_sdpa: anchor_indices scratch is required");
    }
    constexpr int BLKQ = 128;
    const int WARPQ = D == 256 ? 16 : 32;
    const int BLKK = cta_k;
    const int WARPK = cta_k;
    const int padded_Lk = ((Lk + cta_k - 1) / cta_k) * cta_k;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    launch_quant_qk_per_thread_int8(
        make_tensor_arg<4>(q), make_contiguous_tensor_arg<4>(q_int8),
        make_contiguous_tensor_arg<3>(q_scale), make_tensor_arg<4>(k),
        make_contiguous_tensor_arg<4>(k_int8),
        make_contiguous_tensor_arg<3>(k_scale), BLKQ, WARPQ, BLKK, WARPK,
        make_int32_scratch_arg(anchor_indices_ptr, B * H_kv), stream);

    launch_quant_v_int8_kernel(
        make_tensor_arg<4>(v), make_contiguous_tensor_arg<2>(v_int8),
        make_contiguous_tensor_arg<1>(v_scale), padded_Lk, stream);

    auto o_arg = make_contiguous_tensor_arg<4>(o);
    if (output_dtype_code != static_cast<int>(o_arg.meta.dtype)) {
        throw std::runtime_error("sage_sdpa: output dtype mismatch");
    }
    launch_sage_attn_kernel(
        make_contiguous_tensor_arg<4>(q_int8),
        make_contiguous_tensor_arg<4>(k_int8),
        make_packed_sage_v_arg(v_int8, B, H_kv, D, padded_Lk), o_arg,
        make_flat_tensor_arg(q_scale), make_flat_tensor_arg(k_scale),
        make_flat_tensor_arg(v_scale), mask_arg, cta_k, sm_scale, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("sage_sdpa kernel launch failed: ") + cudaGetErrorString(err));
    }
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
    auto x_arg = make_contiguous_tensor_arg<2>(x);
    auto smooth_arg = make_contiguous_tensor_arg<1>(smooth);
    auto lora_down_arg = make_contiguous_tensor_arg<2>(lora_down);
    auto q_x_arg = make_contiguous_tensor_arg<2>(q_x);
    auto ascales_arg = make_contiguous_tensor_arg<2>(ascales);
    auto lora_act_arg = make_contiguous_tensor_arg<2>(lora_act);
    if ((x_arg.meta.dtype != DType::Float16 && x_arg.meta.dtype != DType::BFloat16) ||
        smooth_arg.meta.dtype != x_arg.meta.dtype ||
        lora_down_arg.meta.dtype != x_arg.meta.dtype ||
        ascales_arg.meta.dtype != x_arg.meta.dtype ||
        q_x_arg.meta.dtype != DType::UInt8 ||
        (lora_act_arg.meta.dtype != x_arg.meta.dtype &&
         lora_act_arg.meta.dtype != DType::Float32)) {
        throw std::runtime_error("svdquant_quantize_w4a4: incompatible tensor dtypes");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_svdquant_quantize_w4a4_kernel(
        x_arg, smooth_arg, lora_down_arg, q_x_arg, ascales_arg, lora_act_arg,
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
    auto act_arg = make_contiguous_tensor_arg<2>(act);
    auto wgt_arg = make_flat_tensor_arg(wgt);
    auto ascales_arg = make_contiguous_tensor_arg<2>(ascales);
    auto wscales_arg = make_flat_tensor_arg(wscales);
    auto lora_act_arg = make_contiguous_tensor_arg<2>(lora_act_in);
    auto lora_up_arg = make_flat_tensor_arg(lora_up);
    auto bias_arg = make_flat_tensor_arg(bias);
    auto out_arg = make_contiguous_tensor_arg<2>(out);
    const int K = static_cast<int>(act_arg.meta.sizes[1]) * 2;
    const bool tile_packed = (wgt.ndim() == 4);
    if (act_arg.meta.dtype != DType::UInt8 || wgt_arg.meta.dtype != DType::UInt8 ||
        (out_arg.meta.dtype != DType::Float16 && out_arg.meta.dtype != DType::BFloat16) ||
        ascales_arg.meta.dtype != out_arg.meta.dtype ||
        wscales_arg.meta.dtype != out_arg.meta.dtype ||
        lora_up_arg.meta.dtype != out_arg.meta.dtype ||
        (bias_arg.meta.sizes[0] > 0 && bias_arg.meta.dtype != out_arg.meta.dtype)) {
        throw std::runtime_error("svdquant_scaled_mm_w4a4: incompatible tensor dtypes");
    }
    if (fuse_lora && lora_act_arg.meta.dtype != out_arg.meta.dtype) {
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

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_svdquant_scaled_mm_w4a4_kernel(
        act_arg, wgt_arg, ascales_arg, wscales_arg,
        lora_act_arg, lora_up_arg, bias_arg, out_arg,
        static_cast<int>(act_unsigned),
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
    auto x_arg = make_contiguous_tensor_arg<2>(x);
    auto qweight_arg = make_contiguous_tensor_arg<2>(qweight);
    auto wscales_arg = make_contiguous_tensor_arg<2>(wscales);
    auto wzeros_arg = make_contiguous_tensor_arg<2>(wzeros);
    auto out_arg = make_contiguous_tensor_arg<2>(out);
    if (x_arg.meta.dtype != DType::Float16 && x_arg.meta.dtype != DType::BFloat16) {
        throw std::runtime_error("awq_w4a16: only fp16 and bf16 activations supported");
    }
    if (qweight_arg.meta.dtype != DType::UInt8 ||
        wscales_arg.meta.dtype != x_arg.meta.dtype ||
        wzeros_arg.meta.dtype != x_arg.meta.dtype ||
        out_arg.meta.dtype != x_arg.meta.dtype) {
        throw std::runtime_error("awq_w4a16: incompatible tensor dtypes");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_awq_w4a16_kernel(
        x_arg, qweight_arg, wscales_arg, wzeros_arg, out_arg,
        group_size, stream);
}

// Nanobind wrapper for fused 3D neighborhood attention
void na3d(
    nb::ndarray<nb::device::cuda> q,
    nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> v,
    nb::ndarray<nb::device::cuda> out,
    int64_t batch, int64_t t_size, int64_t h_size, int64_t w_size,
    int64_t num_heads, int64_t head_dim,
    int64_t kt, int64_t kh, int64_t kw,
    int causal_t, int causal_h, int causal_w,
    float scale,
    int dtype_code,
    uintptr_t stream_ptr)
{
    auto q_arg = make_contiguous_tensor_arg<6>(q);
    auto k_arg = make_contiguous_tensor_arg<6>(k);
    auto v_arg = make_contiguous_tensor_arg<6>(v);
    auto out_arg = make_contiguous_tensor_arg<6>(out);
    const int64_t expected[6] = {
        batch, t_size, h_size, w_size, num_heads, head_dim};
    for (int axis = 0; axis < 6; ++axis) {
        if (q_arg.meta.sizes[axis] != expected[axis] ||
            k_arg.meta.sizes[axis] != expected[axis] ||
            v_arg.meta.sizes[axis] != expected[axis] ||
            out_arg.meta.sizes[axis] != expected[axis]) {
            throw std::runtime_error("Invalid tensor shape for na3d");
        }
    }
    if (static_cast<int>(q_arg.meta.dtype) != dtype_code ||
        k_arg.meta.dtype != q_arg.meta.dtype ||
        v_arg.meta.dtype != q_arg.meta.dtype ||
        out_arg.meta.dtype != q_arg.meta.dtype) {
        throw std::runtime_error("Invalid tensor dtype for na3d");
    }
    launch_na3d_kernel(
        q_arg, k_arg, v_arg, out_arg, static_cast<int>(kt),
        static_cast<int>(kh), static_cast<int>(kw), causal_t, causal_h,
        causal_w, scale, reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Nanobind wrapper for fused AdaLN (LayerNorm statistics)
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
    auto x_arg = make_contiguous_tensor_arg<2>(x);
    auto scale_arg = make_contiguous_tensor_arg<2>(scale);
    auto shift_arg = make_contiguous_tensor_arg<2>(shift);
    auto out_arg = make_contiguous_tensor_arg<2>(out);
    if (x_arg.meta.sizes[0] != N || x_arg.meta.sizes[1] != D ||
        out_arg.meta.sizes[0] != N || out_arg.meta.sizes[1] != D ||
        scale_arg.meta.sizes[1] != D || shift_arg.meta.sizes[1] != D ||
        static_cast<int>(x_arg.meta.dtype) != dtype_code ||
        scale_arg.meta.dtype != x_arg.meta.dtype ||
        shift_arg.meta.dtype != x_arg.meta.dtype ||
        out_arg.meta.dtype != x_arg.meta.dtype) {
        throw std::runtime_error("Invalid tensor contract for adaln");
    }
    launch_adaln_kernel(
        x_arg, scale_arg, shift_arg, out_arg, scale_group, shift_group, eps,
        true, reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Nanobind wrapper for fused AdaLN with RMSNorm statistics
void rms_adaln(
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
    auto x_arg = make_contiguous_tensor_arg<2>(x);
    auto scale_arg = make_contiguous_tensor_arg<2>(scale);
    auto shift_arg = make_contiguous_tensor_arg<2>(shift);
    auto out_arg = make_contiguous_tensor_arg<2>(out);
    if (x_arg.meta.sizes[0] != N || x_arg.meta.sizes[1] != D ||
        out_arg.meta.sizes[0] != N || out_arg.meta.sizes[1] != D ||
        scale_arg.meta.sizes[1] != D || shift_arg.meta.sizes[1] != D ||
        static_cast<int>(x_arg.meta.dtype) != dtype_code ||
        scale_arg.meta.dtype != x_arg.meta.dtype ||
        shift_arg.meta.dtype != x_arg.meta.dtype ||
        out_arg.meta.dtype != x_arg.meta.dtype) {
        throw std::runtime_error("Invalid tensor contract for rms_adaln");
    }
    launch_adaln_kernel(
        x_arg, scale_arg, shift_arg, out_arg, scale_group, shift_group, eps,
        false, reinterpret_cast<cudaStream_t>(stream_ptr));
}

// Python module definition
extern "C" {
    void launch_cublas_gemm_int8_kernel(
        TensorArg<2> a, TensorArg<2> b, TensorArg<2> c, TensorArg<1> workspace,
        cudaStream_t stream);

    void launch_quantize_int8_rowwise_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> scales,
        bool stochastic,
        uint64_t seed,
        cudaStream_t stream);

    void launch_quantize_int4_rowwise_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> scales,
        bool stochastic,
        uint64_t seed,
        cudaStream_t stream);

    void launch_quantize_int4_rowwise_convrot64_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> scales,
        int group_size,
        bool stochastic,
        uint64_t seed,
        cudaStream_t stream);

    void launch_quantize_int4_rowwise_convrot64_to_int8_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> scales,
        int group_size,
        bool stochastic,
        uint64_t seed,
        cudaStream_t stream);

    void launch_dequantize_int4_convrot64_kernel(
        TensorArg<2> input, TensorArg<1> scales, TensorArg<2> output,
        int group_size,
        cudaStream_t stream);

    void launch_int4_linear_kernel(
        TensorArg<2> act, TensorArg<2> weight, TensorArg<1> x_scales,
        TensorArg<1> weight_scales, TensorArg<1> bias, TensorArg<2> output,
        cudaStream_t stream);

    void launch_unpack_int4_to_int8_kernel(
        TensorArg<2> input, TensorArg<2> output,
        cudaStream_t stream);

    void launch_int4_weight_int8_act_gemv_dequant_kernel(
        TensorArg<2> input, TensorArg<2> weight, TensorArg<2> x_scales,
        TensorArg<1> weight_scales, TensorArg<1> bias, TensorArg<2> output,
        cudaStream_t stream);

    void launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(
        TensorArg<2> input, TensorArg<2> weight, TensorArg<2> x_scales,
        TensorArg<1> weight_scales, TensorArg<1> bias, TensorArg<2> output,
        TensorArg<2> weight_workspace, TensorArg<2> acc_workspace,
        TensorArg<1> cublas_workspace, int64_t chunk_cols,
        bool allow_sm80_cutlass, cudaStream_t stream);

    bool launch_cutlass_int8_dequant(
        TensorArg<2> a, TensorArg<2> b, TensorArg<1> xs, TensorArg<1> ws,
        TensorArg<1> bias, TensorArg<2> output, cudaStream_t stream);

    bool launch_cutlass_int8_dequant_config(
        TensorArg<2> a, TensorArg<2> b, TensorArg<1> xs, TensorArg<1> ws,
        TensorArg<2> output, int config, cudaStream_t stream);

    bool launch_cutlass_turing_int8_dequant(
        TensorArg<2> a, TensorArg<2> b, TensorArg<1> xs, TensorArg<1> ws,
        TensorArg<1> bias, TensorArg<2> output, cudaStream_t stream);

    bool launch_cutlass_int4_dequant(
        TensorArg<2> a, TensorArg<2> b, TensorArg<1> xs, TensorArg<1> ws,
        TensorArg<1> bias, TensorArg<2> output, cudaStream_t stream);

    bool launch_cutlass_turing_int4_dequant(
        TensorArg<2> a, TensorArg<2> b, TensorArg<1> xs, TensorArg<1> ws,
        TensorArg<1> bias, TensorArg<2> output, cudaStream_t stream);

    void launch_dequant_int4_grouped_to_int8(
        TensorArg<2> weight, TensorArg<2> scale, TensorArg<1> codebook,
        TensorArg<2> output, int64_t group_size, cudaStream_t stream);

    void launch_dequant_int4_grouped_to_int8_e4m3(
        TensorArg<2> weight, TensorArg<2> scale, TensorArg<1> codebook,
        TensorArg<2> output, int64_t group_size, cudaStream_t stream);

    bool launch_quantize_w4a8_convrot(
        TensorArg<2> rotated, TensorArg<1> codebook, TensorArg<2> packed,
        TensorArg<2> scale, TensorArg<1> channel_scale, bool stochastic,
        uint64_t seed, cudaStream_t stream);

    bool launch_w4a8_codebook_gemm_chunked(
        TensorArg<2> activation, TensorArg<2> weight, TensorArg<2> scale,
        TensorArg<1> codebook, TensorArg<1> channel_scale,
        TensorArg<1> activation_scale, TensorArg<1> bias, TensorArg<2> workspace,
        TensorArg<2> output, int64_t group_size, int64_t chunk_cols,
        cudaStream_t stream);

    void launch_quantize_int8_rowwise_convrot_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> scales,
        int group_size,
        bool stochastic,
        uint64_t seed,
        cudaStream_t stream);

    void launch_rotate_int8_convrot_weight_kernel(
        TensorArg<2> input, TensorArg<2> output, int group_size,
        cudaStream_t stream);

    void launch_quantize_int8_convrot_staged_kernel(
        TensorArg<2> input, TensorArg<2> rotated, TensorArg<2> partial_absmax,
        TensorArg<2> output, TensorArg<2> scales, int group_size,
        bool stochastic,
        uint64_t seed,
        cudaStream_t stream);

    void launch_quantize_int8_rowwise_convrot64_kernel(
        TensorArg<2> input, TensorArg<2> output, TensorArg<2> scales,
        int group_size,
        bool stochastic,
        int act_code,
        uint64_t seed,
        cudaStream_t stream);

    void launch_dequantize_int8_linear_kernel(
        TensorArg<2> input, TensorArg<2> x_scales, TensorArg<1> weight_scales,
        TensorArg<1> bias, TensorArg<2> output,
        cudaStream_t stream);

    void launch_int8_gemv_dequant_kernel(
        TensorArg<2> input, TensorArg<2> weight, TensorArg<2> x_scales,
        TensorArg<1> weight_scales, TensorArg<1> bias, TensorArg<2> output,
        cudaStream_t stream);

    void launch_dequantize_int8_simple_kernel(
        TensorArg<1> input, TensorArg<1> scales, TensorArg<1> output,
        int64_t inner_dim, int scale_mode,
        cudaStream_t stream);

    void launch_dequantize_int8_convrot_kernel(
        TensorArg<2> input, TensorArg<1> scales, TensorArg<2> output,
        int group_size,
        cudaStream_t stream);

    void launch_flash_decode(
        TensorArg<3> q, TensorArg<4> k, TensorArg<4> v,
        TensorArg<1> kv_lengths, TensorArg<3> output,
        TensorArg<1> softmax_lse, TensorArg<1> softmax_lse_accum,
        TensorArg<1> output_accum, int num_splits,
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
        make_contiguous_tensor_arg<2>(a), make_contiguous_tensor_arg<2>(b),
        make_contiguous_tensor_arg<2>(c), make_flat_tensor_arg(workspace), stream);
}

void quantize_int8_rowwise(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    bool stochastic,
    uint64_t seed,
    uintptr_t stream_ptr) {

    auto input_arg = make_contiguous_tensor_arg<2>(input);
    auto output_arg = make_contiguous_tensor_arg<2>(output);
    auto scales_arg = make_contiguous_tensor_arg<2>(scales);
    const int64_t M = input_arg.meta.sizes[0];
    const int64_t K = input_arg.meta.sizes[1];

    if (output_arg.meta.sizes[0] != M || output_arg.meta.sizes[1] != K) {
        throw std::runtime_error("INT8 rowwise quantization output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT8 rowwise quantization scale shape mismatch");
    }
    const int input_dtype_code = static_cast<int>(input_arg.meta.dtype);
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT8 rowwise quantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_kernel(
        input_arg, output_arg, scales_arg, stochastic, seed, stream);
}

void quantize_int4_rowwise(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    bool stochastic,
    uint64_t seed,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    if (K % 64 != 0) {
        throw std::runtime_error("INT4 rowwise quantization requires K divisible by 64");
    }
    if (output.shape(0) != M || output.shape(1) != K / 2) {
        throw std::runtime_error("INT4 rowwise quantization output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT4 rowwise quantization scale shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT4 rowwise quantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int4_rowwise_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(scales), stochastic, seed, stream);
}

void quantize_int4_rowwise_convrot64(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    int group_size,
    bool stochastic,
    uint64_t seed,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    if (group_size != 16 && group_size != 64 && group_size != 256) {
        throw std::runtime_error("INT4 ConvRot quantization requires group_size 16, 64, or 256");
    }
    if (K % group_size != 0) {
        throw std::runtime_error("INT4 ConvRot quantization requires K divisible by group_size");
    }
    if (output.shape(0) != M || output.shape(1) != K / 2) {
        throw std::runtime_error("INT4 ConvRot quantization output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT4 ConvRot quantization scale shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT4 ConvRot quantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int4_rowwise_convrot64_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(scales), group_size, stochastic, seed, stream);
}

void quantize_int4_rowwise_convrot64_to_int8(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    int group_size,
    bool stochastic,
    uint64_t seed,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("INT4 ConvRot fallback activation output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT4 ConvRot fallback activation scales must have shape [M, 1]");
    }
    if (group_size != 256 || K % group_size != 0) {
        throw std::runtime_error("INT4 ConvRot fallback activation quantization requires group_size 256 and divisible K");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT4 ConvRot fallback activation quantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int4_rowwise_convrot64_to_int8_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(scales),
        group_size,
        stochastic,
        seed,
        stream);
}

void dequantize_int4_convrot64(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<float, nb::ndim<1>, nb::device::cuda> scales,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int group_size,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1) * 2;
    if (group_size != 16 && group_size != 64 && group_size != 256) {
        throw std::runtime_error("INT4 ConvRot dequantization requires group_size 16, 64, or 256");
    }
    if (K % group_size != 0) {
        throw std::runtime_error("INT4 ConvRot dequantization requires K divisible by group_size");
    }
    if (output.shape(0) != M || output.shape(1) != K) {
        throw std::runtime_error("INT4 ConvRot dequantization output shape mismatch");
    }
    if (scales.size() != 1 && scales.size() != static_cast<size_t>(M)) {
        throw std::runtime_error("INT4 ConvRot dequantization scale must be scalar or per-row");
    }
    const int output_dtype_code = map_dtype_to_code(output.dtype());
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for INT4 ConvRot dequantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_int4_convrot64_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_flat_tensor_arg(scales),
        make_contiguous_tensor_arg<2>(output),
        group_size,
        stream);
}

void int4_linear(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> act,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight,
    nb::ndarray<float, nb::device::cuda> x_scales,
    nb::ndarray<float, nb::device::cuda> weight_scales,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    int output_dtype_code,
    uintptr_t stream_ptr) {

    const int64_t M = act.shape(0);
    const int64_t K_half = act.shape(1);
    const int64_t N = weight.shape(0);
    if (weight.shape(1) != K_half) {
        throw std::runtime_error("INT4 linear K dimensions do not match");
    }
    const int64_t K = K_half * 2;
    if (K % 64 != 0) {
        throw std::runtime_error("INT4 linear requires K divisible by 64");
    }
    if (x_scales.size() != static_cast<size_t>(M)) {
        throw std::runtime_error("INT4 linear x_scales must have one value per row");
    }
    if (weight_scales.size() != static_cast<size_t>(N)) {
        throw std::runtime_error("INT4 linear weight_scales must have one value per output channel");
    }
    if (output.shape(0) != M || output.shape(1) != N) {
        throw std::runtime_error("INT4 linear output shape mismatch");
    }
    const int out_dtype = map_dtype_to_code(output.dtype());
    if (out_dtype != output_dtype_code) {
        throw std::runtime_error("INT4 linear output dtype code mismatch");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for INT4 linear");
    }

    const bool has_bias = bias.size() > 0;
    if (has_bias) {
        if (bias.size() != static_cast<size_t>(N)) {
            throw std::runtime_error("INT4 linear bias shape mismatch");
        }
        const int bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT4 linear");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_int4_linear_kernel(
        make_contiguous_tensor_arg<2>(act), make_contiguous_tensor_arg<2>(weight),
        make_flat_tensor_arg(x_scales), make_flat_tensor_arg(weight_scales),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(output),
        stream);
}

void unpack_int4_to_int8(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    uintptr_t stream_ptr) {

    const int64_t rows = input.shape(0);
    const int64_t K_half = input.shape(1);
    if (output.shape(0) != rows || output.shape(1) != K_half * 2) {
        throw std::runtime_error("unpack_int4_to_int8 output shape mismatch");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_unpack_int4_to_int8_kernel(make_contiguous_tensor_arg<2>(input), make_contiguous_tensor_arg<2>(output), stream);
}

void int4_weight_int8_act_gemv_dequant(
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
    if (weight.shape(1) * 2 != K) {
        throw std::runtime_error("packed INT4 weight GEMV weight K mismatch");
    }
    if (x_scales.shape(0) != M || x_scales.shape(1) != 1) {
        throw std::runtime_error("packed INT4 weight GEMV activation scale shape mismatch");
    }
    if (output.shape(0) != M || output.shape(1) != N) {
        throw std::runtime_error("packed INT4 weight GEMV output shape mismatch");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Invalid packed INT4 weight GEMV output dtype code");
    }

    const bool has_bias = bias.data() && bias.size() > 0;
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("packed INT4 weight GEMV bias shape mismatch");
        }
        const int bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for packed INT4 weight GEMV");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_int4_weight_int8_act_gemv_dequant_kernel(
        make_contiguous_tensor_arg<2>(input), make_contiguous_tensor_arg<2>(weight),
        make_contiguous_tensor_arg<2>(x_scales), make_flat_tensor_arg(weight_scales),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(output),
        stream);
}

void int4_weight_int8_act_gemm_dequant_chunked(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> x_scales,
    nb::ndarray<float, nb::device::cuda> weight_scales,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight_workspace,
    nb::ndarray<int32_t, nb::ndim<2>, nb::device::cuda> acc_workspace,
    nb::ndarray<uint8_t, nb::device::cuda> cublas_workspace,
    int64_t chunk_cols,
    bool allow_sm80_cutlass,
    int output_dtype_code,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    const int64_t N = weight.shape(0);
    const int64_t K_half = weight.shape(1);
    if (K_half * 2 != K) {
        throw std::runtime_error("chunked INT4 weight GEMM weight K mismatch");
    }
    if (x_scales.shape(0) != M || x_scales.shape(1) != 1) {
        throw std::runtime_error("chunked INT4 weight GEMM activation scale shape mismatch");
    }
    if (output.shape(0) != M || output.shape(1) != N) {
        throw std::runtime_error("chunked INT4 weight GEMM output shape mismatch");
    }
    if (chunk_cols <= 0 || chunk_cols > N) {
        throw std::runtime_error("chunked INT4 weight GEMM invalid chunk_cols");
    }
    if (weight_workspace.shape(0) < chunk_cols || weight_workspace.shape(1) != K) {
        throw std::runtime_error("chunked INT4 weight GEMM weight workspace shape mismatch");
    }
    if (acc_workspace.shape(0) != M || acc_workspace.shape(1) < chunk_cols) {
        throw std::runtime_error("chunked INT4 weight GEMM accumulator workspace shape mismatch");
    }
    if (weight_scales.size() != 1 && static_cast<int64_t>(weight_scales.size()) != N) {
        throw std::runtime_error("chunked INT4 weight GEMM weight scale shape mismatch");
    }
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Invalid chunked INT4 weight GEMM output dtype code");
    }

    const bool has_bias = bias.data() && bias.size() > 0;
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("chunked INT4 weight GEMM bias shape mismatch");
        }
        const int bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for chunked INT4 weight GEMM");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_int4_weight_int8_act_gemm_dequant_chunked_kernel(
        make_contiguous_tensor_arg<2>(input), make_contiguous_tensor_arg<2>(weight),
        make_contiguous_tensor_arg<2>(x_scales), make_flat_tensor_arg(weight_scales),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(weight_workspace),
        make_contiguous_tensor_arg<2>(acc_workspace), make_flat_tensor_arg(cublas_workspace),
        chunk_cols, allow_sm80_cutlass, stream);
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
    // xs/ws/bias are read as contiguous [M]/[N] vectors; check element counts (via size(),
    // which tolerates the [M,1] scale the int8 caller passes but rejects degenerate shapes
    // like [M,0]). Match the output dtype exactly (fp16 and bf16 share itemsize but the
    // launch selects half_t vs bfloat16_t) so a mismatched code can't reinterpret the buffer.
    if (static_cast<int64_t>(xs.size()) != M) throw std::runtime_error("cutlass_int8_dequant: xs must be a length-M vector");
    if (static_cast<int64_t>(ws.size()) != N) throw std::runtime_error("cutlass_int8_dequant: ws must be a length-N vector");
    if (bias.size() != 0 && static_cast<int64_t>(bias.size()) != N)
        throw std::runtime_error("cutlass_int8_dequant: bias must be empty or a length-N vector");
    validate_fp_dtype_code(d, out_dtype_code, "cutlass_int8_dequant");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    return launch_cutlass_int8_dequant(
        make_contiguous_tensor_arg<2>(a), make_contiguous_tensor_arg<2>(b),
        make_flat_tensor_arg(xs), make_flat_tensor_arg(ws),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(d), stream);
}

bool cutlass_int8_dequant_config(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,
    nb::ndarray<float, nb::device::cuda> xs,
    nb::ndarray<float, nb::device::cuda> ws,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> d,
    int out_dtype_code,
    int config,
    uintptr_t stream_ptr) {
    const int64_t M = a.shape(0);
    const int64_t K = a.shape(1);
    const int64_t N = b.shape(0);
    if (b.shape(1) != K) throw std::runtime_error("cutlass_int8_dequant_config: K mismatch");
    if (d.shape(0) != M || d.shape(1) != N) {
        throw std::runtime_error("cutlass_int8_dequant_config: D shape mismatch");
    }
    validate_fp_dtype_code(d, out_dtype_code, "cutlass_int8_dequant_config");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    return launch_cutlass_int8_dequant_config(
        make_contiguous_tensor_arg<2>(a), make_contiguous_tensor_arg<2>(b),
        make_flat_tensor_arg(xs), make_flat_tensor_arg(ws),
        make_contiguous_tensor_arg<2>(d), config, stream);
}

float benchmark_cutlass_int8_dequant_config(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,
    nb::ndarray<float, nb::device::cuda> xs,
    nb::ndarray<float, nb::device::cuda> ws,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> d,
    int out_dtype_code,
    int config,
    int iterations,
    uintptr_t stream_ptr) {
    if (iterations <= 0) {
        throw std::runtime_error(
            "benchmark_cutlass_int8_dequant_config: iterations must be positive");
    }
    const int64_t M = a.shape(0);
    const int64_t K = a.shape(1);
    const int64_t N = b.shape(0);
    if (b.shape(1) != K) {
        throw std::runtime_error(
            "benchmark_cutlass_int8_dequant_config: K mismatch");
    }
    if (d.shape(0) != M || d.shape(1) != N) {
        throw std::runtime_error(
            "benchmark_cutlass_int8_dequant_config: D shape mismatch");
    }
    validate_fp_dtype_code(d, out_dtype_code, "benchmark_cutlass_int8_dequant_config");

    const auto a_arg = make_contiguous_tensor_arg<2>(a);
    const auto b_arg = make_contiguous_tensor_arg<2>(b);
    const auto xs_arg = make_flat_tensor_arg(xs);
    const auto ws_arg = make_flat_tensor_arg(ws);
    const auto output_arg = make_contiguous_tensor_arg<2>(d);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (!launch_cutlass_int8_dequant_config(
                a_arg, b_arg, xs_arg, ws_arg, output_arg, config, stream)) {
            cudaEventDestroy(start);
            cudaEventDestroy(end);
            return -1.f;
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return elapsed_ms;
}

bool cutlass_turing_int8_dequant(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,
    nb::ndarray<float, nb::device::cuda> xs,
    nb::ndarray<float, nb::device::cuda> ws,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> d,
    int out_dtype_code,
    uintptr_t stream_ptr) {
    const int64_t M = a.shape(0);
    const int64_t K = a.shape(1);
    const int64_t N = b.shape(0);
    if (b.shape(1) != K) throw std::runtime_error("cutlass_turing_int8_dequant: K mismatch");
    if (d.shape(0) != M || d.shape(1) != N) throw std::runtime_error("cutlass_turing_int8_dequant: D shape mismatch");
    validate_fp_dtype_code(d, out_dtype_code, "cutlass_turing_int8_dequant");
    if (xs.size() != static_cast<size_t>(M)) throw std::runtime_error("cutlass_turing_int8_dequant: xs shape mismatch");
    if (ws.size() != 1 && ws.size() != static_cast<size_t>(N)) {
        throw std::runtime_error("cutlass_turing_int8_dequant: ws shape mismatch");
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    return launch_cutlass_turing_int8_dequant(
        make_contiguous_tensor_arg<2>(a), make_contiguous_tensor_arg<2>(b),
        make_flat_tensor_arg(xs), make_flat_tensor_arg(ws),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(d), stream);
}

// INT4 GEMM + fused dequant via CUTLASS. A and B are packed signed int4 in int8 storage.
// Returns true on success; false means caller falls back to the hand-written int4 kernel.
bool cutlass_int4_dequant(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,   // [M, K / 2]
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,   // [N, K / 2]
    nb::ndarray<float, nb::device::cuda> xs,                // [M] per-row act scale
    nb::ndarray<float, nb::device::cuda> ws,                // [N] per-col weight scale
    nb::ndarray<nb::device::cuda> bias,                     // [N] float or empty
    nb::ndarray<nb::ndim<2>, nb::device::cuda> d,           // [M, N] output
    int out_dtype_code,
    uintptr_t stream_ptr) {
    const int64_t M = a.shape(0);
    const int64_t K_half = a.shape(1);
    const int64_t N = b.shape(0);
    if (b.shape(1) != K_half) throw std::runtime_error("cutlass_int4_dequant: K mismatch");
    if (d.shape(0) != M || d.shape(1) != N) throw std::runtime_error("cutlass_int4_dequant: D shape mismatch");
    validate_fp_dtype_code(d, out_dtype_code, "cutlass_int4_dequant");
    if (xs.size() != static_cast<size_t>(M)) throw std::runtime_error("cutlass_int4_dequant: xs shape mismatch");
    if (ws.size() != static_cast<size_t>(N)) throw std::runtime_error("cutlass_int4_dequant: ws shape mismatch");
    const int64_t K = K_half * 2;
    if (K % 64 != 0) throw std::runtime_error("cutlass_int4_dequant: K must be divisible by 64");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    return launch_cutlass_int4_dequant(
        make_contiguous_tensor_arg<2>(a), make_contiguous_tensor_arg<2>(b),
        make_flat_tensor_arg(xs), make_flat_tensor_arg(ws),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(d), stream);
}

bool cutlass_turing_int4_dequant(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> a,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> b,
    nb::ndarray<float, nb::device::cuda> xs,
    nb::ndarray<float, nb::device::cuda> ws,
    nb::ndarray<nb::device::cuda> bias,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> d,
    int out_dtype_code,
    uintptr_t stream_ptr) {
    const int64_t M = a.shape(0);
    const int64_t K_half = a.shape(1);
    const int64_t N = b.shape(0);
    if (b.shape(1) != K_half) throw std::runtime_error("cutlass_turing_int4_dequant: K mismatch");
    if (d.shape(0) != M || d.shape(1) != N) throw std::runtime_error("cutlass_turing_int4_dequant: D shape mismatch");
    validate_fp_dtype_code(d, out_dtype_code, "cutlass_turing_int4_dequant");
    if (xs.size() != static_cast<size_t>(M)) throw std::runtime_error("cutlass_turing_int4_dequant: xs shape mismatch");
    if (ws.size() != static_cast<size_t>(N)) throw std::runtime_error("cutlass_turing_int4_dequant: ws shape mismatch");
    const int64_t K = K_half * 2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    return launch_cutlass_turing_int4_dequant(
        make_contiguous_tensor_arg<2>(a), make_contiguous_tensor_arg<2>(b),
        make_flat_tensor_arg(xs), make_flat_tensor_arg(ws),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(d), stream);
}

// Grouped int4 -> int8 dequant (group scale folded; per-channel scale applied in GEMM).
void dequant_int4_grouped_to_int8(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> qw,     // [N, K/2]
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> s_rel,   // [N, K/G]
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::device::cuda>> codebook,  // [16] or None
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> out,    // [N, K]
    int64_t G, uintptr_t stream_ptr) {
    const int64_t N = qw.shape(0);
    const int64_t K = out.shape(1);
    if (qw.shape(1) != K / 2) throw std::runtime_error("dequant_int4_grouped: K/2 mismatch");
    if (K % 16 != 0) throw std::runtime_error("dequant_int4_grouped: K must be a multiple of 16");
    if (G < 4 || (16 % G != 0 && G % 16 != 0))
        throw std::runtime_error("dequant_int4_grouped: G must be >=4 and divide 16 or be a multiple of 16");
    if (K % G != 0) throw std::runtime_error("dequant_int4_grouped: K must be divisible by G");
    if (static_cast<int64_t>(s_rel.shape(0)) != N || static_cast<int64_t>(s_rel.shape(1)) != K / G)
        throw std::runtime_error("dequant_int4_grouped: s_rel must have shape [N, K/G]");
    if (static_cast<int64_t>(out.shape(0)) != N)
        throw std::runtime_error("dequant_int4_grouped: out must be [N, K]");
    if (codebook.has_value() && static_cast<int64_t>(codebook->shape(0)) != 16)
        throw std::runtime_error("dequant_int4_grouped: codebook must be [16]");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequant_int4_grouped_to_int8(
        make_contiguous_tensor_arg<2>(qw), make_contiguous_tensor_arg<2>(s_rel),
        make_optional_flat_tensor_arg(codebook), make_contiguous_tensor_arg<2>(out),
        G, stream);
}

// fp8 (e4m3) per-group scale: s_rel passed as raw uint8 bits.
void dequant_int4_grouped_to_int8_e4m3(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> qw,     // [N, K/2]
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> s_rel, // [N, K/G] e4m3 bits
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::device::cuda>> codebook,  // [16] or None
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> out,    // [N, K]
    int64_t G, uintptr_t stream_ptr) {
    const int64_t N = qw.shape(0);
    const int64_t K = out.shape(1);
    if (qw.shape(1) != K / 2) throw std::runtime_error("dequant_int4_grouped: K/2 mismatch");
    if (K % 16 != 0) throw std::runtime_error("dequant_int4_grouped: K must be a multiple of 16");
    if (G < 4 || (16 % G != 0 && G % 16 != 0))
        throw std::runtime_error("dequant_int4_grouped: G must be >=4 and divide 16 or be a multiple of 16");
    if (K % G != 0) throw std::runtime_error("dequant_int4_grouped: K must be divisible by G");
    if (static_cast<int64_t>(s_rel.shape(0)) != N || static_cast<int64_t>(s_rel.shape(1)) != K / G)
        throw std::runtime_error("dequant_int4_grouped: s_rel must have shape [N, K/G]");
    if (static_cast<int64_t>(out.shape(0)) != N)
        throw std::runtime_error("dequant_int4_grouped: out must be [N, K]");
    if (codebook.has_value() && static_cast<int64_t>(codebook->shape(0)) != 16)
        throw std::runtime_error("dequant_int4_grouped: codebook must be [16]");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequant_int4_grouped_to_int8_e4m3(
        make_contiguous_tensor_arg<2>(qw), make_contiguous_tensor_arg<2>(s_rel),
        make_optional_flat_tensor_arg(codebook), make_contiguous_tensor_arg<2>(out),
        G, stream);
}

// Fused W4A8 requantize (group_size=16): rotated weight [N,K] -> packed int4
// [N,K/2] + fp8-e4m3 s_rel [N,K/16] + f32 s_channel [N] in one launch.
void quantize_w4a8_convrot(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> rotated,          // [N, K] fp32/fp16/bf16
    nb::ndarray<float, nb::ndim<1>, nb::device::cuda> codebook,  // [16]
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> packed,   // [N, K/2]
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> s_rel,   // [N, K/16] e4m3 bits
    nb::ndarray<float, nb::ndim<1>, nb::device::cuda> s_channel, // [N]
    bool stochastic, uint64_t seed, uintptr_t stream_ptr) {
    const int64_t N = rotated.shape(0);
    const int64_t K = rotated.shape(1);
    const int in_code = map_dtype_to_code(rotated.dtype());
    if (in_code < 0 || in_code > 2)
        throw std::runtime_error("quantize_w4a8_convrot: rotated must be fp32/fp16/bf16");
    if (N <= 0) throw std::runtime_error("quantize_w4a8_convrot: N must be positive");
    if (K % 16 != 0) throw std::runtime_error("quantize_w4a8_convrot: K must be a multiple of 16");
    if (static_cast<int64_t>(packed.shape(0)) != N || static_cast<int64_t>(packed.shape(1)) != K / 2)
        throw std::runtime_error("quantize_w4a8_convrot: packed must be [N, K/2]");
    if (static_cast<int64_t>(s_rel.shape(0)) != N || static_cast<int64_t>(s_rel.shape(1)) != K / 16)
        throw std::runtime_error("quantize_w4a8_convrot: s_rel must be [N, K/16]");
    if (static_cast<int64_t>(s_channel.shape(0)) != N)
        throw std::runtime_error("quantize_w4a8_convrot: s_channel must be [N]");
    if (static_cast<int64_t>(codebook.shape(0)) != 16)
        throw std::runtime_error("quantize_w4a8_convrot: codebook must be [16]");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (!launch_quantize_w4a8_convrot(
            make_contiguous_tensor_arg<2>(rotated), make_flat_tensor_arg(codebook),
            make_contiguous_tensor_arg<2>(packed), make_contiguous_tensor_arg<2>(s_rel),
            make_flat_tensor_arg(s_channel), stochastic, seed, stream))
        throw std::runtime_error(
            "quantize_w4a8_convrot: launch failed (group scales exceed shared memory, or "
            "invalid launch config)");
}

static void validate_w4a8_codebook_gemm_contract(
    int64_t M, int64_t N, int64_t K,
    int64_t weight_khalf,
    int64_t s_rel_n, int64_t s_rel_groups,
    int64_t s_channel_size, int64_t xs_size,
    int64_t codebook_size, int64_t bias_size,
    int64_t workspace_rows, int64_t workspace_cols,
    int64_t out_rows, int64_t out_cols,
    const nb::dlpack::dtype& out_dtype,
    int64_t G, int64_t chunk_cols, int out_dtype_code) {
    if (weight_khalf != K / 2)
        throw std::runtime_error("w4a8_codebook_gemm: K/2 mismatch");
    if (K % 16 != 0)
        throw std::runtime_error("w4a8_codebook_gemm: K must be a multiple of 16");
    if (G < 4 || (16 % G != 0 && G % 16 != 0))
        throw std::runtime_error("w4a8_codebook_gemm: G must be >=4 and divide 16 or be a multiple of 16");
    if (K % G != 0)
        throw std::runtime_error("w4a8_codebook_gemm: K must be divisible by G");
    if (xs_size != M)
        throw std::runtime_error("w4a8_codebook_gemm: xs must have M values");
    if (s_rel_n != N || s_rel_groups != K / G)
        throw std::runtime_error("w4a8_codebook_gemm: s_rel must be [N, K/G]");
    if (s_channel_size != N)
        throw std::runtime_error("w4a8_codebook_gemm: s_channel must be [N]");
    if (codebook_size >= 0 && codebook_size != 16)
        throw std::runtime_error("w4a8_codebook_gemm: codebook must be [16]");
    if (bias_size >= 0 && bias_size != N)
        throw std::runtime_error("w4a8_codebook_gemm: bias must be [N]");
    if (out_dtype_code < 0 || out_dtype_code > 2)
        throw std::runtime_error("w4a8_codebook_gemm: out_dtype_code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    if (map_dtype_to_code(out_dtype) != out_dtype_code)
        throw std::runtime_error("w4a8_codebook_gemm: out dtype does not match out_dtype_code (0=fp32, 1=fp16, 2=bf16)");
    if (out_rows != M || out_cols != N)
        throw std::runtime_error("w4a8_codebook_gemm: out must be [M, N]");
    if (chunk_cols > 0) {
        const int64_t required_rows = (chunk_cols < N) ? chunk_cols : N;
        if (workspace_cols != K || workspace_rows < required_rows)
            throw std::runtime_error("w4a8_codebook_gemm: workspace must be [>=min(chunk_cols,N), K] int8");
    }
}

// Chunked fused W4A8: per-chunk (codebook+s_rel) dequant -> L2-hot int8 -> strided int8 GEMM.
bool w4a8_codebook_gemm_chunked(
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> xq,        // [M, K] int8 act
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight,    // [N, K/2] packed uint4
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> s_rel,    // [N, K/G] e4m3 bits
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::device::cuda>> codebook,  // [16] or None
    nb::ndarray<float, nb::ndim<1>, nb::device::cuda> s_channel,  // [N] fp32
    nb::ndarray<float, nb::ndim<1>, nb::device::cuda> xs,         // [M] fp32
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::device::cuda>> bias,  // [N] or None
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> workspace, // [chunk_cols, K] int8
    nb::ndarray<nb::ndim<2>, nb::device::cuda> out,               // [M, N] out_dtype
    int64_t G, int64_t chunk_cols, int out_dtype_code, uintptr_t stream_ptr) {
    const int64_t M = xq.shape(0);
    const int64_t K = xq.shape(1);
    const int64_t N = weight.shape(0);
    validate_w4a8_codebook_gemm_contract(
        M, N, K,
        weight.shape(1),
        s_rel.shape(0), s_rel.shape(1),
        s_channel.size(), xs.size(),
        codebook.has_value() ? static_cast<int64_t>(codebook->size()) : -1,
        bias.has_value() ? static_cast<int64_t>(bias->size()) : -1,
        workspace.shape(0), workspace.shape(1),
        out.shape(0), out.shape(1), out.dtype(),
        G, chunk_cols, out_dtype_code);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    return launch_w4a8_codebook_gemm_chunked(
        make_contiguous_tensor_arg<2>(xq), make_contiguous_tensor_arg<2>(weight),
        make_contiguous_tensor_arg<2>(s_rel), make_optional_flat_tensor_arg(codebook),
        make_flat_tensor_arg(s_channel), make_flat_tensor_arg(xs),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(workspace),
        make_contiguous_tensor_arg<2>(out), G, chunk_cols, stream);
}

// Common W4A8 inference path: online ConvRot activation quantization followed by the
// chunked int4 decode + strided INT8 GEMM, coordinated through one Python/native call.
bool w4a8_codebook_linear_chunked(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,             // [M, K] fp32/fp16/bf16
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> xq,       // [M, K]
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> xs,        // [M, 1]
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> weight,   // [N, K/2]
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cuda> s_rel,   // [N, K/G]
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::device::cuda>> codebook,
    nb::ndarray<float, nb::ndim<1>, nb::device::cuda> s_channel, // [N]
    std::optional<nb::ndarray<float, nb::ndim<1>, nb::device::cuda>> bias,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> workspace,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> out,
    int64_t convrot_group_size, int64_t G, int64_t chunk_cols,
    int out_dtype_code, uintptr_t stream_ptr) {
    const int64_t M = input.shape(0);
    const int64_t K = input.shape(1);
    const int64_t N = weight.shape(0);
    if (xq.shape(0) != M || xq.shape(1) != K)
        throw std::runtime_error("w4a8_codebook_linear: xq must be [M, K]");
    if (xs.shape(0) != M || xs.shape(1) != 1)
        throw std::runtime_error("w4a8_codebook_linear: xs must be [M, 1]");
    if (input.stride(1) != 1 || input.stride(0) != K
            || xq.stride(1) != 1 || xq.stride(0) != K
            || xs.stride(1) != 1 || xs.stride(0) != 1)
        throw std::runtime_error("w4a8_codebook_linear: input, xq, and xs must be contiguous");
    if (weight.stride(1) != 1 || weight.stride(0) != weight.shape(1)
            || s_rel.stride(1) != 1 || s_rel.stride(0) != s_rel.shape(1)
            || s_channel.stride(0) != 1
            || workspace.stride(1) != 1 || workspace.stride(0) != workspace.shape(1)
            || out.stride(1) != 1 || out.stride(0) != out.shape(1)
            || (codebook.has_value() && codebook->stride(0) != 1)
            || (bias.has_value() && bias->stride(0) != 1))
        throw std::runtime_error("w4a8_codebook_linear: weight metadata, workspace, and out must be contiguous");
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2)
        throw std::runtime_error("w4a8_codebook_linear: input must be fp32, fp16, or bf16");
    validate_w4a8_codebook_gemm_contract(
        M, N, K,
        weight.shape(1),
        s_rel.shape(0), s_rel.shape(1),
        s_channel.size(), xs.size(),
        codebook.has_value() ? static_cast<int64_t>(codebook->size()) : -1,
        bias.has_value() ? static_cast<int64_t>(bias->size()) : -1,
        workspace.shape(0), workspace.shape(1),
        out.shape(0), out.shape(1), out.dtype(),
        G, chunk_cols, out_dtype_code);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_convrot_kernel(
        make_contiguous_tensor_arg<2>(input), make_contiguous_tensor_arg<2>(xq),
        make_contiguous_tensor_arg<2>(xs), static_cast<int>(convrot_group_size),
        false, 0, stream);
    return launch_w4a8_codebook_gemm_chunked(
        make_contiguous_tensor_arg<2>(xq), make_contiguous_tensor_arg<2>(weight),
        make_contiguous_tensor_arg<2>(s_rel), make_optional_flat_tensor_arg(codebook),
        make_flat_tensor_arg(s_channel), make_flat_tensor_arg(xs),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(workspace),
        make_contiguous_tensor_arg<2>(out), G, chunk_cols, stream);
}

void quantize_int8_rowwise_convrot(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    int64_t group_size,
    bool stochastic,
    uint64_t seed,
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

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_convrot_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(scales), static_cast<int>(group_size),
        stochastic, seed, stream);
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
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(output), static_cast<int>(group_size), stream);
}

void quantize_int8_convrot_staged(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<nb::ndim<2>, nb::device::cuda> rotated,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> partial_absmax,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    int64_t group_size,
    bool stochastic,
    uint64_t seed,
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

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_convrot_staged_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(rotated),
        make_contiguous_tensor_arg<2>(partial_absmax),
        make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(scales), static_cast<int>(group_size),
        stochastic, seed, stream);
}

void quantize_int8_rowwise_convrot64(
    nb::ndarray<nb::ndim<2>, nb::device::cuda> input,
    nb::ndarray<int8_t, nb::ndim<2>, nb::device::cuda> output,
    nb::ndarray<float, nb::ndim<2>, nb::device::cuda> scales,
    int64_t group_size,
    bool stochastic,
    int64_t act_code,
    uint64_t seed,
    uintptr_t stream_ptr) {

    const int64_t M = input.shape(0);
    // K is the activated (quantized) row width; the SwiGLU pair reads a
    // [gate | up] input row twice as wide.
    const int64_t K = output.shape(1);
    const int64_t in_width = (act_code == comfy::kActSwiGLU) ? 2 : 1;

    if (output.shape(0) != M || input.shape(1) != K * in_width) {
        throw std::runtime_error("INT8 rowwise convrot64 output shape mismatch");
    }
    if (scales.shape(0) != M || scales.shape(1) != 1) {
        throw std::runtime_error("INT8 rowwise convrot64 scale shape mismatch");
    }
    const int input_dtype_code = map_dtype_to_code(input.dtype());
    if (input_dtype_code < 0 || input_dtype_code > 2) {
        throw std::runtime_error("Unsupported input dtype for INT8 rowwise convrot64 quantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_quantize_int8_rowwise_convrot64_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(output),
        make_contiguous_tensor_arg<2>(scales), static_cast<int>(group_size),
        stochastic, static_cast<int>(act_code), seed, stream);
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
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("INT8 linear bias shape mismatch");
        }
        const int bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT8 linear");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_int8_linear_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(x_scales), make_flat_tensor_arg(weight_scales),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(output), stream);
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
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("INT8 GEMV bias shape mismatch");
        }
        const int bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT8 GEMV");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_int8_gemv_dequant_kernel(
        make_contiguous_tensor_arg<2>(input),
        make_contiguous_tensor_arg<2>(weight),
        make_contiguous_tensor_arg<2>(x_scales), make_flat_tensor_arg(weight_scales),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(output), stream);
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
    if (has_bias) {
        if (bias.shape(0) != N) {
            throw std::runtime_error("INT8 M=1 linear bias shape mismatch");
        }
        const int bias_dtype_code = map_dtype_to_code(bias.dtype());
        if (bias_dtype_code < 0 || bias_dtype_code > 2) {
            throw std::runtime_error("Unsupported bias dtype for INT8 M=1 linear");
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (convrot) {
        launch_quantize_int8_rowwise_convrot64_kernel(
            make_contiguous_tensor_arg<2>(input),
            make_contiguous_tensor_arg<2>(q_scratch),
            make_contiguous_tensor_arg<2>(x_scales), group_size, false,
            /*act_code=*/0, 0, stream);
    } else {
        launch_quantize_int8_rowwise_kernel(
            make_contiguous_tensor_arg<2>(input),
            make_contiguous_tensor_arg<2>(q_scratch),
            make_contiguous_tensor_arg<2>(x_scales), false, 0, stream);
    }
    launch_int8_gemv_dequant_kernel(
        make_contiguous_tensor_arg<2>(q_scratch),
        make_contiguous_tensor_arg<2>(weight),
        make_contiguous_tensor_arg<2>(x_scales), make_flat_tensor_arg(weight_scales),
        make_optional_flat_tensor_arg(bias), make_contiguous_tensor_arg<2>(output), stream);
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
    auto input_arg = make_flat_tensor_arg(input);
    auto scale_arg = make_flat_tensor_arg(scale);
    auto output_arg = make_flat_tensor_arg(output);
    const int output_dtype_code = static_cast<int>(output_arg.meta.dtype);
    if (output_dtype_code < 0 || output_dtype_code > 2) {
        throw std::runtime_error("Unsupported output dtype for INT8 simple dequantization");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    launch_dequantize_int8_simple_kernel(
        input_arg, scale_arg, output_arg, inner_dim, scale_mode, stream);
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
        make_contiguous_tensor_arg<2>(input), make_flat_tensor_arg(scale),
        make_contiguous_tensor_arg<2>(output), static_cast<int>(group_size), stream);
}

void flash_attention_decode(
    nb::ndarray<nb::ndim<3>, nb::device::cuda> q,
    nb::ndarray<nb::ndim<4>, nb::device::cuda> k,
    nb::ndarray<nb::ndim<4>, nb::device::cuda> v,
    nb::ndarray<int32_t, nb::ndim<1>, nb::device::cuda> kv_lengths,
    nb::ndarray<nb::ndim<3>, nb::device::cuda> output,
    nb::ndarray<float, nb::device::cuda> softmax_lse,
    nb::ndarray<float, nb::device::cuda> softmax_lse_accum,
    nb::ndarray<float, nb::device::cuda> output_accum,
    int num_splits,
    uintptr_t stream_ptr) {
    const int batch = k.shape(0);
    const int kv_capacity = k.shape(1);
    const int heads = k.shape(2);
    const int query_length = q.shape(0) / batch;
    if (batch <= 0 || kv_capacity <= 0 || heads <= 0 || query_length <= 0 || q.shape(0) != batch * query_length || q.shape(1) != heads || q.shape(2) != 128) {
        throw std::runtime_error("Invalid Flash Attention decode dimensions");
    }
    if (v.shape(0) != batch || v.shape(1) != kv_capacity || v.shape(2) != heads || v.shape(3) != 128 || k.shape(3) != 128) {
        throw std::runtime_error("Flash Attention k/v shape mismatch");
    }
    if (output.shape(0) != q.shape(0) || output.shape(1) != heads || output.shape(2) != 128 || kv_lengths.size() != static_cast<size_t>(batch)) {
        throw std::runtime_error("Flash Attention output or length shape mismatch");
    }
    if (map_dtype_to_code(q.dtype()) != 2 || map_dtype_to_code(k.dtype()) != 2 || map_dtype_to_code(v.dtype()) != 2 || map_dtype_to_code(output.dtype()) != 2) {
        throw std::runtime_error("Flash Attention tensors must have bfloat16 dtype");
    }
    const size_t lse_size = static_cast<size_t>(batch) * heads * query_length;
    if (softmax_lse.size() != lse_size || num_splits < 1 || num_splits > 32 || (num_splits > 1 && (softmax_lse_accum.size() != lse_size * num_splits || output_accum.size() != lse_size * 128 * num_splits))) {
        throw std::runtime_error("Invalid Flash Attention split workspace");
    }
    if (k.stride(3) != 1 || v.stride(3) != 1 ||
        q.stride(2) != 1 || output.stride(2) != 1) {
        throw std::runtime_error("Unsupported Flash Attention tensor strides");
    }

    launch_flash_decode(
        make_tensor_arg<3>(q), make_tensor_arg<4>(k), make_tensor_arg<4>(v),
        make_contiguous_tensor_arg<1>(kv_lengths), make_tensor_arg<3>(output),
        make_flat_tensor_arg(softmax_lse), make_flat_tensor_arg(softmax_lse_accum),
        make_flat_tensor_arg(output_accum), num_splits,
        reinterpret_cast<cudaStream_t>(stream_ptr));
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
          nb::arg("stochastic"),
          nb::arg("seed"),
          nb::arg("stream_ptr"));

    m.def("quantize_int4_rowwise", &quantize_int4_rowwise,
          "Rowwise signed INT4 quantization for CUDA activations/weights",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("stochastic"),
          nb::arg("seed"),
          nb::arg("stream_ptr"));

    m.def("quantize_int4_rowwise_convrot64", &quantize_int4_rowwise_convrot64,
          "Fused regular ConvRot-256 activation rotation plus rowwise signed INT4 quantization",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("seed"),
          nb::arg("stream_ptr"));

    m.def("quantize_int4_rowwise_convrot64_to_int8", &quantize_int4_rowwise_convrot64_to_int8,
          "Fused ConvRot-256 activation rotation plus rowwise INT4-scale quantization into INT8 storage",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("seed"),
          nb::arg("stream_ptr"));

    m.def("dequantize_int4_convrot64", &dequantize_int4_convrot64,
          "Fused packed signed INT4 dequantization plus regular ConvRot-256 inverse rotation",
          nb::arg("input"),
          nb::arg("scales"),
          nb::arg("output"),
          nb::arg("group_size"),
          nb::arg("stream_ptr"));

    m.def("int4_linear", &int4_linear,
          "Signed INT4 GEMM with rowwise x colwise dequantization, bias, and output cast",
          nb::arg("act"),
          nb::arg("weight"),
          nb::arg("x_scales"),
          nb::arg("weight_scales"),
          nb::arg("bias"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("unpack_int4_to_int8", &unpack_int4_to_int8,
          "Unpack row-major packed signed INT4 matrix to row-major INT8 matrix",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("stream_ptr"));

    m.def("int4_weight_int8_act_gemv_dequant", &int4_weight_int8_act_gemv_dequant,
          "M=1 GEMV using INT8 activation and packed row-major INT4 weight with fused dequant",
          nb::arg("input"),
          nb::arg("weight"),
          nb::arg("x_scales"),
          nb::arg("weight_scales"),
          nb::arg("bias"),
          nb::arg("output"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("int4_weight_int8_act_gemm_dequant_chunked", &int4_weight_int8_act_gemm_dequant_chunked,
          "Chunked INT8 GEMM using INT8 activation and packed row-major INT4 weight with fused dequant",
          nb::arg("input"),
          nb::arg("weight"),
          nb::arg("x_scales"),
          nb::arg("weight_scales"),
          nb::arg("bias"),
          nb::arg("output"),
          nb::arg("weight_workspace"),
          nb::arg("acc_workspace"),
          nb::arg("cublas_workspace"),
          nb::arg("chunk_cols"),
          nb::arg("allow_sm80_cutlass"),
          nb::arg("output_dtype_code"),
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

    m.def("cutlass_int8_dequant_config", &cutlass_int8_dequant_config,
          "Benchmark one fused CUTLASS INT8 kernel configuration",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("xs"),
          nb::arg("ws"),
          nb::arg("d"),
          nb::arg("out_dtype_code"),
          nb::arg("config"),
          nb::arg("stream_ptr"));

    m.def("benchmark_cutlass_int8_dequant_config",
          &benchmark_cutlass_int8_dequant_config,
          "Time a tight loop of one fused CUTLASS INT8 kernel configuration",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("xs"),
          nb::arg("ws"),
          nb::arg("d"),
          nb::arg("out_dtype_code"),
          nb::arg("config"),
          nb::arg("iterations"),
          nb::arg("stream_ptr"));

    m.def("cutlass_turing_int8_dequant", &cutlass_turing_int8_dequant,
          "Turing INT8 tensor-core GEMM with fused row/column dequantization",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("xs"),
          nb::arg("ws"),
          nb::arg("bias"),
          nb::arg("d"),
          nb::arg("out_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("cutlass_int4_dequant", &cutlass_int4_dequant,
          "INT4 GEMM + fused rowwise x colwise dequant + bias via CUTLASS; false -> fall back to hand kernel",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("xs"),
          nb::arg("ws"),
          nb::arg("bias"),
          nb::arg("d"),
          nb::arg("out_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("cutlass_turing_int4_dequant", &cutlass_turing_int4_dequant,
          "Turing packed INT4 tensor-core GEMM with fused row/column dequantization",
          nb::arg("a"),
          nb::arg("b"),
          nb::arg("xs"),
          nb::arg("ws"),
          nb::arg("bias"),
          nb::arg("d"),
          nb::arg("out_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("dequant_int4_grouped_to_int8", &dequant_int4_grouped_to_int8,
          "Grouped int4 -> int8 dequant (group scale folded into int8); optional 16-entry codebook",
          nb::arg("qw"), nb::arg("s_rel"), nb::arg("codebook").none(), nb::arg("out"),
          nb::arg("g"), nb::arg("stream_ptr"));

    m.def("dequant_int4_grouped_to_int8_e4m3", &dequant_int4_grouped_to_int8_e4m3,
          "Grouped int4 -> int8 dequant with fp8 e4m3 per-group scale; optional 16-entry codebook",
          nb::arg("qw"), nb::arg("s_rel"), nb::arg("codebook").none(), nb::arg("out"),
          nb::arg("g"), nb::arg("stream_ptr"));

    m.def("quantize_w4a8_convrot", &quantize_w4a8_convrot,
          "Fused W4A8 requant (group_size=16): rotated weight -> packed int4 + fp8 s_rel + f32 s_channel",
          nb::arg("rotated"), nb::arg("codebook"), nb::arg("packed"), nb::arg("s_rel"),
          nb::arg("s_channel"), nb::arg("stochastic"), nb::arg("seed"), nb::arg("stream_ptr"));

    m.def("w4a8_codebook_gemm_chunked", &w4a8_codebook_gemm_chunked,
          "Chunked fused W4A8: per-chunk codebook+s_rel dequant -> L2-hot int8 -> strided int8 GEMM",
          nb::arg("xq"), nb::arg("weight"), nb::arg("s_rel"), nb::arg("codebook").none(),
          nb::arg("s_channel"), nb::arg("xs"), nb::arg("bias").none(), nb::arg("workspace"),
          nb::arg("out"), nb::arg("g"), nb::arg("chunk_cols"), nb::arg("out_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("w4a8_codebook_linear_chunked", &w4a8_codebook_linear_chunked,
          "Fused W4A8 inference orchestration: ConvRot activation quantization followed by chunked decode/GEMM",
          nb::arg("input"), nb::arg("xq"), nb::arg("xs"), nb::arg("weight"),
          nb::arg("s_rel"), nb::arg("codebook").none(), nb::arg("s_channel"),
          nb::arg("bias").none(), nb::arg("workspace"), nb::arg("out"),
          nb::arg("convrot_group_size"), nb::arg("g"), nb::arg("chunk_cols"),
          nb::arg("out_dtype_code"), nb::arg("stream_ptr"));

    m.def("quantize_int8_rowwise_convrot", &quantize_int8_rowwise_convrot,
          "Fused ConvRot Hadamard rotation + rowwise INT8 quantization",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("seed"),
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
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("seed"),
          nb::arg("stream_ptr"));

    m.def("quantize_int8_rowwise_convrot64", &quantize_int8_rowwise_convrot64,
          "Fused ConvRot rowwise INT8 quantization using 64-lane FHT groups. "
          "act_code applies an elementwise activation to the input first "
          "(0 = none, 1 = gelu tanh-approx), folding an MLP's activation into "
          "the quantizer instead of round-tripping it through HBM.",
          nb::arg("input"),
          nb::arg("output"),
          nb::arg("scales"),
          nb::arg("group_size"),
          nb::arg("stochastic"),
          nb::arg("act_code"),
          nb::arg("seed"),
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

    m.def("rms_rope", &rms_rope,
          "Fused RMSNorm and interleaved RoPE for Q/K tensors", nb::arg("q"),
          nb::arg("k"), nb::arg("freqs"), nb::arg("q_scale"),
          nb::arg("k_scale"), nb::arg("q_out"), nb::arg("k_out"),
          nb::arg("epsilon"), nb::arg("stream_ptr"),
          nb::arg("split_half") = false, nb::arg("rot_dim") = 0);

    m.def("rms_rope1", &rms_rope1, "Fused RMSNorm and RoPE for a single tensor",
          nb::arg("q"), nb::arg("freqs"), nb::arg("q_scale"), nb::arg("q_out"),
          nb::arg("epsilon"), nb::arg("stream_ptr"),
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

    m.def("_quant_v_int8", &quant_v_int8,
          "Quantize V [B,H,N,D] to signed INT8 rows [B*H*D,padded_N] with per-row scale",
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
          nb::arg("anchor_indices_ptr"));

    m.def("_sage_attn", &sage_attn,
          "Pure INT8 QK / U8-softmax / INT8-V attention kernel",
          nb::arg("q"),
          nb::arg("k"),
          nb::arg("v"),
          nb::arg("o"),
          nb::arg("q_scale"),
          nb::arg("k_scale"),
          nb::arg("v_scale"),
          nb::arg("sm_scale"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"));

    m.def("sage_sdpa_quantize", &sage_sdpa_quantize,
          "Prequantize Q/K/V for split pure-INT8 SDPA",
          nb::arg("q"),
          nb::arg("k"),
          nb::arg("v"),
          nb::arg("q_int8"),
          nb::arg("q_scale"),
          nb::arg("k_int8"),
          nb::arg("k_scale"),
          nb::arg("v_int8"),
          nb::arg("v_scale"),
          nb::arg("cta_k"),
          nb::arg("input_dtype_code"),
          nb::arg("stream_ptr"),
          nb::arg("anchor_indices_ptr"));

    m.def("sage_sdpa_prequantized", &sage_sdpa_prequantized,
          "Run pure-INT8 SDPA from prequantized Q/K/V",
          nb::arg("q_int8"),
          nb::arg("k_int8"),
          nb::arg("v_int8"),
          nb::arg("o"),
          nb::arg("q_scale"),
          nb::arg("k_scale"),
          nb::arg("v_scale"),
          nb::arg("cta_k"),
          nb::arg("sm_scale"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"),
          nb::arg("attn_mask") = nb::none());

    m.def("sage_sdpa", &sage_sdpa,
          "Fused pure-INT8 SDPA: quant_qk + quant_v + attention in one call",
          nb::arg("q"),
          nb::arg("k"),
          nb::arg("v"),
          nb::arg("o"),
          nb::arg("q_int8"),
          nb::arg("q_scale"),
          nb::arg("k_int8"),
          nb::arg("k_scale"),
          nb::arg("v_int8"),
          nb::arg("v_scale"),
          nb::arg("sm_scale"),
          nb::arg("input_dtype_code"),
          nb::arg("output_dtype_code"),
          nb::arg("stream_ptr"),
          nb::arg("anchor_indices_ptr"),
          nb::arg("attn_mask") = nb::none(),
          nb::arg("cta_k") = 0);

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

    m.def("na3d", &na3d,
          "Fused 3D neighborhood attention (NATTEN na3d semantics)",
          nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("out"),
          nb::arg("batch"), nb::arg("t_size"), nb::arg("h_size"), nb::arg("w_size"),
          nb::arg("num_heads"), nb::arg("head_dim"),
          nb::arg("kt"), nb::arg("kh"), nb::arg("kw"),
          nb::arg("causal_t"), nb::arg("causal_h"), nb::arg("causal_w"),
          nb::arg("scale"), nb::arg("dtype_code"), nb::arg("stream_ptr"));

    m.def("flash_attention_decode", &flash_attention_decode,
          "Flash Attention decode over a fixed-capacity variable-length KV cache",
          nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("kv_lengths"),
          nb::arg("output"), nb::arg("softmax_lse"), nb::arg("softmax_lse_accum"),
          nb::arg("output_accum"), nb::arg("num_splits"), nb::arg("stream_ptr"));

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

    m.def("rms_adaln", &rms_adaln,
          "Fused AdaLN: rmsnorm(x) * (1 + scale) + shift",
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

    m.attr("__nanobind__") = true;
    m.attr("__stable_abi__") = true;
}
