#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void launch_mixed_attention_sm89(
    int8_t* q8,
    int8_t* k8,
    __nv_fp8_e4m3* v8,
    half* q16,
    half* k16,
    half* v16,
    half* k_mean,
    half* output,
    int32_t* fp8_lut,
    int32_t* fp8_count,
    int32_t* fp16_lut,
    int32_t* fp16_count,
    float* q_scale,
    float* k_scale,
    float* v_scale,
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);
