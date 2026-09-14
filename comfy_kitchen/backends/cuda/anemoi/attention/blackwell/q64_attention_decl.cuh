#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

// Q64 x K64 native SM120 attention.  The phase-list-shaped ABI is retained so
// this debugging specialization can grow back into the mixed-precision MPA
// family without changing its routing contract.  The current public wrapper
// instantiates only <D128, FP16>.
template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void launch_mixed_attention_sm120_q64(
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
    uint8_t* q_scale,
    uint8_t* k_scale,
    uint8_t* v_scale,
    const int32_t* valid_k_counts,
    float* lse,
    uint32_t fp16_prefix_stages,
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void launch_mixed_attention_sm120_q128_fp16(
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
    uint8_t* q_scale,
    uint8_t* k_scale,
    uint8_t* v_scale,
    const int32_t* valid_k_counts,
    float* lse,
    uint32_t fp16_prefix_stages,
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void launch_mixed_attention_sm120_q128_mxfp8(
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
    uint8_t* q_scale,
    uint8_t* k_scale,
    uint8_t* v_scale,
    const int32_t* valid_k_counts,
    float* lse,
    uint32_t fp16_prefix_stages,
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);

template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK>
void launch_mixed_attention_sm120_q128_mxfp8_compact(
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
    uint8_t* q_scale,
    uint8_t* k_scale,
    uint8_t* v_scale,
    const int32_t* valid_k_counts,
    float* lse,
    uint32_t fp16_prefix_stages,
    uint32_t batch_size,
    uint32_t qo_len,
    uint32_t kv_len,
    uint32_t padded_kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    float softmax_scale,
    cudaStream_t stream);

#define MPA_DECLARE_NVFP4_LAUNCH(name) \
template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK> \
void name( \
    int8_t* q4, int8_t* k4, __nv_fp8_e4m3* v4, \
    half* q16, half* k16, half* v16, half* k_mean, half* output, \
    int32_t* nvfp4_lut, int32_t* nvfp4_count, int32_t* fp16_lut, \
    int32_t* fp16_count, uint8_t* q_scale, uint8_t* k_scale, \
    uint8_t* v_scale, const float* q_global_scale, \
    const float* k_global_scale, const float* v_global_scale, \
    const int32_t* valid_k_counts, float* lse, \
    uint32_t fp16_prefix_stages, uint32_t batch_size, uint32_t qo_len, \
    uint32_t kv_len, uint32_t padded_kv_len, uint32_t num_qo_heads, \
    uint32_t num_kv_heads, \
    float softmax_scale, \
    cudaStream_t stream)

MPA_DECLARE_NVFP4_LAUNCH(launch_mixed_attention_sm120_q64_nvfp4);
MPA_DECLARE_NVFP4_LAUNCH(launch_mixed_attention_sm120_q128_nvfp4);
#undef MPA_DECLARE_NVFP4_LAUNCH

#define MPA_DECLARE_INT8_LAUNCH(name) \
template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK> \
void name( \
    int8_t* q8, int8_t* k8, __nv_fp8_e4m3* v8, \
    half* q16, half* k16, half* v16, half* k_mean, half* output, \
    int32_t* fp8_lut, int32_t* fp8_count, int32_t* fp16_lut, \
    int32_t* fp16_count, float* q_scale, float* k_scale, float* v_scale, \
    const int32_t* valid_k_counts, float* lse, uint32_t fp16_prefix_stages, \
    uint32_t batch_size, uint32_t qo_len, uint32_t kv_len, \
    uint32_t padded_kv_len, uint32_t num_qo_heads, uint32_t num_kv_heads, \
    float softmax_scale, cudaStream_t stream)

MPA_DECLARE_INT8_LAUNCH(launch_mixed_attention_sm120_q64_int8);
MPA_DECLARE_INT8_LAUNCH(launch_mixed_attention_sm120_q64_int8_fp16);
MPA_DECLARE_INT8_LAUNCH(launch_mixed_attention_sm120_q128_int8);
MPA_DECLARE_INT8_LAUNCH(launch_mixed_attention_sm120_q64_int8_dense);
MPA_DECLARE_INT8_LAUNCH(launch_mixed_attention_sm120_q128_int8_dense);
#undef MPA_DECLARE_INT8_LAUNCH

#define MPA_DECLARE_THREE_PHASE_LAUNCH(name, scale_type) \
template <uint32_t HeadDim, bool HasFp8, bool HasFp16, bool SmoothK> \
void name( \
    int8_t* q8, int8_t* k8, __nv_fp8_e4m3* v8, \
    half* q16, half* k16, half* v16, half* k_mean, half* output, \
    int32_t* block_ids, int32_t* nvfp4_count, int32_t* fp16_lut, \
    int32_t* fp16_count, scale_type* q_scale, scale_type* k_scale, \
    scale_type* v_scale, int8_t* q4, int8_t* k4, __nv_fp8_e4m3* v4, \
    uint8_t* q4_scale, uint8_t* k4_scale, uint8_t* v4_scale, \
    int32_t* middle_count, const float* q_global_scale, \
    const float* k_global_scale, const float* v_global_scale, \
    const int32_t* valid_k_counts, float* lse, uint32_t fp16_prefix_stages, \
    uint32_t batch_size, uint32_t qo_len, uint32_t kv_len, \
    uint32_t padded_kv_len, uint32_t num_qo_heads, uint32_t num_kv_heads, \
    float softmax_scale, cudaStream_t stream)

MPA_DECLARE_THREE_PHASE_LAUNCH(
    launch_mixed_attention_sm120_q64_nv_mx_fp16, uint8_t);
MPA_DECLARE_THREE_PHASE_LAUNCH(
    launch_mixed_attention_sm120_q128_nv_mx_fp16, uint8_t);
MPA_DECLARE_THREE_PHASE_LAUNCH(
    launch_mixed_attention_sm120_q128_nv_int8_fp16, float);
MPA_DECLARE_THREE_PHASE_LAUNCH(
    launch_mixed_attention_sm120_q64_nv_int8_fp16, float);
#undef MPA_DECLARE_THREE_PHASE_LAUNCH
