#pragma once
#include "attention_view.h"
#include <vector>
using anemoi_sm120::TensorView;
void sm120_q64_fp16_attention_forward(
    TensorView query,
    TensorView key,
    TensorView value,
    TensorView block_ids,
    TensorView block_counts,
    TensorView valid_k_counts,
    double softmax_scale, TensorView output, cudaStream_t stream);
void sm120_q128_fp16_attention_forward(
    TensorView query,
    TensorView key,
    TensorView value,
    TensorView block_ids,
    TensorView block_counts,
    TensorView valid_k_counts,
    double softmax_scale, TensorView output, cudaStream_t stream);
void sm120_q64_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView int8_block_counts,
    TensorView fp16_block_counts, TensorView q_scale,
    TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q128_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView int8_block_counts,
    TensorView fp16_block_counts, TensorView q_scale,
    TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q64_prefix_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q_scale, TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t prefix_tokens,
    double softmax_scale, TensorView output, cudaStream_t stream);
void sm120_q128_prefix_int8_attention_forward(
    TensorView q8, TensorView k8, TensorView v8,
    TensorView q_scale, TensorView k_scale, TensorView v_scale,
    TensorView valid_k_counts, int64_t prefix_tokens,
    double softmax_scale, TensorView output, cudaStream_t stream);
void sm120_q64_mxfp8_attention_forward(
    TensorView q_mxfp8, TensorView q_mxfp8_scale,
    TensorView k_mxfp8, TensorView k_mxfp8_scale,
    TensorView v_mxfp8, TensorView v_mxfp8_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView mxfp8_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q128_mxfp8_attention_forward(
    TensorView q_mxfp8, TensorView q_mxfp8_scale,
    TensorView k_mxfp8, TensorView k_mxfp8_scale,
    TensorView v_mxfp8, TensorView v_mxfp8_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView mxfp8_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q128_mxfp8_compact_attention_forward(
    TensorView q_mxfp8, TensorView q_mxfp8_scale,
    TensorView k_mxfp8, TensorView k_mxfp8_scale,
    TensorView v_mxfp8, TensorView v_mxfp8_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView mxfp8_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q64_nvfp4_attention_forward(
    TensorView q_nvfp4, TensorView q_nvfp4_scale,
    TensorView k_nvfp4, TensorView k_nvfp4_scale,
    TensorView v_nvfp4, TensorView v_nvfp4_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView nvfp4_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    TensorView q_global_scale, TensorView k_global_scale,
    TensorView v_global_scale, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q128_nvfp4_attention_forward(
    TensorView q_nvfp4, TensorView q_nvfp4_scale,
    TensorView k_nvfp4, TensorView k_nvfp4_scale,
    TensorView v_nvfp4, TensorView v_nvfp4_scale,
    TensorView q_fp16, TensorView k_fp16, TensorView v_fp16,
    TensorView block_ids, TensorView nvfp4_block_counts,
    TensorView fp16_block_counts, TensorView valid_k_counts,
    TensorView q_global_scale, TensorView k_global_scale,
    TensorView v_global_scale, int64_t fp16_prefix_blocks,
    double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q64_nv_int8_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q128_nv_int8_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q64_nv_mx_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
void sm120_q128_nv_mx_fp16_attention_forward(
    TensorView q4, TensorView q4_scale,
    TensorView k4, TensorView k4_scale,
    TensorView v4, TensorView v4_scale,
    TensorView q8, TensorView q8_scale,
    TensorView k8, TensorView k8_scale,
    TensorView v8, TensorView v8_scale,
    TensorView q16, TensorView k16, TensorView v16,
    TensorView block_ids, TensorView nvfp4_counts,
    TensorView middle_counts, TensorView fp16_counts,
    TensorView valid_k_counts, TensorView q_global_scale,
    TensorView k_global_scale, TensorView v_global_scale,
    int64_t fp16_prefix_blocks, double softmax_scale, bool active_fp16, TensorView output, cudaStream_t stream);
std::vector<int64_t> sm120_q64_fp16_kernel_metadata();
std::vector<int64_t> sm120_q64_mxfp8_kernel_metadata();
