// Adapted from Anemoi 4e85afba741bdeaf2d9486cab19cb76d3e7985a4; see LICENSE.
#include "native_tensor.h"
#pragma once


#include <optional>
#include <tuple>

using H3SM89Int8Prepared = std::tuple<
    draft_native::Tensor, draft_native::Tensor, draft_native::Tensor, draft_native::Tensor,
    draft_native::Tensor, draft_native::Tensor, draft_native::Tensor, draft_native::Tensor,
    draft_native::Tensor, draft_native::Tensor, draft_native::Tensor, draft_native::Tensor,
    draft_native::Tensor, draft_native::Tensor>;

// Architecture-owned DraftMap scorer. Mean and optional MaxPool logits use
// Tensor Core GEMMs; the interior-weight path normalizes and fuses both maps
// in one CUDA kernel while weights 0 and 1 retain single-GEMM fast paths.
draft_native::Tensor sm89_h3_draft_probability(
    draft_native::Tensor q_pool,
    draft_native::Tensor k_pool,
    std::optional<draft_native::Tensor> q_max_pool,
    std::optional<draft_native::Tensor> k_max_pool,
    double maxpool_weight);

// Stable global DraftMap route with compact Anchor correction. The returned
// logical IDs are phase-packed but do not yet contain physical prefix blocks.
std::tuple<draft_native::Tensor, draft_native::Tensor, draft_native::Tensor, draft_native::Tensor>
sm89_h3_route_precision(
    draft_native::Tensor probability,
    int64_t n16,
    int64_t n8,
    int64_t n4,
    std::optional<draft_native::Tensor> anchors,
    std::optional<draft_native::Tensor> anchor_ids,
    int64_t anchor_count);

// SM89 Q64 physical lowering. INT8 prefix blocks are explicit at the front of
// the low phase; FP16 prefix blocks retain the established implicit ordering.
std::tuple<draft_native::Tensor, draft_native::Tensor, draft_native::Tensor, draft_native::Tensor>
sm89_h3_materialize_route(
    draft_native::Tensor logical_ids,
    draft_native::Tensor low_counts,
    draft_native::Tensor middle_counts,
    draft_native::Tensor high_counts,
    int64_t query_block_size,
    int64_t prefix_blocks,
    int64_t prefix_phase,
    bool prefix_first,
    bool has_high);

// Native Q64 x K64 FP16 attention used by both contiguous-1D and compact
// aligned-8x8 layouts. block_ids are absolute K64 block indices; every
// physical block has an explicit valid-token count so partial compact blocks
// are masked in both score and value accumulation.
std::tuple<draft_native::Tensor, draft_native::Tensor> k64_fp16_attention_forward(
    draft_native::Tensor query,
    draft_native::Tensor key,
    draft_native::Tensor value,
    draft_native::Tensor block_ids,
    draft_native::Tensor block_counts,
    draft_native::Tensor valid_k_counts,
    double softmax_scale);

// Audit-only Q128 counterpart used to measure the standalone FP16 phase
// floor on the same physical K64 route as the Q128 mixed specialization.
std::tuple<draft_native::Tensor, draft_native::Tensor> q128_k64_fp16_attention_forward(
    draft_native::Tensor query,
    draft_native::Tensor key,
    draft_native::Tensor value,
    draft_native::Tensor block_ids,
    draft_native::Tensor block_counts,
    draft_native::Tensor valid_k_counts,
    double softmax_scale);

// Native Q64xK64 mixed executor. Q/K use the project INT8 tensor-core
// representation and V uses E4M3; selected rescue blocks consume the exact
// FP16 operands. The two list tensors must alias one compact absolute-stage
// list: [INT8 prefix stages when selected][FP8 video stages]
// [FP16 video stages][unused]. Exact FP16 prefix stages remain implicit in the
// FP16 count and do not occupy list slots.
std::tuple<draft_native::Tensor, draft_native::Tensor> k64_mixed_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q16,
    draft_native::Tensor k16,
    draft_native::Tensor v16,
    draft_native::Tensor fp8_block_ids,
    draft_native::Tensor fp8_block_counts,
    draft_native::Tensor fp16_block_ids,
    draft_native::Tensor fp16_block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale);

std::tuple<draft_native::Tensor, draft_native::Tensor> q128_k64_mixed_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q16,
    draft_native::Tensor k16,
    draft_native::Tensor v16,
    draft_native::Tensor fp8_block_ids,
    draft_native::Tensor fp8_block_counts,
    draft_native::Tensor fp16_block_ids,
    draft_native::Tensor fp16_block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale);

// K-smooth counterparts consume centered INT8 K plus the source-bound FP16
// mean. Only FP16 Q and the mean cross the phase boundary; K/V rescue state
// remains phase-local.
std::tuple<draft_native::Tensor, draft_native::Tensor> k64_smooth_mixed_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q16,
    draft_native::Tensor k16,
    draft_native::Tensor v16,
    draft_native::Tensor key_mean,
    draft_native::Tensor fp8_block_ids,
    draft_native::Tensor fp8_block_counts,
    draft_native::Tensor fp16_block_ids,
    draft_native::Tensor fp16_block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale);

std::tuple<draft_native::Tensor, draft_native::Tensor>
q128_k64_smooth_mixed_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q16,
    draft_native::Tensor k16,
    draft_native::Tensor v16,
    draft_native::Tensor key_mean,
    draft_native::Tensor fp8_block_ids,
    draft_native::Tensor fp8_block_counts,
    draft_native::Tensor fp16_block_ids,
    draft_native::Tensor fp16_block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    int64_t fp16_prefix_blocks,
    double softmax_scale);

// Audit-only pure low-precision entry for isolating the inherited
// Sparge/Sage wait_group<1> pipeline on the same absolute Q64xK64 route.
std::tuple<draft_native::Tensor, draft_native::Tensor> k64_fp8_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor block_ids,
    draft_native::Tensor block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    double softmax_scale);

std::tuple<draft_native::Tensor, draft_native::Tensor> q128_k64_fp8_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor block_ids,
    draft_native::Tensor block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    double softmax_scale);

std::tuple<draft_native::Tensor, draft_native::Tensor> k64_smooth_fp8_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q16,
    draft_native::Tensor key_mean,
    draft_native::Tensor block_ids,
    draft_native::Tensor block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    double softmax_scale);

std::tuple<draft_native::Tensor, draft_native::Tensor>
q128_k64_smooth_fp8_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q16,
    draft_native::Tensor key_mean,
    draft_native::Tensor block_ids,
    draft_native::Tensor block_counts,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    double softmax_scale);

// Native producer for the dense prefix-query path. It reads the caller's
// positive-stride BHSD FP16/BF16 view directly and emits padded Q64 INT8 plus
// one FP32 symmetric scale per physical query block.
std::tuple<draft_native::Tensor, draft_native::Tensor> prepare_sm89_prefix_q_int8(
    draft_native::Tensor query,
    int64_t prefix_tokens);

// Native contiguous-Q/K producer used by the package-level SM89 executor.
// Q owns one scale per Q64/Q128 block and K owns one scale per K64 block.
// Optional mean subtraction preserves the established FP16 smooth-K contract.
std::tuple<draft_native::Tensor, draft_native::Tensor, draft_native::Tensor, draft_native::Tensor>
quantize_sm89_qk_int8(
    draft_native::Tensor query,
    draft_native::Tensor key,
    std::optional<draft_native::Tensor> key_mean,
    int64_t query_block_size);

// Dense-sequential prefix-query provider using the same SM89 INT8-QK/E4M3-PV
// phase body and the already prepared full K/V operands.
draft_native::Tensor sm89_q64_prefix_int8_attention_forward(
    draft_native::Tensor q8,
    draft_native::Tensor k8,
    draft_native::Tensor v8,
    draft_native::Tensor q_scale,
    draft_native::Tensor k_scale,
    draft_native::Tensor v_scale,
    draft_native::Tensor valid_k_counts,
    int64_t prefix_tokens,
    double softmax_scale);

// Integration-private adaptive-2D preparation.  A cached logical-to-physical
// token map drives one fused BF16/FP16 -> FP16 Q/K/V pack into K64 blocks;
// invalid physical lanes are written as exact positive zero.
std::tuple<draft_native::Tensor, draft_native::Tensor, draft_native::Tensor>
pack_indexed_k64_qkv_fp16(
    draft_native::Tensor query,
    draft_native::Tensor key,
    draft_native::Tensor value,
    draft_native::Tensor token_indices,
    draft_native::Tensor slot_valid);

// Released-H3 specialization of the same preparation boundary.  Prefix K/V
// is copied into leading K64 blocks while indexed video Q/K/V is packed into
// the suffix, avoiding separate prefix padding and K/V concatenation kernels.
std::tuple<draft_native::Tensor, draft_native::Tensor, draft_native::Tensor>
pack_h3_k64_qkv_fp16(
    draft_native::Tensor query,
    draft_native::Tensor key,
    draft_native::Tensor value,
    draft_native::Tensor video_token_indices,
    draft_native::Tensor video_slot_valid,
    int64_t prefix_tokens);

// Donor-first SM89 INT8 preparation.  The no-smooth specialization reads each
// raw Q/K element once and produces packed FP16 rescue operands, DraftMap
// pools, INT8 operands, and their scales from the same shared-memory tile.
// Smooth-K retains the one raw-K read while deferring centered K quantization
// until a compact block-sum reduction has produced the per-head K mean.
H3SM89Int8Prepared prepare_h3_sm89_int8_operands(
    draft_native::Tensor query,
    draft_native::Tensor key,
    draft_native::Tensor value,
    draft_native::Tensor video_token_indices,
    draft_native::Tensor video_slot_valid,
    draft_native::Tensor video_valid_counts,
    int64_t prefix_tokens,
    int64_t query_block_size,
    bool smooth_k,
    bool has_maxpool);

// One-pass inverse-raster scatter, prefix append, BHSD->BSHD transform, and
// FP16-video conversion to the original H3 input/output dtype.
draft_native::Tensor assemble_h3_k64_output(
    draft_native::Tensor prefix_output_bhsd,
    draft_native::Tensor video_output_bhsd_fp16,
    draft_native::Tensor video_inverse_indices,
    std::optional<draft_native::ScalarType> output_dtype,
    std::optional<draft_native::Tensor> low_counts,
    std::optional<draft_native::Tensor> middle_counts,
    std::optional<draft_native::Tensor> high_counts,
    int64_t query_block_size);

std::tuple<draft_native::Tensor, draft_native::Tensor> preprocess_v_fp8(
    draft_native::Tensor value);

// Quantization-only half of V preprocessing for a caller-provided
// channel-major, 16-token-permuted FP16 [B,Hkv,D,K] tensor.
std::tuple<draft_native::Tensor, draft_native::Tensor> quantize_permuted_v_fp8(
    draft_native::Tensor permuted_value);

// Package-private released-H3 specialization.  Its producer-provided
// per-physical-stage/channel absmax removes the first full read of permuted V.
std::tuple<draft_native::Tensor, draft_native::Tensor>
quantize_permuted_v_fp8_h3_vpartials(
    draft_native::Tensor permuted_value,
    draft_native::Tensor value_stage_amax);

// Package-private no-permuted-V production consumer.  Consumes the exact-H3
// physical K64-stage packed V [1,14,42624,128] and producer stage/channel
// absmax, transposes/permutates into the inherited FP8 V layout while quantizing.
std::tuple<draft_native::Tensor, draft_native::Tensor>
quantize_packed_v_fp8_h3_vpartials(
    draft_native::Tensor packed_value,
    draft_native::Tensor value_stage_amax);

// Production raw-video preprocessing boundary.  Converts same-dtype
// FP16/BF16 frame-major raster [B,H,F*Y*X,D] into FP16 logical 8x16 patch
// order [B,H,R*128,D]. BF16 is narrowed before storage and every virtual edge
// slot is exact positive FP16 zero.
std::tuple<draft_native::Tensor, draft_native::Tensor, draft_native::Tensor>
pack_raster_qkv_fp16(
    draft_native::Tensor query,
    draft_native::Tensor key,
    draft_native::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width);

// Candidate/production boundary for the address-mapped Q/K path.  Only V is
// physically materialized in logical 8x16 patch order; raw Q/K stay in their
// caller-owned frame-major raster layout.
draft_native::Tensor pack_raster_v_fp16(
    draft_native::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width);

// Experimental phase-aware boundary: K/V are materialized together while Q
// remains caller-owned raw raster FP16.  This reuses the production fused KV
// copy rather than launching two independent pack kernels.
std::tuple<draft_native::Tensor, draft_native::Tensor> pack_raster_kv_fp16(
    draft_native::Tensor key,
    draft_native::Tensor value,
    int64_t frames,
    int64_t height,
    int64_t width);

// Final Draft-facing output assembly.  The video partition arrives in the
// native attention ABI's contiguous BHSD layout while both dense partitions
// use contiguous BSHD.  The two visual key partitions are normalized with
// their FP32 natural-log LSE states, then the text partition is copied and an
// optional CUDA bool [B,S_text] mask writes exact positive BF16 zero.  An
// empty CUDA bool tensor is the no-mask sentinel.
draft_native::Tensor assemble_video_text_output(
    draft_native::Tensor video_output_bhsd,
    draft_native::Tensor video_lse_bhs,
    draft_native::Tensor visual_text_output_bshd,
    draft_native::Tensor visual_text_lse_bhs,
    draft_native::Tensor text_output_bshd,
    draft_native::Tensor text_mask);

// Final layout join after visual queries have already traversed both
// video and dense-text K/V in one CTA.  No LSE merge is performed here.
draft_native::Tensor assemble_fused_visual_text_output(
    draft_native::Tensor visual_output_bhsd,
    draft_native::Tensor text_output_bshd,
    draft_native::Tensor text_mask);
