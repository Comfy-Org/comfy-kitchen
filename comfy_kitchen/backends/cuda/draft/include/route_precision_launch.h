#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

// Raw contiguous device buffers, on the caller's current device and stream.
// Invalid pointers/scalars/geometry return cudaErrorInvalidValue. No allocation,
// synchronization, device switching, or inspection of device contents is done.
// Probability must be finite nonnegative FP16 [segments, rows, rows]. Anchor IDs
// must be unique, in [0, rows*rows), and exactly describe the bool anchor mask.
// All buffers must be mutually disjoint (except the documented internal reuse).
// Scratch/output lifetimes extend until stream completion.
// Query returns CUB temporary bytes only, not the typed scratch buffers below.
cudaError_t sm120_h3_route_precision_workspace_size(
    int64_t segments, int64_t rows, size_t* workspace_bytes, cudaStream_t stream);

// N = segments*rows*rows. precision_map: uint8[N]; input_keys/sorted_keys:
// uint32[N]; sorted_ids/block_ids: int32[N]; each count output: int32[segments*rows].
// block_ids is reused internally as the initial CUB values. Workspace must have
// at least the queried size and CUDA allocation alignment. anchors/anchor_ids
// may both be null only when anchor_count is zero; an empty anchor list may use
// a null anchor_ids pointer with a nonnull mask. Outputs pack low/middle/high.
cudaError_t sm120_h3_route_precision(
    const void* probability, const bool* anchors, const int* anchor_ids,
    int64_t segments, int64_t rows, int64_t n16, int64_t n8, int64_t n4,
    int64_t anchor_count, uint8_t* precision_map, uint32_t* input_keys,
    uint32_t* sorted_keys, int* sorted_ids, int* block_ids, int* low_counts,
    int* middle_counts, int* high_counts, void* workspace,
    size_t workspace_bytes, cudaStream_t stream);

// logical_ids: int32[row_count,logical_columns]; count inputs/outputs:
// int32[row_count]. physical_ids: int32[row_count,logical_columns*factor+prefix_blocks],
// factor=query_block_size/64 (1 or 2). high pointers may be null iff !has_high.
// Caller validates nonnegative device counts with sum <= logical_columns and
// active IDs in [0,logical_columns); input/output buffers must not alias.
cudaError_t sm120_h3_materialize_route(
    const int* logical_ids, const int* low_counts, const int* middle_counts,
    const int* high_counts, int64_t row_count, int64_t logical_columns,
    int64_t query_block_size, int64_t prefix_blocks, int64_t prefix_phase,
    bool prefix_first, bool has_high, int* physical_ids,
    int* physical_low_counts, int* physical_middle_counts,
    int* physical_high_counts, cudaStream_t stream);
