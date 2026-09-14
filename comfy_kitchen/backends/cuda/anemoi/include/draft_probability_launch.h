#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace anemoi_sm120 {

// All floating-point buffers are contiguous FP16 on the caller's current CUDA
// device. Caller validates device, shape, alignment, non-aliasing and stream,
// and keeps all buffers alive until stream work completes. Calls are asynchronous.
// Q/Qmax: [B,Hq,R,D]; K/Kmax: [B,Hkv,R,D]; out: [B,Hq,R,R].
// max_logits: [B,Hq,R,R], required only for 0 < weight < 1; otherwise nullable.
// Qmax/Kmax are required for weight > 0; D is 64 or 128.
// A private thread-local, per-device cuBLAS handle uses the explicit stream,
// HOST pointer mode and DEFAULT math mode. cuBLAS manages its own opaque
// resources/workspace; no caller-provided cuBLAS workspace is required. Initial
// handle creation can allocate internally; warm up before CUDA graph capture.
void launch_draft_probability(
    const void* q, const void* k, const void* q_max, const void* k_max,
    void* out, void* max_logits, int64_t B, int64_t Hq, int64_t Hkv,
    int64_t R, int64_t D, double weight, cudaStream_t stream);

// D=128, tail=1 or 2. Q: [B,Hq,R,128]; K: [B,Hkv,R,128].
// packed_k: [B,Hkv,(prefix+R)*64,128]; valid_counts: int32 [R], each
// value in [1,64] (caller must ensure this device-data invariant).
// out: [B,Hq,R,R]; descriptors: [B,Hkv,R*(tail+1),128];
// expanded_logits: [B,Hq,R,R*(tail+1)]. All outputs/workspaces are supplied
// by the caller, mutually non-overlapping and non-overlapping with inputs.
void launch_k_tail_probability(
    const void* q, const void* k, const void* packed_k,
    const int32_t* valid_counts, void* out, void* descriptors,
    void* expanded_logits, int64_t B, int64_t Hq, int64_t Hkv,
    int64_t R, int64_t prefix, int tail, cudaStream_t stream);

}  // namespace anemoi_sm120
