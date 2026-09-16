/* Native Q64 x K64 FP16 and FP8+FP16 specializations for the SM89 mainline. */
#define MPA_CTA_Q 64
#define MPA_WARP_Q 16
#define MPA_K64_BLOCK_MODE 1
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm89_k64_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm89_k64
#include "../mixed_attention.cuh"

template void launch_mixed_attention_sm89_k64<128, false, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);

template void launch_mixed_attention_sm89_k64<128, true, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);

template void launch_mixed_attention_sm89_k64<128, true, true, true>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);

template void launch_mixed_attention_sm89_k64<128, true, false, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);
