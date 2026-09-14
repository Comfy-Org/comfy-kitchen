/* SM120 Q128xK64 MXFP8 phases using the shared optimized mainloop. */
#define MPA_CTA_Q 128
#define MPA_WARP_Q 32
#define MPA_K64_BLOCK_MODE 1
#define MPA_MIDDLE_MXFP8 1
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm120_q128_mxfp8_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm120_q128_mxfp8
#include "../q64_attention.cuh"

template void launch_mixed_attention_sm120_q128_mxfp8<128, true, false, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);

template void launch_mixed_attention_sm120_q128_mxfp8<128, true, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);
