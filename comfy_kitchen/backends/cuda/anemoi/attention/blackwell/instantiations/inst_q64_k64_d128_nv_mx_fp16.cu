/* SM120 Q64 NVFP4 -> MXFP8 -> FP16 phase stack. */
#define MPA_CTA_Q 64
#define MPA_WARP_Q 16
#define MPA_K64_BLOCK_MODE 1
#define MPA_LOW4_NVFP4 1
#define MPA_MIDDLE_MXFP8 1
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm120_q64_nv_mx_fp16_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm120_q64_nv_mx_fp16
#include "../q64_attention.cuh"

template void launch_mixed_attention_sm120_q64_nv_mx_fp16<128, true, false, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*,
    int8_t*, int8_t*, __nv_fp8_e4m3*, uint8_t*, uint8_t*, uint8_t*,
    int32_t*, const float*, const float*, const float*, const int32_t*,
    float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, float, cudaStream_t);

template void launch_mixed_attention_sm120_q64_nv_mx_fp16<128, true, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*,
    int8_t*, int8_t*, __nv_fp8_e4m3*, uint8_t*, uint8_t*, uint8_t*,
    int32_t*, const float*, const float*, const float*, const int32_t*,
    float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, float, cudaStream_t);
