/* Standalone Q128xK64 FP16 floor, isolated from mixed code generation. */
#define MPA_CTA_Q 128
#define MPA_WARP_Q 32
#define MPA_K64_BLOCK_MODE 1
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm89_q128_k64_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm89_q128_k64
#include "../mixed_attention.cuh"

template void launch_mixed_attention_sm89_q128_k64<128, false, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, float*, float*, float*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);
