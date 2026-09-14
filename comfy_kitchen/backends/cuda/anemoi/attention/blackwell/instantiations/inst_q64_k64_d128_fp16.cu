/* SM120 Q64xK64 all-FP16 foundation derived from Anemoi SM89@7ebeb084. */
#define MPA_CTA_Q 64
#define MPA_WARP_Q 16
#define MPA_K64_BLOCK_MODE 1
#define MPA_ATTENTION_KERNEL_ENTRY mixed_attention_sm120_q64_kernel
#define MPA_ATTENTION_LAUNCH_ENTRY launch_mixed_attention_sm120_q64
#include "../q64_attention.cuh"

#include <cstdint>
#include <vector>

template void launch_mixed_attention_sm120_q64<128, false, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);

// D64 uses the same phase implementation with half the channels.
template void launch_mixed_attention_sm120_q64<64, false, true, false>(
    int8_t*, int8_t*, __nv_fp8_e4m3*, half*, half*, half*, half*, half*,
    int32_t*, int32_t*, int32_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*,
    const int32_t*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, float, cudaStream_t);

std::vector<int64_t> sm120_q64_fp16_kernel_metadata() {
  constexpr int kThreads = 128;
  constexpr int kDynamicSmemBytes = 32768;
  auto kernel =
      mpa::attention::mixed_attention_sm120_q64_kernel<128, false, true, false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kDynamicSmemBytes));
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  int active_ctas = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_ctas,
      kernel,
      kThreads,
      kDynamicSmemBytes));
  return {
      attributes.numRegs,
      static_cast<int64_t>(attributes.sharedSizeBytes),
      attributes.maxDynamicSharedSizeBytes,
      active_ctas,
  };
}
