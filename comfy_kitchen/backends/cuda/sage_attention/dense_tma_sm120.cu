#include <cassert>
// SPDX-License-Identifier: Apache-2.0
#include "dense_tma_sm120.cuh"
#include "dense_tma_sm120.h"
#include <stdexcept>
#include <string>

bool launch_dense_tma_sm120(const DenseTmaArgs &a, cudaStream_t stream) {
  // Only contiguous BHSD quantized tensors and output. Other layouts, masks,
  // dtypes and wide offsets are handled by the original launcher.
  if (a.k_stride_s != 128 || a.k_stride_h != uint64_t(a.kv_len) * 128 ||
      a.k_stride_b != uint64_t(a.kv_heads) * a.k_stride_h ||
      a.v_stride_h != uint64_t(128) * a.v_stride_d ||
      a.v_stride_b != uint64_t(a.kv_heads) * a.v_stride_h ||
      a.q_stride_s != 128 || a.q_stride_h != uint64_t(a.qo_len) * 128 ||
      a.q_stride_b != uint64_t(a.qo_heads) * a.q_stride_h ||
      a.o_stride_s != 128 || a.o_stride_h != uint64_t(a.qo_len) * 128 ||
      a.o_stride_b != uint64_t(a.qo_heads) * a.o_stride_h ||
      a.v_stride_d % 128 || a.v_stride_d < a.kv_len ||
      (reinterpret_cast<uintptr_t>(a.k) & 15) ||
      (reinterpret_cast<uintptr_t>(a.v) & 15))
    return false;
  int device = 0, major = 0, minor = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   device);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  if (major != 12 || minor != 0)
    return false;

  using Encode = CUresult(CUDAAPI *)(
      CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
      const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
      const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
      CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  // Resolve through libcudart so the wheel does not require a new libcuda link.
  static Encode encode = [] {
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                         cudaEnableDefault,
                                         &status) != cudaSuccess ||
        status != cudaDriverEntryPointSuccess)
      return static_cast<Encode>(nullptr);
    return reinterpret_cast<Encode>(fn);
  }();
  if (!encode)
    return false;
  alignas(64) CUtensorMap mapK, mapV;
  uint64_t kd[3] = {128, a.kv_len, uint64_t(a.batch) * a.kv_heads};
  uint64_t ks[2] = {a.k_stride_s, a.k_stride_h};
  uint64_t vd[3] = {a.v_stride_d, 128, uint64_t(a.batch) * a.kv_heads};
  uint64_t vs[2] = {a.v_stride_d, a.v_stride_h};
  uint32_t box[3] = {128, 128, 1}, element_strides[3] = {1, 1, 1};
  auto make_map = [&](CUtensorMap *map, void *ptr, uint64_t *dims,
                      uint64_t *strides) {
    return encode(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, dims, strides,
                  box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                  CU_TENSOR_MAP_SWIZZLE_128B,
                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
  };
  if (!make_map(&mapK, a.k, kd, ks) || !make_map(&mapV, a.v, vd, vs))
    return false;
  constexpr int shared_bytes = 4 * 128 * 128 + 1024;
  // Separate instantiations keep the common positive-scale loop unchanged.
  auto kernel = a.sm_scale > 0 ? comfy_sm120::dense_tma<true>
                              : comfy_sm120::dense_tma<false>;
  error = cudaFuncSetAttribute(kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               shared_bytes);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  kernel<<<dim3((a.qo_len + 127) / 128, a.qo_heads, a.batch),
                           dim3(32, 12), shared_bytes, stream>>>(
      static_cast<int8_t *>(a.q), static_cast<int8_t *>(a.k),
      static_cast<int8_t *>(a.v), static_cast<nv_bfloat16 *>(a.o),
      static_cast<float *>(a.q_scale), static_cast<float *>(a.k_scale),
      static_cast<float *>(a.v_scale), a.qo_len, a.kv_len,
      a.qo_heads / a.kv_heads, a.q_stride_b, a.q_stride_s, a.q_stride_h,
      a.k_stride_b, a.k_stride_s, a.k_stride_h, a.v_stride_b, a.v_stride_h,
      a.v_stride_d, a.o_stride_b, a.o_stride_s, a.o_stride_h, a.sm_scale, mapK,
      mapV);
  error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("SM120 TMA attention: ") +
                             cudaGetErrorString(error));
  return true;
}
