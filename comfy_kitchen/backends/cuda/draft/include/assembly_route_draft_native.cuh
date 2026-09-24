// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "native_tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <optional>
#include <tuple>
#include <vector>

namespace assembly_route_draft_native {
inline void cuda_check(cudaError_t status) {
  DRAFT_CHECK(status == cudaSuccess, "CUDA failure: ", cudaGetErrorString(status));
}
// Raw device selection only: no framework guard or framework stream lookup.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    cuda_check(cudaGetDevice(&previous_));
    if (previous_ != device) cuda_check(cudaSetDevice(device));
  }
  ~DeviceGuard() { (void)cudaSetDevice(previous_); }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
 private:
  int previous_ = 0;
};
inline cudaDeviceProp device_properties(int device) {
  // Served from the immutable per-device cache in native_tensor.h; a direct
  // cudaGetDeviceProperties here costs over a millisecond per launch phase.
  return *draft_native::cuda::deviceProperties(device);
}
inline bool shared_device(const cudaDeviceProp* p) {
  return draft_native::ada_serves_device(p) ||
         (p->major == 12 && p->minor == 0);
}
inline std::vector<int64_t> shape(const draft_native::Tensor& tensor) {
  std::vector<int64_t> result;
  result.reserve(tensor.dim());
  for (int i = 0; i < tensor.dim(); ++i) result.push_back(tensor.size(i));
  return result;
}
}  // namespace assembly_route_draft_native
