// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace h3_qkv {
template <typename Kernel> bool loadable(Kernel kernel) {
  cudaFuncAttributes attributes;
  const auto error = cudaFuncGetAttributes(&attributes, kernel);
  if (error == cudaSuccess)
    return true;
  if (error == cudaErrorInvalidDeviceFunction ||
      error == cudaErrorNoKernelImageForDevice) {
    cudaGetLastError(); // Consume only the unavailable-image probe error.
    return false;
  }
  throw std::runtime_error(std::string("H3 kernel probe failed: ") +
                           cudaGetErrorString(error));
}
} // namespace h3_qkv
