#pragma once

#include "../native_tensor.h"
#include <cuda_runtime.h>

namespace mpa::attention {

inline bool sm89_or_sm120_execution_device(const cudaDeviceProp* properties) {
  return anemoi_native::ada_serves_device(properties) ||
      (properties->major == 12 && properties->minor == 0);
}

}  // namespace mpa::attention
