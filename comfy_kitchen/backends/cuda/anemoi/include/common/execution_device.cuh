#pragma once

#include <cuda_runtime.h>

namespace mpa::attention {

inline bool sm89_or_sm120_execution_device(const cudaDeviceProp* properties) {
  return (properties->major == 8 && properties->minor == 9) ||
      (properties->major == 12 && properties->minor == 0);
}

}  // namespace mpa::attention
