// SPDX-License-Identifier: Apache-2.0
// Non-owning raw device-memory descriptors; no framework ABI or allocation.
#pragma once
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "native_tensor.h"

namespace anemoi_sm120 {
enum class ScalarType { Half, Float, Byte, Char, Int, Float8_e4m3fn };
struct Shape : std::vector<int64_t> {
  using std::vector<int64_t>::vector;
  template <size_t N>
  Shape(const std::array<int64_t, N> &a)
      : std::vector<int64_t>(a.begin(), a.end()) {}
};
template <class... T> inline void check(bool ok, const T &...text) {
  if (!ok) {
    std::ostringstream s;
    (s << ... << text);
    throw std::invalid_argument(s.str());
  }
}
inline void cuda_check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct TensorView {
  void *pointer = nullptr;
  Shape shape;
  ScalarType dtype;
  int device_id = -1;
  int64_t size(size_t i) const { return shape.at(i); }
  int64_t dim() const { return shape.size(); }
  const Shape &sizes() const { return shape; }
  int64_t numel() const {
    int64_t n = 1;
    for (auto v : shape)
      n *= v;
    return n;
  }
  bool is_cuda() const { return device_id >= 0; }
  bool is_contiguous() const { return true; } // importer validates strides
  int device() const { return device_id; }
  ScalarType scalar_type() const { return dtype; }
  template <class T = void> T *data_ptr() const {
    return static_cast<T *>(pointer);
  }
};
inline void check_output(const TensorView &out, const TensorView &q,
                         cudaStream_t stream) {
  check(out.dtype == ScalarType::Half && out.shape == q.shape &&
            out.device_id == q.device_id,
        "output must be contiguous FP16 matching complete padded query shape "
        "and device");
  check(q.size(0) <= 65535 && q.size(1) <= 65535,
        "attention batch/head grid exceeds CUDA y/z limits");
  int current = -1;
  cuda_check(cudaGetDevice(&current));
  check(current == q.device_id,
        "explicit current CUDA device must match operands");
  const cudaDeviceProp &prop = *anemoi_native::cuda::deviceProperties(current);
  check(prop.major == 12 && prop.minor == 0,
        "Anemoi SM120 requires exact compute capability 12.0");
  int stream_device = -1;
  cuda_check(cudaStreamGetDevice(stream, &stream_device));
  check(stream_device == current,
        "explicit stream must belong to the current CUDA device");
}
} // namespace anemoi_sm120
#define ANEMOI_SM120_CHECK(...) ::anemoi_sm120::check(__VA_ARGS__)
#define ANEMOI_SM120_CUDA_CHECK(expr) ::anemoi_sm120::cuda_check(expr)
