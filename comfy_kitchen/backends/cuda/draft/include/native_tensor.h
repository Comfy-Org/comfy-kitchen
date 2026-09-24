// SPDX-License-Identifier: Apache-2.0
// Kitchen raw-pointer host descriptors; no framework tensor ABI or CUDA
// allocator.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace draft_native {
template <class... A> inline void check(bool ok, A &&...a) {
  if (!ok) {
    std::ostringstream s;
    (s << ... << a);
    throw std::invalid_argument(s.str());
  }
}
inline void cuda_check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
enum class ScalarType {
  Half,
  BFloat16,
  Float,
  Int,
  Long,
  Byte,
  Char,
  Bool,
  Float8_e4m3fn
};
using Half = half;
using BFloat16 = __nv_bfloat16;
using IntArrayRef = std::vector<int64_t>;
inline size_t itemsize(ScalarType t) {
  switch (t) {
  case ScalarType::Half:
  case ScalarType::BFloat16:
    return 2;
  case ScalarType::Float:
  case ScalarType::Int:
    return 4;
  case ScalarType::Long:
    return 8;
  default:
    return 1;
  }
}
inline const char *dtype_name(ScalarType t) {
  switch (t) {
  case ScalarType::Half:
    return "float16";
  case ScalarType::BFloat16:
    return "bfloat16";
  case ScalarType::Float:
    return "float32";
  case ScalarType::Int:
    return "int32";
  case ScalarType::Long:
    return "int64";
  case ScalarType::Byte:
    return "uint8";
  case ScalarType::Char:
    return "int8";
  case ScalarType::Bool:
    return "bool";
  default:
    return "float8_e4m3fn";
  }
}
struct TensorOptions {
  ScalarType type = ScalarType::Half;
  int ordinal = 0;
  TensorOptions dtype(ScalarType t) const {
    auto o = *this;
    o.type = t;
    return o;
  }
};
struct Tensor {
  void *pointer = nullptr;
  IntArrayRef shape, steps;
  TensorOptions metadata;
  bool present = false;
  int buffer_index = -1;
  bool defined() const { return present; }
  int64_t dim() const { return shape.size(); }
  int64_t size(int i) const { return shape.at(i < 0 ? i + shape.size() : i); }
  int64_t stride(int i) const { return steps.at(i < 0 ? i + steps.size() : i); }
  const IntArrayRef &sizes() const { return shape; }
  int64_t numel() const {
    if (!present)
      return 0;
    int64_t n = 1;
    for (auto s : shape) {
      check(s >= 0, "negative dimension");
      if (!s)
        return 0;
      check(n <= INT64_MAX / s, "tensor size overflow");
      n *= s;
    }
    return n;
  }
  bool is_cuda() const { return present; }
  bool is_contiguous() const {
    int64_t st = 1;
    for (int i = int(shape.size()) - 1; i >= 0; --i) {
      if (shape[i] == 0)
        return true;
      if (shape[i] != 1 && steps[i] != st)
        return false;
      check(st <= INT64_MAX / std::max<int64_t>(shape[i], 1),
            "stride overflow");
      st *= shape[i];
    }
    return true;
  }
  ScalarType scalar_type() const { return metadata.type; }
  int device() const { return metadata.ordinal; }
  int get_device() const { return device(); }
  TensorOptions options() const { return metadata; }
  Tensor narrow(int axis, int64_t start, int64_t length) const {
    check(axis >= 0 && axis < dim() && start >= 0 && length >= 0 &&
              start <= size(axis) && length <= size(axis) - start,
          "invalid narrow");
    auto t = *this;
    if (start) {
      check(pointer != nullptr, "cannot offset an unbacked descriptor");
      t.pointer = static_cast<char *>(pointer) +
                  start * stride(axis) * itemsize(scalar_type());
    }
    t.shape[axis] = length;
    return t;
  }
  void *data_ptr() const { return pointer; }
  template <class T> T *data_ptr() const { return static_cast<T *>(pointer); }
};
// Compatibility view consumer for native launches. Every buffer is supplied
// from the executor's explicit workspace recipe; this never plans or allocates.
struct ExecutionContext {
  cudaStream_t stream = nullptr;
  int device = 0;
  size_t next = 0;
  std::vector<Tensor> buffers;
};
inline thread_local ExecutionContext *active_context = nullptr;
inline cudaStream_t getCurrentCUDAStream(int = -1) {
  check(active_context != nullptr,
        "native call requires explicit execution context");
  return active_context->stream;
}
inline Tensor allocate(IntArrayRef shape, TensorOptions options, bool zero) {
  check(active_context != nullptr, "missing caller allocation context");
  auto &c = *active_context;
  Tensor t;
  t.shape = shape;
  t.metadata = options;
  t.present = true;
  t.steps.resize(shape.size());
  int64_t stride = 1;
  for (int i = int(shape.size()) - 1; i >= 0; --i) {
    check(shape[i] >= 0, "negative allocation size");
    t.steps[i] = stride;
    check(stride <= INT64_MAX / std::max<int64_t>(shape[i], 1),
          "allocation size overflow");
    stride *= shape[i];
  }
  check(t.numel() <= INT32_MAX,
        "allocation exceeds native signed indexing bound");
  t.buffer_index = int(c.next++);
  check(size_t(t.buffer_index) < c.buffers.size(),
        "native launch exceeds caller workspace buffer recipe");
  auto b = c.buffers[t.buffer_index];
  check(b.sizes() == shape && b.scalar_type() == options.type &&
            b.device() == options.ordinal && b.is_contiguous(),
        "buffer ", t.buffer_index,
        " shape/dtype/device/layout differs from allocation plan");
  t.pointer = b.pointer;
  if (zero && t.numel())
    cuda_check(cudaMemsetAsync(
        t.pointer, 0, size_t(t.numel()) * itemsize(options.type), c.stream));
  return t;
}
inline Tensor empty(IntArrayRef s, TensorOptions o) {
  return allocate(std::move(s), o, false);
}
inline Tensor zeros(IntArrayRef s, TensorOptions o) {
  return allocate(std::move(s), o, true);
}
inline Tensor empty_like(const Tensor &t) {
  return empty(t.sizes(), t.options());
}
inline Tensor empty_like(const Tensor &t, TensorOptions o) {
  return empty(t.sizes(), o);
}
namespace cuda {
class CUDAGuard {
  int old_ = 0;
  bool changed_ = false;

public:
  explicit CUDAGuard(int d) {
    cuda_check(cudaGetDevice(&old_));
    changed_ = d != old_;
    if (changed_)
      cuda_check(cudaSetDevice(d));
  }
  ~CUDAGuard() {
    if (changed_)
      cudaSetDevice(old_);
  }
  CUDAGuard(const CUDAGuard &) = delete;
};
using draft_native::getCurrentCUDAStream;
// cudaGetDeviceProperties costs over a millisecond on current drivers and the
// Draft launch path queries it once per phase (preparation, draft, routing,
// assembly, planning). Device properties are immutable for the life of the
// process, so serve them from a per-device cache.
inline const cudaDeviceProp *deviceProperties(int device) {
  static std::mutex mutex;
  static std::vector<cudaDeviceProp> cache;
  static std::vector<bool> filled;
  if (device < 0)
    throw std::invalid_argument("Draft queried a negative CUDA device ordinal");
  std::lock_guard<std::mutex> lock(mutex);
  if (size_t(device) >= cache.size()) {
    cache.resize(device + 1);
    filled.resize(device + 1, false);
  }
  if (!filled[device]) {
    cuda_check(cudaGetDeviceProperties(&cache[device], device));
    filled[device] = true;
  }
  return &cache[device];
}
inline const cudaDeviceProp *getCurrentDeviceProperties() {
  int d;
  cuda_check(cudaGetDevice(&d));
  return deviceProperties(d);
}
} // namespace cuda
// The Ada (SM89) kernel set serves every capability in [89, 120); SM120+
// devices use the Blackwell kernels.
inline bool ada_serves_device(const cudaDeviceProp *properties) {
  int architecture = properties->major * 10 + properties->minor;
  return architecture >= 89 && architecture < 120;
}
inline void require_ada_gpu() {
  auto properties = cuda::getCurrentDeviceProperties();
  check(ada_serves_device(properties),
        "Draft Ada kernels serve SM89 to SM119 (SM120+ uses the Blackwell kernels)");
}
} // namespace draft_native
#define DRAFT_CHECK(...) ::draft_native::check(__VA_ARGS__)
#define DRAFT_CUDA_CHECK(expr) ::draft_native::cuda_check(expr)
#define DRAFT_CUDA_KERNEL_LAUNCH_CHECK()                                      \
  do {                                                                         \
    DRAFT_CUDA_CHECK(cudaGetLastError());                                     \
  } while (0)
