#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cstdint>

// Non-owning, allocation-free launch interface; outputs and workspace are caller-owned.
namespace anemoi::sm120::preparation {
enum class DType { F16, BF16, U8, I8, F32, I64, I32, Bool, E4M3 };
struct TensorView {
  void* data = nullptr;
  std::array<int64_t, 4> shape{};
  std::array<int64_t, 4> strides{};
  int ndim = 0;
  DType dtype = DType::U8;
  int device = -1;
  int64_t size(int i) const { return shape[i]; }
  int64_t stride(int i) const { return strides[i]; }
  int64_t numel() const {
    if (!data) return 0;
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return n;
  }
  template<class T = void> T* data_ptr() const { return static_cast<T*>(data); }
};
using SixOutputs = std::array<TensorView, 6>;
using H3Outputs = std::array<TensorView, 27>;
struct H3Options {
  int64_t prefix_tokens;
  int64_t query_block_size;
  bool has_nvfp4, has_int8, has_mxfp8, has_fp16;
  bool has_prefix_query_int8, has_maxpool;
};
// These low-level launchers require metadata validated by the native executor.
void prepare_mxfp8(TensorView query, TensorView key, TensorView value,
                   const SixOutputs& outputs, cudaStream_t stream);
void prepare_q64_nvfp4(TensorView query, TensorView key, TensorView value,
                      TensorView q_global_scale, TensorView k_global_scale,
                      TensorView v_global_scale, const SixOutputs& outputs,
                      cudaStream_t stream);
void prepare_q128_nvfp4(TensorView query, TensorView key, TensorView value,
                       TensorView q_global_scale, TensorView k_global_scale,
                       TensorView v_global_scale, const SixOutputs& outputs,
                       cudaStream_t stream);
void prepare_h3_sm120_operands(TensorView query, TensorView key, TensorView value,
    TensorView video_token_indices, TensorView video_slot_valid,
    TensorView video_valid_counts, H3Options options,
    TensorView q_global_scale, TensorView k_global_scale, TensorView v_global_scale,
    const H3Outputs& outputs, TensorView workspace, cudaStream_t stream);
} // namespace anemoi::sm120::preparation
