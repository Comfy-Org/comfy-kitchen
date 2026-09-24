// Adapted from Anemoi 4e85afba741bdeaf2d9486cab19cb76d3e7985a4; see LICENSE.
#include "../native_tensor.h"
#pragma once


#include <cstdint>

namespace mpa {

// The public API accepts physically contiguous BHSD or BSHD. Python exposes
// both to native code as a logical [B,H,S,D] tensor. A BSHD tensor therefore
// arrives as the metadata-only view with strides [S*H*D,D,H*D,1]. Restricting
// the accepted layouts to these two dense forms keeps vector alignment and
// prevents an accidental general-strided slow path.
inline bool is_supported_raw_bhsd_view(const draft_native::Tensor& tensor) {
  if (tensor.dim() != 4 || tensor.stride(3) != 1) {
    return false;
  }
  if (tensor.is_contiguous()) {
    return true;
  }
  const int64_t heads = tensor.size(1);
  const int64_t tokens = tensor.size(2);
  const int64_t head_dim = tensor.size(3);
  return tensor.stride(1) == head_dim &&
      tensor.stride(2) == heads * head_dim &&
      tensor.stride(0) == tokens * heads * head_dim;
}

// The released MiniMax-H3 integration receives BF16 Q/K/V as three projections
// split from one packed [S,H,3D] Ulysses buffer.  The stable public operator
// still rejects general strided inputs; a package-private Python scope admits
// only these exact objects.  Native pooling/preparation repeats the exact
// shape/stride/alignment check so direct extension calls remain fail-closed.
inline bool is_supported_raw_bhsd_input_view(const draft_native::Tensor& tensor) {
  if (is_supported_raw_bhsd_view(tensor)) {
    return true;
  }
  constexpr int64_t kBatch = 1;
  constexpr int64_t kHeads = 14;
  constexpr int64_t kVideoTokens = 37296;
  constexpr int64_t kOfficialPrefixTokens = 951;
  constexpr int64_t kFullTokens = kVideoTokens + kOfficialPrefixTokens;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kProjectionStride = 384;
  constexpr int64_t kTokenStride = kHeads * kProjectionStride;
  const int64_t tokens = tensor.dim() == 4 ? tensor.size(2) : 0;
  const bool is_released_h3_extent =
      tokens == kVideoTokens || tokens == kFullTokens;
  return tensor.dim() == 4 &&
      tensor.scalar_type() == draft_native::ScalarType::BFloat16 &&
      tensor.size(0) == kBatch && tensor.size(1) == kHeads &&
      is_released_h3_extent && tensor.size(3) == kHeadDim &&
      tensor.stride(0) == tokens * kTokenStride &&
      tensor.stride(1) == kProjectionStride &&
      tensor.stride(2) == kTokenStride && tensor.stride(3) == 1 &&
      reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0;
}

inline void check_supported_raw_bhsd_view(
    const draft_native::Tensor& tensor,
    const char* name) {
  DRAFT_CHECK(
      is_supported_raw_bhsd_view(tensor),
      name,
      " must be contiguous BHSD or a contiguous-BSHD logical view");
}

inline void check_supported_raw_bhsd_input_view(
    const draft_native::Tensor& tensor,
    const char* name) {
  DRAFT_CHECK(
      is_supported_raw_bhsd_input_view(tensor),
      name,
      " must be contiguous BHSD, a contiguous-BSHD logical view, or the "
      "private aligned H3 packed-input view");
}

}  // namespace mpa
