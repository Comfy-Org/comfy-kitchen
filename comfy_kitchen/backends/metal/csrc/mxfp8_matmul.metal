// SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native MXFP8 GEMM via Metal TensorOps matmul2d + E8M0 scale planes.
// Computes C = A @ B.T for A (M,K), B (N,K) in kitchen row-major layout.
// Scales are unswizzled (M, K/32) / (N, K/32) E8M0 bytes.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int TILE_M = 32;
constant constexpr int TILE_N = 32;

using scales_plane =
    tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using mxfp8_tensor =
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_plane>;
using out_tensor = tensor<device half, dextents<int, 2>, tensor_inline>;

struct MxParams {
  uint M;
  uint N;
  uint K;
};

kernel void scaled_mm_mxfp8_tensorops(
    device uchar* a_data [[buffer(0)]],
    device uchar* a_scales [[buffer(1)]],
    device uchar* b_data [[buffer(2)]],
    device uchar* b_scales [[buffer(3)]],
    device half* c_data [[buffer(4)]],
    constant MxParams& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = int(p.M);
  const int N = int(p.N);
  const int K = int(p.K);
  const int row0 = int(tgid.y) * TILE_M;
  const int col0 = int(tgid.x) * TILE_N;
  if (row0 >= M || col0 >= N) {
    return;
  }

  array<int, 2> a_strides = {1, K};
  array<int, 2> b_strides = {1, K};
  array<int, 2> c_strides = {1, N};

  scales_plane a_sp(a_scales);
  scales_plane b_sp(b_scales);
  mxfp8_tensor A(a_data, dextents<int, 2>(K, M), a_strides, a_sp);
  mxfp8_tensor B(b_data, dextents<int, 2>(K, N), b_strides, b_sp);
  out_tensor C(c_data, dextents<int, 2>(N, M), c_strides);

  auto tA = A.slice(0, row0);
  auto tB = B.slice(0, col0);
  auto tC = C.slice(col0, row0);

  constexpr auto desc = matmul2d_descriptor(
      TILE_M, TILE_N, static_cast<int>(dynamic_extent),
      /*transpose_left=*/false,
      /*transpose_right=*/true);
  matmul2d<desc, execution_simdgroups<4>> op;
  op.run(tA, tB, tC);
}
