# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metal 4 fused INT8 linear kernels for Apple silicon."""

from __future__ import annotations

import math
import threading

import torch


class FusedMPSUnsupportedError(RuntimeError):
    """Raised when the fused path cannot handle an input safely."""


_SHADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

kernel void quantize_rowwise_bf16(
    device const bfloat* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& columns [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
  threadgroup float maxima[32];
  float local_max = 0.0f;
  int base = int(row) * columns;
  for (int column = int(tid); column < columns; column += int(threads_per_group)) {
    local_max = max(local_max, abs(float(input[base + column])));
  }
  local_max = simd_max(local_max);
  if (lane == 0) {
    maxima[simdgroup] = local_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup == 0) {
    uint simdgroup_count = (threads_per_group + 31) / 32;
    float group_max = lane < simdgroup_count ? maxima[lane] : 0.0f;
    group_max = simd_max(group_max);
    if (lane == 0) {
      maxima[0] = max(group_max / 127.0f, 1.0e-30f);
      scales[row] = maxima[0];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  bfloat scale = bfloat(maxima[0]);
  for (int column = int(tid); column < columns; column += int(threads_per_group)) {
    bfloat scaled = bfloat(input[base + column] / scale);
    float value = rint(float(scaled));
    output[base + column] = int8_t(clamp(value, -128.0f, 127.0f));
  }
}

kernel void int8_linear_mpp_bf16(
    device int8_t* activation [[buffer(0)]],
    device int8_t* weight [[buffer(1)]],
    device float* row_scale [[buffer(2)]],
    device float* weight_scale [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    device bfloat* bias [[buffer(5)]],
    constant int& rows [[buffer(6)]],
    constant int& inner [[buffer(7)]],
    constant int& columns [[buffer(8)]],
    constant int& scale_count [[buffer(9)]],
    constant int& has_bias [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr int BM = 64;
  constexpr int BN = 32;
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      true,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> op;

  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> a(
      activation,
      dextents<int32_t, 2>(inner, rows),
      array<int32_t, 2>{1, inner});
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> b(
      weight,
      dextents<int32_t, 2>(inner, columns),
      array<int32_t, 2>{1, inner});

  auto a_tile = a.slice(0, tgid.y * BM);
  auto b_tile = b.slice(0, tgid.x * BN);
  auto accum = op.get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), int32_t>();

  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      accum[i] = 0;
    }
  }

  op.run(a_tile, b_tile, accum);

  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      auto index = accum.get_multidimensional_index(i);
      int column = int(tgid.x) * BN + index[0];
      int row = int(tgid.y) * BM + index[1];
      if (row < rows && column < columns) {
        float scale = row_scale[row] * weight_scale[scale_count == 1 ? 0 : column];
        bfloat value = bfloat(float(accum[i]) * scale);
        if (has_bias != 0) {
          value = bfloat(value + bias[column]);
        }
        output[row * columns + column] = value;
      }
    }
  }
}
"""


_compile_lock = threading.Lock()
_kernels: tuple[object, object] | None = None
_compile_error: Exception | None = None


def _get_kernels() -> tuple[object, object]:
    global _kernels, _compile_error
    if _kernels is not None:
        return _kernels
    if _compile_error is not None:
        raise FusedMPSUnsupportedError("Metal 4 fused kernels are unavailable") from _compile_error
    with _compile_lock:
        if _kernels is not None:
            return _kernels
        try:
            library = torch.mps.compile_shader(_SHADER)
            _kernels = (library.quantize_rowwise_bf16, library.int8_linear_mpp_bf16)
        except Exception as error:
            _compile_error = error
            raise FusedMPSUnsupportedError("Metal 4 fused kernels failed to compile") from error
    return _kernels


def int8_linear_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run row-wise activation quantization and INT8 linear in two Metal kernels."""
    if x.device.type != "mps" or x.dtype != torch.bfloat16:
        raise FusedMPSUnsupportedError("the fused path currently requires BF16 MPS activations")
    if weight.dtype != torch.int8 or weight.dim() != 2:
        raise FusedMPSUnsupportedError("the fused path requires a 2D INT8 weight")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Input and weight inner dimensions must match, got {x.shape[-1]} and {weight.shape[-1]}"
        )

    original_shape = x.shape
    inner = x.shape[-1]
    rows = x.numel() // inner
    columns = weight.shape[0]
    if rows == 0 or inner == 0 or columns == 0:
        raise FusedMPSUnsupportedError("empty matrices use the floating-point fallback")
    int32_max = torch.iinfo(torch.int32).max
    if rows > int32_max or inner > int32_max or columns > int32_max:
        raise FusedMPSUnsupportedError("matrix dimensions exceed Metal tensor index limits")

    x_2d = x.reshape(rows, inner).contiguous()
    weight_mps = weight.to(device=x.device).contiguous()
    scale_mps = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    if scale_mps.numel() not in (1, columns):
        raise ValueError(
            f"INT8 weight scale must be scalar or per-output-channel, got {tuple(scale_mps.shape)} "
            f"for weight shape {tuple(weight.shape)}"
        )
    bias_mps = (
        None
        if bias is None
        else bias.to(device=x.device, dtype=torch.bfloat16).reshape(-1).contiguous()
    )
    if bias_mps is not None and bias_mps.numel() != columns:
        raise ValueError(f"bias must have {columns} values, got {bias_mps.numel()}")

    quantize_kernel, linear_kernel = _get_kernels()
    quantized = torch.empty_like(x_2d, dtype=torch.int8)
    row_scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    output = torch.empty((rows, columns), device=x.device, dtype=torch.bfloat16)

    quantize_kernel(
        x_2d,
        quantized,
        row_scale,
        rows,
        inner,
        threads=rows * 256,
        group_size=256,
        arg_casts={3: "int32", 4: "int32"},
    )

    group_width = 4 * linear_kernel.thread_execution_width
    linear_kernel(
        quantized,
        weight_mps,
        row_scale,
        scale_mps,
        output,
        output if bias_mps is None else bias_mps,
        rows,
        inner,
        columns,
        scale_mps.numel(),
        int(bias_mps is not None),
        threads=(math.ceil(columns / 32) * group_width, math.ceil(rows / 64)),
        group_size=(group_width, 1),
        arg_casts={6: "int32", 7: "int32", 8: "int32", 9: "int32", 10: "int32"},
    )
    return output.reshape(*original_shape[:-1], columns)
