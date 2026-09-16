/*
 * Copyright 2026 Mixed Attention Project Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Backend-neutral address contract shared by architecture-owned kernels.
namespace mpa::attention {

constexpr int32_t kRasterPatchHeight = 8;
constexpr int32_t kRasterPatchWidth = 16;
constexpr int32_t kRasterPatchTokens =
    kRasterPatchHeight * kRasterPatchWidth;
constexpr int32_t kRasterAlignedPatchWidth = 8;
constexpr int32_t kRasterAlignedPatchTokens =
    kRasterPatchHeight * kRasterAlignedPatchWidth;
constexpr int32_t kRasterPhysicalStageTokens = 64;

// Compile-time-only profiling switches.  Production remains materialized;
// experiments toggle these in a dedicated rebuild so the inactive branch is
// eliminated and cannot add a runtime predicate to the accepted kernel.
constexpr bool kExperimentRawFp16Q = false;
constexpr bool kExperimentRawFp16K = false;

struct RasterTokenAddress {
  int64_t raw_token;
  bool valid;
};

/*
 * Arithmetic-only address contract for 8x16 and aligned 8x8 raster patches.
 *
 * The logical patch count is
 * F*ceil(H/patch_height)*ceil(W/patch_width). In particular, it is never
 * inferred as ceil(F*H*W/patch_tokens). An 8x16 logical patch owns two virtual
 * K64 stages; an 8x8 logical patch owns one. The validity bit, rather than
 * sequence-tail arithmetic, identifies virtual padding.
 */
struct Raster2DLayout {
  int64_t frames;
  int64_t height;
  int64_t width;
  int32_t patch_height;
  int32_t patch_width;
  int32_t patch_tokens;
  int32_t physical_stages_per_patch;
  int64_t patches_h;
  int64_t patches_w;
  int64_t patches_per_frame;
  int64_t patch_count;

  __host__ __device__ Raster2DLayout(
      int64_t frames_value,
      int64_t height_value,
      int64_t width_value,
      int32_t patch_width_value = kRasterPatchWidth)
      : frames(frames_value),
        height(height_value),
        width(width_value),
        patch_height(kRasterPatchHeight),
        patch_width(patch_width_value),
        patch_tokens(kRasterPatchHeight * patch_width_value),
        physical_stages_per_patch(
            patch_tokens / kRasterPhysicalStageTokens),
        patches_h((height_value + patch_height - 1) / patch_height),
        patches_w((width_value + patch_width - 1) / patch_width),
        patches_per_frame(patches_h * patches_w),
        patch_count(frames_value * patches_per_frame) {}

  __host__ __device__ RasterTokenAddress logical_token(
      int64_t logical_patch,
      int32_t local_token) const {
    if (logical_patch < 0 || logical_patch >= patch_count ||
        local_token < 0 || local_token >= patch_tokens) {
      return {-1, false};
    }

    const int64_t frame = logical_patch / patches_per_frame;
    const int64_t spatial_patch =
        logical_patch - frame * patches_per_frame;
    const int64_t patch_h = spatial_patch / patches_w;
    const int64_t patch_w = spatial_patch - patch_h * patches_w;
    const int32_t local_h = local_token / patch_width;
    const int32_t local_w = local_token - local_h * patch_width;
    const int64_t raster_h = patch_h * patch_height + local_h;
    const int64_t raster_w = patch_w * patch_width + local_w;
    if (frame >= frames || raster_h >= height || raster_w >= width) {
      return {-1, false};
    }
    return {
        frame * height * width + raster_h * width + raster_w,
        true};
  }

  __host__ __device__ RasterTokenAddress physical_stage_token(
      int64_t physical_stage,
      int32_t stage_token) const {
    if (physical_stage < 0 ||
        physical_stage >= physical_stages_per_patch * patch_count ||
        stage_token < 0 ||
        stage_token >= kRasterPhysicalStageTokens) {
      return {-1, false};
    }
    const int64_t logical_patch =
        physical_stage / physical_stages_per_patch;
    const int32_t stage = static_cast<int32_t>(
        physical_stage % physical_stages_per_patch);
    const int32_t local_token =
        stage * kRasterPhysicalStageTokens + stage_token;
    return logical_token(logical_patch, local_token);
  }
};

// Compatibility name for existing 8x16 callers. New SM120 code supplies an
// explicit width of eight for the native 64x64 routing specialization.
using Raster8x16Layout = Raster2DLayout;

enum class PackedPrecisionPhase : int32_t {
  kInt4 = 0,
  kInt8 = 1,
  kFp16 = 2,
  kFp4 = kInt4,
  kFp8 = kInt8,
};

struct PackedPhysicalStage {
  int32_t logical_id;
  int32_t physical_stage;
  bool valid;
};

/*
 * Read-only iterator over one router row with layout
 *
 *   [INT4 logical IDs][INT8 logical IDs][FP16 logical IDs][zero tail].
 *
 * It deliberately does not build an intermediate list or an expanded physical LUT.
 * Every counted logical ID yields its two same-precision physical stages in
 * order.  The caller owns the backing and count lifetime.
 */
struct PackedPhaseStageIterator {
  const int32_t* row_backing;
  int32_t logical_capacity;
  int32_t phase_base;
  int32_t phase_count;

  __host__ __device__ PackedPhaseStageIterator(
      const int32_t* row_backing_value,
      int32_t logical_capacity_value,
      int32_t fp8_count,
      int32_t fp16_count,
      int32_t fp4_count,
      PackedPrecisionPhase phase)
      : row_backing(row_backing_value),
        logical_capacity(logical_capacity_value),
        phase_base(0),
        phase_count(0) {
    if (phase == PackedPrecisionPhase::kInt4) {
      phase_base = 0;
      phase_count = fp4_count;
    } else if (phase == PackedPrecisionPhase::kInt8) {
      phase_base = fp4_count;
      phase_count = fp8_count;
    } else {
      phase_base = fp4_count + fp8_count;
      phase_count = fp16_count;
    }
  }

  __host__ __device__ int32_t iterations() const {
    return 2 * phase_count;
  }

  __host__ __device__ PackedPhysicalStage at(int32_t iteration) const {
    if (iteration < 0 || iteration >= iterations()) {
      return {-1, -1, false};
    }
    const int32_t logical_position = phase_base + iteration / 2;
    if (logical_position < 0 || logical_position >= logical_capacity) {
      return {-1, -1, false};
    }
    const int32_t logical_id = row_backing[logical_position];
    if (logical_id < 0 || logical_id >= logical_capacity) {
      return {logical_id, -1, false};
    }
    return {
        logical_id,
        2 * logical_id + iteration % 2,
        true};
  }
};

}  // namespace mpa::attention
