// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Codes for the optional activation a quantizer absorbs on the way in. Shared
// by the CUDA kernels and the nanobind layer; must match INPUT_ACT_TO_CODE in
// comfy_kitchen/backends/_activations.py.
#pragma once

namespace comfy {

// SwiGLU and SiTU are gated pairs: the raw row is two K-wide halves, [gate | up]
// for silu(gate) * up and [hidden | gate] for SiTU's 4*tanh(gate/4)*sigmoid(gate) *
// 25*tanh(hidden/25), and the activated row is K wide. RmsNorm carries a weight
// pointer and eps.
enum : int { kActNone = 0, kActGeluTanh = 1, kActSwiGLU = 2, kActRmsNorm = 3, kActSiTU = 4 };

// Raw input columns per activated column.
constexpr int input_act_width(int act) { return (act == kActSwiGLU || act == kActSiTU) ? 2 : 1; }

}  // namespace comfy
