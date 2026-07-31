// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Codes for the optional activation a quantizer absorbs on the way in. Shared
// by the CUDA kernels and the nanobind layer; must match INPUT_ACT_TO_CODE in
// comfy_kitchen/backends/_activations.py.
#pragma once

namespace comfy {

// SwiGLU is the gated pair: the raw row is [gate | up] (2*K wide) and the
// activated row silu(gate) * up is K wide; the others are elementwise.
enum : int { kActNone = 0, kActGeluTanh = 1, kActSwiGLU = 2 };

}  // namespace comfy
