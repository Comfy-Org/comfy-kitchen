// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The dense INT8 attention kernels for head_dim 256 with __half output; see int8_attn.h.
#include "int8_attn.h"

namespace comfy::hip_backend::sage::int8_attn {
template void launch_int8_attn<256, __half>(const Int8AttnLaunch&);
}  // namespace comfy::hip_backend::sage::int8_attn
