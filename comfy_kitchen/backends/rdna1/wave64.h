// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Force-included into the files built wave64 (CMakeLists.txt). HIP's __shfl* lower to
// __builtin_amdgcn_wave_shuffle, which crashes the gfx1010 code generator at wave64
// (LLVM 24, ROCm 10.1). ds_bpermute is what it lowers to at wave32, and the kernels
// only shuffle within 32 lanes.
#pragma once

#define __builtin_amdgcn_wave_shuffle(v, i) __builtin_amdgcn_ds_bpermute((i) << 2, v)
